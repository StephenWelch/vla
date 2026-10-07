"""Replay recorded actions through the policy-evaluation environment, without a policy."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vla_tools.config import parse_args
from vla_tools.tracking import write_json

from .config import (
    ABSOLUTE_ACTION,
    ABSOLUTE_GRIPPER_ACTION,
    ACTION,
    FIELDS,
    TASK,
    validate_action,
)
from .lerobot_env import OCBenchEnvConfig, OCBenchVectorEnv


@dataclass
class Config:
    source: Path
    output: Path
    episodes: tuple[int, ...] = (0, 2, 3)
    atol: float = 1e-3
    absolute_gripper: bool = False
    absolute_arm: bool = False
    checkpoint: Path | None = None


def run(config):
    if config.output.exists():
        raise FileExistsError(config.output)
    if not config.episodes or len(set(config.episodes)) != len(config.episodes):
        raise ValueError("Choose distinct raw episode IDs")
    if not np.isfinite(config.atol) or config.atol <= 0:
        raise ValueError("atol must be finite and positive")
    rows, arrays = [], []
    for episode in config.episodes:
        row = json.loads(
            (config.source / "raw" / f"episode-{episode:06}.json").read_text()
        )
        validate_action(row["action_profile"])
        if row["env_id"] != TASK:
            raise ValueError("Replay requires the supported stacking task")
        with np.load(config.source / "raw" / row["archive"]) as saved:
            data = {k: saved[k].copy() for k in saved.files}
        n = len(data["action"])
        if n == 0 or any(not np.isfinite(v).all() for v in data.values()):
            raise ValueError("Replay requires nonempty, finite trajectories")
        if n != row["length"] or any(len(data[f"sim/{k}"]) != n + 1 for k in FIELDS):
            raise ValueError(f"Invalid state/action alignment in episode {episode}")
        rows.append(row)
        arrays.append(data)
    if len({r["seed"] for r in rows}) != len(rows):
        raise ValueError("Replay requires distinct reset seeds")
    profile = ABSOLUTE_GRIPPER_ACTION if config.absolute_gripper else ACTION
    if config.absolute_arm:
        profile = ABSOLUTE_ACTION
    roundtrip_error = 0.0
    if config.checkpoint:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors

        profile = json.loads((config.checkpoint / "action_profile.json").read_text())
        validate_action(profile)
        policy_config = PreTrainedConfig.from_pretrained(config.checkpoint)
        pre, post = make_pre_post_processors(
            policy_config,
            str(config.checkpoint),
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
    replay_actions = []
    for data in arrays:
        actions = data["action"].copy()
        if profile in (ABSOLUTE_GRIPPER_ACTION, ABSOLUTE_ACTION):
            from .actions import absolute_targets

            actions = absolute_targets(
                actions, data["state"], absolute_arm=profile == ABSOLUTE_ACTION
            )
        if config.checkpoint:
            # Exercise saved processors with the same action-chunk shape as ACT.
            restored = []
            for start in range(0, len(actions), policy_config.chunk_size):
                chunk = actions[start : start + policy_config.chunk_size]
                padded = np.pad(
                    chunk,
                    ((0, policy_config.chunk_size - len(chunk)), (0, 0)),
                    mode="edge",
                )
                normalized = pre(
                    {
                        "action": torch.from_numpy(padded[None]),
                        "observation.state": torch.from_numpy(
                            data["state"][start][None]
                        ),
                    }
                )["action"]
                restored.append(post(normalized).numpy()[0, : len(chunk)])
            restored = np.concatenate(restored)
            if not np.isfinite(restored).all():
                raise ValueError("Saved action processors produced non-finite actions")
            roundtrip_error = max(
                roundtrip_error, float(np.max(np.abs(restored - actions)))
            )
            actions = restored
        replay_actions.append(actions)
    env = OCBenchVectorEnv(
        OCBenchEnvConfig(
            max_steps=max(r["length"] for r in rows), action_profile=profile
        ),
        len(rows),
    )
    report = {
        "status": "running",
        "source": str(config.source),
        "atol": config.atol,
        "scope": "Recorded actions through OCBenchVectorEnv including rendering; no learned policy. Saved processors included only when checkpoint is specified.",
        "action_profile": profile,
        "checkpoint": str(config.checkpoint) if config.checkpoint else None,
        "processor_roundtrip_max_abs_error": roundtrip_error
        if config.checkpoint
        else None,
        "processor_roundtrip_passed": roundtrip_error <= 1e-5
        if config.checkpoint
        else None,
        "episodes": [],
    }
    config.output.mkdir(parents=True)
    write_json(config.output / "report.json", report)
    try:
        initial = {k: np.stack([a[f"sim/{k}"][0] for a in arrays]) for k in FIELDS}
        obs, _ = env.reset(seed=[r["seed"] for r in rows], options={"states": initial})
        errors = [{k: 0.0 for k in (*FIELDS, "observation.state")} for _ in rows]
        first_bad = [None for _ in rows]
        finish = [None for _ in rows]
        for t in range(max(r["length"] for r in rows)):
            active = [
                i for i, r in enumerate(rows) if finish[i] is None and t < r["length"]
            ]
            for i in active:
                err = float(
                    np.max(np.abs(obs["observation.state"][i] - arrays[i]["state"][t]))
                )
                errors[i]["observation.state"] = max(
                    errors[i]["observation.state"], err
                )
                if (not np.isfinite(err) or err > config.atol) and first_bad[i] is None:
                    first_bad[i] = {
                        "frame": t,
                        "field": "observation.state",
                        "error": err,
                    }
            actions = np.zeros((len(rows), 7), dtype=np.float32)
            for i in active:
                actions[i] = replay_actions[i][t]
            obs, _, terminated, truncated, _ = env.step(actions)
            state = env.sim.snapshot()
            for i in active:
                for k in FIELDS:
                    err = float(
                        np.max(np.abs(state[k][i] - arrays[i][f"sim/{k}"][t + 1]))
                    )
                    errors[i][k] = max(errors[i][k], err)
                    if (not np.isfinite(err) or err > config.atol) and first_bad[
                        i
                    ] is None:
                        first_bad[i] = {"frame": t + 1, "field": k, "error": err}
                if terminated[i] or truncated[i] or t + 1 == rows[i]["length"]:
                    finish[i] = t + 1
                    env.done[i] = True
            if all(x is not None for x in finish):
                break
            if t % 100 == 0:
                write_json(
                    config.output / "progress.json",
                    {"frame": t + 1, "finished": finish},
                )
        for i, row in enumerate(rows):
            success = bool(env.native[i])
            report["episodes"].append(
                {
                    "episode_id": row["episode_id"],
                    "seed": row["seed"],
                    "recorded_steps": row["length"],
                    "replay_steps": finish[i],
                    "recorded_native_success": row["native_success"],
                    "recorded_audited_success": bool(
                        row["native_success"] and row["physical_valid"]
                    ),
                    "replay_native_success": success,
                    "replay_audited_success": bool(success and env.valid[i]),
                    "max_abs_error": errors[i],
                    "first_mismatch": first_bad[i],
                    "passed": first_bad[i] is None
                    and finish[i] == row["length"]
                    and success == row["native_success"]
                    and bool(success and env.valid[i])
                    == bool(row["native_success"] and row["physical_valid"]),
                }
            )
        report["passed"] = (
            all(r["passed"] for r in report["episodes"]) and roundtrip_error <= 1e-5
        )
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="error", error=repr(error))
        raise
    finally:
        write_json(config.output / "report.json", report)
        env.close()
    return report


def main():
    report = run(parse_args(Config))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
