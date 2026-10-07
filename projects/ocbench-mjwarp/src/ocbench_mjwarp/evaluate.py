"""Typed OCBench evaluation using LeRobot's standard rollout/metrics tools."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

from vla_tools.config import parse_args
from vla_tools.policy import load_policy, runtime_environment
from vla_tools.tracking import Tracker, WandbConfig, evaluation_log


@dataclass
class EvalConfig:
    checkpoint: Path
    output: Path
    dataset: Path
    episodes: int = 10
    batch_size: int = 5
    seed: int = 2027
    env: str = "block-double-task2-v0"
    task_ids: tuple[int, ...] = (2,)
    max_steps: int = 2500
    videos: int = 2
    device: str = "cuda"
    hf_home: Path | None = None
    seeds: list[int] | None = None
    observation_compression: Literal["dataset", "none"] = "dataset"
    rollout_backend: Literal["auto", "lerobot", "chunked"] = "auto"
    observation_decoder: Literal["pyav", "nvdec"] = "pyav"
    render_batch_frames: Literal[1, 2, 4] = 1
    profile: bool = False
    wandb: WandbConfig = field(default_factory=WandbConfig)


def prepare_evaluation(config):
    from .lerobot_env import OCBenchEnvConfig
    from .profile import profiles
    from .video_codec import dataset_encoding

    rendering, actions = profiles(config.dataset, config.checkpoint)
    if config.output.exists():
        raise FileExistsError(config.output)
    if min(config.episodes, config.batch_size, config.max_steps) < 1:
        raise ValueError("Positive evaluation budgets required")
    if config.seeds is not None and len(config.seeds) != config.episodes:
        raise ValueError("Expected one reset seed per episode")
    env = OCBenchEnvConfig(
        task=config.env,
        rendering=rendering,
        action_profile=actions,
        max_steps=config.max_steps,
        encoding=dataset_encoding(config.dataset, config.observation_compression),
    )
    return env, {
        "config": asdict(config),
        "status": "running",
        "tasks": {},
        "versions": {},
        "observation_encoding": env.encoding,
        "checkpoint_sha256": hashlib.sha256(
            (config.checkpoint / "model.safetensors").read_bytes()
        ).hexdigest(),
        "success_definition": "Native termination with physical contact and numerical audits",
    }


def evaluate_task(config, env_config, policy, pre, post, task_id, *, trace=None):
    """Run one task and merge ordered LeRobot metrics with contact diagnostics."""
    import torch
    from lerobot.envs import make_env, make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy

    from .eval_profile import EvaluationProfile
    from .eval_video import RolloutRecorder, record_environment
    from .rollout import chunked_evaluate, device_postprocessor, resolve_backend
    from .video_codec import dataset_encoding

    backend = resolve_backend(config.rollout_backend, policy)
    if config.render_batch_frames not in (1, 2, 4):
        raise ValueError("Temporal render batch must be 1, 2 or 4")
    if backend == "lerobot" and (
        config.render_batch_frames != 1 or config.observation_decoder != "pyav"
    ):
        raise ValueError("Temporal batching and NVDEC require rollout_backend=chunked")
    if (
        config.observation_decoder == "nvdec"
        and config.observation_compression == "none"
    ):
        raise ValueError("NVDEC requires compressed observations")

    task_config = replace(
        env_config,
        task_ids=[task_id],
        encoding=dataset_encoding(config.dataset, config.observation_compression),
    )
    # Two NVENC sessions per world; use the same per-process cap as GPU export.
    batch_size = min(config.batch_size, config.episodes)
    if task_config.encoding is not None:
        batch_size = min(batch_size, 4)
    if backend == "chunked":
        from .device_env import DeviceEnvironment

        env = DeviceEnvironment(task_config, batch_size)
    else:
        env = make_env(task_config, n_envs=batch_size)[config.env][task_id]
    recorder = None
    profile = EvaluationProfile(config.profile)
    try:
        if config.videos > 0:
            recorder = RolloutRecorder(
                config.output / f"task-{task_id}" / "videos",
                min(config.videos, config.episodes),
            )
        manifest = json.loads((config.dataset / "manifest.json").read_text())
        env.initial_states = {
            r["seed"]: config.dataset / r["replay"] for r in manifest["episodes"]
        }
        replay_rows = {
            r["seed"]: r
            for r in manifest["episodes"]
            if r.get("seed_kind") == "replay_id"
        }
        if replay_rows:
            seeds = config.seeds
            if seeds is None:
                seeds = [
                    r["seed"]
                    for r in manifest["episodes"]
                    if r["dataset_split"] == "val"
                ][: config.episodes]
            if len(seeds) != config.episodes or any(
                s not in replay_rows for s in seeds
            ):
                raise ValueError(
                    "Imported evaluation needs one recorded replay ID per episode; default selection uses the official validation split"
                )
            config = replace(config, seeds=seeds)
        seed_map = None
        if config.seeds is not None:
            # LeRobot enumerates consecutive seed IDs; map those IDs to the fixed suite.
            seed_map = dict(
                zip(
                    range(config.seed, config.seed + config.episodes),
                    config.seeds,
                    strict=True,
                )
            )
            for index in range(config.episodes, config.episodes + env.num_envs):
                extra = max(config.seeds) + index + 1
                seed_map[config.seed + index] = extra
            reset = env.reset

            def reset_suite(*, seed=None, options=None):
                return reset(seed=[seed_map[int(s)] for s in seed], options=options)

            env.reset = reset_suite
        env_pre, env_post = make_env_pre_post_processors(task_config, policy.config)
        from vla_tools.preprocessing import policy_autocast

        if trace is not None:
            import numpy as np

            method = "advance" if backend == "chunked" else "step"
            advance = getattr(env, method)

            def traced(action):
                result = advance(action)
                trace.append(
                    {
                        "action": action.detach().cpu().numpy().copy()
                        if torch.is_tensor(action)
                        else np.array(action, copy=True),
                        "qpos": env.sim.data.qpos.numpy().copy(),
                        "done": env.done.copy(),
                    }
                )
                return result

            setattr(env, method, traced)

        with torch.inference_mode(), policy_autocast(policy):
            if backend == "chunked":
                post = device_postprocessor(post, config.device)
                metrics = chunked_evaluate(
                    config, env, policy, env_pre, env_post, pre, post, recorder, profile
                )
            else:
                if recorder:
                    record_environment(env, recorder)
                env.reset = profile.wrap(env.reset, "initialization")
                env.step = profile.wrap(env.step, "reference_step")
                env.observation = profile.wrap(env.observation, "observation")
                select_action = policy.select_action
                policy.select_action = profile.wrap(select_action, "inference")
                try:
                    metrics = eval_policy(
                        env,
                        policy,
                        env_pre,
                        env_post,
                        profile.wrap(pre, "preprocessing"),
                        profile.wrap(post, "action_processing"),
                        n_episodes=config.episodes,
                        start_seed=config.seed,
                        max_episodes_rendered=0,
                    )
                finally:
                    policy.select_action = select_action
        if recorder:
            with profile.stage("video_finalization"):
                recorder.close()
            metrics["video_paths"] = [str(p) for p in recorder.paths]
        # Worlds finish out of order; extra worlds in the last batch are omitted.
        records = {row["seed"]: row for row in env.records}
        for row in metrics["per_episode"]:
            if seed_map is not None:
                row["seed"] = seed_map[row["seed"]]
            row.update(records[row["seed"]])
            if row["seed"] in replay_rows:
                row["seed_kind"] = "replay_id"
                row["source"] = replay_rows[row["seed"]]["source"]
        metrics["aggregated"].update(
            {
                f"pc_{name}": 100
                * sum(row[name] for row in metrics["per_episode"])
                / config.episodes
                for name in ("task_success", "contact_valid")
            }
        )
        metrics["observation_encoding"] = task_config.encoding
        metrics["environment_batch_size"] = batch_size
        metrics["rollout_backend"] = backend
        metrics["observation_decoder"] = config.observation_decoder
        metrics["render_batch_frames"] = config.render_batch_frames
        metrics["performance"] = profile.report(
            sum(row["steps"] for row in metrics["per_episode"])
        )
        if recorder:
            metrics["performance"].update(
                video_queue_peak=recorder.peak_queue_frames,
                video_backpressure_seconds=recorder.backpressure_seconds,
            )
        return metrics
    finally:
        try:
            if recorder:
                recorder.close()
        finally:
            profile.close()
            env.close()


def evaluate(config):
    os.environ.update(runtime_environment(config.hf_home))
    os.environ.setdefault("MUJOCO_GL", "egl")
    from lerobot.utils.random_utils import set_seed
    from vla_tools.tracking import write_json

    env_config, report = prepare_evaluation(config)
    config.output.mkdir(parents=True)
    report_path = config.output / "eval_info.json"
    write_json(report_path, report)
    parent = config.checkpoint / "tracking.json"
    lineage = json.loads(parent.read_text()) if parent.exists() else {}
    if config.wandb.group is None:
        config.wandb.group = lineage.get("group")
    tracker = Tracker(
        config.output,
        config.wandb,
        "eval",
        {
            "evaluation": report["config"],
            "versions": report["versions"],
            "parent_run": lineage,
            "checkpoint_sha256": report["checkpoint_sha256"],
        },
    )
    try:
        set_seed(config.seed)
        policy, pre, post = load_policy(config.checkpoint, config.device)
        # One task at a time keeps GPU memory bounded.
        for task_id in config.task_ids:
            report["tasks"][str(task_id)] = evaluate_task(
                config, env_config, policy, pre, post, task_id
            )
            write_json(report_path, report)
        report["status"] = "complete"
        evaluation_log(tracker, report, config.output)
    except BaseException:
        report["status"] = "failed_or_interrupted"
        raise
    finally:
        write_json(report_path, report)
        try:
            tracker.log({}, reports=[report_path, config.checkpoint / "rendering.json"])
        finally:
            tracker.finish(report["status"] != "complete")
    return {task: metrics["aggregated"] for task, metrics in report["tasks"].items()}


def main():
    print(json.dumps(evaluate(parse_args(EvalConfig)), indent=2))


if __name__ == "__main__":
    main()
