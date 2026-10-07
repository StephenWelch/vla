"""Typed OCBench evaluation using LeRobot's standard rollout/metrics tools."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

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
    wandb: WandbConfig = field(default_factory=WandbConfig)


def prepare_evaluation(config):
    from .lerobot_env import OCBenchEnvConfig
    from .profile import profiles

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
    )
    return env, {
        "config": asdict(config),
        "status": "running",
        "tasks": {},
        "versions": {},
        "checkpoint_sha256": hashlib.sha256(
            (config.checkpoint / "model.safetensors").read_bytes()
        ).hexdigest(),
        "success_definition": "Native termination with physical contact and numerical audits",
    }


def evaluate_task(config, env_config, policy, pre, post, task_id):
    """Run one task and merge ordered LeRobot metrics with contact diagnostics."""
    import torch
    from lerobot.envs import make_env, make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy

    task_config = replace(env_config, task_ids=[task_id])
    env = make_env(task_config, n_envs=min(config.batch_size, config.episodes))[
        config.env
    ][task_id]
    try:
        manifest = json.loads((config.dataset / "manifest.json").read_text())
        env.initial_states = {
            r["seed"]: config.dataset / r["replay"] for r in manifest["episodes"]
        }
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

        with torch.inference_mode(), policy_autocast(policy):
            metrics = eval_policy(
                env,
                policy,
                env_pre,
                env_post,
                pre,
                post,
                n_episodes=config.episodes,
                start_seed=config.seed,
                max_episodes_rendered=min(config.videos, config.episodes),
                videos_dir=config.output / f"task-{task_id}" / "videos",
            )
        # Worlds finish out of order; extra worlds in the last batch are omitted.
        records = {row["seed"]: row for row in env.records}
        for row in metrics["per_episode"]:
            if seed_map is not None:
                row["seed"] = seed_map[row["seed"]]
            row.update(records[row["seed"]])
        metrics["aggregated"].update(
            {
                f"pc_{name}": 100
                * sum(row[name] for row in metrics["per_episode"])
                / config.episodes
                for name in ("task_success", "contact_valid")
            }
        )
        return metrics
    finally:
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
