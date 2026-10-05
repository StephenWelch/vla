"""Typed OGBench evaluation using LeRobot's standard rollout/metrics tools."""

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
    env: str = "cube-single-v0"
    task_ids: tuple[int, ...] = (1,)
    max_steps: int = 250
    videos: int = 2
    device: str = "cuda"
    hf_home: Path | None = None
    seeds: list[int] | None = None
    wandb: WandbConfig = field(default_factory=WandbConfig)


def prepare_evaluation(config):
    """Validate inputs and collect the environment contract and run provenance."""
    import torch

    from ogbench_mjwarp.io import versions
    from ogbench_mjwarp.lerobot_env import OGBenchEnvConfig

    if (
        min(config.episodes, config.batch_size, config.max_steps) < 1
        or min(config.videos, config.seed) < 0
    ):
        raise ValueError("Positive budgets and nonnegative video count/seed required")
    if config.output.exists():
        raise FileExistsError(f"Choose a new output directory: {config.output}")
    if config.seeds is not None and (
        len(config.seeds) != config.episodes
        or len(set(config.seeds)) != len(config.seeds)
        or min(config.seeds) < 0
    ):
        raise ValueError(
            "Explicit seeds must be unique, nonnegative, and match episodes"
        )
    if torch.device(config.device).type != "cuda":
        raise ValueError("MJWarp evaluation requires a CUDA device")
    if not (config.checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing checkpoint: {config.checkpoint}")
    metadata = json.loads((config.dataset / "meta/info.json").read_text())
    manifest = json.loads((config.dataset / "manifest.json").read_text())
    from ogbench_mjwarp.profile import ogbench_profile

    rendering = ogbench_profile(config.dataset, config.checkpoint)
    if rendering is None:
        raise ValueError("OGBench evaluation requires a v2 OGBench dataset")
    features = metadata["features"]
    images = [
        features[f"observation.images.{view}"]["shape"] for view in ("front", "wrist")
    ]
    if images[0] != images[1] or len(images[0]) != 3 or images[0][0] != 3:
        raise ValueError("Front/wrist cameras must have matching RGB CHW shapes")
    policy_config = json.loads((config.checkpoint / "config.json").read_text())
    if policy_config["input_features"]["observation.state"]["shape"] != [
        18
    ] or policy_config["output_features"]["action"]["shape"] != [5]:
        raise ValueError("Checkpoint must use OGBench's 18-state/5-action contract")
    processor = json.loads((config.checkpoint / "policy_preprocessor.json").read_text())
    rename = {}
    for step in processor["steps"]:
        if step["registry_name"] == "rename_observations_processor":
            rename = step["config"]["rename_map"]
    for view in ("front", "wrist"):
        key = f"observation.images.{view}"
        feature = policy_config["input_features"].get(rename.get(key, key))
        if (
            feature is None
            or feature["type"] != "VISUAL"
            or feature["shape"] != images[0]
        ):
            raise ValueError(f"Checkpoint camera mapping/shape does not match {key}")
    env_config = OGBenchEnvConfig(
        rendering=rendering,
        task=config.env,
        task_ids=list(config.task_ids),
        image_size=tuple(images[0][1:]),
        max_steps=config.max_steps,
        device="cuda:0" if config.device == "cuda" else config.device,
    )
    with (config.checkpoint / "model.safetensors").open("rb") as weights:
        checkpoint_hash = hashlib.file_digest(weights, "sha256").hexdigest()
    report = {
        "status": "running",
        "config": asdict(config),
        "versions": versions(),
        "checkpoint_sha256": checkpoint_hash,
        "training_reset_seeds": sorted({row["seed"] for row in manifest["episodes"]}),
        "success_definition": "Task reached with finite physics and no contact tolerance violations during the episode",
        "contact_thresholds_m": {
            "nonpad": env_config.max_nonpad_penetration,
            "all_robot": env_config.max_penetration,
        },
        "tasks": {},
    }
    return env_config, report


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
        with torch.inference_mode():
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

    from ogbench_mjwarp.io import write_json

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
