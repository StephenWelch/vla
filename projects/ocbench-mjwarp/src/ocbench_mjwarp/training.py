"""OCBench contracts, action views and periodic simulation evaluation."""

from pathlib import Path


def prepare_views(datasets, stats, settings):
    from .actions import prepare_training_views

    mode = settings.get("action_mode", "delta")
    if mode != "delta" or settings.get("percentile_normalization"):
        prepare_training_views(
            datasets,
            stats,
            mode != "delta",
            settings.get("percentile_normalization", False),
            absolute_arm=mode == "absolute",
        )


def evaluate_rollout(
    settings,
    dataset,
    task,
    seeds,
    name,
    checkpoint_dir,
    step,
    policy,
    preprocessor,
    postprocessor,
):
    from . import evaluate as module
    from .lerobot_env import OCBenchEnvConfig

    output = Path(settings["output"])
    cfg = module.EvalConfig(
        checkpoint=checkpoint_dir / "pretrained_model",
        dataset=Path(settings["dataset"]),
        output=output / "eval" / f"{step:06d}" / name,
        episodes=len(seeds),
        batch_size=min(5, len(seeds)),
        seed=settings["seed"],
        env=task["env_id"],
        task_ids=(task["task_id"],),
        seeds=seeds,
        videos=1,
        max_steps=settings["eval_max_steps"],
        device=settings["device"],
    )
    env = OCBenchEnvConfig(
        task=task["env_id"],
        task_ids=[task["task_id"]],
        image_size=tuple(
            dataset.meta.features["observation.images.front"]["shape"][1:]
        ),
        rendering=settings.get("rendering"),
        action_profile=settings.get("action_profile"),
        max_steps=cfg.max_steps,
        device="cuda:0" if cfg.device == "cuda" else cfg.device,
    )
    metrics = module.evaluate_task(
        cfg,
        env,
        policy,
        preprocessor,
        postprocessor,
        task["task_id"],
    )
    return metrics


def validate_profiles(config):
    import json

    from .config import ABSOLUTE_ACTION, ABSOLUTE_GRIPPER_ACTION, ACTION
    from .profile import profiles

    rendering, saved = profiles(config.dataset, config.resume or config.policy)
    actions = {
        "delta": ACTION,
        "absolute_gripper": ABSOLUTE_GRIPPER_ACTION,
        "absolute": ABSOLUTE_ACTION,
    }[config.action_mode]
    if (config.resume or config.policy) and saved != actions:
        raise ValueError("Checkpoint action representation differs from training")
    manifest = json.loads((config.dataset / "manifest.json").read_text())
    if manifest["action_profile"] != ACTION:
        raise ValueError("Training action views require native delta source data")
    if (
        config.action_mode != "delta" or config.percentile_normalization
    ) and not config.validation_fraction:
        raise ValueError("Action views require a train/validation split")
    return rendering, actions


if __name__ == "__main__":
    from vla_tools.hooks import run_worker

    run_worker(prepare_views, evaluate_rollout)
