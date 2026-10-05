"""Offline first-action prediction checks on recorded LeRobot episodes."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from vla_tools.config import parse_args
from vla_tools.policy import load_policy, runtime_environment


@dataclass
class EvalConfig:
    """Evaluate a checkpoint on recorded frames; this does not run the simulator."""

    dataset: Path
    checkpoint: Path
    output: Path
    repo_id: str | None = None
    samples_per_episode: int = 20
    seed: int = 1000
    device: str = "cuda"
    hf_home: Path | None = None


def evaluate(config):
    if config.samples_per_episode < 1:
        raise ValueError("samples_per_episode must be positive")
    os.environ.update(runtime_environment(config.hf_home))
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.random_utils import set_seed

    manifest_path = config.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    from ogbench_mjwarp.profile import ogbench_profile

    ogbench_profile(config.dataset, config.checkpoint)
    repo_id = config.repo_id or manifest.get("repo_id")
    if not repo_id:
        raise ValueError("Pass --repo-id for datasets without repo_id in manifest.json")
    checkpoint_config = json.loads((config.checkpoint / "config.json").read_text())
    set_seed(config.seed)
    dataset = LeRobotDataset(repo_id, root=config.dataset, video_backend="pyav")
    policy, pre, post = load_policy(config.checkpoint, config.device)
    errors, targets, records = [], [], []
    for episode in range(dataset.num_episodes):
        start = int(dataset.meta.episodes["dataset_from_index"][episode])
        end = int(dataset.meta.episodes["dataset_to_index"][episode])
        indices = np.linspace(
            start, end - 1, min(config.samples_per_episode, end - start), dtype=int
        )
        for index in indices:
            frame = dataset[int(index)]
            target = frame["action"].numpy()
            observation = {
                k: v
                for k, v in frame.items()
                if k.startswith("observation.") or k == "task"
            }
            policy.reset()  # Evaluate the first action of a fresh chunk at every sample.
            with torch.inference_mode():
                proposed = (
                    post(policy.select_action(pre(observation)))
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
            if proposed.shape != target.shape or not np.isfinite(proposed).all():
                raise ValueError(
                    "Prediction is nonfinite or has the wrong action shape"
                )
            errors.append(proposed - target)
            targets.append(target)
            records.append(
                {
                    "episode": episode,
                    "frame": int(index),
                    "prediction": proposed.tolist(),
                    "target": target.tolist(),
                }
            )
    errors, targets = np.asarray(errors), np.asarray(targets)
    report = {
        "policy_type": checkpoint_config["type"],
        "checkpoint": str(config.checkpoint.resolve()),
        "dataset": str(config.dataset.resolve()),
        "seed": config.seed,
        "samples": len(records),
        "action_names": dataset.meta.features["action"].get("names"),
        "mae": float(np.abs(errors).mean()),
        "mse": float(np.square(errors).mean()),
        "per_action_mae": np.abs(errors).mean(0).tolist(),
        "zero_action_mae": float(np.abs(targets).mean()),
        "zero_action_mse": float(np.square(targets).mean()),
        "records": records,
        "scope": "Recorded-frame first-action prediction only. If these episodes were used for training, this is an in-sample fit check; it does not measure task success or generalization.",
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(report, indent=2) + "\n")
    return {k: v for k, v in report.items() if k != "records"}


def main():
    config = parse_args(EvalConfig)
    print(json.dumps(evaluate(config), indent=2))


if __name__ == "__main__":
    main()
