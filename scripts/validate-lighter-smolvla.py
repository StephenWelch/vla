"""Compare a fine-tuned SmolVLA's one-step actions with held-out demonstrations."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-root", type=Path, default=Path("E:/vla-smolvla/datasets/lighter_left_curated"))
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Directory containing model.safetensors and policy processor JSON files")
    parser.add_argument("--samples-per-episode", type=int, default=20)
    args = parser.parse_args()
    manifest = json.loads((args.curated_root / "manifest.json").read_text(encoding="utf-8"))
    dataset = LeRobotDataset(manifest["source_repo_id"] + "_val", root=args.curated_root / "val",
                             video_backend="pyav")
    if (list(dataset.meta.tasks.index) != [manifest["task"]] or
            any(episode["tasks"] != [manifest["task"]] for episode in dataset.meta.episodes)):
        raise ValueError("Validation task differs from dataset frames; recurate the dataset")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy.config, str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    groups = defaultdict(lambda: {"model": [], "hold": []})
    for episode, row in enumerate(manifest["splits"]["val"]):
        start = dataset.meta.episodes["dataset_from_index"][episode]
        end = dataset.meta.episodes["dataset_to_index"][episode]
        samples = np.linspace(start, end - 1, min(args.samples_per_episode, end - start), dtype=int)
        group = "recovery" if row["recovery"] else "clean"
        for index in samples:
            frame = dataset[int(index)]
            image = frame["observation.images.left_wrist_cam"]
            image = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            state = frame["observation.state"].numpy().astype(np.float32)
            target = frame["action"].numpy().astype(np.float32)
            observation = prepare_observation_for_inference(
                {"observation.state": state, "observation.images.camera2": image},
                torch.device("cuda"), manifest["task"], "so101_follower",
            )
            policy.reset()
            with torch.inference_mode():
                proposed = post(policy.select_action(pre(observation))).squeeze(0).cpu().numpy()[:6]
            groups[group]["model"].append(float(np.mean((proposed - target) ** 2)))
            groups[group]["hold"].append(float(np.mean((state - target) ** 2)))
    for group in ("clean", "recovery"):
        scores = groups[group]
        print(f"{group}: {len(scores['model'])} sampled frames, "
              f"action MSE {np.mean(scores['model']):.3f}, hold-state MSE {np.mean(scores['hold']):.3f}")


if __name__ == "__main__":
    main()
