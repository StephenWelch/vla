"""Leakage and checkpoint-selection regression checks."""

import json
from typing import ClassVar

import pytest
from vla_tools.hooks import retain_checkpoints, scene_split


def write_manifest(root, seeds, fingerprints=None):
    rows = [
        {
            "episode_index": i,
            "env_id": "cube-single-v0",
            "task_id": 1,
            "seed": seed,
            "randomization": {"initial_state_fingerprint": (fingerprints or seeds)[i]},
        }
        for i, seed in enumerate(seeds)
    ]
    (root / "manifest.json").write_text(json.dumps({"episodes": rows}))


def test_variants_and_state_aliases_stay_together(tmp_path):
    write_manifest(tmp_path, [1, 1, 2, 3, 4, 5], ["a", "a", "a", "b", "c", "d"])
    split = scene_split(tmp_path)
    assert split == scene_split(tmp_path)
    for name in ("train", "val"):
        ids = {r["episode_index"] for r in split[name]}
        assert len(ids & {0, 1, 2}) in (0, 3)
    assert not {r["seed"] for r in split["train"]} & {r["seed"] for r in split["val"]}


def test_same_scene_cannot_validate(tmp_path):
    write_manifest(tmp_path, [2026, 2026])
    with pytest.raises(ValueError, match="independent scenes"):
        scene_split(tmp_path)


def test_checkpoint_best_and_latest_retention(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    first = checkpoints / "001000"
    first.mkdir(parents=True)
    report = {"step": 1000, "val/rollout": {"pc_success": 50}, "val/probe": {"loss": 1}}
    retain_checkpoints(tmp_path, first, report)
    second = checkpoints / "002000"
    second.mkdir()
    retain_checkpoints(
        tmp_path, second, {**report, "step": 2000, "val/probe": {"loss": 2}}
    )
    assert first.exists() and second.exists()
    third = checkpoints / "003000"
    third.mkdir()
    retain_checkpoints(
        tmp_path, third, {**report, "step": 3000, "val/probe": {"loss": 0.5}}
    )
    assert third.exists() and not first.exists() and not second.exists()


def test_loss_probe_weights_partial_batches():
    from types import SimpleNamespace

    import torch
    from vla_tools.hooks import loss_probe

    class Dataset:
        episodes: ClassVar = [0]
        absolute_to_relative_idx: ClassVar = dict(enumerate(range(5)))
        meta = SimpleNamespace(
            episodes=[{"dataset_from_index": 0, "dataset_to_index": 5}], camera_keys=[]
        )

        def __getitem__(self, index):
            return {
                "action": torch.tensor([float(index + 1)]),
                "action_is_pad": torch.tensor([False]),
            }

    class Policy:
        config = SimpleNamespace(type="smolvla")

        def forward(self, batch):
            return batch["action"].mean(), {}

    metrics = loss_probe(Policy(), lambda x: x, Dataset(), 5, 2, 1000)
    assert metrics["samples"] == 5
    assert metrics["loss"] == pytest.approx(3)


@pytest.mark.parametrize("overfit", [False, True])
def test_checkpoint_probe_preserves_rng_mode_and_queues(monkeypatch, tmp_path, overfit):
    import random
    from collections import deque
    from types import SimpleNamespace

    import numpy as np
    import pyarrow as pa
    import torch
    from lerobot.datasets import lerobot_dataset
    from pyarrow import parquet
    from vla_tools.hooks import install_hooks

    stats = {
        "action": {
            "min": np.array([0.0]),
            "max": np.array([2.0]),
            "mean": np.array([1.0]),
            "std": np.array([1.0]),
            "count": np.array([2]),
        }
    }
    rows = [
        {
            "dataset_from_index": i * 2,
            "dataset_to_index": (i + 1) * 2,
            **{
                f"stats/action/{key}": value.tolist()
                for key, value in stats["action"].items()
            },
        }
        for i in range(2)
    ]
    metadata = tmp_path / "meta/episodes/chunk-000"
    rows[1].update(
        {
            "stats/action/min": [100.0],
            "stats/action/max": [102.0],
            "stats/action/mean": [101.0],
        }
    )
    metadata.mkdir(parents=True)
    parquet.write_table(
        pa.Table.from_pylist(
            [{**row, "episode_index": i} for i, row in enumerate(rows)]
        ),
        metadata / "file-000.parquet",
    )

    class Dataset:
        image_transforms = None
        instances: ClassVar = []

        def __init__(self, *args, episodes=None, **kwargs):
            self.instances.append(self)
            self.episodes = episodes
            self.meta = SimpleNamespace(
                episodes=rows, camera_keys=[], features={}, fps=20
            )
            self.absolute_to_relative_idx = {
                i: i - episodes[0] * 2
                for i in range(episodes[0] * 2, episodes[0] * 2 + 2)
            }

        def __getitem__(self, index):
            return {"action": torch.tensor([float(index)])}

    class Policy(torch.nn.Module):
        config = SimpleNamespace(
            type="smolvla",
            reward_delta_indices=None,
            action_delta_indices=None,
            observation_delta_indices=None,
        )

        def __init__(self):
            super().__init__()
            self._queues = {"action": deque([torch.tensor([7.0])])}

        def forward(self, batch):
            random.random()
            np.random.random()
            return torch.rand(1).mean(), {}

        def reset(self):
            self._queues = {}

    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", Dataset)
    trainer = SimpleNamespace(
        make_train_eval_datasets=lambda cfg: (Dataset(episodes=[0]), None),
        save_checkpoint=lambda **kwargs: None,
        update_policy=lambda *a: None,
    )
    settings = {
        "output": str(tmp_path),
        "split": {
            "train": [{"episode_index": 0}],
            "val": [] if overfit else [{"episode_index": 1}],
        },
        "seed": 1000,
        "probe_frames": 2,
        "batch_size": 2,
        "rollout_eval_freq": 4,
        "steps": 4,
    }
    install_hooks(trainer, settings)
    policy = Policy()
    train, _ = trainer.make_train_eval_datasets(
        SimpleNamespace(
            dataset=SimpleNamespace(
                repo_id="test",
                root=tmp_path,
                use_imagenet_stats=False,
                video_backend="pyav",
            ),
            policy=policy.config,
        )
    )
    assert train.meta.stats["action"]["mean"] == pytest.approx([1.0])
    assert train.meta.stats is Dataset.instances[-1].meta.stats
    checkpoint = tmp_path / "checkpoints/000002"
    checkpoint.mkdir(parents=True)
    before = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all(),
    )
    trainer.save_checkpoint(
        step=2, policy=policy, checkpoint_dir=checkpoint, preprocessor=lambda x: x
    )
    report = json.loads((tmp_path / "metrics/000002.json").read_text())
    assert ("val/probe" in report) != overfit
    assert policy.training
    assert policy._queues["action"][0].item() == 7
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
    assert all(
        torch.equal(a, b)
        for a, b in zip(torch.cuda.get_rng_state_all(), before[3], strict=True)
    )
    trainer.save_checkpoint(
        step=2, policy=policy, checkpoint_dir=checkpoint, preprocessor=lambda x: x
    )
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 1


def test_single_episode_overfit_is_not_validation(tmp_path):
    write_manifest(tmp_path, [1, 2])
    split = scene_split(tmp_path, fraction=0, episodes=[1], overfit=True)
    assert split["method"] == "single_episode_overfit"
    assert [r["episode_index"] for r in split["train"]] == [1]
    assert split["val"] == []
    with pytest.raises(ValueError, match="exactly one"):
        scene_split(tmp_path, fraction=0, episodes=[0, 1], overfit=True)
    with pytest.raises(ValueError, match="exactly one"):
        scene_split(tmp_path, episodes=[1], overfit=True)
