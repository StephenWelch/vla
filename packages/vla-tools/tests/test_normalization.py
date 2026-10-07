"""Split loading must preserve RGB preprocessing without leaking held-out states."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import torch
from pyarrow import parquet
from vla_tools.hooks import install_hooks


@pytest.mark.parametrize("imagenet", [True, False])
@pytest.mark.parametrize("backend", ["pyav", "torchcodec"])
def test_split_normalization(monkeypatch, tmp_path, imagenet, backend):
    from lerobot.datasets import lerobot_dataset
    from lerobot.utils.constants import IMAGENET_STATS

    camera = "observation.images.front"
    stats = {}
    for key, mean in [
        ("action", [2.0]),
        ("observation.state", [3.0]),
        (camera, [[[0.1]], [[0.2]], [[0.3]]]),
    ]:
        mean = np.asarray(mean)
        stats[key] = {
            "mean": mean,
            "std": np.ones_like(mean),
            "min": mean - 1,
            "max": mean + 1,
            "count": np.array([10]),
        }
    rows = []
    for i in range(2):
        row = {"episode_index": i}
        for key, fields in stats.items():
            for field, value in fields.items():
                row[f"stats/{key}/{field}"] = (
                    value + (100 if i and field == "mean" else 0)
                ).tolist()
        rows.append(row)
    path = tmp_path / "meta/episodes/chunk-000"
    path.mkdir(parents=True)
    parquet.write_table(pa.Table.from_pylist(rows), path / "file-000.parquet")
    instances = []

    class Dataset:
        image_transforms = None

        def __init__(self, *args, **kwargs):
            if "episodes" in kwargs:
                assert kwargs["video_backend"] == backend
            self.meta = SimpleNamespace(
                camera_keys=[camera],
                features={},
                fps=50,
                stats={camera: {k: torch.tensor(v) for k, v in IMAGENET_STATS.items()}},
            )
            instances.append(self)

    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", Dataset)
    trainer = SimpleNamespace(
        make_train_eval_datasets=lambda cfg: (Dataset(), None),
        save_checkpoint=lambda **kw: None,
        update_policy=lambda *a: None,
    )
    install_hooks(
        trainer,
        {
            "output": str(tmp_path),
            "split": {"train": [{"episode_index": 0}], "val": [{"episode_index": 1}]},
        },
    )
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            root=tmp_path,
            repo_id="test",
            use_imagenet_stats=imagenet,
            video_backend=backend,
        ),
        policy=SimpleNamespace(
            reward_delta_indices=None,
            action_delta_indices=None,
            observation_delta_indices=None,
        ),
    )
    train, _ = trainer.make_train_eval_datasets(cfg)
    assert instances[-1].meta.stats is train.meta.stats
    for key in ("action", "observation.state"):
        np.testing.assert_allclose(train.meta.stats[key]["mean"], stats[key]["mean"])
    for field in ("mean", "std"):
        expected = IMAGENET_STATS[field] if imagenet else stats[camera][field]
        np.testing.assert_allclose(train.meta.stats[camera][field], expected)
