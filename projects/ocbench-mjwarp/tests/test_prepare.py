"""Preparation reuse, episode identity and comparison selections."""

import json
from dataclasses import replace

import pytest
from ocbench_mjwarp.compare import Config, prepare
from ocbench_mjwarp.config import ACTION
from ocbench_mjwarp.prepare import PrepareConfig, prepare_dataset, validate_prepared
from ocbench_mjwarp.train import TrainConfig
from vla_tools.config import parse_args


def prepared(root):
    (root / "meta").mkdir(parents=True)
    (root / "replay").mkdir()
    rows = []
    for i, (success, split, retry) in enumerate(
        [
            (True, "train", 0),
            (True, "train", 1),
            (False, "train", 0),
            (True, "val", 1),
            (False, "val", 0),
        ]
    ):
        replay = f"replay/{i}.npz"
        (root / replay).write_bytes(b"replay")
        rows.append(
            {
                "episode_index": i,
                "episode_id": i,
                "length": 2,
                "source_root": "raw",
                "physical_valid": True,
                "native_success": success,
                "dataset_split": split,
                "seed": i,
                "env_id": "block-double-task2-v0",
                "task_id": 2,
                "replay": replay,
                "randomization": {
                    "plans": [{"num_pick_retries": retry, "is_mistake": 0}]
                },
            }
        )
    manifest = {
        "format": "ocbench-mjwarp-1",
        "repo_id": "local/prepared",
        "action_profile": ACTION,
        "rendering": {"revision": 2},
        "episodes": rows,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "meta/info.json").write_text(
        json.dumps({"total_episodes": 5, "total_frames": 10})
    )
    return manifest


def test_existing_dataset_comparisons_share_unfiltered_holdout(tmp_path):
    dataset = tmp_path / "dataset"
    prepared(dataset)
    config = Config(dataset=dataset, output=tmp_path / "outcomes")
    paths = prepare(config)
    specs = [parse_args(TrainConfig, ["--config", str(path)]) for path in paths]
    assert [s.dataset for s in specs] == [dataset, dataset]
    assert specs[0].episodes == [0, 3, 4]
    assert specs[1].episodes == [0, 1, 2, 3, 4]
    assert not (config.output / "dataset").exists()
    action_config = replace(
        config, output=tmp_path / "actions", comparison="actions", exclude_mistakes=True
    )
    paths = prepare(action_config)
    action_specs = [parse_args(TrainConfig, ["--config", str(path)]) for path in paths]
    assert [s.action_mode for s in action_specs] == ["absolute", "delta"]
    assert action_specs[0].episodes == action_specs[1].episodes == [0, 3, 4]
    reports = [
        json.loads((root / "comparison.json").read_text())
        for root in (config.output, action_config.output)
    ]
    assert reports[0]["holdout_sha256"] == reports[1]["holdout_sha256"]


def test_prepared_dataset_rejects_incomplete_missing_and_duplicate_rows(tmp_path):
    manifest = prepared(tmp_path)
    validate_prepared(tmp_path)
    (tmp_path / "INCOMPLETE").touch()
    with pytest.raises(ValueError, match="Incomplete"):
        validate_prepared(tmp_path)
    (tmp_path / "INCOMPLETE").unlink()
    manifest["episodes"][1]["episode_id"] = 0
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unique"):
        validate_prepared(tmp_path)


def test_reuse_checks_source_provenance(tmp_path):
    import hashlib

    sources = tmp_path / "sources"
    for name in ("successes", "failures"):
        prepared(sources / name)
    output = tmp_path / "combined"
    prepared(output)
    hashes = {
        str((sources / name).resolve()): hashlib.sha256(
            (sources / name / "manifest.json").read_bytes()
        ).hexdigest()
        for name in ("successes", "failures")
    }
    (output / "preparation.json").write_text(
        json.dumps(
            {
                "source_manifest_sha256": hashes,
                "manifest_sha256": hashlib.sha256(
                    (output / "manifest.json").read_bytes()
                ).hexdigest(),
            }
        )
    )
    cfg = PrepareConfig(datasets=sources, output=output)
    assert prepare_dataset(cfg) == output
    path = sources / "successes" / "manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="provenance"):
        prepare_dataset(cfg)


@pytest.mark.dataset
def test_native_aggregation_links_videos_and_preserves_episode_alignment(tmp_path):
    import numpy as np
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sources = tmp_path / "sources"
    for name, success in [("successes", True), ("failures", False)]:
        root = sources / name
        dataset = LeRobotDataset.create(
            f"local/{name}",
            root=root,
            fps=50,
            robot_type="test",
            features={
                "action": {"dtype": "float32", "shape": (7,), "names": None},
                "observation.state": {
                    "dtype": "float32",
                    "shape": (18,),
                    "names": None,
                },
                "observation.images.front": {
                    "dtype": "video",
                    "shape": (3, 32, 32),
                    "names": ["channel", "height", "width"],
                },
            },
            video_backend="pyav",
            rgb_encoder=RGBEncoderConfig(vcodec="h264"),
            image_writer_threads=0,
        )
        rows = []
        for episode in range(2):
            seed = episode + (0 if success else 2)
            for tick in range(2):
                dataset.add_frame(
                    {
                        "action": np.full(7, seed, np.float32),
                        "observation.state": np.full(18, seed, np.float32),
                        "observation.images.front": np.full(
                            (3, 32, 32), 30 + seed * 30, np.uint8
                        ),
                        "task": "stack",
                    }
                )
            dataset.save_episode()
            replay = f"replay/{episode}.npz"
            (root / "replay").mkdir(exist_ok=True)
            np.savez(root / replay, action=np.full((2, 7), seed, np.float32))
            rows.append(
                {
                    "episode_index": episode,
                    "episode_id": seed,
                    "length": 2,
                    "source_root": "raw",
                    "physical_valid": True,
                    "native_success": success,
                    "dataset_split": "train" if episode == 0 else "val",
                    "seed": seed,
                    "env_id": "stack",
                    "task_id": 2,
                    "replay": replay,
                }
            )
        dataset.finalize()
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "ocbench-mjwarp-1",
                    "repo_id": f"local/{name}",
                    "action_profile": ACTION,
                    "rendering": {"revision": 2},
                    "episodes": rows,
                }
            )
        )
    cfg = PrepareConfig(datasets=sources, output=tmp_path / "combined")
    root = prepare_dataset(cfg)
    loaded = LeRobotDataset("local/ocbench-stack-all", root=root, video_backend="pyav")
    assert loaded.num_frames == 8 and loaded.num_episodes == 4
    assert [loaded[2 * i]["action"][0].item() for i in range(4)] == [0, 1, 2, 3]
    assert prepare_dataset(cfg) == root
    source_videos = {p.stat().st_ino for p in sources.glob("*/videos/**/*.mp4")}
    assert {p.stat().st_ino for p in root.glob("videos/**/*.mp4")} <= source_videos
    split = json.loads((root / "split.json").read_text())
    assert [r["seed"] for r in split["val"]] == [1, 3]
