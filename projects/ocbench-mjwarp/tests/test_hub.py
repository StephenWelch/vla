"""Published trajectories retain their identity, split and missing evidence."""

import json
import pickle
from dataclasses import replace

import numpy as np
import pytest
from ocbench_mjwarp.episodes import load_arrays, records, select_episodes
from ocbench_mjwarp.hub import ImportConfig, boundaries, import_dataset


@pytest.fixture
def release(tmp_path, monkeypatch):
    import ocbench

    cache = tmp_path / "cache"
    cache.mkdir()
    env = ocbench.make("block-cpu-double-task2-v0")
    env.reset(seed=83002)
    q, v = env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy()
    env.close()
    paths = []
    for split in ("train", "val"):
        q = q.copy()
        if split == "val":
            q[0] += 0.01
        n = 9
        arrays = {
            "qpos": np.repeat(q[None], n, axis=0).astype(np.float32),
            "qvel": np.repeat(v[None], n, axis=0).astype(np.float32),
            "actions": np.arange(n * 7, dtype=np.float32).reshape(n, 7) / 100,
            "rewards": np.array([0, 0, 1, 0, 0, 0, 0, 0, 0], np.float32),
            "masks": np.array([1, 1, 0, 1, 1, 1, 1, 1, 1], np.float32),
            "terminals": np.array([0, 0, 1, 0, 0, 0, 0, 0, 1], bool),
        }
        path = cache / f"{split}.npz"
        np.savez_compressed(path, **arrays)
        metadata = [
            {"start": 0, "end": 2, "length": 3, "success": True},
            {"start": 3, "end": 8, "length": 6, "success": False},
        ]
        with path.with_name(path.stem + "-metadata.pkl").open("wb") as stream:
            pickle.dump({"episodes": metadata}, stream)
        paths.append(path)
    monkeypatch.setattr(
        ocbench,
        "download_datasets",
        lambda *a, **kw: ([str(paths[0])], [str(paths[1])]),
    )
    return ImportConfig(
        output=tmp_path / "import", cache=cache, train_episodes=2, val_episodes=1
    )


def test_import_resume_alignment_and_official_splits(release):
    assert import_dataset(release)["episodes"] == 3
    rows = records(release.output)
    assert [r["dataset_split"] for r in rows] == ["train", "train", "val"]
    assert all(r["physical_valid"] is None and r["reset_seed"] is None for r in rows)
    assert all(not r["terminal_state_available"] for r in rows)
    arrays = load_arrays(release.output, rows)
    assert [len(a["sim/qpos"]) for a in arrays] == [3, 6, 3]
    np.testing.assert_array_equal(
        arrays[0]["action"], np.arange(21, dtype=np.float32).reshape(3, 7) / 100
    )
    assert arrays[0]["state"].shape == (3, 18)
    assert arrays[0]["terminated"][-1] and arrays[1]["truncated"][-1]
    assert len(select_episodes(release.output, None)) == 3
    assert len(select_episodes(release.output, True)) == 2
    assert import_dataset(release)["episodes"] == 3
    assert records(release.output) == rows
    with pytest.raises(ValueError, match="configuration changed"):
        import_dataset(replace(release, train_episodes=1))
    (release.output / "raw" / rows[0]["archive"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="differs on resume"):
        import_dataset(release)


def test_boundaries_reject_missing_endpoint_and_misaligned_metadata():
    with pytest.raises(ValueError, match="boundaries"):
        boundaries(np.array([0, 0]), [])
    with pytest.raises(ValueError, match="differ"):
        boundaries(np.array([0, 1]), [{"start": 0, "end": 2, "length": 3}])


@pytest.mark.gpu
@pytest.mark.dataset
def test_import_uses_shared_gpu_export(release, tmp_path, monkeypatch):
    import av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from ocbench_mjwarp.dataset import ExportConfig
    from ocbench_mjwarp.materialize import materialize
    from ocbench_mjwarp.prepare import validate_prepared
    from vla_tools.hooks import scene_split

    import_dataset(release)
    cfg = ExportConfig(
        source=release.output,
        output=tmp_path / "dataset",
        successes=None,
        batch_size=2,
        episode_order="source",
        render_batch_frames=4,
    )
    assert materialize(cfg)["episodes"] == 3
    manifest = validate_prepared(cfg.output)
    assert manifest["quality"] == "upstream-outcomes-unaudited"
    split = scene_split(cfg.output)
    assert [r["episode_index"] for r in split["train"]] == [0, 1]
    assert [r["episode_index"] for r in split["val"]] == [2]
    data = LeRobotDataset(cfg.repo_id, root=cfg.output, video_backend="pyav")
    assert data.num_frames == 12
    assert data[0]["observation.images.front"].shape == (3, 480, 640)
    assert data[2]["next.reward"].item() == 1
    assert data[8]["next.truncated"].item()
    assert data[3]["timestamp"].item() == 0
    assert materialize(cfg)["episodes"] == 3
    from ocbench_mjwarp.train import TrainConfig
    from ocbench_mjwarp.training import validate_profiles
    from vla_tools.train import training_plan

    plan = training_plan(
        TrainConfig(
            dataset=cfg.output,
            output=tmp_path / "training",
            validation_fraction=0.2,
            percentile_normalization=True,
            steps=1,
            workers=0,
            dry_run=True,
        ),
        validate_profiles,
    )
    assert [r["episode_index"] for r in plan["split"]["train"]] == [0, 1]
    assert [r["episode_index"] for r in plan["split"]["val"]] == [2]
    from types import SimpleNamespace

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.scripts import lerobot_eval
    from ocbench_mjwarp.evaluate import EvalConfig, evaluate_task
    from ocbench_mjwarp.lerobot_env import OCBenchEnvConfig

    def one_step(env, *args, **kwargs):
        env.reset(seed=[kwargs["start_seed"]])
        with np.load(cfg.output / manifest["episodes"][2]["replay"]) as initial:
            np.testing.assert_array_equal(
                env.sim.data.qpos.numpy()[0], initial["sim/qpos"]
            )
        env.step(np.zeros((1, 7), np.float32))
        return {"per_episode": [{"seed": kwargs["start_seed"]}], "aggregated": {}}

    monkeypatch.setattr(lerobot_eval, "eval_policy", one_step)
    policy = SimpleNamespace(config=ACTConfig(device="cuda"), select_action=lambda x: x)
    result = evaluate_task(
        EvalConfig(
            checkpoint=tmp_path,
            output=tmp_path / "eval",
            dataset=cfg.output,
            episodes=1,
            videos=0,
        ),
        OCBenchEnvConfig(max_steps=1),
        policy,
        lambda x: x,
        lambda x: x,
        2,
    )
    assert result["per_episode"][0]["seed"] == manifest["episodes"][2]["seed"]
    assert result["per_episode"][0]["seed_kind"] == "replay_id"
    for camera in ("front", "wrist"):
        count = 0
        for path in (cfg.output / "videos" / f"observation.images.{camera}").rglob(
            "*.mp4"
        ):
            with av.open(path) as video:
                count += sum(1 for _ in video.decode(video=0))
        assert count == 12


def test_missing_audits_cannot_become_clean_or_replay_successes(release, tmp_path):
    from ocbench_mjwarp.compare import selections
    from ocbench_mjwarp.replay_check import Config, run

    import_dataset(release)
    with pytest.raises(ValueError, match="unknown contact audits"):
        selections({"episodes": records(release.output)})
    with pytest.raises(ValueError, match="lack terminal"):
        run(Config(source=release.output, output=tmp_path / "report", episodes=(0,)))


def test_imported_action_statistics_exclude_validation():
    from types import SimpleNamespace

    from datasets import Dataset
    from ocbench_mjwarp.actions import prepare_training_views

    datasets = {
        "train": SimpleNamespace(
            hf_dataset=Dataset.from_dict(
                {
                    "action": [[0.0] * 7, [1.0] * 7],
                    "observation.state": [[0.0] * 18, [1.0] * 18],
                }
            )
        ),
        "val": SimpleNamespace(
            hf_dataset=Dataset.from_dict(
                {"action": [[999.0] * 7], "observation.state": [[999.0] * 18]}
            )
        ),
    }
    stats = {"action": {}, "observation.state": {}}
    prepare_training_views(datasets, stats, False, True)
    np.testing.assert_allclose(stats["action"]["q99"], 0.99)
    np.testing.assert_allclose(stats["observation.state"]["q99"], 0.99)


def test_replay_identifiers_do_not_alias_real_reset_seeds(tmp_path):
    from vla_tools.hooks import scene_split

    rows = [
        {
            "episode_index": i,
            "env_id": "stack",
            "task_id": 2,
            "seed": 0,
            "dataset_split": split,
        }
        for i, split in enumerate(("train", "val"))
    ]
    rows[1].update(
        seed_kind="replay_id",
        source={
            "repo_id": "official",
            "revision": "pinned",
            "file": "val.npz",
            "episode_index": 0,
        },
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"episodes": rows}))
    split = scene_split(tmp_path)
    assert len(split["train"]) == len(split["val"]) == 1


def test_pilot_yaml_and_nullable_export_selection(tmp_path):
    from pathlib import Path

    from ocbench_mjwarp.dataset import ExportConfig
    from ocbench_mjwarp.pipeline import PipelineConfig
    from vla_tools.config import parse_args

    recipe = Path(__file__).resolve().parents[3] / "configs/ocbench/import-stack.yaml"
    config = parse_args(PipelineConfig, ["--config", str(recipe)])
    assert config.source == "huggingface" and config.training.steps == 0
    assert config.training.overrides["policy.chunk_size"] == "25"
    assert (config.hub.train_episodes, config.hub.val_episodes) == (100, 20)
    export = parse_args(
        ExportConfig,
        [
            "--source",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--successes",
            "None",
        ],
    )
    assert export.successes is None
    (tmp_path / "import.json").write_text("{}")
    with pytest.raises(ValueError, match="resume"):
        select_episodes(tmp_path)
