"""Export ordering, bounded handoff, and resolution without a CUDA workload."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def source(tmp_path):
    from ocbench_mjwarp.config import ACTION, FIELDS

    root = tmp_path / "source"
    (root / "raw").mkdir(parents=True)
    for i, n in enumerate((7, 3, 5, 3, 6)):
        name = f"episode-{i:06d}"
        arrays = {f"sim/{k}": np.full((n + 1, 1), i) for k in FIELDS}
        arrays.update(
            state=np.full((n, 18), i, np.float32),
            action=np.full((n, 7), i / 10, np.float32),
            success=np.ones(n, bool),
        )
        np.savez(root / "raw" / f"{name}.npz", **arrays)
        (root / "raw" / f"{name}.json").write_text(
            json.dumps(
                {
                    "episode_id": i,
                    "seed": i,
                    "length": n,
                    "archive": f"{name}.npz",
                    "physical_valid": True,
                    "native_success": True,
                    "termination_reason": "success",
                    "action_profile": ACTION,
                    "dataset_split": "val" if i == 3 else "train",
                }
            )
        )
    return root


def test_schedule_selection_ties_resume_and_legacy(source, tmp_path):
    from ocbench_mjwarp.dataset import ExportConfig
    from ocbench_mjwarp.materialize import export_schedule

    cfg = ExportConfig(source=source, output=tmp_path / "dataset", limit=4)
    rows = export_schedule(cfg)
    assert [r["episode_id"] for r in rows] == [1, 3, 2, 0]
    assert export_schedule(cfg) == rows
    assert rows[1]["dataset_split"] == "val"
    for changed in (
        replace(cfg, image_size=(240, 320)),
        replace(cfg, limit=3),
        replace(cfg, episode_order="source"),
    ):
        with pytest.raises(FileExistsError):
            export_schedule(changed)
    legacy = replace(cfg, output=tmp_path / "legacy")
    legacy.output.mkdir()
    assert [r["episode_id"] for r in export_schedule(legacy)] == [0, 1, 2, 3]
    assert export_schedule(legacy) == export_schedule(legacy)


def test_resolution_provenance_and_spaces():
    from ocbench_mjwarp.dataset import dataset_features
    from ocbench_mjwarp.lerobot_env import OCBenchEnvConfig, OCBenchVectorEnv

    config = OCBenchEnvConfig(rendering={"resolution": [240, 320]})
    assert config.image_size == (240, 320)
    assert config.features["observation.images.front"].shape == (3, 240, 320)
    env = OCBenchVectorEnv(config, 2)
    assert env.single_observation_space["pixels"]["wrist"].shape == (240, 320, 3)
    assert dataset_features(image_size=config.image_size)["observation.images.front"][
        "shape"
    ] == (3, 240, 320)
    with pytest.raises(ValueError, match="resolution"):
        OCBenchEnvConfig(image_size=(480, 640), rendering={"resolution": [240, 320]})


@pytest.fixture
def fake_sim(monkeypatch):
    from ocbench_mjwarp import materialize as module

    instances = []

    class Simulation:
        def __init__(self, seeds, *, image_size, audit):
            self.worlds = len(seeds)
            self.size = image_size
            self.renderer = SimpleNamespace(profile={"resolution": list(image_size)})
            self.resets = 0
            self.closed = False
            instances.append(self)

        def reset_render(self, seeds):
            assert len(seeds) == self.worlds
            self.resets += 1

        def restore(self, values, forward=False):
            self.colors = values["qpos"][:, 0] * 30

        def render(self):
            shape = (self.worlds, *self.size, 3)
            return {
                view: np.broadcast_to(
                    (self.colors[:, None, None, None] + offset).astype(np.uint8), shape
                ).copy()
                for view, offset in (("front", 30), ("wrist", 60))
            }

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "Simulation", Simulation)
    return instances


@pytest.mark.dataset
@pytest.mark.parametrize("overlap", [False, True])
def test_reordered_export_reuses_context_and_preserves_alignment(
    source, fake_sim, tmp_path, overlap
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from ocbench_mjwarp.dataset import ExportConfig
    from ocbench_mjwarp.materialize import materialize

    cfg = ExportConfig(
        source=source,
        output=tmp_path / "dataset",
        batch_size=2,
        encoder_backend="cpu",
        image_size=(32, 48),
        overlap_commits=overlap,
    )
    result = materialize(cfg)
    assert result["episodes"] == 5 and result["frames"] == 24
    assert len(fake_sim) == 2 and fake_sim[0].resets == 1
    assert all(sim.closed for sim in fake_sim)
    data = LeRobotDataset(cfg.repo_id, root=cfg.output, video_backend="pyav")
    manifest = json.loads((cfg.output / "manifest.json").read_text())
    offset = 0
    for row in manifest["episodes"]:
        i, n = row["episode_id"], row["length"]
        np.testing.assert_allclose(data[offset]["action"], i / 10)
        assert data[offset]["timestamp"].item() == 0
        assert data[offset]["observation.images.front"].shape == (3, 32, 48)
        assert data[offset + n - 1]["next.done"].item()
        with np.load(cfg.output / row["replay"]) as state:
            assert state["sim/qpos"][0] == i
        offset += n
    assert [r["episode_id"] for r in manifest["episodes"]] == [1, 3, 2, 4, 0]
    assert materialize(cfg)["episodes"] == 5
    assert not list((cfg.output / ".encoding").glob("batch-*"))


@pytest.mark.dataset
@pytest.mark.parametrize("failure", ["render", "commit"])
def test_failure_preserves_checkpoint_and_rejects_partial_commit(
    source, fake_sim, tmp_path, monkeypatch, failure
):
    from ocbench_mjwarp import materialize as module
    from ocbench_mjwarp.dataset import ExportConfig

    cfg = ExportConfig(
        source=source,
        output=tmp_path / "dataset",
        batch_size=2,
        encoder_backend="cpu",
        image_size=(32, 48),
        overlap_commits=False,
    )
    target = "load_arrays" if failure == "render" else "save_encoded_episode"
    original = getattr(module, target)
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        # One durable batch; commit failure follows one extra episode saved.
        if calls == (2 if failure == "render" else 4):
            raise RuntimeError("injected failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, target, fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        module.materialize(cfg)
    checkpoint = json.loads((cfg.output / "checkpoint.json").read_text())
    assert len(checkpoint["episodes"]) == 2
    monkeypatch.setattr(module, target, original)
    if failure == "render":
        assert module.materialize(cfg)["episodes"] == 5
    else:
        with pytest.raises(ValueError, match="durable checkpoint"):
            module.materialize(cfg)


def test_benchmark_subset_covers_length_range_and_splits(source):
    from ocbench_mjwarp.benchmark_export import select_subset
    from ocbench_mjwarp.episodes import records

    rows = records(source)
    rows = [
        dict(r, dataset_split=split, episode_id=r["episode_id"] + offset)
        for split, offset in (("train", 0), ("val", 10))
        for r in rows
    ]
    selected = select_subset(rows, 3)
    assert [r["length"] for r in selected] == [3, 5, 7, 3, 5, 7]
    assert [r["dataset_split"] for r in selected] == ["train"] * 3 + ["val"] * 3
    with pytest.raises(ValueError, match="Need"):
        select_subset(rows, 10)


def test_low_resolution_recipe_and_cli_override(monkeypatch):
    from pathlib import Path

    from ocbench_mjwarp.pipeline import PipelineConfig
    from vla_tools.config import parse_args

    recipe = (
        Path(__file__).resolve().parents[3] / "configs/ocbench/import-stack-320.yaml"
    )
    monkeypatch.setattr(
        "sys.argv", ["pipeline", "--config", str(recipe), "--image-size", "120", "160"]
    )
    config = parse_args(PipelineConfig)
    assert config.image_size == (120, 160)
    assert config.episode_order == "length" and config.overlap_commits


@pytest.mark.dataset
def test_commit_overlaps_render_but_writer_has_one_owner(
    source, fake_sim, tmp_path, monkeypatch
):
    import threading

    from ocbench_mjwarp import materialize as module
    from ocbench_mjwarp.dataset import ExportConfig

    entering_commit, next_render = threading.Event(), threading.Event()
    owners = []
    original = module.save_encoded_episode
    reset = module.Simulation.reset_render

    def commit(*args):
        owners.append(threading.get_ident())
        if len(owners) == 1:
            entering_commit.set()
            assert next_render.wait(10), "Next batch did not render during commit"
        return original(*args)

    def render(self, seeds):
        assert entering_commit.wait(10)
        next_render.set()
        return reset(self, seeds)

    monkeypatch.setattr(module, "save_encoded_episode", commit)
    monkeypatch.setattr(module.Simulation, "reset_render", render)
    cfg = ExportConfig(
        source=source,
        output=tmp_path / "dataset",
        batch_size=2,
        encoder_backend="cpu",
        image_size=(32, 48),
        overlap_commits=True,
    )
    assert module.materialize(cfg)["episodes"] == 5
    assert len(set(owners)) == 1 and owners[0] != threading.get_ident()
