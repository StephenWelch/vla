import numpy as np
import pytest


@pytest.mark.dataset
def test_concurrent_videos_commit_with_native_stats_and_alignment(tmp_path):
    import av
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from ocbench_mjwarp.dataset import dataset_features
    from ocbench_mjwarp.materialize import KEYS, episode_buffer
    from vla_tools.encoding import EpisodeVideoEncoder, save_encoded_episode

    features = dataset_features()
    for key in KEYS:
        features[key]["shape"] = (3, 32, 32)
    dataset = LeRobotDataset.create(
        "local/test-direct",
        root=tmp_path / "dataset",
        fps=50,
        robot_type="test",
        features=features,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18),
        image_writer_threads=0,
    )
    encoders = [
        EpisodeVideoEncoder(
            fps=50,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18),
            queue_maxsize=1,
            encoder_threads=1,
            expected_frames=n,
        )
        for n in (7, 11)
    ]
    try:
        for encoder in encoders:
            encoder.start_episode(KEYS, temp_dir=dataset.root)
        for tick in range(11):
            for world, n in enumerate((7, 11)):
                if tick < n:
                    for camera, key in enumerate(KEYS):
                        encoders[world].feed_frame(
                            key,
                            np.full(
                                (32, 32, 3),
                                30 + world * 100 + camera * 40 + tick,
                                np.uint8,
                            ),
                        )
        for world, (n, encoder) in enumerate(zip((7, 11), encoders, strict=True)):
            a = {
                "action": np.full((n, 7), world, np.float32),
                "state": np.full((n, 18), world, np.float32),
                "success": np.zeros(n, bool),
            }
            save_encoded_episode(
                dataset,
                episode_buffer(
                    dataset, {"length": n, "termination_reason": "horizon"}, a
                ),
                encoder,
            )
        dataset.finalize()
    finally:
        for encoder in encoders:
            encoder.close()
    loaded = LeRobotDataset(
        "local/test-direct", root=dataset.root, video_backend="pyav"
    )
    assert loaded.num_episodes == 2 and loaded.num_frames == 18
    assert float(loaded[7]["action"][0]) == 1
    assert bool(loaded[6]["next.done"])
    assert loaded[7]["timestamp"].item() == 0
    assert (
        loaded[7]["observation.images.front"].float().mean()
        > loaded[0]["observation.images.front"].float().mean()
    )
    assert loaded[0][KEYS[1]].mean() > loaded[0][KEYS[0]].mean()
    assert np.isfinite(loaded.meta.stats[KEYS[0]]["mean"]).all()
    for key in KEYS:
        count = 0
        for path in (dataset.root / "videos" / key).rglob("*.mp4"):
            with av.open(str(path)) as container:
                count += sum(1 for _ in container.decode(video=0))
        assert count == 18


@pytest.mark.gpu
@pytest.mark.dataset
def test_default_gpu_video_across_batches_and_partial_tail(tmp_path, monkeypatch):
    import json

    import av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from ocbench_mjwarp import materialize as module
    from ocbench_mjwarp.config import ACTION, FIELDS
    from ocbench_mjwarp.dataset import ExportConfig
    from ocbench_mjwarp.environment import Simulation
    from ocbench_mjwarp.materialize import materialize as export

    source = tmp_path / "source"
    (source / "raw").mkdir(parents=True)
    sim = Simulation([83000], audit=False)
    try:
        initial = sim.snapshot()
    finally:
        sim.close()
    lengths = [3, 4, 5, 6, 7]
    for i, n in enumerate(lengths):
        name = f"episode-{i:06d}"
        arrays = {
            f"sim/{k}": np.repeat(initial[k][0][None], n + 1, axis=0) for k in FIELDS
        }
        arrays.update(
            action=np.full((n, 7), i / 10, np.float32),
            state=np.full((n, 18), i, np.float32),
            success=np.ones(n, bool),
        )
        np.savez_compressed(source / "raw" / f"{name}.npz", **arrays)
        (source / "raw" / f"{name}.json").write_text(
            json.dumps(
                {
                    "episode_id": i,
                    "seed": 83000,
                    "length": n,
                    "archive": f"{name}.npz",
                    "physical_valid": True,
                    "native_success": True,
                    "termination_reason": "success",
                    "action_profile": ACTION,
                }
            )
        )
    config = ExportConfig(source=source, output=tmp_path / "dataset")
    calls = 0

    def interrupted_simulation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected restart between batches")
        return Simulation(*args, **kwargs)

    monkeypatch.setattr(module, "Simulation", interrupted_simulation)
    with pytest.raises(RuntimeError, match="injected restart"):
        export(config)
    checkpoint = json.loads((config.output / "checkpoint.json").read_text())
    assert len(checkpoint["episodes"]) == 4
    before = {p: p.read_bytes() for p in (config.output / "videos").rglob("*.mp4")}
    monkeypatch.setattr(module, "Simulation", Simulation)
    result = export(config)
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert result["episodes"] == 5 and result["frames"] == sum(lengths)
    manifest = json.loads((config.output / "manifest.json").read_text())
    assert manifest["video_materialization"] == "async-gpu-v1"
    assert manifest["encoding"]["render_batch_frames"] == 4
    assert manifest["encoding"]["image_stats"] == "uint8"
    assert not (config.output / "INCOMPLETE.json").exists()
    assert not list((config.output / ".encoding").iterdir())
    assert len(list((config.output / "previews").glob("*.mp4"))) == 6
    loaded = LeRobotDataset(config.repo_id, root=config.output, video_backend="pyav")
    offset = 0
    for i, n in enumerate(lengths):
        np.testing.assert_allclose(loaded[offset]["action"], i / 10)
        np.testing.assert_array_equal(
            loaded[offset]["observation.state"], np.full(18, i)
        )
        assert loaded[offset]["timestamp"].item() == 0
        assert loaded[offset + n - 1]["next.done"].item()
        assert (config.output / manifest["episodes"][i]["replay"]).exists()
        offset += n
    for view in ("front", "wrist"):
        count = 0
        for path in (config.output / "videos" / f"observation.images.{view}").rglob(
            "*.mp4"
        ):
            with av.open(str(path)) as container:
                for tick, frame in enumerate(container.decode(video=0)):
                    assert abs(float(frame.pts * frame.time_base) - tick / 50) < 1e-5
                    count += 1
        assert count == sum(lengths)
    assert export(config)["episodes"] == 5
    config.encoder_backend = "cpu"
    with pytest.raises(FileExistsError):
        export(config)
