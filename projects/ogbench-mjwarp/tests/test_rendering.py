import json
from threading import Event

import numpy as np
import pytest
import torch
from ogbench_mjwarp.recording import ArchiveWriter, EpisodeBuffer
from ogbench_mjwarp.rendering import dataset_profile, rendering_profile


@pytest.mark.gpu
@pytest.mark.parametrize("env_id", ["cube-single-v0", "scene-v0", "puzzle-3x3-v0"])
def test_visibility_matches_native_mujoco_after_motion(env_id, monkeypatch):
    import mujoco
    from ogbench_mjwarp import rendering
    from ogbench_mjwarp.config import PlannerConfig
    from ogbench_mjwarp.environment import BatchEnvironment
    from ogbench_mjwarp.tasks import make_env, restore_cpu

    create = rendering.mjw.create_render_context
    monkeypatch.setattr(
        rendering.mjw,
        "create_render_context",
        lambda *args, **kwargs: create(*args, render_seg=True, **kwargs),
    )
    height, width = 120, 160
    env = make_env(env_id, seed=42, size=(height, width))
    try:
        sim = BatchEnvironment(env, 2, PlannerConfig(episodes=2))
        sim.reset([42, 43], [1, 1])
        groups = sim.host_model.geom_group.copy()
        with mujoco.Renderer(sim.host_model, height=height, width=width) as native:
            native.enable_segmentation_rendering()
            for pose in range(3):
                states = sim.cpu_states()
                if pose:
                    states["qpos"][:, int(sim.arm_q[0])] += 0.2
                    if sim.cube_q:
                        states["qpos"][:, sim.cube_q[0]] += 0.12
                if pose == 2 and sim.cube_q:
                    # A close cube must survive the wrist camera's near plane.
                    cam = sim.host_model.camera("ur5e/wrist").id
                    for world in range(2):
                        restore_cpu(env, {k: v[world] for k, v in states.items()})
                        data = env.unwrapped._data
                        pos = data.cam_xpos[cam] + data.cam_xmat[cam].reshape(3, 3) @ [
                            0,
                            0,
                            -0.05,
                        ]
                        start = sim.cube_q[0]
                        states["qpos"][world, start : start + 3] = pos
                sim.restore(states)
                sim.render_batch()
                segmentation = sim.renderer.context.seg_data.numpy().reshape(
                    2, 2, height, width, 2
                )
                np.testing.assert_array_equal(sim.host_model.geom_group, groups)
                for world in range(2):
                    restore_cpu(env, {k: v[world] for k, v in states.items()})
                    for camera, name in enumerate(("front", "ur5e/wrist")):
                        native.update_scene(env.unwrapped._data, camera=name)
                        expected = native.render()[..., 0]
                        actual = segmentation[world, camera, ..., 0]
                        # Rasterization and ray tracing differ at silhouettes.
                        assert np.mean(expected == actual) > 0.995
                        for hidden in sim.renderer.profile["hidden_geoms"]:
                            assert not np.any(actual == hidden)
                        if sim.cube_q and (
                            pose == 2 and camera == 1 or pose == 0 and camera == 0
                        ):
                            cube = sim.host_model.geom("object_0").id
                            reference, rendered = expected == cube, actual == cube
                            assert reference.sum() > 10
                            assert (reference & rendered).sum() / (
                                reference | rendered
                            ).sum() > 0.98
    finally:
        env.close()


@pytest.mark.gpu
def test_so101_camera_resolution():
    from ogbench_mjwarp.config import PlannerConfig
    from ogbench_mjwarp.environment import BatchEnvironment
    from ogbench_mjwarp.lerobot_env import OGBenchEnvConfig
    from ogbench_mjwarp.tasks import make_env

    config = OGBenchEnvConfig()
    assert config.features["observation.images.wrist"].shape == (3, 480, 640)
    env = make_env("cube-single-v0")
    try:
        sim = BatchEnvironment(env, 2, PlannerConfig(episodes=2))
        sim.reset([42, 43], [1, 1])
        images = sim.render_batch()
        assert sim.renderer.profile["resolution"] == [480, 640]
        for image in images.values():
            assert image.shape == (2, 480, 640, 3)
            assert image.dtype == np.uint8 and image.std() > 1
    finally:
        env.close()


def test_legacy_rejection_and_checkpoint_contract(tmp_path):
    assert dataset_profile(tmp_path) is None
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"format": "ogbench-mjwarp-1"}))
    with pytest.raises(ValueError, match="Legacy"):
        dataset_profile(tmp_path)
    profile = {"backend": "mujoco-warp", "revision": 2, "resolution": [32, 32]}
    manifest.write_text(
        json.dumps({"format": "ogbench-mjwarp-2", "rendering": profile})
    )
    checkpoint = tmp_path / "policy"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="differ"):
        dataset_profile(tmp_path, checkpoint)
    (checkpoint / "rendering.json").write_text(json.dumps(profile))
    assert dataset_profile(tmp_path, checkpoint) == profile
    profile.pop("revision")
    manifest.write_text(
        json.dumps({"format": "ogbench-mjwarp-2", "rendering": profile})
    )
    with pytest.raises(ValueError, match="faulty renderer"):
        dataset_profile(tmp_path)


def test_archive_commit_order_and_failure(tmp_path, monkeypatch):
    buffer = EpisodeBuffer(tmp_path, 0, {})
    buffer.add(None, np.zeros(18), np.zeros(5), False, True, True, 0)
    import ogbench_mjwarp.recording as module

    original = module.write_json

    def commit(path, metadata):
        assert path.with_suffix(".npz").is_file()
        original(path, metadata)

    monkeypatch.setattr(module, "write_json", commit)
    writer = ArchiveWriter()
    writer.submit(buffer, "failure", "timeout", {})
    writer.close()
    assert (tmp_path / "episode-000000.json").exists()
    bad = EpisodeBuffer(tmp_path, 1, {})

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(bad, "save", fail)
    writer = ArchiveWriter()
    writer.submit(bad, "failure", "timeout", {})
    with pytest.raises(OSError, match="disk full"):
        writer.close()
    assert not (tmp_path / "episode-000001.json").exists()


def test_writer_bounds_pending_work():
    gate = Event()

    class SlowBuffer:
        def save(self, *args):
            assert gate.wait(5)

    writer = ArchiveWriter(workers=2, capacity=4)
    for _ in range(4):
        writer.submit(SlowBuffer(), None, None, None)
    assert len(writer.pending) == 4
    gate.set()
    writer.submit(SlowBuffer(), None, None, None)
    assert len(writer.pending) <= 4
    writer.close()


@pytest.mark.gpu
@pytest.mark.parametrize("env_id", ["scene-v0", "puzzle-3x3-v0"])
def test_batch_colors_cameras_and_physics_isolation(env_id):
    from ogbench_mjwarp.config import PlannerConfig
    from ogbench_mjwarp.environment import BatchEnvironment
    from ogbench_mjwarp.tasks import make_env

    env = make_env(env_id, seed=42, size=32)
    other = make_env(env_id, seed=43, size=32)
    try:
        assert rendering_profile(env.unwrapped._model, 32) == rendering_profile(
            other.unwrapped._model, 32
        )
        sim = BatchEnvironment(env, 2, PlannerConfig(episodes=2))
        snapshot = sim.snapshot()
        snapshot["buttons"][0] = 0
        snapshot["buttons"][1] = 1
        sim.restore(snapshot)
        before = sim.snapshot()
        images = sim.render_batch()
        for key, value in before.items():
            torch.testing.assert_close(sim.snapshot()[key], value, rtol=0, atol=0)
        assert images["front"].shape == (2, 32, 32, 3)
        assert images["front"].dtype == np.uint8
        assert not np.array_equal(images["front"][0], images["front"][1])
        assert not np.array_equal(images["front"], images["wrist"])
        # Rendering leaves the next physical transition unchanged.
        action = sim.tensor([[0.1, 0, 0, 0, 0]] * 2)
        sim.step(action)
        expected = sim.snapshot()
        sim.restore(before)
        sim.render_batch()
        sim.step(action)
        for key, value in expected.items():
            # Float32 contact reductions can vary by a few ulps across launches.
            torch.testing.assert_close(sim.snapshot()[key], value, rtol=1e-5, atol=1e-5)
        assert not np.array_equal(sim.render_batch()["wrist"], images["wrist"])
    finally:
        env.close()
        other.close()


@pytest.mark.gpu
@pytest.mark.parametrize("size", [16, (48, 64)])
def test_inline_alignment_and_batched_rerender(tmp_path, size):
    from ogbench_mjwarp.config import PlannerConfig
    from ogbench_mjwarp.recording import generate
    from ogbench_mjwarp.rerender import rerender

    raw = tmp_path / "raw"
    config = PlannerConfig(episodes=2, candidates=2, horizon=2, iterations=1)
    generate(raw, "cube-single-v0", 3, [1], 123, config, size=size, max_steps=2)
    output = tmp_path / "rendered"
    rerender(raw, output, batch_size=2)
    for path in raw.glob("episode-*.npz"):
        with np.load(path) as original, np.load(output / path.name) as restored:
            for key in original.files:
                np.testing.assert_array_equal(original[key], restored[key])
            assert len(original["sim/qpos"]) == len(original["front"]) + 1
    # Completed runs resume without changing committed archives.
    assert (
        generate(raw, "cube-single-v0", 3, [1], 123, config, size=size, max_steps=2)[
            "episodes"
        ]
        == 3
    )
    # A crash before metadata commit must not acknowledge an orphan archive.
    (raw / "episode-000002.json").unlink()
    (raw / "episode-000002.npz").write_bytes(b"interrupted archive")
    result = generate(
        raw, "cube-single-v0", 3, [1], 123, config, size=size, max_steps=2
    )
    assert result["episodes"] == 3
    with np.load(raw / "episode-000002.npz") as recovered:
        assert len(recovered["sim/qpos"]) == len(recovered["front"]) + 1
