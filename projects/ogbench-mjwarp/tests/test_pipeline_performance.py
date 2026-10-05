import json
import queue
import time
from threading import Event, Thread
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("lerobot")
from ogbench_mjwarp.encoding import BlockingVideoEncoder
from ogbench_mjwarp.export_worker import RoundExporter
from ogbench_mjwarp.recording import EpisodeBuffer


def round_data(root, episode=0):
    root.mkdir()
    buffer = EpisodeBuffer(
        root,
        episode,
        {
            "instruction": "Move the cube",
            "image_size": 32,
            "fps": 20,
            "rendering": {
                "backend": "mujoco-warp",
                "revision": 2,
                "resolution": [32, 32],
            },
            "env_id": "cube-single-v0",
            "task_id": 1,
            "seed": episode,
            "contact_quality": {"valid": True},
        },
    )
    for i in range(3):
        buffer.states.append({"qpos": np.array([i], np.float32)})
        buffer.add(
            {
                view: np.full((32, 32, 3), i * 50, np.uint8)
                for view in ("front", "wrist")
            },
            np.zeros(18),
            np.zeros(5),
            i == 2,
            i == 2,
            False,
            0,
        )
    buffer.save("success", "success", {"qpos": np.array([3], np.float32)})
    return root


def encoder():
    result = BlockingVideoEncoder(fps=20, queue_maxsize=1)
    result._episode_active = True
    result._threads["wrist"] = Mock(is_alive=Mock(return_value=True))
    result._frame_queues["wrist"] = queue.Queue(1)
    return result


def test_encoder_backpressure_never_drops_frames():
    writer = encoder()
    image = np.ones((16, 16, 3), np.uint8)
    writer.feed_frame("wrist", image)
    finished = Event()
    feeder = Thread(target=lambda: (writer.feed_frame("wrist", image), finished.set()))
    feeder.start()
    try:
        assert not finished.wait(0.15)
        np.testing.assert_array_equal(writer._frame_queues["wrist"].get(), image)
        assert finished.wait(2)
        image[:] = 9
        assert np.all(writer._frame_queues["wrist"].get() == 1)
        assert not writer._dropped_frames
    finally:
        feeder.join(2)


def test_encoder_failure_and_stall_propagate(monkeypatch):
    writer = encoder()
    writer._threads["wrist"].is_alive.return_value = False
    with pytest.raises(RuntimeError, match="stopped"):
        writer.feed_frame("wrist", np.zeros((16, 16, 3)))
    writer._threads["wrist"].is_alive.return_value = True
    writer._frame_queues["wrist"].put(None)
    times = iter([0, 121])
    monkeypatch.setattr("ogbench_mjwarp.encoding.time.monotonic", lambda: next(times))
    with pytest.raises(RuntimeError, match="stalled"):
        writer.feed_frame("wrist", np.zeros((16, 16, 3)))


def test_export_overlaps_and_preserves_order(tmp_path):
    first, second = (
        round_data(tmp_path / "first", 7),
        round_data(tmp_path / "second", 3),
    )
    events = []
    output = tmp_path / "dataset"
    worker = RoundExporter(output, capacity=1, progress=events.append)
    try:
        worker.submit(first)
        deadline = time.monotonic() + 60
        while not any(event["event"] == "export" for event in events):
            worker.poll()
            assert time.monotonic() < deadline
            time.sleep(0.1)
        # The first round exported before the next round was even submitted.
        assert (output / "INCOMPLETE.json").exists()
        assert not (output / "manifest.json").exists()
        worker.submit(second)
        result = worker.finish()
        assert result["episodes"] == 2 and result["frames"] == 6
        rows = json.loads((output / "manifest.json").read_text())["episodes"]
        assert [row["source_episode_id"] for row in rows] == [7, 3]
        assert not (output / "INCOMPLETE.json").exists()
        assert not list(output.rglob("*.png"))
        import av

        for video in output.rglob("*.mp4"):
            with av.open(str(video)) as container:
                assert len(list(container.decode(video=0))) == 6
    finally:
        worker.close()


def test_export_worker_failure_does_not_hang(tmp_path):
    source = round_data(tmp_path / "source")
    (source / "episode-000000.npz").unlink()
    worker = RoundExporter(tmp_path / "dataset", capacity=1)
    try:
        worker.submit(source)
        with pytest.raises(RuntimeError):
            worker.finish()
        assert (tmp_path / "dataset/INCOMPLETE.json").exists()
    finally:
        worker.close()
    worker = RoundExporter(tmp_path / "dead", capacity=1)
    worker.process.terminate()
    worker.process.join()
    try:
        with pytest.raises(RuntimeError, match="prematurely"):
            worker.submit(source)
    finally:
        worker.close()


@pytest.mark.gpu
def test_batched_host_states_match_and_masked_restore_isolated():
    import torch
    from ogbench_mjwarp.config import PlannerConfig
    from ogbench_mjwarp.environment import BatchEnvironment
    from ogbench_mjwarp.tasks import make_env

    env = make_env("cube-single-v0", size=16)
    try:
        sim = BatchEnvironment(env, 2, PlannerConfig(episodes=2))
        sim.reset([42, 43])
        states = sim.cpu_states()
        for world in range(2):
            for key, value in sim.cpu_state(world).items():
                np.testing.assert_array_equal(states[key][world], value)
                assert states[key].dtype == value.dtype
        sim.overflow[:] = 1
        sim.restore(states, world_mask=torch.tensor([True, False], device=sim.device))
        assert sim.overflow.cpu().tolist() == [0, 1]
        states["qpos"][:] = 999
        assert not np.any(sim.cpu_state(0)["qpos"] == 999)
    finally:
        env.close()


@pytest.mark.gpu
@pytest.mark.parametrize("refill", [True, False])
def test_slot_reuse_timeouts_rng_and_resume(tmp_path, monkeypatch, refill):
    import torch
    from ogbench_mjwarp.config import PlannerConfig
    from ogbench_mjwarp.planner import SamplingMPC
    from ogbench_mjwarp.recording import generate

    calls = []

    def plan(self, references, objectives=None):
        calls.append((self.mean.clone(), [g.initial_seed() for g in self.generators]))
        valid = [True, True]
        if len(calls) == 2:
            valid[0] = False
        if len(calls) == 3:
            valid[1] = False
        self.mean.fill_(99)
        self.previous_action.fill_(99)
        return torch.zeros((self.execution.worlds, 5), device=self.execution.device), {
            "valid": valid,
            "seconds": 0,
            "cost": [0, 0],
        }

    monkeypatch.setattr(SamplingMPC, "plan", plan)
    raw = tmp_path / "raw"
    config = PlannerConfig(episodes=2, candidates=2, horizon=2, iterations=1)
    generate(
        raw,
        "cube-single-v0",
        4,
        [1],
        123,
        config,
        size=16,
        max_steps=3,
        refill_slots=refill,
    )
    rows = [json.loads(p.read_text()) for p in sorted(raw.glob("episode-*.json"))]
    assert [r["length"] for r in rows] == [2, 3, 3, 3]
    assert rows[2]["reason"] == rows[3]["reason"] == "timeout"
    if refill:
        assert calls[2][0][0].count_nonzero() == 0
        assert calls[2][0][1].count_nonzero() > 0
        assert calls[2][1][0] == rows[2]["randomization"]["seeds"]["planner"]
        assert calls[2][1][1] == calls[0][1][1]
    for row in rows:
        with np.load(raw / row["archive"]) as arrays:
            assert len(arrays["sim/qpos"]) == row["length"] + 1
    (raw / "episode-000003.json").unlink()
    result = generate(
        raw,
        "cube-single-v0",
        4,
        [1],
        123,
        config,
        size=16,
        max_steps=3,
        refill_slots=refill,
    )
    assert result["episodes"] == 4
