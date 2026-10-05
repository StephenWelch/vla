import json

import numpy as np
import pytest
from ogbench_mjwarp.cli import parse_args
from ogbench_mjwarp.io import load_sim_states, rollout_path
from ogbench_mjwarp.viewer import Playback


def test_simulator_archive_loading_and_alignment(tmp_path):
    archive = tmp_path / "episode.npz"
    np.savez(
        archive,
        **{
            "sim/qpos": np.zeros((3, 2)),
            "sim/time": np.arange(3),
            "front": np.zeros((2, 16, 16, 3)),
        },
    )
    states = load_sim_states(archive)
    assert set(states) == {"qpos", "time"}
    assert len(states["qpos"]) == 3  # Terminal snapshot is included.
    np.savez(archive, **{"sim/qpos": np.zeros((3, 2)), "sim/time": np.arange(2)})
    with pytest.raises(ValueError, match="aligned"):
        load_sim_states(archive)
    np.savez(archive, action=np.zeros((2, 5)))
    with pytest.raises(ValueError, match="snapshots"):
        load_sim_states(archive)


def test_playback_timing_and_controls():
    playback = Playback(frames=4, fps=20, deadline=0.05)
    playback.update(0.049)
    assert playback.frame == 0
    playback.update(0.05)
    assert playback.frame == 1
    playback.update(0.06, [32])
    playback.update(10)
    assert playback.frame == 1 and playback.paused
    playback.update(10, [262])
    assert playback.frame == 2 and playback.paused
    playback.update(10, [269])
    assert playback.frame == 3
    playback.update(10, [268, 61, 32])
    assert playback.frame == 0 and playback.speed == 2 and not playback.paused
    playback.update(10.1)
    assert playback.frame in range(4)


def test_playback_stops_on_terminal_and_loops():
    playback = Playback(frames=3, fps=20, loop=False, deadline=0.05)
    playback.update(0.3)
    assert playback.frame == 2 and playback.paused
    playback = Playback(frames=3, fps=20, deadline=0.05)
    playback.update(0.16)
    assert playback.frame == 0 and not playback.paused


def test_rollout_resolution_and_view_cli(tmp_path):
    raw = {"episode_id": 7, "archive": "episode-000007.npz"}
    (tmp_path / "episode-000007.json").write_text(json.dumps(raw))
    assert rollout_path(tmp_path, 7) == (raw, tmp_path / raw["archive"])
    with pytest.raises(ValueError, match="not found"):
        rollout_path(tmp_path, 0)
    exported = {"episode_index": 0, "replay": "replay/episode-000000.npz"}
    (tmp_path / "manifest.json").write_text(json.dumps({"episodes": [exported]}))
    assert rollout_path(tmp_path, 0) == (exported, tmp_path / exported["replay"])
    args = parse_args(
        ["view", "--root", str(tmp_path), "--camera", "wrist", "--paused", "--no-loop"]
    )
    assert args.camera == "wrist" and args.paused and not args.loop
