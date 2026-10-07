import json
from types import SimpleNamespace

import numpy as np
import pytest
from ocbench_mjwarp import replay_check
from ocbench_mjwarp.config import ACTION, FIELDS, TASK


@pytest.mark.parametrize("corrupt", [False, True])
def test_replay_detects_action_alignment_errors(monkeypatch, tmp_path, corrupt):
    raw = tmp_path / "raw"
    raw.mkdir()
    actions = np.ones((3, 7), np.float32)
    if corrupt:
        actions[0] = 2
    np.savez(
        raw / "episode-000000.npz",
        **{
            **{f"sim/{k}": np.arange(4, dtype=np.float32)[:, None] for k in FIELDS},
            "state": np.arange(3, dtype=np.float32)[:, None],
            "action": actions,
        },
    )
    (raw / "episode-000000.json").write_text(
        json.dumps(
            {
                "episode_id": 0,
                "seed": 10,
                "length": 3,
                "archive": "episode-000000.npz",
                "native_success": True,
                "physical_valid": True,
                "env_id": TASK,
                "action_profile": ACTION,
            }
        )
    )

    class Env:
        def __init__(self, cfg, n):
            self.value = 0
            self.native = np.array([False])
            self.valid = np.array([True])
            self.done = np.array([False])
            self.sim = SimpleNamespace(
                snapshot=lambda: {k: np.array([[self.value]]) for k in FIELDS}
            )

        def reset(self, seed, options):
            assert seed == [10]
            assert options["states"]["qpos"][0, 0] == 0
            return {"observation.state": np.array([[0.0]])}, {}

        def step(self, action):
            self.value += action[0, 0]
            self.native[:] = self.value >= 3
            return (
                {"observation.state": np.array([[self.value]])},
                None,
                self.native.copy(),
                np.array([False]),
                {},
            )

        def close(self):
            pass

    monkeypatch.setattr(replay_check, "OCBenchVectorEnv", Env)
    result = replay_check.run(replay_check.Config(tmp_path, tmp_path / "report", (0,)))
    assert result["passed"] is (not corrupt)
    assert (result["episodes"][0]["first_mismatch"] is not None) is corrupt
    assert (
        json.loads((tmp_path / "report/report.json").read_text())["status"]
        == "complete"
    )
    with pytest.raises(FileExistsError):
        replay_check.run(replay_check.Config(tmp_path, tmp_path / "report", (0,)))
