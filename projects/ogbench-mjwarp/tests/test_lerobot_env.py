"""Evaluation outcomes must preserve contact failures and freeze finished worlds."""

from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch

pytest.importorskip("lerobot")
from lerobot.envs.factory import make_env_config
from ogbench_mjwarp.lerobot_env import OGBenchEnvConfig, OGBenchVectorEnv


class FakeSim:
    device = "cpu"

    def __init__(self):
        self.state = {"qpos": torch.zeros(2, 1)}
        self.contact_depth = torch.zeros(2, 2)
        self.success = torch.tensor([False, False])
        self.valid = torch.tensor([True, True])
        self.rendered = []

    def tensor(self, value):
        return torch.as_tensor(value)

    def snapshot(self):
        return {key: value.clone() for key, value in self.state.items()}

    def restore(self, state, world_mask=None):
        for key in state:
            self.state[key][world_mask] = state[key][world_mask]

    def step(self, actions):
        self.state["qpos"] += 1
        return self.success, self.valid

    def render_batch(self):
        self.rendered.append("batch")
        image = np.stack(
            [
                np.full((16, 16, 3), int(self.state["qpos"][world, 0]), dtype=np.uint8)
                for world in range(2)
            ]
        )
        return {"front": image, "wrist": image}

    def proprioception(self):
        return self.state["qpos"].expand(2, 18)

    def contact_valid(self):
        return (self.contact_depth[:, 0] <= 0.001) & (self.contact_depth[:, 1] <= 0.003)


def fake_env():
    env = OGBenchVectorEnv.__new__(OGBenchVectorEnv)
    env.num_envs = 2
    env.single_action_space = gym.spaces.Box(-1, 1, (5,), np.float32)
    env.sim = FakeSim()
    env.config = SimpleNamespace(max_steps=3, image_size=16)
    env.done = np.zeros(2, dtype=bool)
    env.elapsed = np.zeros(2, dtype=int)
    env.contact_valid = np.ones(2, dtype=bool)
    env.physics_valid = np.ones(2, dtype=bool)
    env.task_success = np.zeros(2, dtype=bool)
    env.peaks = np.zeros((2, 2))
    env.task_metadata = [{"seed": 7}, {"seed": 8}]
    env.records = []
    env.observe = lambda world_mask=None: {}
    env.closed = True  # No real renderer to release.
    return env


def test_registry_and_contract():
    config = make_env_config("ogbench", task="scene-v0", task_ids=[2, 3])
    assert config.features["observation.state"].shape == (18,)
    assert config.features["action"].shape == (5,)
    assert config.features["observation.images.wrist"].shape == (3, 480, 640)
    with pytest.raises(ValueError):
        OGBenchEnvConfig(task_ids=[])


def test_contact_failure_is_sticky_and_disqualifies_later_success():
    env = fake_env()
    env.sim.contact_depth[0, 0] = 0.002
    env.step(np.zeros((2, 5)))
    env.sim.contact_depth.zero_()
    env.sim.success[0] = True
    _, reward, terminated, _, info = env.step(np.zeros((2, 5)))
    assert terminated.tolist() == [True, False]
    assert reward[0] == 0 and not info["is_success"][0]
    assert env.records[0]["task_success"] and not env.records[0]["contact_valid"]
    assert env.records[0]["peak_nonpad_penetration"] == pytest.approx(0.002)


def test_finished_world_freezes_and_timeout_records_once():
    env = fake_env()
    env.sim.success[0] = True
    _, reward, terminated, truncated, _ = env.step(np.zeros((2, 5)))
    assert reward.tolist() == [1, 0]
    assert terminated.tolist() == [True, False] and not truncated.any()
    frozen = env.sim.state["qpos"][0].clone()
    env.step(np.zeros((2, 5)))
    _, _, _, truncated, _ = env.step(np.zeros((2, 5)))
    assert truncated.tolist() == [False, True]
    assert torch.equal(env.sim.state["qpos"][0], frozen)
    assert len(env.records) == 2
    assert [r["steps"] for r in env.records] == [1, 3]


def test_nonfinite_action_rejected():
    env = fake_env()
    with pytest.raises(ValueError):
        env.step(np.full((2, 5), np.nan))


def test_invalid_physics_disqualifies_task_success():
    env = fake_env()
    env.sim.success[0] = True
    env.sim.valid[0] = False
    _, reward, terminated, _, info = env.step(np.zeros((2, 5)))
    assert terminated[0] and not info["is_success"][0] and reward[0] == 0
    assert not env.records[0]["physics_valid"]


def test_terminal_images_render_once_then_cache():
    env = fake_env()
    del env.observe
    env.observe()
    env.sim.success[0] = True
    obs, *_ = env.step(np.zeros((2, 5)))
    terminal = obs["pixels"]["front"][0].copy()
    previous_active = obs["pixels"]["front"][1]
    obs, *_ = env.step(np.zeros((2, 5)))
    np.testing.assert_array_equal(obs["pixels"]["front"][0], terminal)
    assert env.sim.rendered == ["batch"] * 3
    assert not np.array_equal(obs["pixels"]["front"][1], terminal)
    np.testing.assert_array_equal(previous_active, terminal)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task_ids": [1, 1]},
        {"max_penetration": float("nan")},
        {"max_nonpad_penetration": -1},
    ],
)
def test_invalid_environment_config(kwargs):
    with pytest.raises(ValueError):
        OGBenchEnvConfig(**kwargs)


def test_partial_task_creation_closes_existing_worlds(monkeypatch):
    import ogbench_mjwarp.lerobot_env as module

    env = SimpleNamespace(close=Mock())
    monkeypatch.setattr(
        module,
        "OGBenchVectorEnv",
        Mock(side_effect=[env, RuntimeError("task reset failed")]),
    )
    with pytest.raises(RuntimeError, match="task reset failed"):
        OGBenchEnvConfig(task_ids=[1, 2]).create_envs(1)
    env.close.assert_called_once()


def test_failed_simulator_construction_closes_cpu_env(monkeypatch):
    import ogbench_mjwarp.lerobot_env as module

    env = SimpleNamespace(close=Mock())
    monkeypatch.setattr(module, "make_env", Mock(return_value=env))
    monkeypatch.setattr(
        module, "BatchEnvironment", Mock(side_effect=RuntimeError("CUDA unavailable"))
    )
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        OGBenchVectorEnv(OGBenchEnvConfig(), 1, 1)
    env.close.assert_called_once()


@pytest.mark.gpu
def test_gpu_vector_reset_observation_and_timeout():
    config = OGBenchEnvConfig(max_steps=2, image_size=16)
    env = config.create_envs(2)[config.task][1]
    try:
        obs, _ = env.reset(seed=[7, 8])
        assert obs["agent_pos"].shape == (2, 18)
        np.testing.assert_array_equal(obs["observation.state"], obs["agent_pos"])
        assert obs["pixels"]["wrist"].shape == (2, 16, 16, 3)
        assert np.isfinite(obs["agent_pos"]).all()
        first = obs["agent_pos"].copy()
        env.step(np.zeros((2, 5)))
        _, _, _, truncated, _ = env.step(np.zeros((2, 5)))
        assert truncated.all() and len(env.records) == 2
        assert all(record["physics_valid"] for record in env.records)
        obs, _ = env.reset(seed=[7, 8])
        np.testing.assert_allclose(obs["agent_pos"], first, atol=1e-6)
        assert len(env.get_attr("task_description")) == 2
        assert env.call("render")[0].shape == (16, 32, 3)
    finally:
        env.close()
