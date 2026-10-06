"""LeRobot registration for native OCBench actions and batched cameras."""

from collections import deque
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import warp as wp
from gymnasium.vector.utils import batch_space
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig

from .config import ACTION, TASK, validate_action
from .environment import Simulation


@EnvConfig.register_subclass("ocbench")
@dataclass
class OCBenchEnvConfig(EnvConfig):
    task: str = TASK
    task_ids: list[int] = field(default_factory=lambda: [2])
    fps: int = 50
    image_size: tuple[int, int] = (480, 640)
    max_steps: int = 2500
    device: str = "cuda:0"
    rendering: dict | None = None
    action_profile: dict = field(default_factory=lambda: dict(ACTION))

    def __post_init__(self):
        validate_action(self.action_profile)
        if (
            self.task != TASK
            or self.task_ids != [2]
            or self.fps != 50
            or tuple(self.image_size) != (480, 640)
        ):
            raise ValueError("Expected native block-double-task2 at 50 Hz and 640x480")
        self.features = {
            "observation.state": PolicyFeature(FeatureType.STATE, (18,)),
            "action": PolicyFeature(FeatureType.ACTION, (7,)),
            **{
                f"observation.images.{v}": PolicyFeature(
                    FeatureType.VISUAL, (3, 480, 640)
                )
                for v in ("front", "wrist")
            },
        }
        self.features_map = {k: k for k in self.features}

    @property
    def gym_kwargs(self):
        return {}

    def create_envs(self, n_envs, use_async_envs=False):
        return {self.task: {2: OCBenchVectorEnv(self, n_envs)}}


class OCBenchVectorEnv(gym.vector.VectorEnv):
    def __init__(self, config, n_envs):
        self.config, self.num_envs = config, n_envs
        self.sim = None
        self.records = []
        self.initial_states = {}
        self.metadata = {
            "render_fps": 50,
            "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
        }
        self.single_action_space = gym.spaces.Box(-1, 1, (7,), np.float32)
        self.single_observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(-np.inf, np.inf, (18,), np.float32),
                "pixels": gym.spaces.Dict(
                    {
                        v: gym.spaces.Box(0, 255, (480, 640, 3), np.uint8)
                        for v in ("front", "wrist")
                    }
                ),
            }
        )
        self.action_space = batch_space(self.single_action_space, n_envs)
        self.observation_space = batch_space(self.single_observation_space, n_envs)

    def observation(self):
        self.images = self.sim.render()
        state = self.sim.state()
        return {"agent_pos": state, "observation.state": state, "pixels": self.images}

    def reset(self, *, seed=None, options=None):
        self.close()
        seeds = (
            list(seed)
            if isinstance(seed, (list, np.ndarray))
            else list(range(seed or 0, (seed or 0) + self.num_envs))
        )
        self.seeds = seeds
        self.sim = Simulation(seeds, self.config.task)
        if self.initial_states:
            from .config import FIELDS

            state = self.sim.snapshot()
            for i, s in enumerate(seeds):
                if s in self.initial_states:
                    with np.load(self.initial_states[s]) as saved:
                        for k in FIELDS:
                            state[k][i] = saved[f"sim/{k}"]
            self.sim.restore(state)
        if options and "states" in options:
            self.sim.restore(options["states"])
        self.done = np.zeros(self.num_envs, bool)
        self.native = np.zeros(self.num_envs, bool)
        self.valid = np.ones(self.num_envs, bool)
        self.finite = np.ones(self.num_envs, bool)
        self.steps = np.zeros(self.num_envs, int)
        self.peak = np.zeros((self.num_envs, 2))
        self.history = deque([self.sim.data.qpos.numpy().copy()], maxlen=51)
        observation = self.observation()
        if self.config.rendering and self.sim.renderer.profile != self.config.rendering:
            raise ValueError("Evaluation camera/model profile differs from dataset")
        return observation, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.num_envs, 7) or not np.isfinite(action).all():
            raise ValueError("Expected finite batched seven-dimensional actions")
        active = ~self.done
        frozen = self.sim.snapshot() if self.done.any() else None
        mask = wp.array(
            self.done.astype(np.int32), dtype=wp.int32, device=self.sim.warp_device
        )
        self.sim.env.step_joint_actions_gpu(
            wp.array(action, dtype=float, device=self.sim.warp_device), mask
        )
        self.steps += active
        self.history.append(self.sim.data.qpos.numpy().copy())
        self.peak[active] = np.maximum(
            self.peak[active], self.sim.depth.numpy()[active]
        )
        self.finite &= ~active | self.sim.physics_status()[0]
        healthy = self.sim.env._gpu_healthy.numpy().astype(bool)
        self.valid &= ~active | (
            healthy
            & self.finite
            & (self.peak[:, 0] <= 0.001)
            & (self.peak[:, 1] <= 0.003)
        )
        self.native |= active & self.sim.env._gpu_success.numpy().astype(bool)
        terminated = active & (self.native | ~healthy | ~self.finite)
        truncated = active & (self.steps >= self.config.max_steps) & ~terminated
        for i in np.flatnonzero(terminated | truncated):
            from .collect import stable_stack

            stable = stable_stack(self.sim, {"sim/qpos": np.stack(self.history)[:, i]})
            self.records.append(
                {
                    "seed": int(self.seeds[i]),
                    "steps": int(self.steps[i]),
                    "task_success": bool(self.native[i]),
                    "contact_valid": bool((self.peak[i] <= [0.001, 0.003]).all()),
                    "physics_valid": bool(self.finite[i] and healthy[i]),
                    "success": bool(self.native[i] and self.valid[i]),
                    "truncated": bool(truncated[i]),
                    "peak_penetration": self.peak[i].tolist(),
                    "stable_stack": stable["valid"],
                    "stable_stack_failures": stable["failures"],
                }
            )
        if frozen is not None:
            current = self.sim.snapshot()
            for k in current:
                current[k][self.done] = frozen[k][self.done]
            self.sim.restore(current)
        self.done |= terminated | truncated
        return (
            self.observation(),
            (self.native & self.valid).astype(float),
            terminated,
            truncated,
            {"is_success": self.native & self.valid},
        )

    def close(self):
        if self.sim:
            self.sim.close()
            self.sim = None

    def call(self, name, *args, **kwargs):
        if name == "_max_episode_steps":
            return (self.config.max_steps,) * self.num_envs
        if name in ("task", "task_description"):
            return ("Stack one block on top of the other.",) * self.num_envs
        if name == "render":
            return tuple(
                np.concatenate(
                    [self.images["front"][i], self.images["wrist"][i]], axis=1
                )
                for i in range(self.num_envs)
            )
        raise AttributeError(name)

    def get_attr(self, name):
        return self.call(name)
