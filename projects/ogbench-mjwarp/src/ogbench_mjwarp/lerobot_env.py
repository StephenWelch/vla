"""LeRobot environment registration backed by the generation simulator."""

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig

from .config import CuroboConfig, PlannerConfig
from .environment import BatchEnvironment
from .io import jsonable
from .tasks import image_shape, make_env


@EnvConfig.register_subclass("ogbench")
@dataclass
class OGBenchEnvConfig(EnvConfig):
    task: str = "cube-single-v0"
    task_ids: list[int] = field(default_factory=lambda: [1])
    fps: int = 20
    image_size: int | tuple[int, int] = (480, 640)
    max_steps: int = 250
    device: str = "cuda:0"
    max_nonpad_penetration: float = 0.001
    max_penetration: float = 0.003
    rendering: dict | None = None
    action_profile: dict | None = None

    def __post_init__(self):
        if self.action_profile:
            from .actions import validate_profile

            validate_profile(self.action_profile)
            if self.task != "cube-double-v0" or self.task_ids != [5]:
                raise ValueError("Joint action pilot supports cube-double-v0 task 5")
        if (
            not self.task_ids
            or min(self.task_ids) < 1
            or len(set(self.task_ids)) != len(self.task_ids)
        ):
            raise ValueError("Provide unique positive OGBench task IDs")
        shape = image_shape(self.image_size)
        if self.max_steps < 1 or self.fps != 20:
            raise ValueError("Need image_size >=16, max_steps >0, and native 20 Hz")
        thresholds = (self.max_nonpad_penetration, self.max_penetration)
        if not np.isfinite(thresholds).all() or min(thresholds) < 0:
            raise ValueError("Contact thresholds must be finite and nonnegative")
        self.features = {
            "observation.state": PolicyFeature(FeatureType.STATE, (18,)),
            "action": PolicyFeature(
                FeatureType.ACTION, (7 if self.action_profile else 5,)
            ),
            **{
                f"observation.images.{view}": PolicyFeature(
                    FeatureType.VISUAL, (3, *shape)
                )
                for view in ("front", "wrist")
            },
        }
        self.features_map = {key: key for key in self.features}

    @property
    def gym_kwargs(self):
        return {}

    def create_envs(self, n_envs, use_async_envs=False):
        # MJWarp batches physics on one GPU; no subprocess vectorization is needed.
        if n_envs < 1:
            raise ValueError("n_envs must be positive")
        envs = {}
        try:
            for task_id in self.task_ids:
                envs[task_id] = OGBenchVectorEnv(self, n_envs, task_id)
        except BaseException:
            for env in envs.values():
                env.close()
            raise
        return {self.task: envs}


class OGBenchVectorEnv(gym.vector.VectorEnv):
    """Batched GPU physics and cameras with sticky episode outcomes.

    Finished worlds freeze until the next explicit reset. This matches LeRobot's
    rollout masking and prevents autoreset contacts from contaminating outcomes.
    """

    def __init__(self, config, n_envs, task_id):
        self.closed = True
        self.config, self.num_envs, self.task_id = config, n_envs, task_id
        self.metadata = {
            "render_fps": 20,
            "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
        }
        self.render_mode = "rgb_array"
        self.single_action_space = gym.spaces.Box(-1, 1, (5,), np.float32)
        if config.action_profile:
            bounds = np.asarray(config.action_profile["bounds"], dtype=np.float32)
            self.single_action_space = gym.spaces.Box(bounds[:, 0], bounds[:, 1])
        image = gym.spaces.Box(0, 255, (*image_shape(config.image_size), 3), np.uint8)
        self.single_observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(-np.inf, np.inf, (18,), np.float32),
                "observation.state": gym.spaces.Box(-np.inf, np.inf, (18,), np.float32),
                "pixels": gym.spaces.Dict({"front": image, "wrist": image}),
            }
        )
        self.action_space = batch_space(self.single_action_space, n_envs)
        self.observation_space = batch_space(self.single_observation_space, n_envs)
        env = make_env(config.task, task_id=task_id, size=config.image_size)
        try:
            if config.rendering is not None:
                from .rendering import rendering_profile

                if (
                    rendering_profile(env.unwrapped._model, config.image_size)
                    != config.rendering
                ):
                    raise ValueError(
                        "Simulator and checkpoint rendering profiles differ"
                    )
            self.sim = BatchEnvironment(
                env,
                n_envs,
                PlannerConfig(
                    backend="curobo" if config.action_profile else "cem",
                    curobo=CuroboConfig(
                        **{
                            key: config.action_profile[key]
                            for key in ("max_velocity", "max_acceleration")
                        }
                    )
                    if config.action_profile
                    else CuroboConfig(),
                    max_nonpad_penetration=config.max_nonpad_penetration,
                    max_penetration=config.max_penetration,
                ),
                config.device,
            )
        except BaseException:
            env.close()
            raise
        self.records = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = np.random.default_rng().integers(0, 2**31, self.num_envs).tolist()
        elif isinstance(seed, int):
            seed = list(range(seed, seed + self.num_envs))
        _, self.task_metadata = self.sim.reset(seed, [self.task_id] * self.num_envs)
        if self.config.action_profile:
            self.sim.fields["joint_target_velocity"].zero_()
        self.task_metadata = jsonable(self.task_metadata)
        self.elapsed = np.zeros(self.num_envs, dtype=int)
        self.done = np.zeros(self.num_envs, dtype=bool)
        self.contact_valid = np.ones(self.num_envs, dtype=bool)
        self.peaks = np.zeros((self.num_envs, 2))
        self.task_success = np.zeros(self.num_envs, dtype=bool)
        self.physics_valid = np.ones(self.num_envs, dtype=bool)
        return self.observe(), {}

    def observe(self, world_mask=None):
        if world_mask is None:
            self.images = {
                view: np.empty(
                    (self.num_envs, *image_shape(self.config.image_size), 3),
                    dtype=np.uint8,
                )
                for view in ("front", "wrist")
            }
        worlds = (
            range(self.num_envs) if world_mask is None else np.flatnonzero(world_mask)
        )
        worlds = list(worlds)
        if worlds:
            images = self.sim.render_batch()
            for view in self.images:
                self.images[view][worlds] = images[view][worlds]
        state = self.sim.proprioception().cpu().numpy().copy()
        # LeRobot preprocesses agent_pos, while its recorder reads feature names.
        return {
            "agent_pos": state,
            "observation.state": state,
            "pixels": {view: image.copy() for view, image in self.images.items()},
        }

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if (
            actions.shape != (self.num_envs, *self.single_action_space.shape)
            or not np.isfinite(actions).all()
        ):
            raise ValueError(
                "Actions must be finite and match the environment action space"
            )
        active = ~self.done
        frozen = self.sim.snapshot() if self.done.any() else None
        success, valid = self.sim.step(
            self.sim.tensor(
                np.clip(
                    actions, self.single_action_space.low, self.single_action_space.high
                )
            )
        )
        success, valid = success.cpu().numpy(), valid.cpu().numpy()
        depths = self.sim.contact_depth.cpu().numpy()
        self.peaks[active] = np.maximum(self.peaks[active], depths[active])
        self.contact_valid[active] &= self.sim.contact_valid().cpu().numpy()[active]
        self.physics_valid[active] &= valid[active]
        self.task_success[active] |= success[active]
        self.elapsed[active] += 1
        terminated = active & (success | ~valid)
        truncated = active & ~terminated & (self.elapsed >= self.config.max_steps)
        finished = terminated | truncated
        is_success = (
            finished & self.task_success & self.contact_valid & self.physics_valid
        )
        for i in np.flatnonzero(finished):
            self.records.append(
                {
                    **self.task_metadata[i],
                    "steps": int(self.elapsed[i]),
                    "task_success": bool(self.task_success[i]),
                    "contact_valid": bool(self.contact_valid[i]),
                    "physics_valid": bool(self.physics_valid[i]),
                    "success": bool(is_success[i]),
                    "truncated": bool(truncated[i]),
                    "peak_nonpad_penetration": float(self.peaks[i, 0]),
                    "peak_penetration": float(self.peaks[i, 1]),
                }
            )
        if frozen is not None:
            mask = torch.as_tensor(self.done, device=self.sim.device)
            self.sim.restore(frozen, world_mask=mask)
        self.done |= finished
        return (
            self.observe(active),
            is_success.astype(np.float32),
            terminated,
            truncated,
            {"is_success": is_success},
        )

    def call(self, name, *args, **kwargs):
        if name == "_max_episode_steps":
            return (self.config.max_steps,) * self.num_envs
        if name in ("task", "task_description"):
            return tuple(row["instruction"] for row in self.task_metadata)
        if name == "render":
            return tuple(
                np.concatenate(
                    (self.images["front"][i], self.images["wrist"][i]), axis=1
                )
                for i in range(self.num_envs)
            )
        raise AttributeError(name)

    def get_attr(self, name):
        return self.call(name)

    def close_extras(self, **kwargs):
        self.sim.env.close()
