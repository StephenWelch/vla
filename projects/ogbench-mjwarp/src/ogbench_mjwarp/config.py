from dataclasses import asdict, dataclass
from math import isfinite


@dataclass
class RandomizationConfig:
    """Seeded strategy and waypoint variation; distances in meters, angles in radians."""

    seed: int | None = None
    variants_per_reset: int = 1
    order: bool = False
    cube_grasps: bool = False
    handle_grasps: bool = False
    position_noise: float = 0.0
    yaw_noise: float = 0.0
    duration_scale_min: float = 1.0
    duration_scale_max: float = 1.0

    def __post_init__(self):
        if self.variants_per_reset < 1 or (self.seed is not None and self.seed < 0):
            raise ValueError("Need positive variants_per_reset and nonnegative seed")
        for name in (
            "position_noise",
            "yaw_noise",
            "duration_scale_min",
            "duration_scale_max",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.duration_scale_min <= self.duration_scale_max:
            raise ValueError("Need 0 < duration_scale_min <= duration_scale_max")


@dataclass
class PlannerConfig:
    """Sampling MPC budget, objective weights, and physics capacities."""

    # Concurrent execution worlds; Generate.episodes is the total attempt count.
    episodes: int = 32
    candidates: int = 8
    horizon: int = 8
    iterations: int = 2
    elite_fraction: float = 0.1
    noise: float = 0.12
    gripper_noise: float = 0.0
    # Per-episode uniform offsets to arm joint targets after IK, in radians.
    joint_target_noise: float = 0.0
    min_std: float = 0.015
    tracking_weight: float = 100.0
    action_weight: float = 0.01
    smoothness_weight: float = 0.02
    task_weight: float = 20.0
    contact_weight: float = 100.0
    max_nonpad_penetration: float = 0.001
    max_penetration: float = 0.003
    nconmax: int = 128
    njmax: int = 512

    def __post_init__(self):
        for key in (
            "episodes",
            "candidates",
            "horizon",
            "iterations",
            "nconmax",
            "njmax",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.candidates < 2 or not 0 < self.elite_fraction <= 1:
            raise ValueError("Need >=2 candidates and elite_fraction in (0, 1]")
        for key in (
            "noise",
            "gripper_noise",
            "joint_target_noise",
            "min_std",
            "tracking_weight",
            "action_weight",
            "smoothness_weight",
            "task_weight",
            "contact_weight",
            "max_nonpad_penetration",
            "max_penetration",
        ):
            if not isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")

    def to_dict(self):
        return asdict(self)
