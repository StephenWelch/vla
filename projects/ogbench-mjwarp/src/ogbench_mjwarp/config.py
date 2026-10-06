from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Literal


@dataclass
class CuroboConfig:
    execution: Literal["timed", "waypoint"] = "waypoint"
    ik_seeds: int = 32
    trajectory_seeds: int = 4
    attempts: int = 3
    max_velocity: float = 1.0
    max_acceleration: float = 2.0
    tracking_tolerance: float = 0.05
    phase_timeout: int = 200

    def __post_init__(self):
        if self.execution not in ("timed", "waypoint"):
            raise ValueError("Unknown cuRobo executor")
        if (
            min(self.ik_seeds, self.trajectory_seeds, self.attempts, self.phase_timeout)
            < 1
        ):
            raise ValueError("cuRobo budgets must be positive")
        if any(
            not isfinite(x) or x <= 0
            for x in (self.max_velocity, self.max_acceleration, self.tracking_tolerance)
        ):
            raise ValueError("cuRobo limits must be finite and positive")


@dataclass
class SplineConfig:
    """Whole-attempt sampling in SI units, except explicitly named degree bounds."""

    variation: Literal["nominal", "moderate", "mixed"] = "mixed"
    challenging_fraction: float = 0.2
    grasp_candidates: int = 4
    grasp_selection: Literal["nearest", "random"] = "nearest"
    ik_seeds: int = 16
    trajectory_seeds: int = 2
    max_jerk: float = 50.0
    # Opt-in joint-space corridor around the original curve, in radians.
    waypoint_relaxation: float = 0.0
    guide_position_tolerance: float = 0.03
    guide_rotation_tolerance: float = 0.25
    retiming: Literal["uniform", "local"] = "uniform"
    transit_linear_speed: float = 0.3
    transit_angular_speed: float = 1.5
    grasp_offset: tuple[float, float] = (0.005, 0.008)
    yaw_degrees: tuple[float, float] = (10.0, 15.0)
    tilt_degrees: tuple[float, float] = (10.0, 20.0)
    placement_offset: tuple[float, float] = (0.002, 0.008)
    path_offset: tuple[float, float] = (0.01, 0.03)
    duration_min: tuple[float, float] = (0.85, 0.65)
    duration_max: tuple[float, float] = (1.2, 1.4)
    dwell_min: float = 0.2
    dwell_max: float = 0.6
    approach_offset: tuple[float, float] = (0.0, 0.0)
    approach_height_min: tuple[float, float] = (0.14, 0.14)
    approach_height_max: tuple[float, float] = (0.14, 0.14)
    speed_min: tuple[float, float] = (1.0, 1.0)
    speed_max: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self):
        if self.retiming not in ("uniform", "local"):
            raise ValueError("Unknown spline retiming")
        if self.grasp_selection not in ("nearest", "random"):
            raise ValueError("Unknown grasp selection")
        if any(
            not isfinite(x) or x <= 0
            for x in (self.transit_linear_speed, self.transit_angular_speed)
        ):
            raise ValueError("Event speed limits must be finite and positive")
        if self.variation not in ("nominal", "moderate", "mixed"):
            raise ValueError("Unknown spline variation")
        if not 0 <= self.challenging_fraction <= 1:
            raise ValueError("Challenging fraction must be in [0, 1]")
        if min(self.grasp_candidates, self.ik_seeds, self.trajectory_seeds) < 1:
            raise ValueError("Spline planning budgets must be positive")
        if not isfinite(self.max_jerk) or self.max_jerk <= 0:
            raise ValueError("Spline jerk limit must be positive and finite")
        for name in (
            "waypoint_relaxation",
            "guide_position_tolerance",
            "guide_rotation_tolerance",
        ):
            if not isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Invalid smoothing bound: {name}")
        for name in (
            "grasp_offset",
            "yaw_degrees",
            "tilt_degrees",
            "placement_offset",
            "path_offset",
            "duration_min",
            "duration_max",
            "approach_offset",
            "approach_height_min",
            "approach_height_max",
            "speed_min",
            "speed_max",
        ):
            values = tuple(getattr(self, name))
            if len(values) != 2 or any(not isfinite(x) or x < 0 for x in values):
                raise ValueError(f"Invalid spline bounds: {name}")
            setattr(self, name, values)
        if any(
            not 0 < lo <= hi for lo, hi in zip(self.duration_min, self.duration_max)
        ):
            raise ValueError("Invalid duration bounds")
        if not 0 < self.dwell_min <= self.dwell_max < float("inf"):
            raise ValueError("Invalid dwell bounds")
        for lo, hi in zip(
            self.approach_height_min, self.approach_height_max, strict=True
        ):
            if not 0 < lo <= hi:
                raise ValueError("Invalid approach heights")
        for lo, hi in zip(self.speed_min, self.speed_max, strict=True):
            if not 0 < lo <= hi <= 1:
                raise ValueError("Execution speeds must be in (0, 1]")


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
    backend: Literal["cem", "curobo", "spline"] = "cem"
    curobo: CuroboConfig = field(default_factory=CuroboConfig)
    spline: SplineConfig = field(default_factory=SplineConfig)
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
        if self.backend not in ("cem", "curobo", "spline"):
            raise ValueError("Unknown planner backend")
        if isinstance(self.curobo, dict):
            self.curobo = CuroboConfig(**self.curobo)
        if isinstance(self.spline, dict):
            self.spline = SplineConfig(**self.spline)
        if self.backend == "spline" and self.joint_target_noise:
            raise ValueError("Spline variation replaces post-IK joint target noise")
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

    @property
    def joint_actions(self):
        return self.backend in ("curobo", "spline")

    def to_dict(self):
        return asdict(self)
