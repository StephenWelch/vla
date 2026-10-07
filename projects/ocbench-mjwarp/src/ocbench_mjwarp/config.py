"""Explicit native action and collection contracts."""

from dataclasses import dataclass, field
from pathlib import Path

from vla_tools.tracking import WandbConfig

COMMIT = "e2cd2f72110b66bd65afab1b855d81ebc73aeacc"
TASK = "block-double-task2-v0"
ACTION = {
    "name": "ocbench-normalized-joint-delta-v1",
    "fps": 50,
    "scales": [0.18, 0.18, 0.18, 0.36, 0.36, 0.36, 0.12],
    "bounds": [-1.0, 1.0],
    "gripper": "normalized-opening-delta",
}
FIELDS = ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "time")
ABSOLUTE_GRIPPER_ACTION = ACTION | {
    "name": "ocbench-joint-delta-absolute-gripper-v1",
    "scales": [0.18, 0.18, 0.18, 0.36, 0.36, 0.36, 1.0],
    "gripper": "absolute-normalized-opening",
    "gripper_bounds": [0.0, 1.0],
}
ARM_LIMITS = [6.2831, 6.2831, 3.1415, 6.2831, 6.2831, 6.2831]
ABSOLUTE_ACTION = ABSOLUTE_GRIPPER_ACTION | {
    "name": "ocbench-absolute-joint-position-v1",
    "arm": "absolute-joint-position-radians",
    "scales": [1.0] * 7,
    "bounds": [-6.2831, 6.2831],
    "arm_bounds": [[-limit, limit] for limit in ARM_LIMITS],
}


@dataclass
class CollectionConfig:
    output: Path
    episodes: int = 500
    seed: int = 83000
    oracle_seed: int = 93000
    worlds: int = 32
    max_steps: int = 2500
    task: str = TASK
    max_nonpad_penetration: float = 0.001
    max_penetration: float = 0.003
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(project="vla-ocbench")
    )

    def __post_init__(self):
        if self.task != TASK:
            raise ValueError(
                "The audited integration currently supports block-double-task2-v0"
            )
        if min(self.episodes, self.worlds, self.max_steps) < 1:
            raise ValueError("Episode, world and step budgets must be positive")
        if min(self.max_nonpad_penetration, self.max_penetration) < 0:
            raise ValueError("Penetration tolerances must be nonnegative")


def validate_action(profile):
    if profile not in (ACTION, ABSOLUTE_GRIPPER_ACTION, ABSOLUTE_ACTION):
        raise ValueError("Expected a supported versioned OCBench 50 Hz action contract")
