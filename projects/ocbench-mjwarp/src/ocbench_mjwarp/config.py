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
    if profile != ACTION:
        raise ValueError(
            "Expected OCBench 50 Hz normalized joint deltas; incompatible action contract"
        )
