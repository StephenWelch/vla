"""OCBench configuration for the native LeRobot training worker."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vla_tools.config import parse_args
from vla_tools.train import TrainingConfig as BaseConfig
from vla_tools.train import normalize_record, train

from .training import validate_profiles


@dataclass
class TrainingConfig(BaseConfig):
    policy_type: Literal["act", "smolvla", "pi05"] = "act"
    video_backend: Literal["pyav", "torchcodec"] = "torchcodec"
    eval_max_steps: int = 2500
    action_mode: Literal["delta", "absolute_gripper", "absolute"] = "delta"
    percentile_normalization: bool = False


@dataclass(kw_only=True)
class TrainConfig(TrainingConfig):
    dataset: Path
    output: Path


def read_config(values):
    # Saved specs from the active comparison must still launch its queued arm.
    return normalize_record(values) if "backend" in values else values


def main():
    train(
        parse_args(TrainConfig, transform=read_config),
        validate_profiles,
        "ocbench_mjwarp.training",
    )


if __name__ == "__main__":
    main()
