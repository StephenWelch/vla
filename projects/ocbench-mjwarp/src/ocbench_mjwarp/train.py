"""OCBench defaults for the shared LeRobot trainer."""

from dataclasses import dataclass
from typing import Literal

from vla_tools.config import parse_args
from vla_tools.train import TrainConfig as BaseConfig
from vla_tools.train import train


@dataclass
class TrainConfig(BaseConfig):
    backend: Literal["ocbench"] = "ocbench"
    policy_type: Literal["act", "smolvla", "pi05"] = "act"
    eval_max_steps: int = 2500


def main():
    train(parse_args(TrainConfig))


if __name__ == "__main__":
    main()
