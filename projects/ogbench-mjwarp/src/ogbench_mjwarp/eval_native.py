"""Run the native LeRobot evaluation CLI with the local OGBench extension."""

import json
import sys
from pathlib import Path

from lerobot.scripts.lerobot_eval import eval_main

import ogbench_mjwarp.lerobot_env  # noqa: F401
from ogbench_mjwarp.lerobot_env import OGBenchEnvConfig


def main():
    # Native LeRobot resolves its config itself. Bind the image contract before
    # its factory creates any OGBench environments.
    args = sys.argv[1:]
    checkpoint = next(
        (arg.split("=", 1)[1] for arg in args if arg.startswith("--policy.path=")), None
    )
    if checkpoint is None and "--policy.path" in args:
        checkpoint = args[args.index("--policy.path") + 1]
    if checkpoint is None or not (Path(checkpoint) / "rendering.json").exists():
        raise ValueError(
            "Native OGBench eval requires --policy.path with a v2 rendering.json"
        )
    profile = json.loads((Path(checkpoint) / "rendering.json").read_text())
    action_path = Path(checkpoint) / "action_profile.json"
    actions = json.loads(action_path.read_text()) if action_path.exists() else None
    original = OGBenchEnvConfig.create_envs

    def create_envs(self, *args, **kwargs):
        self.rendering = profile
        self.action_profile = actions
        self.__post_init__()
        return original(self, *args, **kwargs)

    OGBenchEnvConfig.create_envs = create_envs
    eval_main()


if __name__ == "__main__":
    main()
