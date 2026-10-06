"""Native OCBench workflow commands; defaults < YAML < CLI."""

import sys
from dataclasses import dataclass
from pathlib import Path

from vla_tools.config import parse_args

from .config import TASK, CollectionConfig


@dataclass
class ViewConfig:
    root: Path
    episode: int = 0


def view(config):
    import time

    import mujoco
    import mujoco.viewer
    import numpy as np
    import ocbench

    env = ocbench.make("block-cpu-double-task2-v0")
    model = env.unwrapped.model
    data = mujoco.MjData(model)
    with (
        np.load(config.root / "raw" / f"episode-{config.episode:06d}.npz") as archive,
        mujoco.viewer.launch_passive(model, data) as viewer,
    ):
        frame = 0
        while viewer.is_running():
            start = time.monotonic()
            for key in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat"):
                getattr(data, key)[:] = archive[f"sim/{key}"][frame]
            mujoco.mj_forward(model, data)
            viewer.sync()
            frame = (frame + 1) % len(archive["sim/qpos"])
            time.sleep(max(0, 0.02 - (time.monotonic() - start)))
    env.close()


def main():
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "list-tasks"
    if command == "list-tasks":
        print(f"{TASK}: Stack one block on top of the other (audited integration)")
    elif command == "generate":
        from .collect import generate

        generate(parse_args(CollectionConfig))
    elif command == "render":
        from .dataset import RenderConfig, render

        render(parse_args(RenderConfig))
    elif command == "export":
        from .dataset import ExportConfig, export

        print(export(parse_args(ExportConfig)))
    elif command == "view":
        view(parse_args(ViewConfig))
    else:
        raise ValueError("Commands: list-tasks, generate, render, export, view")


if __name__ == "__main__":
    main()
