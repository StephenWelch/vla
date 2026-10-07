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

    from .episodes import records

    row = next(r for r in records(config.root) if r["episode_id"] == config.episode)
    env = ocbench.make("block-cpu-double-task2-v0")
    model = env.unwrapped.model
    data = mujoco.MjData(model)
    with (
        np.load(config.root / "raw" / row["archive"]) as archive,
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
    elif command == "import-hf":
        from .hub import ImportConfig, import_dataset

        print(import_dataset(parse_args(ImportConfig)))
    elif command == "render":
        from .dataset import RenderConfig, render

        render(parse_args(RenderConfig))
    elif command == "export":
        from .dataset import ExportConfig, export

        print(export(parse_args(ExportConfig)))
    elif command == "benchmark-export":
        from .benchmark_export import BenchmarkConfig, run

        run(parse_args(BenchmarkConfig))
    elif command == "prepare-dataset":
        from .prepare import PrepareConfig, prepare_dataset

        print(prepare_dataset(parse_args(PrepareConfig)))
    elif command == "evaluate-recorded":
        from vla_tools.evaluate import EvalConfig, evaluate

        from .actions import absolute_targets
        from .config import ABSOLUTE_ACTION, ACTION
        from .profile import profiles

        config = parse_args(EvalConfig)
        _, actions = profiles(config.dataset, config.checkpoint)
        transform = (
            None
            if actions == ACTION
            else lambda action, state: absolute_targets(
                action, state, actions == ABSOLUTE_ACTION
            )
        )
        print(evaluate(config, profiles, transform))
    elif command == "view":
        view(parse_args(ViewConfig))
    elif command == "replay-check":
        from .replay_check import main as replay_main

        replay_main()
    else:
        raise ValueError(
            "Commands: list-tasks, generate, import-hf, render, export, benchmark-export, prepare-dataset, evaluate-recorded, view, replay-check"
        )


if __name__ == "__main__":
    main()
