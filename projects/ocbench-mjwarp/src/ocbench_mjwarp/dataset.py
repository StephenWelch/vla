"""Deferred batched cameras and LeRobot export with explicit native labels."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from vla_tools.tracking import write_json
from vla_tools.video import VideoWriter

from .config import FIELDS
from .environment import Simulation
from .episodes import load_arrays, select_episodes


@dataclass
class RenderConfig:
    source: Path
    batch_size: int = 4
    limit: int | None = None


def render(config):
    selected = select_episodes(config.source, limit=config.limit)
    output = config.source / "rendered"
    output.mkdir(exist_ok=True)
    for offset in range(0, len(selected), config.batch_size):
        batch = [
            r
            for r in selected[offset : offset + config.batch_size]
            if not (output / f"{r['episode_id']:06d}.json").exists()
        ]
        if not batch:
            continue
        if shutil.disk_usage(output).free < 4 * 2**30:
            raise RuntimeError(
                "Less than 4 GiB free; completed render batches and raw attempts are retained"
            )
        sim = Simulation([r["seed"] for r in batch], audit=False)
        archives = load_arrays(config.source, batch)
        writers = {}
        try:
            for row in batch:
                for view in ("front", "wrist"):
                    path = output / f"{row['episode_id']:06d}-{view}.tmp.mp4"
                    writers[row["episode_id"], view] = VideoWriter(path)
            for tick in range(max(r["length"] for r in batch)):
                sim.restore(
                    {
                        k: np.stack(
                            [
                                a[f"sim/{k}"][min(tick, r["length"] - 1)]
                                for a, r in zip(archives, batch)
                            ]
                        )
                        for k in FIELDS
                    },
                    forward=False,
                )
                images = sim.render()
                for world, row in enumerate(batch):
                    if tick < row["length"]:
                        for view in ("front", "wrist"):
                            writers[row["episode_id"], view].write(images[view][world])
            for process in writers.values():
                process.close()
            for row in batch:
                for view in ("front", "wrist"):
                    (output / f"{row['episode_id']:06d}-{view}.tmp.mp4").replace(
                        output / f"{row['episode_id']:06d}-{view}.mp4"
                    )
                write_json(
                    output / f"{row['episode_id']:06d}.json",
                    {
                        "episode_id": row["episode_id"],
                        "frames": row["length"],
                        "rendering": sim.renderer.profile,
                    },
                )
            print(
                f"Rendered {offset + len(batch)}/{len(selected)} episodes", flush=True
            )
        finally:
            for process in writers.values():
                process.close()
            sim.close()


@dataclass
class ExportConfig:
    source: Path
    output: Path
    successes: bool | None = True
    repo_id: str = "local/ocbench-stack"
    limit: int | None = None
    batch_size: int = 4
    image_size: tuple[int, int] = (480, 640)
    episode_order: Literal["length", "source"] = "length"
    overlap_commits: bool = True
    reuse_simulation: bool = True
    encoder_threads: int = 1
    encoder_queue_size: int = 8
    encoder_backend: str = "async"
    render_batch_frames: int = 4
    buffer_frames: int = 2
    write_buffer_bytes: int = 1048576
    worker_timeout_seconds: float = 180
    worker_retries: int = 2


def export(config):
    from .export_worker import supervise
    from .materialize import materialize

    return (
        supervise(config) if config.encoder_backend == "async" else materialize(config)
    )


def dataset_features(*, rewards=False, image_size=(480, 640)):
    return {
        **(
            {"next.reward": {"dtype": "float32", "shape": (1,), "names": None}}
            if rewards
            else {}
        ),
        "observation.state": {"dtype": "float32", "shape": (18,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        **{
            f"observation.images.{view}": {
                "dtype": "video",
                "shape": (3, *image_size),
                "names": ["channels", "height", "width"],
            }
            for view in ("front", "wrist")
        },
        **{
            f"next.{key}": {"dtype": "bool", "shape": (1,), "names": None}
            for key in ("success", "done", "truncated")
        },
    }
