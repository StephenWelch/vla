"""Deferred batched cameras and LeRobot export with explicit native labels."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vla_tools.tracking import write_json
from vla_tools.video import VideoWriter

from .collect import rows
from .config import ACTION, FIELDS, validate_action
from .environment import Simulation


@dataclass
class RenderConfig:
    source: Path
    batch_size: int = 4
    limit: int | None = None


def render(config):
    selected = [r for r in rows(config.source) if r["physical_valid"]]
    if config.limit is not None:
        selected = selected[: config.limit]
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
        archives = []
        for row in batch:
            with np.load(config.source / "raw" / row["archive"]) as archive:
                archives.append({f"sim/{k}": archive[f"sim/{k}"] for k in FIELDS})
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


def video_frames(path):
    import av

    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


@dataclass
class ExportConfig:
    source: Path
    output: Path
    successes: bool = True
    repo_id: str = "local/ocbench-stack"
    limit: int | None = None


def export(config):
    from lerobot.configs.video import RGBEncoderConfig
    from vla_tools.encoding import StreamingDataset

    selected = [
        r
        for r in rows(config.source)
        if r["physical_valid"] and r["native_success"] == config.successes
    ]
    if config.limit is not None:
        selected = selected[: config.limit]
    if not selected:
        return {"episodes": 0, "frames": 0}
    if config.output.exists():
        manifest = config.output / "manifest.json"
        if not (config.output / "INCOMPLETE.json").exists() and manifest.exists():
            saved = json.loads(manifest.read_text())
            if [r["episode_id"] for r in saved["episodes"]] == [
                r["episode_id"] for r in selected
            ]:
                return {
                    "episodes": len(selected),
                    "frames": sum(r["length"] for r in selected),
                }
        raise FileExistsError(
            "Existing export is incomplete or has different selection; use a fresh output"
        )
    features = {
        "observation.state": {"dtype": "float32", "shape": (18,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        **{
            f"observation.images.{view}": {
                "dtype": "video",
                "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            }
            for view in ("front", "wrist")
        },
        **{
            f"next.{key}": {"dtype": "bool", "shape": (1,), "names": None}
            for key in ("success", "done", "truncated")
        },
    }
    dataset = StreamingDataset.create(
        config.repo_id,
        root=config.output,
        fps=50,
        robot_type="ocbench_ur5e_joint_delta",
        features=features,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18),
        streaming_encoding=True,
        encoder_queue_maxsize=30,
        encoder_threads=2,
        image_writer_threads=0,
    )
    write_json(config.output / "INCOMPLETE.json", {"episodes": 0})
    manifest = []
    profile = None
    for index, row in enumerate(selected):
        validate_action(row["action_profile"])
        record = json.loads(
            (config.source / "rendered" / f"{row['episode_id']:06d}.json").read_text()
        )
        if profile is not None and profile != record["rendering"]:
            raise ValueError("Mixed camera/model profiles")
        profile = record["rendering"]
        streams = {
            v: video_frames(
                config.source / "rendered" / f"{row['episode_id']:06d}-{v}.mp4"
            )
            for v in ("front", "wrist")
        }
        with np.load(config.source / "raw" / row["archive"]) as archive:
            a = {k: archive[k] for k in archive.files}
            for tick in range(row["length"]):
                dataset.add_frame(
                    {
                        "observation.state": a["state"][tick],
                        "action": a["action"][tick],
                        "task": "Stack one block on top of the other.",
                        **{
                            f"observation.images.{v}": next(streams[v]) for v in streams
                        },
                        "next.success": np.array([a["success"][tick]], bool),
                        "next.done": np.array([tick == row["length"] - 1], bool),
                        "next.truncated": np.array(
                            [
                                tick == row["length"] - 1
                                and row["termination_reason"] == "horizon"
                            ],
                            bool,
                        ),
                    }
                )
            for stream in streams.values():
                if next(stream, None) is not None:
                    raise ValueError("Rendered video has extra frames")
            dataset.save_episode()
            replay = config.output / "replay" / f"episode-{index:06d}.npz"
            replay.parent.mkdir(exist_ok=True)
            np.savez_compressed(
                replay, **{k: a[k][0] for k in a if k.startswith("sim/")}
            )
        manifest.append(
            row
            | {
                "episode_index": index,
                "source_root": str(config.source.resolve()),
                "replay": str(replay.relative_to(config.output)),
            }
        )
        write_json(config.output / "INCOMPLETE.json", {"episodes": index + 1})
    dataset.finalize()
    write_json(
        config.output / "manifest.json",
        {
            "format": "ocbench-mjwarp-1",
            "repo_id": config.repo_id,
            "action_profile": ACTION,
            "rendering": profile,
            "episodes": manifest,
            "quality": "audited-native-success"
            if config.successes
            else "audited-native-failure",
        },
    )
    (config.output / "INCOMPLETE.json").unlink()
    return {"episodes": len(manifest), "frames": sum(r["length"] for r in manifest)}
