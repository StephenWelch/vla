"""Batch replay directly into LeRobot videos with bounded GPU/CPU encoders."""

import json
import shutil
import time

import numpy as np
import torch
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from vla_tools.encoding import EpisodeVideoEncoder, save_encoded_episode
from vla_tools.tracking import write_json

from .collect import rows
from .config import ACTION, FIELDS, validate_action
from .dataset import dataset_features
from .environment import Simulation

VIEWS = ("front", "wrist")
KEYS = [f"observation.images.{v}" for v in VIEWS]


def load_arrays(source, records):
    arrays = []
    for row in records:
        validate_action(row["action_profile"])
        with np.load(source / "raw" / row["archive"]) as archive:
            a = {k: archive[k] for k in archive.files}
        if len(a["action"]) != row["length"] or len(a["sim/qpos"]) != row["length"] + 1:
            raise ValueError("Raw action/state alignment differs from episode metadata")
        arrays.append(a)
    return arrays


def replay_images(sim, arrays, frames=None):
    """Yield one GPU batch; completed worlds hold their last observation."""
    lengths = [len(a["action"]) for a in arrays]
    for tick in range(max(lengths) if frames is None else frames):
        indices = [
            min(tick, n - 1)
            if frames is None
            else round(tick * (n - 1) / max(1, frames - 1))
            for n in lengths
        ]
        sim.restore(
            {
                k: np.stack(
                    [a[f"sim/{k}"][i] for a, i in zip(arrays, indices, strict=True)]
                )
                for k in FIELDS
            },
            forward=False,
        )
        yield tick, sim.render()


def episode_buffer(dataset, row, arrays):
    n = row["length"]
    buffer = dataset.writer._create_episode_buffer()
    buffer.update(
        size=n,
        task=["Stack one block on top of the other."] * n,
        frame_index=np.arange(n),
        timestamp=np.arange(n) / 50,
    )
    buffer["observation.state"] = arrays["state"]
    buffer["action"] = arrays["action"]
    buffer["next.success"] = arrays["success"].reshape(n, 1)
    buffer["next.done"] = (np.arange(n) == n - 1).reshape(n, 1)
    buffer["next.truncated"] = buffer["next.done"] & (
        row["termination_reason"] == "horizon"
    )
    for key in KEYS:
        buffer[key] = [None] * n  # Native streaming writer stores video placeholders.
    return buffer


def materialize(config):
    if min(config.batch_size, config.encoder_threads, config.encoder_queue_size) < 1:
        raise ValueError("Positive rendering/encoder budgets required")
    if config.encoder_backend not in ("async", "cpu"):
        raise ValueError("encoder_backend must be async or cpu")
    gpu = config.encoder_backend == "async"
    if gpu and (
        config.batch_size > 4
        or min(config.render_batch_frames, config.buffer_frames) < 1
        or config.write_buffer_bytes < 2
    ):
        raise ValueError(
            "Async export requires 1-4 episodes, positive temporal/ring sizes and write buffer >= 2"
        )
    encoding = {
        "backend": config.encoder_backend,
        "gop": 2,
        "image_stats": "uint8" if gpu else "native",
        "render_batch_frames": config.render_batch_frames if gpu else 1,
        "buffer_frames": config.buffer_frames if gpu else config.encoder_queue_size,
        "write_buffer_bytes": config.write_buffer_bytes if gpu else None,
    }
    selected = [
        r
        for r in rows(config.source)
        if r["physical_valid"] and r["native_success"] == config.successes
    ]
    if config.limit is not None:
        selected = selected[: config.limit]
    if not selected:
        return {"episodes": 0, "frames": 0}
    output = config.output
    manifest, profile = [], None
    if output.exists():
        manifest = output / "manifest.json"
        if manifest.exists() and not (output / "INCOMPLETE.json").exists():
            saved = json.loads(manifest.read_text())
            if (
                [r["episode_id"] for r in saved["episodes"]]
                == [r["episode_id"] for r in selected]
                and saved.get("action_profile") == ACTION
                and saved.get("encoding") == encoding
            ):
                return {
                    "episodes": len(selected),
                    "frames": sum(r["length"] for r in selected),
                }
        checkpoint = output / "checkpoint.json"
        if not checkpoint.exists():
            raise FileExistsError(
                "Export has no durable checkpoint; preserve it and use a fresh output"
            )
        saved = json.loads(checkpoint.read_text())
        manifest, profile = saved["episodes"], saved["rendering"]
        if (
            saved["encoding"] != encoding
            or saved["action_profile"] != ACTION
            or manifest
            != [
                r
                | {
                    "episode_index": i,
                    "source_root": str(config.source.resolve()),
                    "replay": f"replay/episode-{i:06d}.npz",
                }
                for i, r in enumerate(selected[: len(manifest)])
            ]
        ):
            raise FileExistsError("Checkpoint and requested export differ")
    codec = RGBEncoderConfig(
        vcodec="h264_nvenc" if gpu else "h264", crf=18, preset=12 if gpu else None
    )
    options = {
        "repo_id": config.repo_id,
        "root": output,
        "video_backend": "pyav",
        "rgb_encoder": codec,
        "streaming_encoding": False,
        "image_writer_threads": 0,
        "encoder_threads": config.encoder_threads,
    }
    dataset = (
        LeRobotDataset.resume(**options)
        if manifest
        else LeRobotDataset.create(
            **options,
            fps=50,
            robot_type="ocbench_ur5e_joint_delta",
            features=dataset_features(),
            video_files_size_in_mb=1,
        )
    )
    if dataset.num_episodes != len(manifest) or dataset.num_frames != sum(
        r["length"] for r in manifest
    ):
        dataset.finalize()
        raise ValueError(
            "Dataset metadata differs from durable checkpoint; refusing unsafe resume"
        )
    timings = {"render_seconds": 0.0, "enqueue_seconds": 0.0, "commit_seconds": 0.0}
    started = time.perf_counter()
    write_json(
        output / "INCOMPLETE.json",
        {"episodes": len(manifest), "stage": "direct-render-export"},
    )
    try:
        for offset in range(len(manifest), len(selected), config.batch_size):
            if shutil.disk_usage(output).free < 4 * 2**30:
                raise RuntimeError(
                    "Less than 4 GiB free; raw rollouts and incomplete dataset retained"
                )
            batch = selected[offset : offset + config.batch_size]
            if dataset._is_finalized:
                dataset = LeRobotDataset.resume(**options)
            write_json(
                output / "worker-state.json",
                {"stage": "rendering", "episodes": len(manifest)},
            )
            arrays = load_arrays(config.source, batch)
            sim = Simulation(
                [r["seed"] for r in batch] * (config.render_batch_frames if gpu else 1),
                audit=False,
            )
            encoders = []
            pipeline = None
            scratch = output / ".encoding" / f"batch-{offset:06d}"
            try:
                if scratch.exists():
                    shutil.rmtree(
                        scratch
                    )  # Uncommitted encoder scratch from a timed-out worker.
                if gpu:
                    from .async_video import AsyncVideoPipeline
                    from .gpu_video import DeviceReplay

                    replay = DeviceReplay(
                        sim, arrays, render_batch_frames=config.render_batch_frames
                    )
                    pipeline = AsyncVideoPipeline(
                        scratch,
                        sim,
                        [r["length"] for r in batch],
                        config.buffer_frames,
                        image_stats="uint8",
                        render_batch_frames=config.render_batch_frames,
                        write_buffer_bytes=config.write_buffer_bytes,
                    )
                    stamp = time.perf_counter()
                    for tick in range(0, replay.length, config.render_batch_frames):
                        pipeline.render(replay, tick)
                    encoders = pipeline.finish()
                    timings["render_seconds"] += time.perf_counter() - stamp
                for world, row in enumerate([] if gpu else batch):
                    previews = (
                        {
                            key: output
                            / "previews"
                            / f"{offset + world:06d}-{view}.mp4"
                            for key, view in zip(KEYS, VIEWS, strict=True)
                        }
                        if offset + world < 3
                        else {}
                    )
                    encoder = EpisodeVideoEncoder(
                        fps=50,
                        rgb_encoder=codec,
                        queue_maxsize=config.encoder_queue_size,
                        encoder_threads=config.encoder_threads,
                        expected_frames=row["length"],
                        previews=previews,
                    )
                    encoder.start_episode(KEYS, temp_dir=output)
                    encoders.append(encoder)
                if not gpu:
                    replay = iter(replay_images(sim, arrays))
                    for _ in range(max(r["length"] for r in batch)):
                        stamp = time.perf_counter()
                        tick, images = next(replay)
                        timings["render_seconds"] += time.perf_counter() - stamp
                        stamp = time.perf_counter()
                        for world, row in enumerate(batch):
                            if tick < row["length"]:
                                for key, view in zip(KEYS, VIEWS, strict=True):
                                    encoders[world].feed_frame(key, images[view][world])
                        timings["enqueue_seconds"] += time.perf_counter() - stamp
                if profile is not None and profile != sim.renderer.profile:
                    raise ValueError("Mixed camera/model profiles")
                profile = sim.renderer.profile
                write_json(
                    output / "worker-state.json",
                    {"stage": "committing", "episodes": len(manifest)},
                )
                stamp = time.perf_counter()
                for row, a, encoder in zip(batch, arrays, encoders, strict=True):
                    index = len(manifest)
                    if gpu and index < 3:
                        for key, (path, _) in encoder.finish_episode().items():
                            preview = (
                                output
                                / "previews"
                                / f"{index:06d}-{key.rsplit('.', 1)[-1]}.mp4"
                            )
                            preview.parent.mkdir(exist_ok=True)
                            shutil.copyfile(path, preview)
                    save_encoded_episode(
                        dataset, episode_buffer(dataset, row, a), encoder
                    )
                    replay_path = output / "replay" / f"episode-{index:06d}.npz"
                    replay_path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(
                        replay_path, **{k: a[k][0] for k in a if k.startswith("sim/")}
                    )
                    manifest.append(
                        row
                        | {
                            "episode_index": index,
                            "source_root": str(config.source.resolve()),
                            "replay": str(replay_path.relative_to(output)),
                        }
                    )
                    write_json(
                        output / "INCOMPLETE.json",
                        {
                            "episodes": len(manifest),
                            "total": len(selected),
                            "stage": "direct-render-export",
                        },
                    )
                dataset.finalize()  # Persist BOTH data and episode-metadata parquet footers.
                write_json(
                    output / "checkpoint.json",
                    {
                        "episodes": manifest,
                        "rendering": profile,
                        "encoding": encoding,
                        "action_profile": ACTION,
                    },
                )
                write_json(
                    output / "worker-state.json",
                    {"stage": "checkpointed", "episodes": len(manifest)},
                )
                timings["commit_seconds"] += time.perf_counter() - stamp
                write_json(
                    output / "materialization.json",
                    {
                        "episodes": len(manifest),
                        "total": len(selected),
                        "batch_size": config.batch_size,
                        "encoding": encoding,
                        "last_batch": pipeline.metrics
                        if pipeline is not None
                        else None,
                        "seconds": time.perf_counter() - started,
                        **timings,
                    },
                )
                print(
                    f"Direct render/export: {len(manifest)}/{len(selected)} episodes",
                    flush=True,
                )
            finally:
                if pipeline is not None:
                    pipeline.close()
                for encoder in encoders:
                    encoder.close()
                sim.close()
            if scratch.exists():
                shutil.rmtree(scratch)
        dataset.finalize()
    except BaseException:
        dataset.finalize()
        raise
    write_json(
        output / "manifest.json",
        {
            "format": "ocbench-mjwarp-1",
            "repo_id": config.repo_id,
            "action_profile": ACTION,
            "rendering": profile,
            "episodes": manifest,
            "quality": "audited-native-success"
            if config.successes
            else "audited-native-failure",
            "video_materialization": "async-gpu-v1" if gpu else "direct-streaming-v1",
            "encoding": encoding,
        },
    )
    (output / "INCOMPLETE.json").unlink()
    return {
        "episodes": len(manifest),
        "frames": sum(r["length"] for r in manifest),
        "seconds": time.perf_counter() - started,
        **timings,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
