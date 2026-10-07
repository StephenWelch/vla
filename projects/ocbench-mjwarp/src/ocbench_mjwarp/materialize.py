"""Batch replay directly into LeRobot videos with bounded GPU/CPU encoders."""

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from vla_tools.encoding import EpisodeVideoEncoder, save_encoded_episode
from vla_tools.tracking import write_json

from .config import ACTION, FIELDS
from .dataset import dataset_features
from .environment import Simulation
from .episodes import imported, load_arrays, select_episodes
from .rendering import image_shape

VIEWS = ("front", "wrist")
KEYS = [f"observation.images.{v}" for v in VIEWS]


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
    if imported(row):
        buffer["next.reward"] = arrays["reward"].reshape(n, 1)
        buffer["next.done"] = arrays["done"].reshape(n, 1)
        buffer["next.truncated"] = arrays["truncated"].reshape(n, 1)
    for key in KEYS:
        buffer[key] = [None] * n  # Native streaming writer stores video placeholders.
    return buffer


@dataclass(frozen=True)
class EncodedVideos:
    """LeRobot's encoder interface, detached from CUDA resources and threads."""

    results: dict

    def finish_episode(self):
        return self.results

    def close(self):
        pass  # Files belong to the commit job until the checkpoint is durable.


def export_schedule(config):
    selected = select_episodes(config.source, config.successes, config.limit)
    path = config.output.with_name(config.output.name + ".schedule.json")
    schedule_exists = path.exists()
    saved = json.loads(path.read_text()) if schedule_exists else None
    legacy = saved.get("legacy", False) if saved else config.output.exists()
    order = "source" if legacy else config.episode_order
    if order not in ("source", "length"):
        raise ValueError("episode_order must be source or length")
    if order == "length":
        selected = sorted(selected, key=lambda row: row["length"])
    record = {
        "legacy": legacy,
        "source": str(config.source.resolve()),
        "order": order,
        "image_size": list(image_shape(config.image_size)),
        "episodes": selected,
    }
    if schedule_exists:
        # Existing exports always retain their persisted order.
        if saved["order"] != order:
            raise FileExistsError("Export schedule differs; use a fresh output")
        if saved != record:
            raise FileExistsError("Export inputs or resolution differ from schedule")
    elif selected and not legacy:
        write_json(path, record)
    return selected


def materialize(config):
    started = time.perf_counter()
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
    size = image_shape(config.image_size)
    if any(v % 2 for v in size):
        raise ValueError("H.264 export requires even image dimensions")
    encoding = {
        "backend": config.encoder_backend,
        "gop": 2,
        "image_stats": "uint8" if gpu else "native",
        "render_batch_frames": config.render_batch_frames if gpu else 1,
        "buffer_frames": config.buffer_frames if gpu else config.encoder_queue_size,
        "write_buffer_bytes": config.write_buffer_bytes if gpu else None,
    }
    selected = export_schedule(config)
    if not selected:
        return {"episodes": 0, "frames": 0}
    output = config.output
    source_root = str(config.source.resolve())
    manifest, profile = [], None
    if output.exists():
        manifest_path = output / "manifest.json"
        if manifest_path.exists() and not (output / "INCOMPLETE.json").exists():
            saved = json.loads(manifest_path.read_text())
            if (
                len(saved["episodes"]) == len(selected)
                and [
                    {k: r[k] for k in selected[i]}
                    for i, r in enumerate(saved["episodes"])
                ]
                == selected
                and saved.get("action_profile") == ACTION
                and saved.get("encoding") == encoding
                and saved["rendering"]["resolution"] == list(size)
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
            saved["rendering"]["resolution"] != list(size)
            or saved["encoding"] != encoding
            or saved["action_profile"] != ACTION
            or manifest
            != [
                r
                | {
                    "episode_index": i,
                    "source_root": source_root,
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
            features=dataset_features(
                rewards=any(imported(r) for r in selected), image_size=size
            ),
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
    timings = dict.fromkeys(
        (
            "setup_seconds",
            "load_seconds",
            "render_seconds",
            "commit_seconds",
            "commit_wait_seconds",
        ),
        0.0,
    )
    timings["initialization_seconds"] = time.perf_counter() - started
    padded_frames = rendered_frames = 0
    write_json(
        output / "INCOMPLETE.json",
        {"episodes": len(manifest), "stage": "direct-render-export"},
    )
    write_json(
        output / "worker-state.json", {"stage": "starting", "episodes": len(manifest)}
    )

    def progress(stage, active, **values):
        write_json(output / f"worker-{stage}.json", {"active": active, **values})

    def commit(batch, arrays, videos, scratch, batch_metrics, render_timings):
        nonlocal dataset
        progress("commit", True, episodes=len(manifest))
        stamp = time.perf_counter()
        try:
            if dataset._is_finalized:
                dataset = LeRobotDataset.resume(**options)
            for row, a, encoder in zip(batch, arrays, videos, strict=True):
                index = len(manifest)
                if index < 3:
                    for key, (path, _) in encoder.finish_episode().items():
                        preview = (
                            output
                            / "previews"
                            / f"{index:06d}-{key.rsplit('.', 1)[-1]}.mp4"
                        )
                        preview.parent.mkdir(exist_ok=True)
                        shutil.copyfile(path, preview)
                save_encoded_episode(dataset, episode_buffer(dataset, row, a), encoder)
                replay_path = output / "replay" / f"episode-{index:06d}.npz"
                replay_path.parent.mkdir(exist_ok=True)
                np.savez_compressed(
                    replay_path, **{k: a[k][0] for k in a if k.startswith("sim/")}
                )
                manifest.append(
                    row
                    | {
                        "episode_index": index,
                        "source_root": source_root,
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
                progress("commit", True, episodes=len(manifest))
            # Both data and metadata parquet footers precede the checkpoint.
            dataset.finalize()
            write_json(
                output / "checkpoint.json",
                {
                    "episodes": manifest,
                    "rendering": profile,
                    "encoding": encoding,
                    "action_profile": ACTION,
                },
            )
            timings["commit_seconds"] += time.perf_counter() - stamp
            write_json(
                output / "worker-state.json",
                {"stage": "checkpointed", "episodes": len(manifest)},
            )
            write_json(
                output / "materialization.json",
                {
                    "episodes": len(manifest),
                    "total": len(selected),
                    "batch_size": config.batch_size,
                    "encoding": encoding,
                    "last_batch": batch_metrics,
                    "seconds": time.perf_counter() - started,
                    **render_timings,
                    "commit_seconds": timings["commit_seconds"],
                },
            )
            if scratch.exists():
                shutil.rmtree(scratch)
            print(
                f"Direct render/export: {len(manifest)}/{len(selected)} episodes",
                flush=True,
            )
        finally:
            dataset.finalize()
            progress("commit", False, episodes=len(manifest))

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset-commit")
    pending = None
    sim = None
    try:
        for offset in range(len(manifest), len(selected), config.batch_size):
            if pending is not None and pending.done():
                pending.result()
            if shutil.disk_usage(output).free < 4 * 2**30:
                raise RuntimeError(
                    "Less than 4 GiB free; raw rollouts and incomplete dataset retained"
                )
            batch = selected[offset : offset + config.batch_size]
            progress("render", True, offset=offset)
            stamp = time.perf_counter()
            arrays = load_arrays(config.source, batch)
            timings["load_seconds"] += time.perf_counter() - stamp
            stamp = time.perf_counter()
            seeds = [r["seed"] for r in batch] * (
                config.render_batch_frames if gpu else 1
            )
            if sim is None or sim.worlds != len(seeds) or not config.reuse_simulation:
                if sim is not None:
                    sim.close()
                    sim = None
                sim = Simulation(seeds, audit=False, image_size=size)
            else:
                sim.reset_render(seeds)
            timings["setup_seconds"] += time.perf_counter() - stamp
            encoders, pipeline = [], None
            scratch = output / ".encoding" / f"batch-{offset:06d}"
            try:
                if scratch.exists():
                    shutil.rmtree(scratch)  # Only this uncommitted batch's scratch.
                stamp = time.perf_counter()
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
                    for tick in range(0, replay.length, config.render_batch_frames):
                        if pending is not None and pending.done():
                            pending.result()
                        pipeline.render(replay, tick)
                        if tick % (config.render_batch_frames * 25) == 0:
                            progress("render", True, offset=offset, frame=tick)
                    encoders = pipeline.finish()
                    del replay
                else:
                    for row in batch:
                        encoder = EpisodeVideoEncoder(
                            fps=50,
                            rgb_encoder=codec,
                            queue_maxsize=config.encoder_queue_size,
                            encoder_threads=config.encoder_threads,
                            expected_frames=row["length"],
                        )
                        encoder.start_episode(KEYS, temp_dir=output)
                        encoders.append(encoder)
                    for tick, images in replay_images(sim, arrays):
                        if pending is not None and pending.done():
                            pending.result()
                        for world, row in enumerate(batch):
                            if tick < row["length"]:
                                for key, view in zip(KEYS, VIEWS, strict=True):
                                    encoders[world].feed_frame(key, images[view][world])
                        if tick % 100 == 0:
                            progress("render", True, offset=offset, frame=tick)
                videos = [
                    EncodedVideos(encoder.finish_episode()) for encoder in encoders
                ]
                timings["render_seconds"] += time.perf_counter() - stamp
                if profile is not None and profile != sim.renderer.profile:
                    raise ValueError("Mixed camera/model profiles")
                profile = sim.renderer.profile
                metrics = dict(pipeline.metrics) if pipeline is not None else None
            finally:
                if pipeline is not None:
                    pipeline.close()
                    pipeline = None
                for encoder in encoders:
                    encoder.close()
                encoders.clear()
            # No CUDA resources cross the commit boundary. One previous job at most.
            progress("render", False, offset=offset)
            stamp = time.perf_counter()
            if pending is not None:
                pending.result()
            timings["commit_wait_seconds"] += time.perf_counter() - stamp
            temporal = config.render_batch_frames if gpu else 1
            submitted = (
                ((max(r["length"] for r in batch) + temporal - 1) // temporal)
                * temporal
                * len(batch)
            )
            rendered_frames += submitted
            padded_frames += submitted - sum(r["length"] for r in batch)
            snapshot = dict(timings) | {
                "rendered_frames": rendered_frames,
                "padded_frames": padded_frames,
            }
            pending = pool.submit(
                commit, batch, arrays, videos, scratch, metrics, snapshot
            )
            if not config.overlap_commits:
                pending.result()
            del arrays, videos
        if pending is not None:
            pending.result()
    finally:
        # The writer remains owned by the commit thread, including cleanup.
        try:
            pool.submit(lambda: dataset.finalize()).result()
        finally:
            pool.shutdown(wait=True)
            if sim is not None:
                sim.close()
    write_json(
        output / "manifest.json",
        {
            "format": "ocbench-mjwarp-1",
            "repo_id": config.repo_id,
            "action_profile": ACTION,
            "rendering": profile,
            "episodes": manifest,
            "quality": "upstream-outcomes-unaudited"
            if any(imported(r) for r in selected)
            else (
                "audited-native-success"
                if config.successes
                else "audited-native-failure"
            ),
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
        "rendered_frames": rendered_frames,
        "padded_frames": padded_frames,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
