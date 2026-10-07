"""Run collection, deferred cameras, export and ACT training in separate processes."""

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from vla_tools.config import parse_args
from vla_tools.tracking import Tracker, WandbConfig, write_json

from .collect import rows, summarize
from .config import CollectionConfig
from .dataset import ExportConfig
from .hub import HubConfig
from .train import TrainConfig, TrainingConfig


@dataclass
class PipelineConfig:
    output: Path
    source: Literal["generate", "huggingface"] = "generate"
    hub: HubConfig = field(default_factory=HubConfig)
    dataset_root: Path | None = None
    episodes: int = 500
    seed: int = 83000
    oracle_seed: int = 93000
    worlds: int = 32
    max_steps: int = 2500
    render_batch_size: int = 4
    image_size: tuple[int, int] = (480, 640)
    episode_order: Literal["length", "source"] = "length"
    overlap_commits: bool = True
    reuse_simulation: bool = True
    encoder_backend: str = "async"
    render_batch_frames: int = 4
    buffer_frames: int = 2
    write_buffer_bytes: int = 1048576
    worker_timeout_seconds: float = 180
    worker_retries: int = 2
    encoder_threads: int = 1
    encoder_queue_size: int = 8
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(validation_fraction=0.2)
    )
    export_limit: int | None = None
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(
            enable=True, project="vla-ocbench", group="native-stack-500"
        )
    )


def run(config):
    # Keep dict defaults empty for Tyro's arbitrary KEY VALUE override parser.
    config.training.overrides = {
        "policy.chunk_size": "40",
        "policy.n_action_steps": "40",
        "log_freq": "25",
    } | config.training.overrides
    root = config.output
    root.mkdir(parents=True, exist_ok=True)
    dataset_root = config.dataset_root or root / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)
    write_json(root / "pipeline.json", asdict(config))
    if config.source == "huggingface":
        spec = asdict(config.hub) | {"output": root}
        write_json(root / "import-config.json", spec)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ocbench_mjwarp.cli",
                "import-hf",
                "--config",
                str(root / "import-config.json"),
            ],
            check=True,
        )
        record = rows(root)
        summary = {
            "episodes": len(record),
            "upstream_successes": sum(r["native_success"] for r in record),
            "audit_status": "unknown",
        }
        exports = [(None, "all")]
    else:
        collection = CollectionConfig(
            output=root,
            episodes=config.episodes,
            seed=config.seed,
            oracle_seed=config.oracle_seed,
            worlds=config.worlds,
            max_steps=config.max_steps,
            wandb=config.wandb,
        )
        write_json(root / "generate.json", asdict(collection))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ocbench_mjwarp.cli",
                "generate",
                "--config",
                str(root / "generate.json"),
            ],
            check=True,
        )
        record = rows(root)
        summary = summarize(record)
        exports = [(True, "successes"), (False, "failures")]
    # One encoding pass into LeRobot videos, plus a few byte-copied previews.
    frames = sum(
        r["length"]
        for r in record
        if r["physical_valid"] or config.source == "huggingface"
    )
    estimated_bytes = (
        frames * (18000 if config.encoder_backend == "async" else 12000) + 6 * 2**30
    )
    if (
        config.export_limit is None
        and shutil.disk_usage(dataset_root).free < estimated_bytes * 1.2
    ):
        raise RuntimeError(
            f"Collection saved; rendering/training need an estimated {estimated_bytes * 1.2 / 2**30:.1f} GiB free"
        )
    write_json(root / "status.json", {"status": "rendering-export", **summary})
    tracker = Tracker(root, config.wandb, "collection", resume=True)
    failed = True
    try:
        for success, name in exports:
            spec = ExportConfig(
                source=root,
                output=dataset_root / name,
                successes=success,
                repo_id=f"local/ocbench-stack-{name}",
                limit=config.export_limit,
                batch_size=config.render_batch_size,
                image_size=config.image_size,
                episode_order=config.episode_order,
                overlap_commits=config.overlap_commits,
                reuse_simulation=config.reuse_simulation,
                encoder_threads=config.encoder_threads,
                encoder_queue_size=config.encoder_queue_size,
                encoder_backend=config.encoder_backend,
                render_batch_frames=config.render_batch_frames,
                buffer_frames=config.buffer_frames,
                write_buffer_bytes=config.write_buffer_bytes,
                worker_timeout_seconds=config.worker_timeout_seconds,
                worker_retries=config.worker_retries,
            )
            export_config = root / f"export-{name}.json"
            write_json(export_config, asdict(spec))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ocbench_mjwarp.cli",
                    "export",
                    "--config",
                    str(export_config),
                ],
                check=True,
            )
            info_path = spec.output / "meta/info.json"
            info = json.loads(info_path.read_text()) if info_path.exists() else {}
            result = {
                "episodes": info.get("total_episodes", 0),
                "frames": info.get("total_frames", 0),
            }
            if result["episodes"]:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                loaded = LeRobotDataset(
                    f"local/ocbench-stack-{name}",
                    root=dataset_root / name,
                    video_backend="pyav",
                )
                if loaded.num_episodes != result["episodes"] or loaded[0][
                    "action"
                ].shape != (7,):
                    raise ValueError(
                        "Exported dataset count/action verification failed"
                    )
                del loaded
            tracker.log({f"export/{name}": result})
        videos = {}
        for _, name in exports:
            preview_root = dataset_root / name / "previews"
            for path in sorted(preview_root.glob("*.mp4")):
                videos[f"examples/{name}/{path.stem}"] = path
        tracker.log(
            {"collection": summary},
            videos=videos,
            reports=[
                root
                / (
                    "import.json"
                    if config.source == "huggingface"
                    else "collection.json"
                )
            ],
        )
        failed = False
    finally:
        tracker.finish(failed)
    if config.training.steps:
        training = TrainConfig(
            dataset=dataset_root
            / ("all" if config.source == "huggingface" else "successes"),
            output=root / "act",
            **(asdict(config.training) | {"wandb": config.wandb}),
        )
        write_json(root / "train.json", asdict(training))
        write_json(root / "status.json", {"status": "training", **summary})
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ocbench_mjwarp.train",
                "--config",
                str(root / "train.json"),
            ],
            check=True,
        )
    write_json(root / "status.json", {"status": "complete", **summary})


def main():
    config = parse_args(PipelineConfig)
    try:
        run(config)
    except BaseException as error:
        write_json(
            config.output / "status.json",
            {"status": "failed_or_interrupted", "error": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
