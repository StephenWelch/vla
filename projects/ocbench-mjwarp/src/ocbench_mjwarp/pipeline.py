"""Run collection, deferred cameras, export and ACT training in separate processes."""

import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vla_tools.config import parse_args
from vla_tools.tracking import Tracker, WandbConfig, write_json

from .collect import rows, summarize
from .config import CollectionConfig
from .dataset import ExportConfig, export
from .train import TrainConfig


@dataclass
class PipelineConfig:
    output: Path
    episodes: int = 500
    seed: int = 83000
    oracle_seed: int = 93000
    worlds: int = 32
    max_steps: int = 2500
    render_batch_size: int = 4
    train_steps: int = 20000
    eval_max_steps: int = 2500
    loss_eval_freq: int = 1000
    rollout_eval_freq: int = 2000
    eval_episodes: int = 10
    export_limit: int | None = None
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(
            enable=True, project="vla-ocbench", group="native-stack-500"
        )
    )


def run(config):
    root = config.output
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "pipeline.json", asdict(config))
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
    # At most two H.264 copies: deferred camera videos and the LeRobot videos.
    frames = sum(r["length"] for r in record if r["physical_valid"])
    estimated_bytes = frames * 24000 + 6 * 2**30
    if (
        config.export_limit is None
        and shutil.disk_usage(root).free < estimated_bytes * 1.2
    ):
        raise RuntimeError(
            f"Collection saved; rendering/training need an estimated {estimated_bytes * 1.2 / 2**30:.1f} GiB free"
        )
    command = [
        sys.executable,
        "-m",
        "ocbench_mjwarp.cli",
        "render",
        "--source",
        str(root),
        "--batch-size",
        str(config.render_batch_size),
    ]
    # Smoke exports can select a subset, but full collection always retains all attempts.
    write_json(root / "status.json", {"status": "rendering", **summary})
    subprocess.run(command, check=True)
    tracker = Tracker(root, config.wandb, "collection", resume=True)
    failed = True
    try:
        for success, name in ((True, "successes"), (False, "failures")):
            result = export(
                ExportConfig(
                    root,
                    root / "datasets" / name,
                    success,
                    f"local/ocbench-stack-{name}",
                    config.export_limit,
                )
            )
            if result["episodes"]:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                loaded = LeRobotDataset(
                    f"local/ocbench-stack-{name}",
                    root=root / "datasets" / name,
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
        for success, label in ((True, "success"), (False, "failure")):
            samples = [
                r
                for r in record
                if r["physical_valid"] and r["native_success"] == success
            ][:3]
            for row in samples:
                for view in ("front", "wrist"):
                    videos[f"examples/{label}-{row['episode_id']}/{view}"] = (
                        root / "rendered" / f"{row['episode_id']:06d}-{view}.mp4"
                    )
        tracker.log(
            {"collection": summary}, videos=videos, reports=[root / "collection.json"]
        )
        failed = False
    finally:
        tracker.finish(failed)
    if config.train_steps:
        training = TrainConfig(
            dataset=root / "datasets/successes",
            policy=None,
            output=root / "act",
            steps=config.train_steps,
            batch_size=8,
            workers=2,
            validation_fraction=0.2,
            loss_eval_freq=config.loss_eval_freq,
            rollout_eval_freq=config.rollout_eval_freq,
            eval_episodes=config.eval_episodes,
            eval_max_steps=config.eval_max_steps,
            overrides=[
                "policy.chunk_size=40",
                "policy.n_action_steps=40",
                "log_freq=25",
            ],
            wandb=config.wandb,
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
