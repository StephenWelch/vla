"""Collect contact-valid manipulation scenes and export annotated LeRobot data."""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vla_tools.config import parse_args
from vla_tools.tracking import Tracker, WandbConfig

from ogbench_mjwarp.hooks import scene_split


@dataclass
class Config:
    raw: Path
    dataset: Path
    env_id: str = "cube-single-v0"
    task_id: int = 1
    scenes: int = 100
    scenes_per_round: int = 20
    max_scenes: int = 1000
    variants: int = 2
    seed: int = 20000
    size: int | tuple[int, int] = (480, 640)
    validation_fraction: float = 0.2
    generation_batch_size: int = 32
    refill_slots: bool = True
    batched_cpu: bool = True
    streaming_encoding: bool = True
    overlap_export: bool = True
    encoder_queue_size: int = 30
    encoder_threads: int = 2
    export_queue_capacity: int = 2
    reuse_dataset: bool = False
    tracking_output: Path | None = None
    wandb: WandbConfig = field(default_factory=WandbConfig)


def prepare(config):
    tracker = Tracker(
        config.tracking_output or config.raw / "tracking",
        config.wandb,
        "collect",
        {"collection": json.loads(json.dumps(asdict(config), default=str))},
        resume=(config.tracking_output or config.raw / "tracking")
        .joinpath("tracking.json")
        .exists(),
    )
    failed = True
    try:
        result = collect(config, tracker)
        failed = False
        return result
    finally:
        tracker.finish(failed)


def dataset_report(config, tracker):
    from ogbench_mjwarp.profile import ogbench_profile

    if any(
        (config.dataset / name).exists() for name in ("INCOMPLETE", "INCOMPLETE.json")
    ):
        raise ValueError("Dataset export is incomplete")
    if ogbench_profile(config.dataset) is None:
        raise ValueError("Collection requires a v2 OGBench dataset")
    manifest = json.loads((config.dataset / "manifest.json").read_text())
    if any(
        row["env_id"] != config.env_id or row["task_id"] != config.task_id
        for row in manifest["episodes"]
    ):
        raise ValueError("Dataset differs from the requested environment/task")
    split = scene_split(config.dataset, config.validation_fraction, 1000)
    (config.dataset / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    rows = manifest["episodes"]
    tracker.log(
        {
            "collection": {
                "reused": config.reuse_dataset,
                "dataset_episodes": len(rows),
                "dataset_frames": sum(row["length"] for row in rows),
                "independent_scenes": len(
                    {
                        row.get("randomization", {}).get("initial_state_fingerprint")
                        or row["seed"]
                        for row in rows
                    }
                ),
                "train_episodes": len(split["train"]),
                "val_episodes": len(split["val"]),
            }
        },
        tables={
            "collection/dataset_episodes": [
                {
                    key: row.get(key)
                    for key in (
                        "episode_index",
                        "seed",
                        "task_id",
                        "outcome",
                        "length",
                        "reason",
                        "contact_quality",
                        "randomization",
                    )
                }
                for row in rows
            ]
        },
        reports=[
            config.dataset / path
            for path in (
                "manifest.json",
                "generation.json",
                "split.json",
                "meta/info.json",
                "meta/stats.json",
                "export_metrics.json",
                "collection_metrics.json",
            )
        ],
    )
    result = {
        "dataset": str(config.dataset),
        "episodes": {name: len(split[name]) for name in ("train", "val")},
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def collect(config, tracker):
    from ogbench_mjwarp.config import PlannerConfig, RandomizationConfig
    from ogbench_mjwarp.dataset import export_dataset
    from ogbench_mjwarp.recording import generate

    if (
        min(
            config.scenes,
            config.scenes_per_round,
            config.variants,
            config.generation_batch_size,
            config.encoder_queue_size,
            config.encoder_threads,
            config.export_queue_capacity,
        )
        < 1
        or config.max_scenes < config.scenes
    ):
        raise ValueError(
            "Positive generation budgets with max_scenes >= scenes required"
        )
    if config.reuse_dataset:
        return dataset_report(config, tracker)
    started = time.perf_counter()
    committed = {}
    if tracker.run:
        for path in config.raw.glob("round-*/episode-*.json"):
            committed[str(path)] = json.loads(path.read_text())

    def metrics(event, root):
        values = {"episodes_committed": len(committed)}
        if event["event"] == "episode":
            row = event["episode"]
            committed[str(root / f"episode-{row['episode_id']:06d}.json")] = row
            values.update(
                episodes_committed=len(committed),
                successes=sum(
                    row["outcome"] == "success" for row in committed.values()
                ),
                frames=sum(row["length"] for row in committed.values()),
            )
        else:
            values.update(
                {key: value for key, value in event.items() if key != "event"}
            )
        tracker.log({"collection": values})

    export_settings = {
        "streaming_encoding": config.streaming_encoding,
        "encoder_queue_size": config.encoder_queue_size,
        "encoder_threads": config.encoder_threads,
    }
    exporter = None
    if config.overlap_export and not config.dataset.exists():
        from ogbench_mjwarp.export_worker import RoundExporter

        exporter = RoundExporter(
            config.dataset,
            capacity=config.export_queue_capacity,
            progress=lambda event: metrics(event, config.raw),
            **export_settings,
        )
    sources, identities = [], set()

    def on_generation(event, root):
        metrics(event, root)
        if exporter:
            exporter.poll()

    try:
        for offset in range(0, config.max_scenes, config.scenes_per_round):
            root = config.raw / f"round-{offset:06d}"
            planner = PlannerConfig(
                episodes=config.generation_batch_size,
                candidates=8,
                horizon=8,
                iterations=2,
                joint_target_noise=0.01,
            )
            run_path = root / "run.json"
            if run_path.exists():
                # Completed episodes retain their original planner provenance on resume.
                planner = PlannerConfig(**json.loads(run_path.read_text())["planner"])
            print(
                f"Generation round {offset}: up to {planner.episodes} concurrent episodes",
                flush=True,
            )
            summary = generate(
                root,
                config.env_id,
                min(config.scenes_per_round, config.max_scenes - offset)
                * config.variants,
                [config.task_id],
                config.seed + offset,
                planner,
                size=config.size,
                record_images=True,
                metrics=lambda event, root=root: on_generation(event, root),
                refill_slots=config.refill_slots,
                batched_cpu=config.batched_cpu,
                randomization=RandomizationConfig(
                    variants_per_reset=config.variants,
                    order=True,
                    cube_grasps=True,
                    handle_grasps=config.env_id == "scene-v0",
                    position_noise=0.01,
                    yaw_noise=0.1,
                    duration_scale_min=0.9,
                    duration_scale_max=1.2,
                ),
            )
            sources.append(root)
            if exporter:
                exporter.submit(root)
            for path in root.glob("episode-*.json"):
                row = json.loads(path.read_text())
                if row["outcome"] == "success" and row["contact_quality"]["valid"]:
                    identities.add(
                        row.get("randomization", {}).get("initial_state_fingerprint")
                        or row["seed"]
                    )
            print(
                f"Accepted independent scenes: {len(identities)}/{config.scenes}",
                flush=True,
            )
            tracker.log(
                {
                    "collection": {
                        "episodes_committed": len(committed),
                        "accepted_scenes": len(identities),
                        "target_scenes": config.scenes,
                        "round_performance": summary.get("performance", {}),
                    }
                },
                reports=[root / "run.json", root / "summary.json"],
            )
            if len(identities) >= config.scenes:
                break
        else:
            raise RuntimeError(
                f"Generation budget exhausted with {len(identities)} successful scenes; raw data preserved"
            )
        generation_seconds = time.perf_counter() - started
        export_result = None
        if exporter:
            export_result = exporter.finish()
        elif not config.dataset.exists():
            export_result = export_dataset(
                sources,
                config.dataset,
                "local/ogbench-manipulation-training",
                "success",
                True,
                progress=lambda event: metrics(event, config.raw),
                **export_settings,
            )
        if export_result:
            tracker.log({"collection/export": export_result})
    finally:
        if exporter:
            exporter.close()
    performance = {
        "generation_seconds": generation_seconds,
        "pipeline_seconds": time.perf_counter() - started,
        "overlap_export": config.overlap_export,
    }
    (config.dataset / "collection_metrics.json").write_text(
        json.dumps(performance, indent=2) + "\n"
    )
    tracker.log({"collection": performance})
    if any(
        (config.dataset / name).exists() for name in ("INCOMPLETE", "INCOMPLETE.json")
    ):
        raise ValueError(
            "Incomplete dataset export; preserve it and choose a new dataset path"
        )
    manifest = json.loads((config.dataset / "manifest.json").read_text())
    dataset_scenes = {
        row.get("randomization", {}).get("initial_state_fingerprint") or row["seed"]
        for row in manifest["episodes"]
    }
    if dataset_scenes != identities or any(
        row["outcome"] != "success" or not row["contact_quality"]["valid"]
        for row in manifest["episodes"]
    ):
        raise ValueError(
            "Existing dataset differs from the collected successful scenes"
        )
    tracker.log(
        {},
        tables={
            "collection/attempts": [
                {
                    key: row.get(key)
                    for key in (
                        "seed",
                        "task_id",
                        "outcome",
                        "reason",
                        "length",
                        "contact_quality",
                    )
                }
                for row in committed.values()
            ]
        },
    )
    return dataset_report(config, tracker)


def main():
    prepare(parse_args(Config))


if __name__ == "__main__":
    main()
