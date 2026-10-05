"""Compare identical-episode export and eight-scene collection on an idle GPU."""

import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vla_tools.config import parse_args
from vla_tools.tracking import Tracker, WandbConfig, write_json


@dataclass
class Config:
    source: Path
    output: Path
    wait_for_training: Path | None = None
    wandb: WandbConfig = field(default_factory=lambda: WandbConfig(enable=True))


def benchmark(config):
    from ogbench_mjwarp.dataset import export_dataset
    from ogbench_mjwarp.io import episode_metadata

    if config.output.exists():
        raise FileExistsError("Choose a fresh benchmark output directory")
    config.output.mkdir(parents=True)
    status = config.output / "benchmark.json"
    report = {"status": "waiting", "config": asdict(config), "results": {}}
    write_json(status, report)
    if config.wait_for_training:
        record = config.wait_for_training.with_name(
            config.wait_for_training.name + ".experiment.json"
        )
        print(f"Waiting for training to finish: {record}", flush=True)
        while json.loads(record.read_text())["status"] == "running":
            time.sleep(10)
    tracker = Tracker(
        config.output,
        config.wandb,
        "benchmark",
        json.loads(json.dumps(asdict(config), default=str)),
    )
    failed = True
    try:
        report["status"] = "running"
        write_json(status, report)
        subset = config.output / "identical-episodes"
        subset.mkdir()
        rows = episode_metadata(config.source, "success", True)[:8]
        if len(rows) != 8:
            raise ValueError(
                "Export benchmark needs eight successful, contact-valid raw episodes"
            )
        for row in rows:
            shutil.copy2(config.source / row["archive"], subset / row["archive"])
            write_json(subset / f"episode-{row['episode_id']:06d}.json", row)
        for name, streaming in (("baseline", False), ("optimized", True)):
            result = export_dataset(
                subset, config.output / f"export-{name}", streaming_encoding=streaming
            )
            report["results"][f"export_{name}"] = result
            tracker.log({f"benchmark/export_{name}": result})
            write_json(status, report)
        for name, optimized in (("baseline", False), ("optimized", True)):
            settings = {
                "raw": config.output / f"raw-{name}",
                "dataset": config.output / f"dataset-{name}",
                "scenes": 8,
                "scenes_per_round": 4,
                "max_scenes": 8,
                "variants": 2,
                "seed": 20000,
                "size": [480, 640],
                "generation_batch_size": 4,
                "refill_slots": optimized,
                "batched_cpu": optimized,
                "streaming_encoding": optimized,
                "overlap_export": optimized,
                "wandb": {"enable": False},
            }
            path = config.output / f"{name}.json"
            write_json(path, settings)
            with (config.output / f"{name}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ogbench_mjwarp.prepare",
                        "--config",
                        str(path),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            dataset = Path(settings["dataset"])
            result = json.loads((dataset / "collection_metrics.json").read_text())
            result["export"] = json.loads((dataset / "export_metrics.json").read_text())
            result["rounds"] = [
                json.loads(p.read_text())
                for p in sorted(Path(settings["raw"]).glob("round-*/summary.json"))
            ]
            episodes = [
                row
                for root in sorted(Path(settings["raw"]).glob("round-*"))
                for row in episode_metadata(root)
            ]
            result["contact"] = {
                "valid_episodes": sum(
                    row["contact_quality"]["valid"] for row in episodes
                ),
                "peak_nonpad_penetration": max(
                    row["contact_quality"]["peak_nonpad_penetration"]
                    for row in episodes
                ),
                "peak_penetration": max(
                    row["contact_quality"]["peak_penetration"] for row in episodes
                ),
            }
            result["split"] = {
                key: len(value)
                for key, value in json.loads(
                    (dataset / "split.json").read_text()
                ).items()
                if key in ("train", "val")
            }
            report["results"][name] = result
            tracker.log({f"benchmark/{name}": result})
            write_json(status, report)
        results = report["results"]
        report["speedup"] = {
            "export": results["export_baseline"]["export_seconds"]
            / results["export_optimized"]["export_seconds"],
            "pipeline": results["baseline"]["pipeline_seconds"]
            / results["optimized"]["pipeline_seconds"],
        }
        tracker.log({"benchmark/speedup": report["speedup"]})
        report["status"] = "complete"
        failed = False
    finally:
        if failed:
            report["status"] = "failed"
        write_json(status, report)
        tracker.log({}, reports=[status])
        tracker.finish(failed)


if __name__ == "__main__":
    benchmark(parse_args(Config))
