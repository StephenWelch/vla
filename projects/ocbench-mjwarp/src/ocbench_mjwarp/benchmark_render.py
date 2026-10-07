"""Benchmark the production exporter on a bounded episode selection."""

import json
import time
from dataclasses import dataclass

from vla_tools.config import parse_args
from vla_tools.tracking import write_json

from .dataset import ExportConfig, export


@dataclass
class Config(ExportConfig):
    limit: int = 2


def run(config):
    if config.output.exists():
        raise FileExistsError(
            "Benchmark requires a fresh output to measure actual export work"
        )
    if config.limit < 1:
        raise ValueError("Benchmark requires a positive episode limit")
    started = time.perf_counter()
    result = export(config)
    seconds = time.perf_counter() - started
    report = {
        **result,
        "wall_seconds": seconds,
        "frames_per_second": result["frames"] / seconds,
    }
    path = config.output / "materialization.json"
    if path.exists():
        report["materialization"] = json.loads(path.read_text())
    write_json(config.output.with_name(config.output.name + ".benchmark.json"), report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(parse_args(Config)), indent=2))
