"""Restartable collection -> ACT -> SmolVLA research run, all inside WSL."""

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from vla_tools.config import parse_args
from vla_tools.policy import latest_checkpoint
from vla_tools.tracking import WandbConfig, write_json


@dataclass
class Config:
    raw: Path | None = None
    dataset: Path | None = None
    runs: Path = Path("outputs/ogbench/runs")
    scenes: int = 100
    scenes_per_round: int = 20
    env_id: str = "cube-single-v0"
    task_id: int = 1
    variants: int = 2
    eval_max_steps: int = 250
    background: bool = False
    generation_batch_size: int = 32
    experiment: str | None = None
    steps: int = 20_000
    policies: list[Literal["act", "smolvla"]] = field(
        default_factory=lambda: ["act", "smolvla"]
    )
    size: int | tuple[int, int] = (480, 640)
    refill_slots: bool = True
    batched_cpu: bool = True
    streaming_encoding: bool = True
    overlap_export: bool = True
    encoder_queue_size: int = 30
    encoder_threads: int = 2
    export_queue_capacity: int = 2
    wandb: WandbConfig = field(default_factory=lambda: WandbConfig(enable=True))


def pipeline_alive(record):
    """A reboot or reused PID must not leave the launcher permanently locked."""
    path = Path(f"/proc/{record.get('pid', -1)}/cmdline")
    try:
        arguments = path.read_bytes().split(b"\0")
        target = str(Path(__file__).resolve()).encode()
        return (
            target in arguments
            or b"ogbench_mjwarp.pipeline" in arguments
            or any(
                argument.endswith(b"/run-ogbench-training.py") for argument in arguments
            )
        )
    except (FileNotFoundError, ProcessLookupError):
        return False


def run(config):
    if not config.policies or len(set(config.policies)) != len(config.policies):
        raise ValueError("Provide unique policy names")
    config.runs.mkdir(parents=True, exist_ok=True)
    latest_status = config.runs / "long-training-status.json"
    if latest_status.exists():
        previous = json.loads(latest_status.read_text())
        if previous["status"] == "running" and pipeline_alive(previous):
            raise RuntimeError(f"Pipeline already running as PID {previous['pid']}")
    config.experiment = (
        config.experiment
        or datetime.now(UTC).strftime("ogbench-%Y%m%d-%H%M%S-") + uuid4().hex[:4]
    )
    if Path(config.experiment).name != config.experiment or config.experiment in (
        ".",
        "..",
    ):
        raise ValueError("Experiment must be one directory name")
    directory = config.runs / config.experiment
    directory.mkdir(parents=True, exist_ok=True)
    config.raw = config.raw or config.runs.parent / "raw" / config.experiment
    config.dataset = (
        config.dataset or config.runs.parent / "datasets" / config.experiment
    )
    config.wandb.group = config.wandb.group or config.experiment
    status = directory / "pipeline-status.json"
    settings = asdict(config) | {"background": False}
    write_json(directory / "pipeline-config.json", settings)
    if config.background:
        log_path = directory / "pipeline.log"
        with log_path.open("a") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "ogbench_mjwarp.pipeline",
                    "--config",
                    str(directory / "pipeline-config.json"),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(
            json.dumps(
                {"pid": process.pid, "log": str(log_path), "status": str(status)},
                indent=2,
            )
        )
        return
    report = {
        "status": "running",
        "phase": "collect",
        "pid": os.getpid(),
        "experiment": config.experiment,
        "directory": str(directory),
        "stages": {},
    }

    def write():
        write_json(status, report)
        write_json(latest_status, report)

    def stage_config(stage, values):
        tracking = asdict(config.wandb) | {"name": f"{config.experiment}-{stage}"}
        path = directory / f"{stage}-config.json"
        write_json(path, values | {"wandb": tracking})
        return path

    try:
        write()
        collection = stage_config(
            "collect",
            {
                "raw": config.raw,
                "dataset": config.dataset,
                "scenes": config.scenes,
                "scenes_per_round": config.scenes_per_round,
                "env_id": config.env_id,
                "task_id": config.task_id,
                "variants": config.variants,
                "generation_batch_size": config.generation_batch_size,
                "size": config.size,
                "refill_slots": config.refill_slots,
                "batched_cpu": config.batched_cpu,
                "streaming_encoding": config.streaming_encoding,
                "overlap_export": config.overlap_export,
                "encoder_queue_size": config.encoder_queue_size,
                "encoder_threads": config.encoder_threads,
                "export_queue_capacity": config.export_queue_capacity,
                "reuse_dataset": config.dataset.exists(),
                "tracking_output": directory / "collect",
            },
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ogbench_mjwarp.prepare",
                "--config",
                str(collection),
            ],
            check=True,
        )
        for policy in config.policies:
            tracking_path = directory / report["phase"] / "tracking.json"
            if tracking_path.exists():
                report["stages"][report["phase"]] = json.loads(
                    tracking_path.read_text()
                )
            report["phase"] = policy
            write()
            output = directory / policy
            from omegaconf import OmegaConf

            recipe = OmegaConf.to_container(
                OmegaConf.load(
                    Path(__file__).parent
                    / "recipes"
                    / f"train-ogbench-{policy}-wandb.yaml"
                ),
                resolve=True,
            )
            settings_path = stage_config(
                policy,
                recipe
                | {
                    "dataset": config.dataset,
                    "output": output,
                    "steps": config.steps,
                    "eval_max_steps": config.eval_max_steps,
                },
            )
            command = [
                sys.executable,
                "-m",
                "ogbench_mjwarp.train",
                "--config",
                str(settings_path),
                "--dataset",
                str(config.dataset),
                "--output",
                str(output),
            ]
            if output.exists():
                record = output / "experiment.json"
                if not record.exists():
                    record = output.with_name(output.name + ".experiment.json")
                experiment = json.loads(record.read_text())
                if experiment["status"] == "complete":
                    continue
                latest = latest_checkpoint(output)
                command.extend(["--resume", str(latest.resolve())])
            subprocess.run(command, check=True)
            tracking_path = output / "tracking.json"
            if tracking_path.exists():
                report["stages"][policy] = json.loads(tracking_path.read_text())
        report.update(status="complete", phase="complete")
    except BaseException:
        report["status"] = "failed_or_interrupted"
        raise
    finally:
        write()


def main():
    run(parse_args(Config))


if __name__ == "__main__":
    main()
