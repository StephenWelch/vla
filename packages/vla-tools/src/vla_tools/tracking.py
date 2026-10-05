"""W&B run identities, custom metrics, and small research reports."""

import json
import logging
import math
import netrc
import os
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class WandbConfig:
    enable: bool = False
    project: str = "vla-ogbench"
    entity: str | None = None
    mode: Literal["online", "offline", "disabled"] = "online"
    group: str | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=list)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, default=str, indent=2) + "\n")
    temporary.replace(path)


def credentials():
    """Read credentials into memory only; never include them in run metadata."""
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    host = urlparse(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
    paths = [Path.home() / ".netrc"]
    if os.environ.get("VLA_WANDB_NETRC_PATH"):
        paths.insert(0, Path(os.environ["VLA_WANDB_NETRC_PATH"]))
    for path in paths:
        if path.is_file():
            try:
                entry = netrc.netrc(str(path)).authenticators(host)
                if entry:
                    return entry[2]
            except (OSError, netrc.NetrcParseError):
                logger.warning("Could not read W&B credentials from %s", path)
    return None


def scalars(values, prefix=""):
    result = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(scalars(value, name))
        elif (
            isinstance(value, Real)
            and math.isfinite(value)
            or isinstance(value, str)
            and key == "loss_definition"
        ):
            result[name] = value
    return result


class Tracker:
    def __init__(self, output, config, job_type, metadata=None, resume=False):
        self.output, self.config = Path(output), config
        self.run = None
        self.state = {}
        if not config.enable or config.mode == "disabled":
            return
        import wandb

        self.wandb = wandb
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "tracking.json"
        previous = json.loads(self.path.read_text()) if self.path.exists() else {}
        if previous and not resume:
            raise ValueError(
                "Tracking identity already exists; resume or choose a new output"
            )
        for key in ("project", "group"):
            if previous and previous[key] != getattr(config, key):
                raise ValueError(f"Resume changes W&B {key}")
        if previous and config.entity and previous.get("entity") != config.entity:
            raise ValueError("Resume changes W&B entity")
        run_id = previous.get("run_id") or wandb.util.generate_id()
        key = credentials() if config.mode == "online" else None
        mode = config.mode
        fallback = None
        if mode == "online" and not key:
            mode, fallback = (
                "offline",
                "No W&B API key found; log in inside WSL or set WANDB_API_KEY",
            )
        arguments = {
            "id": run_id,
            "project": config.project,
            "entity": config.entity,
            "group": config.group,
            "name": config.name or self.output.name,
            "tags": config.tags,
            "job_type": job_type,
            "dir": str(self.output),
            "config": metadata or {},
            "save_code": False,
        }
        try:
            self.run = wandb.init(
                **arguments,
                mode=mode,
                resume="allow" if resume and mode == "online" else None,
                settings=wandb.Settings(api_key=key, init_timeout=15),
            )
        except (wandb.errors.CommError, wandb.errors.UsageError) as error:
            if mode != "online":
                raise
            mode, fallback = (
                "offline",
                f"Online initialization failed ({type(error).__name__})",
            )
            self.run = wandb.init(**arguments, mode="offline")
        self.event = max(previous.get("last_event", -1) + 1, self.run.step)
        self.run.define_metric("train/update")
        self.run.define_metric("train/*", step_metric="train/update")
        self.run.define_metric("val/*", step_metric="train/update")
        self.run.define_metric("collection/episodes_committed")
        self.run.define_metric(
            "collection/*", step_metric="collection/episodes_committed"
        )
        self.state = {
            "run_id": run_id,
            "project": config.project,
            "entity": self.run.entity,
            "group": config.group,
            "mode": mode,
            "url": self.run.get_url() if mode == "online" else None,
            "status": "running",
            "last_event": self.event - 1,
            "fallback_reason": fallback,
            "segments": previous.get("segments", []) + [str(Path(self.run.dir).parent)],
        }
        if fallback:
            logger.warning("%s. Continuing with offline W&B logging.", fallback)
        self.save()
        print(
            f"W&B: {self.state['url'] or 'offline'}; identity: {self.path}", flush=True
        )

    def save(self):
        if self.run:
            write_json(self.path, self.state)

    def log(self, values, update=None, videos=None, tables=None, reports=()):
        if not self.run:
            return
        payload = scalars(values)
        if update is not None:
            payload["train/update"] = update
        for name, path in (videos or {}).items():
            payload[name] = self.wandb.Video(str(path), format="mp4")
        for name, rows in (tables or {}).items():
            if rows:
                columns = sorted({key for row in rows for key in row})
                payload[name] = self.wandb.Table(
                    columns=columns,
                    data=[
                        [
                            json.dumps(row.get(key), default=str)
                            if isinstance(row.get(key), (dict, list))
                            else row.get(key)
                            for key in columns
                        ]
                        for row in rows
                    ],
                )
        if payload:
            self.run.log(payload, step=self.event)
            self.state["last_event"] = self.event
            self.event += 1
        paths = [Path(path) for path in reports if Path(path).is_file()]
        if paths:
            artifact = self.wandb.Artifact(
                f"reports-{self.state['run_id']}", type="research-report"
            )
            for index, path in enumerate(paths):
                artifact.add_file(str(path), name=f"{index:02d}-{path.name}")
            self.run.log_artifact(artifact)
        self.save()

    def finish(self, failed=False):
        if not self.run:
            return
        self.state["status"] = "failed_or_interrupted" if failed else "complete"
        self.run.summary["status"] = self.state["status"]
        self.save()
        self.run.finish(exit_code=1 if failed else 0)
        if self.state["mode"] == "offline":
            for segment in self.state["segments"]:
                print(
                    f"Sync: wandb sync --append --id {self.state['run_id']} {segment}",
                    flush=True,
                )


def install_native_logging(trainer, settings, record):
    """Use the native logger interface with one shared run and custom update axes."""
    config = WandbConfig(**settings.get("wandb", {}))
    if not config.enable or config.mode == "disabled":
        return lambda: None
    tracker = None

    class Logger(trainer.WandBLogger):
        def __init__(self, cfg):
            nonlocal tracker
            tracker = Tracker(
                cfg.output_dir,
                config,
                "train",
                {
                    "training": record["config"],
                    "dataset": record["dataset"],
                    "metadata_sha256": record["metadata_sha256"],
                    "camera_map": record["camera_map"],
                    "rendering": record.get("rendering"),
                    "versions": record["versions"],
                },
                resume=cfg.resume,
            )
            self.cfg = cfg.wandb
            self.cfg.disable_artifact = True
            self.cfg.run_id = tracker.state["run_id"]

        def log_dict(self, values, step=None, mode="train", custom_step_key=None):
            tracker.log({mode: values}, update=step)

        def log_video(self, path, step, mode="train"):
            tracker.log({}, update=step, videos={f"{mode}/video": path})

    trainer.WandBLogger = Logger
    return lambda: tracker


def evaluation_log(tracker, report, output, step=None):
    """Publish completed probes/rollouts with explicit split and task names."""
    if not tracker or not tracker.run:
        return
    values, tables = {}, {}
    for key, value in report.items():
        if key.endswith(("/probe", "/rollout")):
            values[key] = dict(value)
            if isinstance(value, dict) and value.get("episodes"):
                tables[f"{key}/episodes"] = value["episodes"]
                for metric in ("peak_nonpad_penetration", "peak_penetration"):
                    values[key][metric] = max(
                        row.get(metric, 0) for row in value["episodes"]
                    )
    for task, metrics in report.get("tasks", {}).items():
        aggregate = dict(metrics["aggregated"])
        for metric in ("physics_valid", "truncated"):
            rows = metrics["per_episode"]
            if rows:
                aggregate[f"pc_{metric}"] = (
                    100 * sum(row[metric] for row in rows) / len(rows)
                )
        for metric in ("peak_nonpad_penetration", "peak_penetration"):
            aggregate[metric] = max(
                (row.get(metric, 0) for row in metrics["per_episode"]), default=0
            )
        values[f"eval/task_{task}"] = aggregate
        tables[f"eval/task_{task}/episodes"] = metrics["per_episode"]
    videos = {
        f"{path.relative_to(output).parent.as_posix()}/video": path
        for path in Path(output).rglob("*.mp4")
    }
    tracker.log(values, update=step, videos=videos, tables=tables)
