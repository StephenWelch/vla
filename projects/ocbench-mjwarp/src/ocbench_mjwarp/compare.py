"""Matched ACT comparisons over outcomes or action representations."""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

from vla_tools.config import parse_args
from vla_tools.hooks import scene_split
from vla_tools.tracking import WandbConfig, write_json

from .train import TrainConfig, TrainingConfig


@dataclass
class Config:
    dataset: Path
    output: Path
    training_root: Path | None = None
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            steps=100_000,
            action_mode="delta",
            percentile_normalization=True,
            wandb=WandbConfig(enable=True, project="vla-ocbench"),
            overrides={
                "policy.chunk_size": "25",
                "policy.n_action_steps": "25",
                "log_freq": "25",
            },
        )
    )
    exclude_pick_retries: bool = True
    exclude_mistakes: bool = False
    comparison: Literal["outcomes", "actions"] = "outcomes"
    prepare_only: bool = False
    reuse_prepared: bool = False

    def __post_init__(self):
        if self.comparison == "actions" and not (
            self.exclude_pick_retries and self.exclude_mistakes
        ):
            raise ValueError("Action comparison requires clean successes")


def selections(manifest, exclude_pick_retries=True, exclude_mistakes=False):
    """Filter success training by recorded retries, preserving the shared holdout."""
    rows = manifest["episodes"]
    if any(r.get("source_kind") == "ocbench_hub" for r in rows):
        raise ValueError(
            "Imported data has unknown contact audits and regrasp counts; clean-success comparisons require locally audited episodes. Train the imported dataset directly instead."
        )
    if any(
        not r["physical_valid"] or r.get("dataset_split") not in ("train", "val")
        for r in rows
    ):
        raise ValueError("Comparison requires audited episodes with recorded splits")
    identities = [(r["source_root"], r["episode_id"]) for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("Duplicate source episodes in comparison")
    val = [r["episode_index"] for r in rows if r["dataset_split"] == "val"]
    success_ids = set()
    for row in rows:
        if row["dataset_split"] != "train" or not row["native_success"]:
            continue
        if exclude_pick_retries:
            plans = row.get("randomization", {}).get("plans")
            if not plans or any("num_pick_retries" not in p for p in plans):
                raise ValueError(
                    "Success filtering requires recorded pick-retry annotations"
                )
            if any(p["num_pick_retries"] != 0 for p in plans):
                continue
        if exclude_mistakes:
            plans = row.get("randomization", {}).get("plans")
            if not plans or any("is_mistake" not in p for p in plans):
                raise ValueError("Clean successes require recorded mistake annotations")
            if any(p["is_mistake"] != 0 for p in plans):
                continue
        success_ids.add(row["episode_index"])
    train = {
        name: [
            r["episode_index"]
            for r in rows
            if r["dataset_split"] == "train"
            and (name == "all" or r["episode_index"] in success_ids)
        ]
        for name in ("successes", "all")
    }
    if not val or any(not ids for ids in train.values()):
        raise ValueError("Both training arms and the shared holdout must be nonempty")
    return train, val


def prepare(config):
    from .prepare import validate_prepared

    if config.output.exists():
        raise FileExistsError(f"Comparison output already exists: {config.output}")
    dataset = config.dataset
    combined = validate_prepared(dataset)
    train_ids, val_ids = selections(
        combined, config.exclude_pick_retries, config.exclude_mistakes
    )
    if config.comparison == "actions":
        train_ids = {
            name: list(train_ids["successes"]) for name in ("absolute", "relative")
        }
    names = list(train_ids)
    config.output.mkdir(parents=True)
    splits = {
        name: scene_split(dataset, 0.2, config.training.seed, sorted(ids + val_ids))
        for name, ids in train_ids.items()
    }
    reference = splits[names[0]]
    if any(split["val"] != reference["val"] for split in splits.values()):
        raise ValueError("Holdout differs between comparison arms")
    if any(not r["native_success"] for r in reference["train"]):
        raise ValueError("Failure leaked into success-only training")
    eval_seeds = {
        name: sorted({r["seed"] for r in reference[name]})[
            : config.training.eval_episodes
        ]
        for name in ("train", "val")
    }
    paths = []
    for name in train_ids:
        spec = TrainConfig(
            dataset=dataset,
            output=(config.training_root or config.output) / name,
            **(
                asdict(config.training)
                | {
                    "episodes": sorted(train_ids[name] + val_ids),
                    "action_mode": "absolute"
                    if name == "absolute"
                    else "delta"
                    if name == "relative"
                    else config.training.action_mode,
                    "validation_fraction": 0.2,
                    "train_eval_seeds": eval_seeds["train"],
                    "val_eval_seeds": eval_seeds["val"],
                }
                | {
                    "wandb": replace(
                        config.training.wandb,
                        group=config.training.wandb.group or config.output.name,
                        name=config.training.wandb.name or f"act-{name}",
                        tags=[
                            *config.training.wandb.tags,
                            "act",
                            "shared-holdout",
                            name,
                        ],
                    ),
                }
            ),
        )
        path = config.output / f"train-{name}.json"
        write_json(path, asdict(spec))
        paths.append(path)
    write_json(
        config.output / "comparison.json",
        {
            "config": asdict(config),
            "dataset": str(dataset),
            "train_episode_indices": train_ids,
            "success_filter": {
                "exclude_pick_retries": config.exclude_pick_retries,
                "exclude_mistakes": config.exclude_mistakes,
                "excluded_train_episode_indices": [
                    r["episode_index"]
                    for r in combined["episodes"]
                    if r["dataset_split"] == "train"
                    and r["native_success"]
                    and r["episode_index"] not in train_ids[names[0]]
                ],
            },
            "holdout_episode_indices": val_ids,
            "eval_seeds": eval_seeds,
            "holdout_sha256": hashlib.sha256(
                json.dumps(reference["val"], sort_keys=True).encode()
            ).hexdigest(),
            "dataset_manifest_sha256": hashlib.sha256(
                (dataset / "manifest.json").read_bytes()
            ).hexdigest(),
            "counts": {
                name: {part: len(split[part]) for part in ("train", "val")}
                for name, split in splits.items()
            },
        },
    )
    write_json(
        config.output / "status.json",
        {"status": "prepared", "queued": names},
    )
    return paths


def run(config):
    names = (
        ("absolute", "relative")
        if config.comparison == "actions"
        else ("successes", "all")
    )
    if config.reuse_prepared:
        record = json.loads((config.output / "comparison.json").read_text())
        current = json.loads(json.dumps(asdict(config), default=str))
        ignored = {"prepare_only", "reuse_prepared"}
        if {k: v for k, v in current.items() if k not in ignored} != {
            k: v for k, v in record["config"].items() if k not in ignored
        }:
            raise ValueError("Prepared comparison configuration differs")
        if (
            hashlib.sha256((config.dataset / "manifest.json").read_bytes()).hexdigest()
            != record["dataset_manifest_sha256"]
        ):
            raise ValueError("Prepared dataset changed")
        paths = [config.output / f"train-{name}.json" for name in names]
    else:
        paths = prepare(config)
    if config.prepare_only:
        return
    for index, (name, path) in enumerate(zip(names, paths, strict=True)):
        write_json(
            config.output / "status.json",
            {
                "status": "training",
                "active": name,
                "queued": list(names[index + 1 :]),
            },
        )
        try:
            subprocess.run(
                [sys.executable, "-m", "ocbench_mjwarp.train", "--config", str(path)],
                check=True,
            )
        except BaseException as error:
            write_json(
                config.output / "status.json",
                {"status": "failed", "active": name, "error": str(error)},
            )
            raise
    write_json(config.output / "status.json", {"status": "complete"})


if __name__ == "__main__":
    run(parse_args(Config))
