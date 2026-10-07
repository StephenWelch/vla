"""Prepare one immutable combined dataset for many training comparisons."""

import errno
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vla_tools.config import parse_args
from vla_tools.hooks import scene_split
from vla_tools.tracking import write_json

from .profile import profiles


@dataclass
class PrepareConfig:
    datasets: Path
    output: Path


def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, target)


def validate_prepared(root):
    profiles(root)
    if (root / "INCOMPLETE").exists():
        raise ValueError("Incomplete prepared dataset")
    manifest = json.loads((root / "manifest.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    rows = manifest["episodes"]
    if [r["episode_index"] for r in rows] != list(range(info["total_episodes"])):
        raise ValueError("Prepared episode indices/count differ from metadata")
    if sum(row["length"] for row in rows) != info["total_frames"]:
        raise ValueError("Prepared frame count differs from metadata")
    identities = [(r["source_root"], r["episode_id"]) for r in rows]
    if len(set(identities)) != len(rows) or any(
        not r["physical_valid"] or r.get("dataset_split") not in ("train", "val")
        for r in rows
    ):
        raise ValueError(
            "Prepared dataset needs unique audited episodes and recorded splits"
        )
    if any(not (root / r["replay"]).is_file() for r in rows):
        raise ValueError("Prepared replay data missing")
    scene_split(root)
    return manifest


def prepare_dataset(config):
    sources = [config.datasets / name for name in ("successes", "failures")]
    contracts = [profiles(source) for source in sources]
    if contracts[0] != contracts[1]:
        raise ValueError("Source action/rendering profiles differ")
    if any((source / "INCOMPLETE").exists() for source in sources):
        raise ValueError("Incomplete source dataset")
    hashes = {
        str(source.resolve()): hashlib.sha256(
            (source / "manifest.json").read_bytes()
        ).hexdigest()
        for source in sources
    }
    if config.output.exists():
        validate_prepared(config.output)
        saved = json.loads((config.output / "preparation.json").read_text())
        if saved["source_manifest_sha256"] != hashes:
            raise ValueError("Prepared dataset source provenance differs")
        if (
            saved["manifest_sha256"]
            != hashlib.sha256(
                (config.output / "manifest.json").read_bytes()
            ).hexdigest()
        ):
            raise ValueError("Prepared dataset manifest changed")
        return config.output
    manifests = [
        json.loads((source / "manifest.json").read_text()) for source in sources
    ]
    combined = manifests[0] | {
        "repo_id": "local/ocbench-stack-all",
        "quality": "audited-successes-and-failures",
        "episodes": [],
    }
    for source, manifest in zip(sources, manifests, strict=True):
        for row in manifest["episodes"]:
            combined["episodes"].append(
                row
                | {
                    "episode_index": len(combined["episodes"]),
                    "source_dataset": str(source.resolve()),
                    "source_episode_index": row["episode_index"],
                }
            )
    from lerobot.datasets import aggregate

    config.output.parent.mkdir(parents=True, exist_ok=True)
    # Aggregate requires an absent root. A sibling marker covers failure before it creates one.
    marker = config.output.with_name(config.output.name + ".INCOMPLETE")
    marker.write_text("preparing\n")
    try:
        with patch.object(aggregate, "shutil", SimpleNamespace(copy=link_or_copy)):
            aggregate.aggregate_datasets(
                [m["repo_id"] for m in manifests],
                combined["repo_id"],
                roots=sources,
                aggr_root=config.output,
                concatenate_videos=False,
                concatenate_data=False,
            )
        for row in combined["episodes"]:
            target = (
                config.output / "replay" / f"episode-{row['episode_index']:06d}.npz"
            )
            target.parent.mkdir(exist_ok=True)
            link_or_copy(Path(row["source_dataset"]) / row["replay"], target)
            row["replay"] = str(target.relative_to(config.output))
        write_json(config.output / "manifest.json", combined)
        validate_prepared(config.output)
        write_json(config.output / "split.json", scene_split(config.output))
        write_json(
            config.output / "preparation.json",
            {
                "source_manifest_sha256": hashes,
                "manifest_sha256": hashlib.sha256(
                    (config.output / "manifest.json").read_bytes()
                ).hexdigest(),
            },
        )
    except BaseException:
        if config.output.exists():
            (config.output / "INCOMPLETE").touch()
        raise
    marker.unlink()
    return config.output


if __name__ == "__main__":
    print(prepare_dataset(parse_args(PrepareConfig)))
