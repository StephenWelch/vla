"""The small shared boundary between recorded episodes and image export."""

import hashlib
import json

import numpy as np

from .config import FIELDS, validate_action


def imported(row):
    return row.get("source_kind") == "ocbench_hub"


def records(root):
    return [
        json.loads(p.read_text()) for p in sorted((root / "raw").glob("episode-*.json"))
    ]


def select_episodes(source, successes=None, limit=None):
    if (source / "import.json").exists():
        status = source / "import-status.json"
        if (
            not status.exists()
            or json.loads(status.read_text()).get("status") != "complete"
        ):
            raise ValueError("Finish or resume the Hub import before exporting")
    rows = [
        r
        for r in records(source)
        if (
            r["physical_valid"] is True or (imported(r) and r["physical_valid"] is None)
        )
        and (successes is None or r["native_success"] == successes)
    ]
    return rows if limit is None else rows[:limit]


def load_arrays(source, rows):
    result = []
    for row in rows:
        validate_action(row["action_profile"])
        path = source / "raw" / row["archive"]
        if imported(row):
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != row.get("archive_sha256"):
                raise ValueError("Imported archive checksum differs from provenance")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        n = row["length"]
        expected = n if imported(row) else n + 1
        if len(arrays["action"]) != n or any(
            len(arrays["sim/" + key]) != expected for key in FIELDS
        ):
            raise ValueError("Raw action/state alignment differs from episode metadata")
        if imported(row) and (
            row.get("state_alignment") != "pre_action"
            or row.get("terminal_state_available") is not False
        ):
            raise ValueError(
                "Imported trajectories require explicit pre-action alignment"
            )
        result.append(arrays)
    return result
