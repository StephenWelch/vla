import json
from pathlib import Path


def ogbench_profile(dataset, checkpoint=None):
    path = Path(dataset) / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    if not manifest.get("format", "").startswith("ogbench-mjwarp-"):
        return None
    if manifest["format"] != "ogbench-mjwarp-2" or not manifest.get("rendering"):
        raise ValueError("Legacy OGBench dataset: collect a fresh MJWarp v2 dataset")
    profile = manifest["rendering"]
    if profile.get("revision") != 2:
        raise ValueError(
            "OGBench images use a faulty renderer; collect a fresh dataset"
        )
    if checkpoint is not None:
        saved = Path(checkpoint) / "rendering.json"
        if not saved.exists() or json.loads(saved.read_text()) != profile:
            raise ValueError("Checkpoint and dataset rendering profiles differ")
    return profile
