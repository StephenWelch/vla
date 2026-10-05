"""Download pinned policy assets and preserve their Hugging Face revision."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from policy_utils import parse_args, runtime_environment


@dataclass
class DownloadConfig:
    """Cache a policy checkpoint; optional tokenizer access is checked first."""

    repo_id: str
    output: Path
    revision: str | None = None
    hf_home: Path | None = None
    tokenizer_repo: str | None = None


def main():
    config = parse_args(DownloadConfig)
    os.environ.update(runtime_environment(config.hf_home))
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    if config.tokenizer_repo:
        hf_hub_download(config.tokenizer_repo, "tokenizer_config.json")
    info = HfApi().model_info(config.repo_id, revision=config.revision)
    source = {"repo_id": config.repo_id, "revision": info.sha}
    source_path = config.output / "source.json"
    if source_path.exists() and json.loads(source_path.read_text()) != source:
        raise ValueError(
            "Output belongs to another model revision; choose a new directory"
        )
    path = snapshot_download(
        config.repo_id,
        revision=info.sha,
        local_dir=config.output,
        allow_patterns=["*.json", "*.safetensors"],
    )
    source_path.write_text(json.dumps(source, indent=2) + "\n")
    print(json.dumps({**source, "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
