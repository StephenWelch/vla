"""Download the exact SmolVLA checkpoint and LIBERO assets for Docker mounts."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    root = args.data_root.resolve()
    model_dir = root / "models" / "smolvla_libero"
    assets_dir = root / "libero-assets"

    snapshot_download(
        repo_id="HuggingFaceVLA/smolvla_libero",
        revision=MODEL_REVISION,
        local_dir=model_dir,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "*.safetensors",
        ],
    )
    snapshot_download(
        repo_id="lerobot/libero-assets",
        repo_type="dataset",
        revision=ASSETS_REVISION,
        local_dir=assets_dir,
    )

    config = json.loads((model_dir / "config.json").read_text())
    expected_inputs = {
        "observation.images.image": [3, 256, 256],
        "observation.images.image2": [3, 256, 256],
        "observation.state": [8],
    }
    for name, shape in expected_inputs.items():
        actual = config["input_features"][name]["shape"]
        if actual != shape:
            raise ValueError(f"{name}: expected {shape}, got {actual}")
    if config["output_features"]["action"]["shape"] != [7]:
        raise ValueError("This checkpoint does not have LIBERO's 7D action shape")

    print(f"Model: {model_dir} ({MODEL_REVISION})")
    print(f"Assets: {assets_dir} ({ASSETS_REVISION})")


if __name__ == "__main__":
    main()
