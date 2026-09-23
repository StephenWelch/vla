import sys

from huggingface_hub import snapshot_download


if __name__ == "__main__":
    print(
        snapshot_download(
            "jubba/smolvla_pusht_20k_10_14_2025",
            local_dir=sys.argv[1],
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
                "*.safetensors",
            ],
        )
    )
