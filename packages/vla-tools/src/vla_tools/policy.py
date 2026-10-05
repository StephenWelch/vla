import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def checkpoint_step(pretrained_model):
    """Validate a completed native save without loading model/optimizer tensors."""
    from safetensors import safe_open

    model = Path(pretrained_model).resolve()
    state = model.parent / "training_state"
    required = [
        model / "model.safetensors",
        model / "config.json",
        model / "train_config.json",
        state / "training_step.json",
        state / "optimizer_param_groups.json",
        state / "optimizer_state.safetensors",
        state / "rng_state.safetensors",
    ]
    if any(not path.is_file() for path in required):
        raise ValueError("Checkpoint is missing model or resumable training state")
    try:
        for path in model.parent.rglob("*.json"):
            json.loads(path.read_text())
        for path in model.parent.rglob("*.safetensors"):
            with safe_open(str(path), framework="pt", device="cpu") as file:
                if not file.keys():
                    raise ValueError(f"Empty tensor archive: {path.name}")
        step = json.loads((state / "training_step.json").read_text())["step"]
    except Exception as error:
        raise ValueError(
            f"Invalid checkpoint {model.parent.name}: {type(error).__name__}"
        ) from error
    if (
        not isinstance(step, int)
        or step < 1
        or (model.parent.name.isdecimal() and int(model.parent.name) != step)
    ):
        raise ValueError("Checkpoint directory and saved update do not match")
    return step


def latest_checkpoint(output):
    paths = sorted(
        (Path(output) / "checkpoints").iterdir(),
        key=lambda path: int(path.name) if path.name.isdecimal() else -1,
        reverse=True,
    )
    for path in paths:
        if path.name.isdecimal():
            try:
                checkpoint_step(path / "pretrained_model")
                return path / "pretrained_model"
            except ValueError as error:
                logger.warning("Skipping damaged/incomplete checkpoint: %s", error)
    raise ValueError(f"No valid resumable checkpoint in {output}")


def runtime_environment(hf_home=None, environment=None):
    """Preserve existing authentication when relocating the model cache."""
    result = dict(os.environ if environment is None else environment)
    if hf_home is not None:
        original_home = Path(result.get("HF_HOME", Path.home() / ".cache/huggingface"))
        original_token = original_home / "token"
        if (
            "HF_TOKEN_PATH" not in result
            and not (hf_home / "token").is_file()
            and original_token.is_file()
        ):
            result["HF_TOKEN_PATH"] = str(original_token.resolve())
        result["HF_HOME"] = str(hf_home.resolve())
    result["PYTHONUNBUFFERED"] = "1"
    return result


def load_policy(checkpoint, device):
    """Load a policy and its saved camera mapping, normalization, and processors."""
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    config = json.loads((checkpoint / "config.json").read_text())
    policy = (
        get_policy_class(config["type"]).from_pretrained(checkpoint).to(device).eval()
    )
    pre, post = make_pre_post_processors(
        policy.config,
        str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post
