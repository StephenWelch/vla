"""Checkpoint resizing must survive reload and precede RGB normalization."""

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_pre_post_processors
from torchvision.transforms.functional import resize
from vla_tools.preprocessing import resize_preprocessor


def test_resize_normalization_and_checkpoint_reload(tmp_path, monkeypatch):
    camera = "observation.images.front"
    cfg = ACTConfig(
        device="cpu",
        input_features={
            camera: PolicyFeature(FeatureType.VISUAL, (3, 240, 320)),
            "observation.state": PolicyFeature(FeatureType.STATE, (18,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
    )
    stats = {
        camera: {
            "mean": torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
            "std": torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1),
        },
        "observation.state": {"mean": torch.zeros(18), "std": torch.ones(18)},
        "action": {"mean": torch.full((7,), 0.1234567), "std": torch.ones(7)},
    }
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    resize_preprocessor(pre, (240, 320))
    resize_preprocessor(pre, (240, 320))  # Resume must not append a second resize.
    batch = {camera: torch.rand(2, 3, 480, 640), "observation.state": torch.rand(2, 18)}
    expected = (
        resize(batch[camera], [240, 320], antialias=True) - stats[camera]["mean"]
    ) / (stats[camera]["std"] + 1e-8)
    actual = pre(batch)[camera]
    torch.testing.assert_close(actual, expected)
    assert actual.min() < 0  # Resizing after normalization would clamp this away.
    from types import SimpleNamespace

    import accelerate
    from vla_tools import preprocessing

    monkeypatch.setattr(accelerate, "Accelerator", lambda **kw: SimpleNamespace(**kw))

    monkeypatch.setattr(preprocessing, "configure_precision", lambda settings: None)
    trainer = SimpleNamespace(
        make_pre_post_processors=lambda: (pre, post),
        save_checkpoint=lambda **kw: None,
        train=lambda **kw: kw["accelerator"],
        update_policy=lambda **kw: None,
    )
    preprocessing.configure_training(
        trainer, {"use_amp": True, "amp_dtype": "bfloat16", "device": "cpu"}
    )
    trainer.make_pre_post_processors()
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded, loaded_post = make_pre_post_processors(cfg, pretrained_path=tmp_path)
    action = loaded_post(torch.zeros(2, 7, dtype=torch.bfloat16))
    assert action.dtype == torch.float32
    torch.testing.assert_close(action, stats["action"]["mean"].expand(2, 7))
    torch.testing.assert_close(loaded(batch)[camera], actual)
    with pytest.raises(ValueError, match="differs"):
        resize_preprocessor(loaded, (288, 384))


def test_amp_configures_accelerate_and_rejects_silent_fallback(monkeypatch):
    from types import SimpleNamespace

    import accelerate
    from vla_tools import preprocessing

    monkeypatch.setattr(accelerate, "Accelerator", lambda **kw: SimpleNamespace(**kw))

    monkeypatch.setattr(preprocessing, "configure_precision", lambda settings: None)
    trainer = SimpleNamespace(
        make_pre_post_processors=lambda: (None, None),
        save_checkpoint=lambda **kwargs: None,
        train=lambda **kwargs: kwargs["accelerator"],
        update_policy=lambda **kwargs: (
            SimpleNamespace(
                loss=SimpleNamespace(val=1.0), grad_norm=SimpleNamespace(val=2.0)
            ),
            {},
        ),
    )
    preprocessing.configure_training(
        trainer, {"use_amp": True, "amp_dtype": "bfloat16"}
    )
    accelerator = trainer.train()
    assert accelerator.mixed_precision == "bf16"
    from vla_tools.hooks import install_hooks

    trainer.make_train_eval_datasets = lambda cfg: None
    install_hooks(
        trainer,
        {"output": ".", "split": None, "use_amp": True, "amp_dtype": "bfloat16"},
    )
    with pytest.raises(RuntimeError, match="Expected bf16"):
        trainer.update_policy(accelerator=SimpleNamespace(mixed_precision="no"))
