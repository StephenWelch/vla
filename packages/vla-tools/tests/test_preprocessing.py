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


@pytest.mark.parametrize("execute_steps", [1, 2, 3])
def test_per_offset_actions_decode_before_queue_and_survive_reload(
    tmp_path, execute_steps
):
    from lerobot.configs import NormalizationMode
    from lerobot.policies.act.modeling_act import ACTPolicy
    from vla_tools.preprocessing import configure_chunk_normalization

    cfg = ACTConfig(
        device="cpu",
        chunk_size=3,
        n_action_steps=execute_steps,
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (2,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        normalization_mapping={
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.QUANTILES,
        },
    )
    low = torch.tensor([[0.0, 10.0], [100.0, 200.0], [-10.0, -20.0]])
    high = low + torch.tensor([[2.0, 4.0], [10.0, 20.0], [4.0, 8.0]])
    normalized = torch.tensor([[[-1.0, 1.0], [0.0, -0.5], [0.5, 0.0]]]).repeat(2, 1, 1)
    physical = (normalized + 1) * (high - low) / 2 + low
    pre, post = make_pre_post_processors(
        cfg, dataset_stats={"action": {"q01": low, "q99": high}}
    )

    class Policy:
        config = cfg
        reset = ACTPolicy.reset
        select_action = ACTPolicy.select_action

        def eval(self):
            return self

        def predict_action_chunk(self, batch):
            return normalized.to(torch.bfloat16)

    for reload in (False, True):
        if reload:
            pre, post = make_pre_post_processors(cfg, pretrained_path=tmp_path)
        policy = Policy()
        configure_chunk_normalization(policy, pre, post)
        configure_chunk_normalization(policy, pre, post)  # Idempotent.
        torch.testing.assert_close(
            pre({"action": physical, "observation.state": torch.zeros(2, 2)})["action"],
            normalized,
        )
        torch.testing.assert_close(post(policy.predict_action_chunk({})), physical)
        for _ in range(2):
            policy.reset()
            for i in range(execute_steps * 2):
                actual = post(policy.select_action({}))
                assert actual.shape == (2, 2) and actual.dtype == torch.float32
                torch.testing.assert_close(actual, physical[:, i % execute_steps])
        pre.save_pretrained(tmp_path)
        post.save_pretrained(tmp_path)
