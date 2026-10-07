"""Saved LeRobot image resizing and consistent ACT mixed precision."""


def configure_precision(settings):
    import torch

    if settings.get("use_amp", False):
        dtype = settings.get("amp_dtype", "bfloat16")
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested BF16 AMP requires a supported CUDA device")
        torch.set_autocast_dtype("cuda", getattr(torch, dtype))


def policy_autocast(policy):
    import torch

    return torch.autocast(
        device_type=torch.device(getattr(policy.config, "device", "cpu")).type,
        enabled=getattr(policy.config, "use_amp", False),
    )


def resize_preprocessor(pre, size):
    """Insert native, serializable antialiased resizing before normalization."""
    from lerobot.processor import ImageCropResizeProcessorStep, NormalizerProcessorStep

    existing = [s for s in pre.steps if isinstance(s, ImageCropResizeProcessorStep)]
    if existing:
        if len(existing) != 1 or tuple(existing[0].resize_size or ()) != tuple(size):
            raise ValueError("Checkpoint image resize differs from requested size")
        return
    index = next(
        i for i, s in enumerate(pre.steps) if isinstance(s, NormalizerProcessorStep)
    )
    pre.steps.insert(index, ImageCropResizeProcessorStep(resize_size=tuple(size)))


def configure_chunk_normalization(policy, pre, post):
    """Decode offset-specific ACT outputs before native select_action queues them.

    Training forward/probes remain in per-offset normalized coordinates. Inference
    predict_action_chunk returns physical actions, so the saved postprocessor skips
    action unnormalization. Call this after loading the saved processors as well.
    """
    from dataclasses import replace

    import torch
    from lerobot.configs import FeatureType, NormalizationMode
    from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep

    if policy.config.type != "act":
        return
    normalizer = next(s for s in pre.steps if isinstance(s, NormalizerProcessorStep))
    low = torch.as_tensor(normalizer.stats.get("action", {}).get("q01", []))
    if low.ndim != 2:
        return
    if policy.config.type != "act" or policy.config.temporal_ensemble_coeff is not None:
        raise ValueError("Per-timestep normalization requires open-loop ACT")
    if normalizer.norm_map[FeatureType.ACTION] != NormalizationMode.QUANTILES:
        raise ValueError("Per-timestep normalization requires action quantiles")
    if low.shape != (policy.config.chunk_size, policy.config.action_feature.shape[0]):
        raise ValueError("Action statistics do not match the ACT chunk shape")
    if getattr(policy, "_per_timestep_normalization", False):
        return
    low = low.to(device=policy.config.device, dtype=torch.float32)
    high = torch.as_tensor(
        normalizer.stats["action"]["q99"], device=low.device, dtype=torch.float32
    )
    span = high - low
    if not torch.isfinite(span).all() or not (span > 0).all():
        raise ValueError("Action quantile spans must be finite and positive")
    original = policy.predict_action_chunk

    def predict(batch):
        normalized = original(batch).float()
        return (normalized + 1) * span / 2 + low

    policy.predict_action_chunk = predict
    policy._per_timestep_normalization = True
    for i, step in enumerate(post.steps):
        if isinstance(step, UnnormalizerProcessorStep):
            post.steps[i] = replace(
                step,
                norm_map={
                    **step.norm_map,
                    FeatureType.ACTION: NormalizationMode.IDENTITY,
                },
            )


def configure_training(trainer, settings):
    configure_precision(settings)
    original = trainer.make_pre_post_processors
    policy = None
    if settings.get("per_timestep_normalization"):
        make_policy = trainer.make_policy

        def capture_policy(*args, **kwargs):
            nonlocal policy
            policy = make_policy(*args, **kwargs)
            return policy

        trainer.make_policy = capture_policy

    def processors(*args, **kwargs):
        pre, post = original(*args, **kwargs)
        if settings.get("image_size"):
            resize_preprocessor(pre, settings["image_size"])
        if settings.get("use_amp"):
            from lerobot.processor import DeviceProcessorStep

            # Unnormalize in float32 as well as returning NumPy-compatible actions.
            if not (
                isinstance(post.steps[0], DeviceProcessorStep)
                and post.steps[0].float_dtype == "float32"
            ):
                post.steps.insert(
                    0,
                    DeviceProcessorStep(
                        device=settings["device"], float_dtype="float32"
                    ),
                )
        if settings.get("per_timestep_normalization"):
            configure_chunk_normalization(policy, pre, post)
        return pre, post

    trainer.make_pre_post_processors = processors
    if settings.get("use_amp"):
        expected = {"bfloat16": "bf16", "float16": "fp16"}[settings["amp_dtype"]]
        from functools import partial

        from accelerate import Accelerator
        from accelerate.utils import DistributedDataParallelKwargs

        # Use the trainer's explicit Accelerator argument; ACT has no dtype field.
        trainer.train = partial(
            trainer.train,
            accelerator=Accelerator(
                mixed_precision=expected,
                step_scheduler_with_optimizer=False,
                kwargs_handlers=[
                    DistributedDataParallelKwargs(find_unused_parameters=True)
                ],
            ),
        )
