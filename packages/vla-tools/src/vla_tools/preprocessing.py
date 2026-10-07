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


def configure_training(trainer, settings):
    configure_precision(settings)
    original = trainer.make_pre_post_processors

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
