"""Exercise buffer reuse and failures with deliberately slow encoder consumers."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.gpu
@pytest.mark.parametrize(
    "depth,fail,batch",
    [(1, False, 1), (3, False, 1), (2, True, 1), (1, False, 4), (2, True, 4)],
)
def test_async_buffer_ownership_and_worker_failure(
    tmp_path, monkeypatch, depth, fail, batch
):
    from ocbench_mjwarp import gpu_video
    from ocbench_mjwarp.async_video import AsyncVideoPipeline

    lengths = [9, 6]
    encoders = []

    class Encoder:
        def __init__(self, root, world, sim, length, stream, **kwargs):
            self.world, self.length = world, length
            self.frames, self.samples = [], []
            self.closed = False
            self.timings = {}
            encoders.append(self)

        def feed(self, pixels, samples):
            time.sleep(0.003 * (self.world + 1))
            if fail and self.world == 1 and len(self.frames) == 2:
                raise RuntimeError("injected encoder failure")
            self.frames.append(pixels.cpu().numpy().copy())
            self.samples.append(samples.copy())

        def finish_episode(self):
            assert len(self.frames) == self.length
            return {}

        def close(self):
            self.closed = True

    monkeypatch.setattr(gpu_video, "DeviceEncoder", Encoder)
    stream = torch.cuda.Stream()
    sim = SimpleNamespace(
        warp_device="cuda:0",
        torch_stream=stream,
        renderer=SimpleNamespace(height=8, width=12),
    )
    with torch.cuda.stream(stream):
        base = torch.arange(16, dtype=torch.uint8, device="cuda").reshape(2, 2, 1, 1, 4)
        pixels = torch.empty((batch, 2, 2, 8, 12, 4), dtype=torch.uint8, device="cuda")
        offsets = torch.arange(batch, dtype=torch.uint8, device="cuda").reshape(
            batch, 1, 1, 1, 1, 1
        )
    stream.synchronize()
    expected = base.cpu().numpy()

    class Replay:
        def render(self, tick):
            # Deliberately overwrite the same renderer-owned buffer every tick.
            pixels.copy_(base + tick + offsets)
            return pixels

    pipeline = AsyncVideoPipeline(
        tmp_path, sim, lengths, depth, profile=depth == 3, render_batch_frames=batch
    )
    try:
        if fail:
            with pytest.raises(RuntimeError, match="injected encoder failure"):
                for tick in range(0, max(lengths), batch):
                    pipeline.render(Replay(), tick)
                pipeline.finish()
        else:
            for tick in range(0, max(lengths), batch):
                pipeline.render(Replay(), tick)
            pipeline.finish()
            assert pipeline.metrics["submitted_frames"] == sum(lengths)
            assert pipeline.metrics["gpu_buffer_bytes"] == depth * pixels.numel()
            if depth == 3:
                assert pipeline.metrics["render_interval_seconds"] > 0
                assert len(pipeline.metrics["encoder_workers"]) == len(lengths)
            for encoder in encoders:
                for tick, (frame, samples) in enumerate(
                    zip(encoder.frames, encoder.samples, strict=True)
                ):
                    reference = np.broadcast_to(
                        expected[encoder.world] + tick, frame.shape
                    )
                    np.testing.assert_array_equal(frame, reference)
                    np.testing.assert_array_equal(
                        samples, reference[:, ::4, ::4, :3][..., ::-1]
                    )
    finally:
        pipeline.close()
    assert len(encoders) == 2 and all(e.closed for e in encoders)
