"""Bounded GPU snapshots and ordered, independent per-world encoding workers."""

import time
from concurrent.futures import ThreadPoolExecutor

import torch


class AsyncVideoPipeline:
    """Keep each snapshot until its encoder stream and CPU statistics finish.

    One worker owns two persistent camera sessions per world. This first pilot
    preserves continuous video streams. Temporal batches share these sessions;
    larger rendering batches do not require additional encoders. The caller
    limits concurrent episodes to four.
    """

    def __init__(
        self,
        root,
        sim,
        lengths,
        buffer_frames,
        *,
        profile=False,
        image_stats="native",
        render_batch_frames=1,
        write_buffer_bytes=8192,
        gop=2,
    ):
        from .gpu_video import DeviceEncoder

        if buffer_frames < 1 or not 1 <= len(lengths) <= 4 or min(lengths) < 1:
            raise ValueError("Positive buffer depth and 1-4 worlds required")
        if image_stats not in ("native", "uint8"):
            raise ValueError("Image statistics must be native or uint8")
        if render_batch_frames < 1:
            raise ValueError("Positive temporal batch required")
        self.render_batch_frames = render_batch_frames
        self.sim, self.lengths = sim, lengths
        self.profile = profile
        self.pools, self.workers, self.slots = [], [], []
        self.metrics = {
            "buffer_frames": buffer_frames,
            "render_batch_frames": render_batch_frames,
            "backpressure_seconds": 0.0,
            "submit_seconds": 0.0,
            "drain_seconds": 0.0,
            "worker_seconds": 0.0,
            "submitted_frames": 0,
            "ready_wait_seconds": 0.0,
            "stream_wait_seconds": 0.0,
            "render_interval_seconds": 0.0,
            "copy_interval_seconds": 0.0,
        }
        self.next_tick = 0
        self.closed = False

        def initialize(world):
            with torch.cuda.device(str(sim.warp_device)):
                stream = torch.cuda.Stream(device=str(sim.warp_device))
                with torch.cuda.stream(stream):
                    encoder = DeviceEncoder(
                        root,
                        world,
                        sim,
                        lengths[world],
                        stream.cuda_stream,
                        write_buffer_bytes=write_buffer_bytes,
                        gop=gop,
                    )
                    encoder.profile = profile
                    if image_stats == "uint8":
                        from .image_stats import ImageStats

                        encoder.stats = {key: ImageStats() for key in encoder.stats}
                return encoder, stream

        try:
            # Driver session creation is serialized; each session still belongs
            # to its encoding worker. Encoding itself remains concurrent.
            for world in range(len(lengths)):
                pool = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=f"nvenc-{world}"
                )
                self.pools.append(pool)
                self.workers.append(pool.submit(initialize, world).result(timeout=120))
            shape = (
                render_batch_frames,
                len(lengths),
                2,
                sim.renderer.height,
                sim.renderer.width,
                4,
            )
            sample_shape = (
                *shape[:3],
                len(range(0, shape[3], 4)),
                len(range(0, shape[4], 4)),
                3,
            )
            with torch.cuda.stream(sim.torch_stream):
                for _ in range(buffer_frames):
                    self.slots.append(
                        {
                            "pixels": torch.empty(
                                shape, dtype=torch.uint8, device=str(sim.warp_device)
                            ),
                            "samples": torch.empty(
                                sample_shape, dtype=torch.uint8, pin_memory=True
                            ),
                            "start": torch.cuda.Event(enable_timing=profile),
                            "rendered": torch.cuda.Event(enable_timing=profile),
                            "ready": torch.cuda.Event(enable_timing=profile),
                            "pending": [],
                        }
                    )
            self.metrics["gpu_buffer_bytes"] = sum(
                s["pixels"].numel() for s in self.slots
            )
            self.metrics["pinned_buffer_bytes"] = sum(
                s["samples"].numel() for s in self.slots
            )
        except BaseException:
            self.close()
            raise

    @staticmethod
    def consume(worker, slot, world, count):
        encoder, stream = worker
        started = time.perf_counter()
        # Wait for this snapshot only; the renderer can fill another slot.
        slot["ready"].synchronize()
        ready = time.perf_counter()
        with torch.cuda.stream(stream):
            try:
                samples = slot["samples"][:, world].numpy()
                for frame in range(count):
                    encoder.feed(slot["pixels"][frame, world], samples[frame])
            finally:
                # NVENC copies into internal surfaces on its input CUDA stream.
                # Do not recycle the slot while that copy may still be in flight.
                stamp = time.perf_counter()
                stream.synchronize()
        ended = time.perf_counter()
        return ended - started, ready - started, ended - stamp

    def reclaim(self, slot):
        if slot["pending"] and self.profile:
            slot["ready"].synchronize()
            self.metrics["render_interval_seconds"] += (
                slot["start"].elapsed_time(slot["rendered"]) / 1000
            )
            self.metrics["copy_interval_seconds"] += (
                slot["rendered"].elapsed_time(slot["ready"]) / 1000
            )
        for future in slot["pending"]:
            elapsed, ready, stream = future.result(timeout=120)
            self.metrics["worker_seconds"] += elapsed
            self.metrics["ready_wait_seconds"] += ready
            self.metrics["stream_wait_seconds"] += stream
        slot["pending"].clear()

    def render(self, replay, tick):
        if self.closed or tick != self.next_tick or tick >= max(self.lengths):
            raise ValueError("Frames must be submitted once, in increasing order")
        slot = self.slots[(tick // self.render_batch_frames) % len(self.slots)]
        stamp = time.perf_counter()
        self.reclaim(slot)
        self.metrics["backpressure_seconds"] += time.perf_counter() - stamp
        stamp = time.perf_counter()
        with torch.cuda.stream(self.sim.torch_stream):
            if self.profile:
                slot["start"].record(self.sim.torch_stream)
            bgra = replay.render(tick).reshape_as(slot["pixels"])
            if self.profile:
                slot["rendered"].record(self.sim.torch_stream)
            # The renderer owns/reuses bgra; this copy establishes slot ownership.
            slot["pixels"].copy_(bgra, non_blocking=True)
            slot["samples"].copy_(
                bgra[:, :, :, ::4, ::4, :3].flip(-1), non_blocking=True
            )
            slot["ready"].record(self.sim.torch_stream)
        for world, length in enumerate(self.lengths):
            if tick < length:
                slot["pending"].append(
                    self.pools[world].submit(
                        self.consume,
                        self.workers[world],
                        slot,
                        world,
                        min(self.render_batch_frames, length - tick),
                    )
                )
                self.metrics["submitted_frames"] += min(
                    self.render_batch_frames, length - tick
                )
        self.next_tick += self.render_batch_frames
        self.metrics["submit_seconds"] += time.perf_counter() - stamp

    def finish(self):
        stamp = time.perf_counter()
        for slot in self.slots:
            self.reclaim(slot)
        # Flush and remux on the session's owning worker before dataset commits.
        futures = [
            pool.submit(worker[0].finish_episode)
            for pool, worker in zip(self.pools, self.workers, strict=True)
        ]
        for future in futures:
            future.result(timeout=120)
        if self.profile:
            self.metrics["encoder_workers"] = [
                worker[0].timings for worker in self.workers
            ]
        self.metrics["drain_seconds"] += time.perf_counter() - stamp
        return [worker[0] for worker in self.workers]

    def close(self):
        if self.closed:
            return
        self.closed = True
        for pool in self.pools:
            pool.shutdown(wait=True, cancel_futures=True)
        self.sim.torch_stream.synchronize()
        for encoder, stream in self.workers:
            stream.synchronize()
            encoder.close()
