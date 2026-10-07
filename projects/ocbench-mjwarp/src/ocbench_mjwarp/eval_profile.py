"""Evaluation timing and sampled process memory, without W&B metric proliferation."""

import time
from collections import defaultdict
from contextlib import contextmanager
from threading import Event, Thread


class EvaluationProfile:
    def __init__(self, detailed=False):
        self.detailed = detailed
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.peak_rss = 0
        self.peak_cuda_reserved = 0
        self.peak_cuda_allocated = 0
        self.stop = Event()
        self.thread = Thread(target=self._sample, daemon=True)
        self.thread.start()
        self.started = time.perf_counter()

    def _sample(self):
        import psutil

        process = psutil.Process()
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, process.memory_info().rss)
            self.stop.wait(0.05)

    @contextmanager
    def stage(self, name):
        import torch

        if self.detailed and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.detailed and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            self.seconds[name] += time.perf_counter() - start
            self.calls[name] += 1
            # CUDA queries stay on the calling thread, outside graph capture.
            if torch.cuda.is_initialized():
                self.peak_cuda_reserved = max(
                    self.peak_cuda_reserved, torch.cuda.memory_reserved()
                )
                self.peak_cuda_allocated = max(
                    self.peak_cuda_allocated, torch.cuda.memory_allocated()
                )

    def close(self):
        self.stop.set()
        self.thread.join()

    def wrap(self, function, name):
        def measured(*args, **kwargs):
            with self.stage(name):
                return function(*args, **kwargs)

        return measured

    def report(self, frames):
        elapsed = time.perf_counter() - self.started
        return {
            "wall_seconds": elapsed,
            "environment_steps_per_second": frames / elapsed if elapsed else 0,
            "stage_wall_seconds": dict(self.seconds),
            "stage_calls": dict(self.calls),
            "synchronized_stage_timings": self.detailed,
            "peak_process_rss_bytes": self.peak_rss,
            "peak_torch_reserved_bytes": self.peak_cuda_reserved,
            "peak_torch_allocated_bytes": self.peak_cuda_allocated,
        }
