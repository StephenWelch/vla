"""Bounded, ordered rollout video recording; never retain an entire episode."""

import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from queue import Full, Queue

import numpy as np


class RolloutRecorder:
    def __init__(self, directory, limit, fps=50):
        self.directory, self.limit, self.fps = directory, limit, fps
        self.paths, self.counts = [], []
        self.selected = {}
        self.queue = Queue(maxsize=4)
        self.peak_queue_frames = 0
        self.backpressure_seconds = 0.0
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-video")
        self.future = self.pool.submit(self._write)
        self.closed = False

    def _put(self, item):
        start = time.perf_counter()
        while True:
            if self.future.done():
                self.future.result()  # Propagate the original encoder exception.
                raise RuntimeError("Rollout recorder stopped unexpectedly")
            try:
                self.queue.put(item, timeout=0.1)
                self.peak_queue_frames = max(self.peak_queue_frames, self.queue.qsize())
                self.backpressure_seconds += time.perf_counter() - start
                return
            except Full:
                if time.perf_counter() - start > 120:
                    raise TimeoutError("Rollout video encoder stalled")

    def reset(self, worlds):
        self._put(("reset", None, None))
        self.selected = {}
        for world in range(min(worlds, self.limit - len(self.paths))):
            index = len(self.paths)
            self.selected[world] = index
            self.paths.append(self.directory / f"eval_episode_{index}.mp4")
            self.counts.append(0)

    def append(self, images, active):
        for world, index in self.selected.items():
            if not active[world]:
                continue
            views = [images[name][world] for name in ("front", "wrist")]
            views = [v.cpu().numpy() if hasattr(v, "cpu") else v for v in views]
            # concatenate owns its storage even if the producer reuses its buffers.
            frame = np.concatenate(views, axis=1)
            self._put(("frame", index, frame))
            self.counts[index] += 1

    def _write(self):
        import av

        videos = {}

        def finish():
            pending = list(videos.values())
            videos.clear()
            error = None
            for container, stream, _ in pending:
                try:
                    try:
                        for packet in stream.encode():
                            container.mux(packet)
                    finally:
                        container.close()
                except BaseException as exc:  # noqa: BLE001 -- close all streams, then re-raise
                    error = error or exc
            if error is not None:
                raise error

        try:
            while True:
                kind, index, frame = self.queue.get()
                if kind in ("reset", "close"):
                    finish()
                    if kind == "close":
                        return
                    continue
                if index not in videos:
                    self.directory.mkdir(parents=True, exist_ok=True)
                    container = av.open(str(self.paths[index]), "w")
                    try:
                        stream = container.add_stream("libx264", rate=self.fps)
                        stream.width, stream.height = frame.shape[1], frame.shape[0]
                        stream.pix_fmt = "yuv420p"
                        stream.codec_context.thread_count = 1
                        stream.options = {"crf": "23", "preset": "veryfast"}
                    except BaseException:
                        container.close()
                        raise
                    videos[index] = [container, stream, 0]
                container, stream, tick = videos[index]
                image = av.VideoFrame.from_ndarray(frame, format="rgb24")
                image.pts, image.time_base = tick, Fraction(1, self.fps)
                for packet in stream.encode(image):
                    container.mux(packet)
                videos[index][2] += 1
        finally:
            finish()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._put(("close", None, None))
            self.future.result(timeout=120)
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)


def record_environment(env, recorder):
    """Use native LeRobot evaluation without its episode-sized video accumulator."""
    reset, step = env.reset, env.step

    def recorded_reset(*args, **kwargs):
        observation, info = reset(*args, **kwargs)
        recorder.reset(env.num_envs)
        recorder.append(observation["pixels"], np.ones(env.num_envs, bool))
        return observation, info

    def recorded_step(action):
        active = ~env.done.copy()
        result = step(action)
        recorder.append(result[0]["pixels"], active)
        return result

    env.reset, env.step = recorded_reset, recorded_step
