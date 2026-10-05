"""LeRobot's streaming encoder with lossless queue backpressure."""

import importlib.metadata
import queue
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import StreamingVideoEncoder


class BlockingVideoEncoder(StreamingVideoEncoder):
    def feed_frame(self, video_key, image):
        if not self._episode_active:
            raise RuntimeError("No active encoding episode")
        frame = image.copy()
        deadline = time.monotonic() + 120
        while True:
            if not self._threads[video_key].is_alive():
                raise RuntimeError(f"Encoder stopped for {video_key}")
            try:
                self._frame_queues[video_key].put(frame, timeout=0.1)
                return
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Encoder stalled for {video_key}")


class StreamingDataset(LeRobotDataset):
    @staticmethod
    def _build_streaming_encoder(
        fps, rgb_encoder, depth_encoder, encoder_queue_maxsize, encoder_threads
    ):
        if importlib.metadata.version("lerobot") != "0.6.1":
            raise RuntimeError("Blocking encoder adapter requires LeRobot 0.6.1")
        return BlockingVideoEncoder(
            fps=fps,
            rgb_encoder=rgb_encoder,
            depth_encoder=depth_encoder,
            queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
        )
