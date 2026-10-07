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


class EpisodeVideoEncoder(BlockingVideoEncoder):
    """Encode one world concurrently; validate before LeRobot commits its videos."""

    def __init__(self, *args, expected_frames, previews=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_frames = expected_frames
        self.previews = previews or {}

    def finish_episode(self):
        import shutil

        import av

        results = super().finish_episode()
        for key, (path, stats) in results.items():
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                if (
                    stream.frames != self.expected_frames
                    or stream.average_rate != self.fps
                ):
                    raise RuntimeError(
                        f"Incomplete video {key}: {stream.frames}/{self.expected_frames} frames"
                    )
            if self.expected_frames >= 2 and stats is None:
                raise RuntimeError(f"Missing encoder statistics for {key}")
            if key in self.previews:
                destination = self.previews[key]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)  # Encoded bytes, never re-encode.
        return results


def save_encoded_episode(dataset, buffer, encoder):
    """Hand a concurrent encoder to pinned LeRobot's sequential episode writer.

    LeRobot consumes the encoded paths/statistics through finish_episode(), then
    moves or remuxes the packets into its dataset layout. No image decoding.
    """
    if importlib.metadata.version("lerobot") != "0.6.1":
        raise RuntimeError("Concurrent episode handoff requires LeRobot 0.6.1")
    if dataset.writer._streaming_encoder is not None:
        raise RuntimeError("Dataset already owns an encoder")
    dataset.writer._streaming_encoder = encoder
    try:
        dataset.save_episode(episode_data=buffer)
    finally:
        dataset.writer._streaming_encoder = None
        encoder.close()
