"""Bounded-memory RGB video writing using PyAV's bundled codecs."""

import av


class VideoWriter:
    def __init__(self, path, fps=50, size=(640, 480)):
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height = size
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "fast", "threads": "2"}
        self.closed = False

    def write(self, image):
        for packet in self.stream.encode(
            av.VideoFrame.from_ndarray(image, format="rgb24")
        ):
            self.container.mux(packet)

    def close(self):
        if not self.closed:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            finally:
                self.container.close()
                self.closed = True
