"""GPU replay and CUDA-input video encoding used by production export."""

import time
from fractions import Fraction
from typing import Any

import av
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from lerobot.datasets.compute_stats import RunningQuantileStats

from .config import FIELDS
from .rendering import BatchRenderer

KEYS = ("observation.images.front", "observation.images.wrist")


@wp.kernel
def restore_frame(
    states: wp.array2d(dtype=Any),
    dest: wp.array(dtype=Any),
    tick: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    dest[i] = states[tick[0], i]


class DeviceReplay:
    """Upload trajectories once; map consecutive timestamps to rendering worlds.

    Worlds are ordered timestamp-major, then episode. End states are padded;
    the async consumer exports only each episode's actual frames.
    """

    def __init__(self, sim, arrays, graph=False, render_batch_frames=1):
        self.sim = sim
        self.render_batch_frames = render_batch_frames
        if sim.worlds != len(arrays) * render_batch_frames:
            raise ValueError(
                "Simulation worlds must match episodes times temporal batch"
            )
        if graph and render_batch_frames != 1:
            raise ValueError("Temporal batching does not support CUDA graph replay")
        self.length = max(len(a["action"]) for a in arrays)
        self.dest = {k: wp.to_torch(getattr(sim.data, k)) for k in FIELDS}
        with torch.cuda.stream(sim.torch_stream):
            self.states = {
                k: torch.as_tensor(
                    np.stack(
                        [
                            a[f"sim/{k}"][
                                np.minimum(
                                    np.arange(self.length + render_batch_frames - 1),
                                    len(a["action"]) - 1,
                                )
                            ]
                            for a in arrays
                        ],
                        axis=1,
                    ),
                    dtype=self.dest[k].dtype,
                    device=self.dest[k].device,
                )
                for k in FIELDS
            }
        if sim.renderer is None:
            sim.renderer = BatchRenderer(sim)
        self.graph = None
        if graph:
            self.tick = wp.zeros(1, dtype=wp.int32, device=sim.warp_device)
            self.inputs = {
                k: wp.from_torch(v.reshape(self.length, -1))
                for k, v in self.states.items()
            }
            self.outputs = {
                k: wp.from_torch(v.reshape(-1)) for k, v in self.dest.items()
            }
            self.launch()
            wp.synchronize_stream(sim.stream)
            with wp.ScopedCapture(stream=sim.stream) as capture:
                self.launch()
            self.graph = capture.graph

    def launch(self):
        with wp.ScopedStream(self.sim.stream):
            for k in FIELDS:
                wp.launch(
                    restore_frame,
                    dim=self.outputs[k].size,
                    inputs=[self.inputs[k], self.outputs[k], self.tick],
                    device=self.sim.warp_device,
                )
            mjw.kinematics(self.sim.model, self.sim.data)
            self.pixels = self.sim.renderer.render(device=True)

    def render(self, tick):
        sim = self.sim
        if self.graph is not None:
            with wp.ScopedStream(sim.stream):
                self.tick.fill_(tick)
                wp.capture_launch(self.graph, stream=sim.stream)
            return self.pixels
        with torch.cuda.stream(sim.torch_stream):
            for k in FIELDS:
                self.dest[k].copy_(
                    self.states[k][tick : tick + self.render_batch_frames].reshape_as(
                        self.dest[k]
                    )
                )
        with wp.ScopedStream(sim.stream):
            mjw.kinematics(sim.model, sim.data)
        return sim.renderer.render(device=True)


class DeviceEncoder:
    """Two CUDA-input sessions per world, with native LeRobot image statistics."""

    def __init__(
        self, root, world, sim, length, stream=None, *, write_buffer_bytes=8192, gop=2
    ):
        import PyNvVideoCodec as nvc

        self.length, self.count = length, 0
        self.results = None
        self.profile = False
        self.timings = {
            "encode_seconds": 0.0,
            "write_seconds": 0.0,
            "statistics_seconds": 0.0,
        }
        self.encoders, self.files, self.stats, self.paths = {}, {}, {}, {}
        try:
            for key in KEYS:
                folder = root / "encoded" / str(world) / key
                folder.mkdir(parents=True)
                self.paths[key] = folder / "video.h264"
                self.files[key] = self.paths[key].open(
                    "wb", buffering=write_buffer_bytes
                )
                self.stats[key] = RunningQuantileStats()
                self.encoders[key] = nvc.CreateEncoder(
                    640,
                    480,
                    "ARGB",
                    False,
                    codec="h264",
                    preset="P1",
                    rc="constqp",
                    constqp="18",
                    fps="50",
                    gop=str(gop),
                    bf="0",
                    cudastream=sim.stream.cuda_stream if stream is None else stream,
                )
        except BaseException:
            self.close()
            raise

    def feed(self, bgra, sampled_rgb):
        for camera, key in enumerate(KEYS):
            stamp = time.perf_counter() if self.profile else 0.0
            packets = self.encoders[key].Encode(bgra[camera])
            if self.profile:
                self.timings["encode_seconds"] += time.perf_counter() - stamp
                stamp = time.perf_counter()
            for packet in packets:
                self.files[key].write(packet["data"])
            if self.profile:
                self.timings["write_seconds"] += time.perf_counter() - stamp
                stamp = time.perf_counter()
            self.stats[key].update(sampled_rgb[camera].reshape(-1, 3))
            if self.profile:
                self.timings["statistics_seconds"] += time.perf_counter() - stamp
        self.count += 1

    def finish_episode(self):
        if self.results is not None:
            return self.results
        if self.count != self.length:
            raise ValueError(f"Frame mismatch: {self.count}/{self.length}")
        results = {}
        for key in KEYS:
            for packet in self.encoders[key].EndEncode():
                self.files[key].write(packet["data"])
            self.files[key].close()
            raw = self.paths[key]
            output = raw.with_suffix(".mp4")
            # B-frames are disabled; assign exact 50 Hz timestamps while remuxing.
            with (
                av.open(str(raw), format="h264") as src,
                av.open(str(output), "w") as dst,
            ):
                stream = dst.add_stream_from_template(src.streams.video[0])
                count = 0
                for packet in src.demux(video=0):
                    if not packet.size:
                        continue
                    packet.pts = packet.dts = count
                    packet.duration = 1
                    packet.time_base = Fraction(1, 50)
                    packet.stream = stream
                    dst.mux(packet)
                    count += 1
            if count != self.length:
                raise ValueError("Encoded packet count differs from trajectory")
            raw.unlink()
            results[key] = (output, self.stats[key].get_statistics())
        self.encoders.clear()
        self.results = results
        return results

    def close(self):
        self.encoders.clear()
        for f in self.files.values():
            f.close()
