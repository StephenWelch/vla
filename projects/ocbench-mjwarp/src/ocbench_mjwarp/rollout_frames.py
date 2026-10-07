"""Two-slot temporal rendering with one ordered camera-codec worker."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import warp as wp

from .environment import Simulation
from .rendering import BatchRenderer
from .video_codec import LiveVideoCodec

# Preserve the live renderer's transforms. Recomputing kinematics from qpos
# advances the geometry past the last physics forward pass.
RENDER_FIELDS = ("xpos", "xquat", "subtree_com", "geom_xpos", "geom_xmat")


class RolloutFrames:
    def __init__(self, sim, batch, encoding, decoder, recorder, profile):
        self.sim, self.batch = sim, batch
        self.recorder, self.profile = recorder, profile
        self.render_sim = (
            sim
            if batch == 1
            else Simulation(
                [0] * (batch * sim.worlds),
                task=sim.task,
                audit=False,
                image_size=sim.image_size,
            )
        )
        if self.render_sim.renderer is None:
            self.render_sim.renderer = BatchRenderer(self.render_sim)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-codec")
        self.codec = None
        self.encoding, self.decoder = encoding, decoder
        self.slots = []
        try:
            with torch.cuda.stream(sim.torch_stream):
                for _ in range(2):
                    self.slots.append(
                        {
                            "states": {
                                k: torch.empty(
                                    (batch, *wp.to_torch(getattr(sim.data, k)).shape),
                                    dtype=wp.to_torch(getattr(sim.data, k)).dtype,
                                    device=str(sim.warp_device),
                                )
                                for k in RENDER_FIELDS
                            },
                            "pixels": torch.empty(
                                (batch, sim.worlds, 2, *sim.image_size, 4),
                                dtype=torch.uint8,
                                device=str(sim.warp_device),
                            ),
                            "ready": torch.cuda.Event(),
                            "future": None,
                            "active": [],
                        }
                    )
            self.reset()
        except BaseException:
            self.close()
            raise

    def reset(self):
        for slot in self.slots:
            self._reclaim(slot)
            slot["active"].clear()
        self.index = self.count = self.input_tick = 0
        self.last_tick = -1
        self.last_images = None
        self.pending = deque()

        def initialize():
            torch.cuda.set_device(self.sim.warp_device.ordinal)
            if self.codec is not None:
                self.codec.close()
            self.stream = torch.cuda.Stream(device=str(self.sim.warp_device))
            with torch.cuda.stream(self.stream):
                self.codec = (
                    LiveVideoCodec(
                        self.sim.worlds,
                        self.stream,
                        self.encoding,
                        self.decoder,
                        image_size=self.sim.image_size,
                    )
                    if self.encoding
                    else None
                )

        self.pool.submit(initialize).result(timeout=120)

    @staticmethod
    def _reclaim(slot):
        if slot["future"] is not None:
            slot["future"].result(timeout=120)
            slot["future"] = None

    def capture(self, active, *, decision=False):
        # NVENC/decoder synchronization must not overlap OCBench's first CUDA
        # graph capture. Drain startup frames until the physics graph exists.
        startup = self.sim.env._use_cuda_graph and self.sim.env._step_graph is None
        drain = decision or startup
        slot = self.slots[self.index]
        if self.count == 0:
            self._reclaim(slot)
            slot["active"].clear()
        with torch.cuda.stream(self.sim.torch_stream):
            for k in RENDER_FIELDS:
                slot["states"][k][self.count].copy_(
                    wp.to_torch(getattr(self.sim.data, k))
                )
        slot["active"].append((self.input_tick, np.asarray(active).copy()))
        self.input_tick += 1
        self.count += 1
        if self.count == self.batch or drain:
            self._submit(slot, drain)
            self.count = 0
            self.index = (self.index + 1) % len(self.slots)
        if drain:
            self._reclaim(slot)
            if self.pending or self.last_tick != self.input_tick - 1:
                raise RuntimeError("Decision observation is delayed or missing")
            if decision:
                return self.last_images
        return None

    def _submit(self, slot, decision):
        sim = self.render_sim
        with (
            self.profile.stage("rendering"),
            torch.cuda.stream(sim.torch_stream),
            wp.ScopedStream(sim.stream),
        ):
            if self.batch != 1:
                for k in RENDER_FIELDS:
                    states = slot["states"][k]
                    # Last partial temporal batch pads only rendering worlds.
                    if self.count < self.batch:
                        states[self.count :].copy_(states[self.count - 1 : self.count])
                    wp.to_torch(getattr(sim.data, k)).copy_(
                        states.reshape_as(wp.to_torch(getattr(sim.data, k)))
                    )
            pixels = sim.renderer.render(device=True)
            slot["pixels"].copy_(pixels.reshape_as(slot["pixels"]))
            slot["ready"].record(sim.torch_stream)
        slot["future"] = self.pool.submit(self._consume, slot, self.count, decision)

    def _consume(self, slot, count, decision):
        with (
            torch.cuda.device(str(self.sim.warp_device)),
            torch.cuda.stream(self.stream),
        ):
            self.stream.wait_event(slot["ready"])
            for tick in range(count):
                self.pending.append(slot["active"][tick])
                with self.profile.stage("compression"):
                    if self.codec:
                        frames = self.codec.feed(
                            slot["pixels"][tick], flush=decision and tick == count - 1
                        )
                    else:
                        rgb = slot["pixels"][tick, ..., :3].flip(-1)
                        frames = [
                            {
                                name: rgb[:, i]
                                for i, name in enumerate(("front", "wrist"))
                            }
                        ]
                for images in frames:
                    frame_tick, active = self.pending.popleft()
                    if self.recorder:
                        with self.profile.stage("recording"):
                            self.recorder.append(images, active)
                    self.last_images, self.last_tick = images, frame_tick
            # All reads and output copies complete before this slot can be reused.
            self.stream.synchronize()

    def close(self):
        try:
            for slot in self.slots:
                self._reclaim(slot)
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)
            if self.codec:
                self.codec.close()
                self.codec = None
            if self.render_sim is not self.sim:
                self.render_sim.close()
