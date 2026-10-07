"""Shared export encoding and causal compression of policy camera observations."""

import json


def encoder_options(gop=2):
    # These are the actual async-gpu-v1 settings, not LeRobot's generic video metadata.
    return {
        "codec": "h264",
        "preset": "P1",
        "rc": "constqp",
        "constqp": "18",
        "fps": "50",
        "gop": str(gop),
        "bf": "0",
    }


def create_encoder(stream, options, image_size=(480, 640)):
    import PyNvVideoCodec as nvc

    return nvc.CreateEncoder(
        image_size[1], image_size[0], "ARGB", False, cudastream=stream, **options
    )


def dataset_encoding(dataset, mode="dataset"):
    if mode == "none":
        return None
    if mode != "dataset":
        raise ValueError(f"Unknown observation compression: {mode}")
    manifest = json.loads((dataset / "manifest.json").read_text())
    encoding = manifest.get("encoding", {})
    if (
        manifest.get("video_materialization") != "async-gpu-v1"
        or encoding.get("backend") != "async"
        or encoding.get("gop") != 2
    ):
        raise ValueError(
            "Dataset compression requires the async-gpu-v1 GOP-2 export; "
            "use observation_compression=none for uncompressed evaluation"
        )
    return encoder_options()


class LiveVideoCodec:
    """One persistent H.264 stream per world/camera, reset with the environment.

    EndEncode drains PyNvVideoCodec's three-frame output queue without resetting
    its reference frames. With this no-B-frame encoder, the next Encode continues
    the same I/P sequence. Require one decoded current frame on every call.
    """

    def __init__(self, worlds, stream, options, decoder="pyav", image_size=(480, 640)):
        import av

        if decoder not in ("pyav", "nvdec"):
            raise ValueError(f"Unknown observation decoder: {decoder}")
        self.image_size = tuple(image_size)
        self.decoder_backend = decoder
        self.pending = 0
        self.stream = stream
        self.encoders, self.decoders = [], []
        try:
            for _ in range(worlds * 2):
                self.encoders.append(
                    create_encoder(stream.cuda_stream, options, self.image_size)
                )
                if self.decoder_backend == "nvdec":
                    self.decoders.append(NvDecoder(stream))
                else:
                    context = av.CodecContext.create("h264", "r")
                    context.thread_count = 1
                    self.decoders.append(context)
        except BaseException:
            self.close()
            raise

    def __call__(self, bgra):
        frames = self.feed(bgra, flush=True)
        if len(frames) != 1:
            raise RuntimeError("Live compression must return exactly one current frame")
        return frames[0]

    def feed(self, bgra, *, flush=False):
        """Feed an ordered frame; draining at a decision boundary adds no frame delay."""
        import av
        import numpy as np
        import torch

        # The renderer lends its GPU output until the next render. Drain before
        # returning, so no encoder retains an outstanding read of that buffer.
        self.stream.synchronize()
        images = []
        self.pending += 1
        for frame, encoder, decoder in zip(
            bgra.flatten(0, 1), self.encoders, self.decoders, strict=True
        ):
            packets = encoder.Encode(frame)
            if flush:
                packets += encoder.EndEncode()
            if self.decoder_backend == "nvdec":
                decoded = [
                    image
                    for packet in packets
                    for image in decoder.decode(packet["data"])
                ]
            else:
                decoded = [
                    image.to_ndarray(format="rgb24")
                    for packet in packets
                    for image in decoder.decode(av.Packet(bytes(packet["data"])))
                ]
            images.append(decoded)
        count = len(images[0])
        if any(len(v) != count for v in images) or (flush and count != self.pending):
            raise RuntimeError("Camera codecs lost frame alignment")
        self.pending -= count
        frames = []
        for tick in range(count):
            values = [camera[tick] for camera in images]
            rgb = (
                torch.stack(values)
                if self.decoder_backend == "nvdec"
                else np.stack(values)
            ).reshape(-1, 2, *self.image_size, 3)
            frames.append(
                {name: rgb[:, i] for i, name in enumerate(("front", "wrist"))}
            )
        return frames

    def close(self):
        self.encoders.clear()
        self.decoders.clear()


class NvDecoder:
    """Explicit opt-in: hardware RGB conversion may differ from FFmpeg's rounding."""

    def __init__(self, stream):
        import PyNvVideoCodec as nvc

        self.decoder = nvc.CreateDecoder(
            gpuid=stream.device.index,
            codec=nvc.cudaVideoCodec.H264,
            cudastream=stream.cuda_stream,
            usedevicememory=True,
            outputColorType=nvc.OutputColorType.RGB,
            latency=nvc.DisplayDecodeLatencyType.ZERO,
        )

    def decode(self, data):
        import numpy as np
        import PyNvVideoCodec as nvc
        import torch

        buffer = np.frombuffer(data, dtype=np.uint8)
        packet = nvc.PacketData()
        packet.bsl_data, packet.bsl = buffer.ctypes.data, buffer.size
        packet.decode_flag = int(nvc.VideoPacketFlag.ENDOFPICTURE)
        # Clone while the decoder surface is valid; subsequent Decode reuses it.
        return [
            torch.from_dlpack(frame).clone() for frame in self.decoder.Decode(packet)
        ]
