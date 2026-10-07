import json

import numpy as np
import pytest
from ocbench_mjwarp.video_codec import (
    LiveVideoCodec,
    create_encoder,
    dataset_encoding,
    encoder_options,
)


def test_dataset_encoding_requires_known_export(tmp_path):
    assert dataset_encoding(tmp_path, "none") is None
    manifest = {
        "video_materialization": "async-gpu-v1",
        "encoding": {"backend": "async", "gop": 2},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert dataset_encoding(tmp_path) == encoder_options()
    for bad in ({"backend": "cpu", "gop": 2}, {"backend": "async", "gop": 30}, {}):
        manifest["encoding"] = bad
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="async-gpu-v1"):
            dataset_encoding(tmp_path)


@pytest.mark.gpu
@pytest.mark.parametrize("image_size", [(480, 640), (240, 320)])
def test_live_codec_matches_export_without_frame_delay(image_size):
    import av
    import torch

    stream = torch.cuda.Stream()
    base = np.random.default_rng(7).integers(0, 256, (2, 2, *image_size, 4), np.uint8)
    base[..., 3] = 255
    with torch.cuda.stream(stream):
        inputs = [
            torch.tensor(np.roll(base, t * 9, axis=3), device="cuda") for t in range(8)
        ]
    stream.synchronize()
    expected = []
    for world in range(2):
        for camera in range(2):
            encoder = create_encoder(stream.cuda_stream, encoder_options(), image_size)
            decoder = av.CodecContext.create("h264", "r")
            decoder.thread_count = 1
            packets = [
                packet
                for frame in inputs
                for packet in encoder.Encode(frame[world, camera])
            ]
            packets += encoder.EndEncode()
            expected.append(
                np.stack(
                    [
                        image.to_ndarray(format="rgb24")
                        for packet in packets
                        for image in decoder.decode(av.Packet(bytes(packet["data"])))
                    ]
                )
            )
            del encoder, decoder
    expected = np.stack(expected).reshape(2, 2, 8, *image_size, 3)
    # Recreating the codec must reset references and GOP position for a new episode.
    for _ in range(2):
        codec = LiveVideoCodec(2, stream, encoder_options(), image_size=image_size)
        try:
            for t, frame in enumerate(inputs):
                actual = codec(frame)
                for camera, name in enumerate(("front", "wrist")):
                    np.testing.assert_array_equal(actual[name], expected[:, camera, t])
        finally:
            codec.close()


@pytest.mark.gpu
def test_compressed_environment_reset_and_step():
    from ocbench_mjwarp.lerobot_env import OCBenchEnvConfig, OCBenchVectorEnv

    env = OCBenchVectorEnv(OCBenchEnvConfig(encoding=encoder_options()), 1)
    try:
        first, _ = env.reset(seed=[83002])
        stepped, *_ = env.step(np.zeros((1, 7), np.float32))
        again, _ = env.reset(seed=[83002])
        for key in ("front", "wrist"):
            assert stepped["pixels"][key].shape == (1, 480, 640, 3)
            assert stepped["pixels"][key].dtype == np.uint8
            np.testing.assert_array_equal(first["pixels"][key], again["pixels"][key])
        assert env.codec is not None
    finally:
        env.close()
    assert env.codec is None


@pytest.mark.gpu
def test_nvdec_current_frame_and_buffer_ownership():
    import torch

    stream = torch.cuda.Stream()
    codec = LiveVideoCodec(1, stream, encoder_options(), decoder="nvdec")
    frames = []
    try:
        for value in (20, 80, 160, 220):
            with torch.cuda.stream(stream):
                pixels = torch.full(
                    (1, 2, 480, 640, 4), value, dtype=torch.uint8, device="cuda"
                )
                pixels[..., 3] = 255
                decoded = codec(pixels)
                frames.append(decoded)
        stream.synchronize()
        for expected, images in zip((20, 80, 160, 220), frames, strict=True):
            for image in images.values():
                assert image.is_cuda and image.shape == (1, 480, 640, 3)
                assert image.float().mean().item() == pytest.approx(expected, abs=3)
    finally:
        codec.close()


@pytest.mark.gpu
def test_low_resolution_compressed_eval_and_temporal_render():
    from ocbench_mjwarp.eval_profile import EvaluationProfile
    from ocbench_mjwarp.lerobot_env import OCBenchEnvConfig, OCBenchVectorEnv
    from ocbench_mjwarp.rollout_frames import RolloutFrames

    config = OCBenchEnvConfig(image_size=(240, 320), encoding=encoder_options())
    env = OCBenchVectorEnv(config, 1)
    frames = None
    profile = EvaluationProfile()
    try:
        first, _ = env.reset(seed=[83002])
        assert first["pixels"]["front"].shape == (1, 240, 320, 3)
        env.codec.close()
        env.codec = None
        frames = RolloutFrames(env.sim, 4, encoder_options(), "pyav", None, profile)
        images = frames.capture(np.array([True]), decision=True)
        assert images["front"].shape == (1, 240, 320, 3)
    finally:
        if frames is not None:
            frames.close()
        env.close()
        profile.close()
