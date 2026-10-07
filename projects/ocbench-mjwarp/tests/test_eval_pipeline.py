from types import SimpleNamespace

import av
import numpy as np
import pytest
from ocbench_mjwarp.eval_video import RolloutRecorder, record_environment
from ocbench_mjwarp.rollout import resolve_backend


def test_streaming_video_order_reset_and_bounded_queue(tmp_path):
    recorder = RolloutRecorder(tmp_path, 3)
    images = {k: np.zeros((2, 16, 16, 3), np.uint8) for k in ("front", "wrist")}
    try:
        recorder.reset(2)
        for t in range(2501):
            images["front"][:] = t % 200
            images["wrist"][:] = 230
            recorder.append(images, [True, t < 7])
        recorder.reset(1)
        recorder.append(images, [True])
    finally:
        recorder.close()
    recorder.close()
    assert recorder.peak_queue_frames <= 4
    assert recorder.counts == [2501, 7, 1]
    for path, count in zip(recorder.paths, recorder.counts, strict=True):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            assert stream.average_rate == 50
            frames = list(container.decode(stream))
        assert len(frames) == count
        assert [float(f.pts * f.time_base) for f in frames] == pytest.approx(
            np.arange(count) / 50
        )
        rgb = frames[0].to_ndarray(format="rgb24")
        assert rgb.shape == (16, 32, 3)
        assert rgb[:, 16:].mean() == pytest.approx(230, abs=3)


def test_video_worker_error_is_propagated(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("injected disk failure")

    monkeypatch.setattr(av, "open", fail)
    recorder = RolloutRecorder(tmp_path, 1)
    recorder.reset(1)
    images = {k: np.zeros((1, 16, 16, 3), np.uint8) for k in ("front", "wrist")}
    recorder.append(images, [True])
    with pytest.raises(OSError, match="disk failure"):
        recorder.close()


def test_native_video_wrapper_includes_initial_and_terminal_frames(tmp_path):
    class Env:
        num_envs = 1
        done = np.zeros(1, bool)

        def reset(self, **kwargs):
            self.done[:] = False
            return {"pixels": images}, {}

        def step(self, action):
            self.done[:] = True
            return {"pixels": images}, np.ones(1), self.done.copy(), ~self.done, {}

    images = {k: np.zeros((1, 16, 16, 3), np.uint8) for k in ("front", "wrist")}
    env = Env()
    recorder = RolloutRecorder(tmp_path, 1)
    record_environment(env, recorder)
    try:
        env.reset()
        env.step(None)
        env.step(None)
    finally:
        recorder.close()
    assert recorder.counts == [2]


def test_backend_is_explicit_until_promotion():
    policy = SimpleNamespace(
        config=SimpleNamespace(type="act", temporal_ensemble_coeff=None)
    )
    assert resolve_backend("chunked", policy) == "chunked"
    assert resolve_backend("auto", policy) == "lerobot"

    policy.config.temporal_ensemble_coeff = 0.1
    with pytest.raises(ValueError, match="open-loop"):
        resolve_backend("chunked", policy)
    assert resolve_backend("auto", policy) == "lerobot"


@pytest.mark.parametrize("execution_steps", [1, 2, 3])
def test_chunk_loop_processes_once_per_decision_and_masks_finished_worlds(
    monkeypatch, execution_steps
):
    from contextlib import nullcontext

    import torch
    from ocbench_mjwarp import rollout_frames
    from ocbench_mjwarp.rollout import chunked_evaluate

    calls, actions_seen, captures = [], [], []

    class Env:
        num_envs = 2
        config = SimpleNamespace(encoding=None, rendering=None)

        def reset(self, seed):
            self.sim = self
            self.tick = 0
            self.done = np.zeros(2, bool)
            self.native, self.valid = np.ones(2, bool), np.ones(2, bool)

        def state(self):
            return np.full((2, 18), self.tick, np.float32)

        def call(self, name):
            return ("stack", "stack")

        def advance(self, action):
            actions_seen.append(action.numpy())
            self.tick += 1
            self.done |= self.tick >= np.array([2, 5])
            return self.done.astype(float), self.done, ~self.done

    class Frames:
        def __init__(self, *args):
            pass

        def reset(self):
            pass

        def capture(self, active, decision=False):
            captures.append(active.copy())
            return {k: np.zeros((2, 16, 16, 3), np.uint8) for k in ("front", "wrist")}

        def close(self):
            pass

    class Policy:
        config = SimpleNamespace(n_action_steps=execution_steps)
        training = True

        def eval(self):
            self.training = False

        def train(self, value):
            self.training = value

        def reset(self):
            pass

        def predict_action_chunk(self, observation):
            anchor = observation["observation.state"][0, 0]
            return (anchor + torch.arange(3)[None, :, None]).expand(2, -1, 7)

    def preprocess(observation):
        calls.append(observation["observation.state"][0, 0].item())
        return observation

    monkeypatch.setattr(rollout_frames, "RolloutFrames", Frames)
    config = SimpleNamespace(
        episodes=3, seed=100, render_batch_frames=1, observation_decoder="pyav"
    )
    profile = SimpleNamespace(stage=lambda name: nullcontext())
    policy = Policy()
    result = chunked_evaluate(
        config,
        Env(),
        policy,
        lambda x: x,
        lambda x: x,
        preprocess,
        lambda x: x * 3,
        None,
        profile,
    )
    assert calls == list(range(0, 5, execution_steps)) * 2
    assert [a[0, 0] for a in actions_seen] == list(np.arange(5) * 3) * 2
    assert [row["sum_reward"] for row in result["per_episode"]] == [1, 1, 1]
    assert [row["seed"] for row in result["per_episode"]] == [100, 101, 102]
    assert captures[3].tolist() == [False, True]
    assert policy.training


@pytest.mark.gpu
def test_device_steps_resets_and_history_match_reference():
    import torch
    from ocbench_mjwarp.device_env import DeviceEnvironment
    from ocbench_mjwarp.lerobot_env import OCBenchEnvConfig, OCBenchVectorEnv

    config = OCBenchEnvConfig(max_steps=55)
    reference = OCBenchVectorEnv(config, 2)
    fast = DeviceEnvironment(config, 2)
    saved, repeated = [], []
    actions = [np.full((2, 7), 0.01 * (t % 3 - 1), np.float32) for t in range(55)]
    try:
        for target in (saved, repeated):
            reference.reset(seed=[83002, 83003])
            reference.done[0] = True  # Exercise the reference frozen-world path.
            for action in actions:
                observation, reward, term, trunc, _ = reference.step(action)
                target.append(
                    (
                        observation["agent_pos"],
                        reward,
                        term,
                        trunc,
                        reference.sim.data.qpos.numpy().copy(),
                    )
                )
        records = reference.records[-1:]
        reference.close()
        q_tolerance = max(
            1e-6, 8 * max(np.abs(a[4] - b[4]).max() for a, b in zip(saved, repeated))
        )
        state_tolerance = max(
            1e-6, 8 * max(np.abs(a[0] - b[0]).max() for a, b in zip(saved, repeated))
        )
        for _ in range(2):
            fast.reset(seed=[83002, 83003])
            fast.done[0] = True
            for action, expected in zip(actions, saved, strict=True):
                result = fast.advance(torch.tensor(action, device="cuda"))
                for actual, value in zip(result, expected[1:4], strict=True):
                    np.testing.assert_array_equal(actual, value)
                np.testing.assert_allclose(
                    fast.sim.data.qpos.numpy(), expected[4], atol=q_tolerance, rtol=0
                )
                np.testing.assert_allclose(
                    fast.state(), expected[0], atol=state_tolerance, rtol=0
                )
                # State packing is checked independently of physics repeatability.
                np.testing.assert_allclose(
                    fast.state(), fast.sim.state(), atol=1e-7, rtol=1e-6
                )
            for key, value in records[0].items():
                if key == "peak_penetration":
                    np.testing.assert_allclose(fast.records[-1][key], value, atol=1e-6)
                else:
                    assert fast.records[-1][key] == value
    finally:
        reference.close()
        fast.close()


@pytest.mark.gpu
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_temporal_frames_are_pixel_exact_for_identical_states(batch):
    from ocbench_mjwarp.environment import Simulation
    from ocbench_mjwarp.eval_profile import EvaluationProfile
    from ocbench_mjwarp.rendering import BatchRenderer
    from ocbench_mjwarp.rollout_frames import RolloutFrames
    from ocbench_mjwarp.video_codec import LiveVideoCodec, encoder_options

    sim = Simulation([83002], audit=False)
    sim.env._use_cuda_graph = (
        False  # This test restores states without stepping physics.
    )
    sim.renderer = BatchRenderer(sim)
    reference = LiveVideoCodec(1, sim.torch_stream, encoder_options())
    profile = EvaluationProfile()
    recorded = []
    recorder = SimpleNamespace(append=lambda images, active: recorded.append(images))
    frames = RolloutFrames(sim, batch, encoder_options(), "pyav", recorder, profile)
    states = sim.snapshot()
    try:
        for episode in range(2):
            if episode:
                reference.close()
                reference = LiveVideoCodec(1, sim.torch_stream, encoder_options())
                frames.reset()
            expected = []
            recorded.clear()
            for tick in range(10):
                current = {k: v.copy() for k, v in states.items()}
                current["qpos"][:, 0] += tick * 0.01
                sim.restore(current, forward=False)
                import mujoco_warp as mjw

                mjw.step(sim.model, sim.data)
                expected.append(reference(sim.renderer.render(device=True)))
                frames.capture([True], decision=tick in (0, 5, 9))
            assert len(recorded) == len(expected)
            for actual, saved in zip(recorded, expected, strict=True):
                for camera in ("front", "wrist"):
                    np.testing.assert_array_equal(actual[camera], saved[camera])
    finally:
        frames.close()
        reference.close()
        profile.close()
        sim.close()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "absolute,per_timestep",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("image_size", [(480, 640), (240, 320)])
def test_real_act_reference_and_chunked_actions(
    tmp_path, absolute, per_timestep, image_size
):
    import json

    import torch
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from ocbench_mjwarp.config import ABSOLUTE_ACTION, ACTION
    from ocbench_mjwarp.environment import Simulation
    from ocbench_mjwarp.evaluate import EvalConfig, evaluate_task
    from ocbench_mjwarp.lerobot_env import OCBenchEnvConfig
    from vla_tools.preprocessing import (
        configure_chunk_normalization,
        resize_preprocessor,
    )

    torch.manual_seed(42)
    sim = Simulation([83002], audit=False)
    center = torch.tensor(sim.state()[0, :6])
    sim.close()
    cameras = ("observation.images.front", "observation.images.wrist")
    cfg = ACTConfig(
        device="cuda",
        chunk_size=25,
        n_action_steps=25,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (18,)),
            **{k: PolicyFeature(FeatureType.VISUAL, (3, 32, 32)) for k in cameras},
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        },
    )
    low, high = torch.full((7,), -0.01), torch.full((7,), 0.01)
    if absolute:
        low[:6], high[:6] = center - 0.01, center + 0.01
        low[6], high[6] = 0, 1
    if per_timestep:
        low = low[None].repeat(25, 1) + torch.linspace(0, 0.001, 25)[:, None]
        high = high[None].repeat(25, 1) + torch.linspace(0, 0.001, 25)[:, None]
    stats = {
        "action": {"q01": low, "q99": high},
        "observation.state": {
            "q01": torch.full((18,), -5),
            "q99": torch.full((18,), 5),
        },
        **{
            k: {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)}
            for k in cameras
        },
    }
    policy = ACTPolicy(cfg).cuda().eval()
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    resize_preprocessor(pre, (32, 32))
    configure_chunk_normalization(policy, pre, post)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "video_materialization": "async-gpu-v1",
                "encoding": {"backend": "async", "gop": 2},
                "episodes": [],
            }
        )
    )
    env = OCBenchEnvConfig(
        action_profile=ABSOLUTE_ACTION if absolute else ACTION,
        max_steps=27,
        image_size=image_size,
    )
    traces, metrics = [], []
    for trial, backend in enumerate(("lerobot", "lerobot", "chunked")):
        trace = []
        config = EvalConfig(
            checkpoint=tmp_path,
            output=tmp_path / str(trial),
            dataset=tmp_path,
            episodes=1,
            batch_size=1,
            max_steps=27,
            videos=1,
            seed=83002,
            rollout_backend=backend,
            render_batch_frames=4 if backend == "chunked" else 1,
        )
        metrics.append(evaluate_task(config, env, policy, pre, post, 2, trace=trace))
        traces.append(trace)
    assert all(len(trace) == 27 for trace in traces)
    reference, repeat, actual = [
        np.stack([r["action"] for r in trace]) for trace in traces
    ]
    # Identical reset observations must produce identical first chunks. Later
    # decisions include the simulator's independently measured repeatability.
    np.testing.assert_allclose(actual[:25], reference[:25], atol=1e-7, rtol=0)
    tolerance = max(1e-6, 8 * float(np.abs(reference - repeat).max()))
    np.testing.assert_allclose(actual, reference, atol=tolerance, rtol=0)
    for a, b in zip(traces[0], traces[2], strict=True):
        np.testing.assert_array_equal(a["done"], b["done"])
    for name in (
        "success",
        "steps",
        "task_success",
        "contact_valid",
        "physics_valid",
        "stable_stack",
    ):
        assert metrics[0]["per_episode"][0][name] == metrics[2]["per_episode"][0][name]
    for result in metrics:
        with av.open(result["video_paths"][0]) as video:
            assert video.streams.video[0].frames == 28
        assert result["performance"]["video_queue_peak"] <= 4
    assert metrics[2]["performance"]["stage_calls"]["inference"] == 2
    assert post.steps[-1].device == "cpu"  # Evaluation did not mutate saved processors.
