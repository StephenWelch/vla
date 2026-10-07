"""Chunk-aware ACT evaluation using the saved LeRobot processing pipelines."""

import time
from dataclasses import replace

import numpy as np
import torch


def device_postprocessor(post, device):
    """Preserve saved action transforms but bypass their final CPU transport step."""
    from lerobot.processor import DeviceProcessorStep

    return replace(
        post,
        steps=[
            replace(step, device=device)
            if isinstance(step, DeviceProcessorStep)
            and torch.device(step.device).type == "cpu"
            else step
            for step in post.steps
        ],
    )


def resolve_backend(requested, policy):
    supported = (
        policy.config.type == "act" and policy.config.temporal_ensemble_coeff is None
    )
    if requested == "chunked" and not supported:
        raise ValueError("Chunked evaluation requires open-loop ACT")
    if requested not in ("auto", "lerobot", "chunked"):
        raise ValueError(f"Unknown rollout backend: {requested}")
    # The explicit path is available for parity tests; promotion follows benchmarks.
    return "lerobot" if requested == "auto" else requested


def policy_observation(env, images):
    from lerobot.envs.utils import preprocess_observation

    state = env.state()
    if isinstance(images["front"], np.ndarray):
        observation = preprocess_observation({"pixels": images, "agent_pos": state})
    else:
        observation = {"observation.state": torch.from_numpy(state)}
        for name, image in images.items():
            observation[f"observation.images.{name}"] = (
                image.permute(0, 3, 1, 2).contiguous().float() / 255
            )
    observation["task"] = list(env.call("task_description"))
    return observation


def chunked_evaluate(
    config, env, policy, env_pre, env_post, pre, post, recorder, profile
):
    from .rollout_frames import RolloutFrames

    started = time.perf_counter()
    rows, frames = [], None
    was_training = policy.training
    policy.eval()
    try:
        for offset in range(0, config.episodes, env.num_envs):
            seeds = list(
                range(config.seed + offset, config.seed + offset + env.num_envs)
            )
            with profile.stage("initialization"):
                env.reset(seed=seeds)
                policy.reset()
                if recorder:
                    recorder.reset(min(env.num_envs, config.episodes - offset))
                if frames is None:
                    frames = RolloutFrames(
                        env.sim,
                        config.render_batch_frames,
                        env.config.encoding,
                        config.observation_decoder,
                        recorder,
                        profile,
                    )
                else:
                    frames.reset()
                if (
                    env.config.rendering
                    and frames.render_sim.renderer.profile != env.config.rendering
                ):
                    raise ValueError(
                        "Evaluation camera/model profile differs from dataset"
                    )
                images = frames.capture(np.ones(env.num_envs, bool), decision=True)
            rewards = np.zeros(env.num_envs)
            maxima = np.zeros(env.num_envs)
            while not env.done.all():
                with profile.stage("preprocessing"):
                    observation = pre(env_pre(policy_observation(env, images)))
                with profile.stage("inference"):
                    actions = policy.predict_action_chunk(observation)[
                        :, : policy.config.n_action_steps
                    ]
                if actions.ndim != 3 or actions.shape[1] == 0:
                    raise ValueError(
                        "Policy must return a nonempty batched action chunk"
                    )
                for tick in range(actions.shape[1]):
                    with profile.stage("action_processing"):
                        action = env_post({"action": post(actions[:, tick])})["action"]
                    active = ~env.done.copy()
                    with profile.stage("simulation"):
                        reward, _, _ = env.advance(action)
                    rewards += reward * active
                    maxima = np.maximum(maxima, reward * active)
                    boundary = tick == actions.shape[1] - 1 or env.done.all()
                    images = frames.capture(active, decision=boundary)
                    if env.done.all():
                        break
            for world in range(min(env.num_envs, config.episodes - offset)):
                rows.append(
                    {
                        "episode_ix": offset + world,
                        "seed": seeds[world],
                        "sum_reward": float(rewards[world]),
                        "max_reward": float(maxima[world]),
                        "success": bool(env.native[world] and env.valid[world]),
                    }
                )
        elapsed = time.perf_counter() - started
        return {
            "per_episode": rows,
            "aggregated": {
                "avg_sum_reward": float(np.mean([r["sum_reward"] for r in rows])),
                "avg_max_reward": float(np.mean([r["max_reward"] for r in rows])),
                "pc_success": 100 * float(np.mean([r["success"] for r in rows])),
                "eval_s": elapsed,
                "eval_ep_s": elapsed / len(rows),
            },
        }
    finally:
        try:
            if frames:
                frames.close()
        finally:
            policy.train(was_training)
