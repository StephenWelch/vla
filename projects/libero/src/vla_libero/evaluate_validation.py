"""Evaluate ACT from matched held-out LIBERO demonstration states."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vla_tools.config import parse_args
from vla_tools.policy import latest_checkpoint, load_policy, runtime_environment
from vla_tools.tracking import Tracker, WandbConfig, write_json


@dataclass
class Config:
    experiment: Path = Path("outputs/libero-drawer-act.experiment.json")
    source_repo: str = "clip-rt/modified_libero_hdf5"
    source_revision: str = "6a6659f8ac7d580fd594173a0e3abf880c843130"
    source_file: str = (
        "libero_goal_no_noops/open_the_middle_drawer_of_the_cabinet_demo.hdf5"
    )
    source_cache: Path = Path("outputs/libero/modified-source")
    device: str = "cuda"
    seed: int = 3000
    log_wandb: bool = True


def match_episode(actions, demonstrations, tolerance=1e-6):
    """Require a unique complete action-sequence match, allowing leading removed frames."""
    matches = []
    for name, group in demonstrations.items():
        source = np.asarray(group["actions"])
        if len(source) < len(actions):
            continue
        for offset in range(len(source) - len(actions) + 1):
            error = float(
                np.max(np.abs(source[offset : offset + len(actions)] - actions))
            )
            if error <= tolerance:
                matches.append((name, offset, error))
    if len(matches) != 1:
        raise ValueError(f"Expected one full demonstration match; found {len(matches)}")
    return matches[0]


def prepare_states(record, source_path):
    import h5py
    import pyarrow.dataset as ds

    data = ds.dataset(Path(record["dataset"]["root"]) / "data", format="parquet")
    mapping, states, observations = {}, {}, {}
    with h5py.File(source_path) as source:
        for split in ("train", "val"):
            for index in record["split"][split]:
                rows = (
                    data.to_table(
                        filter=ds.field("episode_index") == index,
                        columns=["action", "observation.state", "frame_index"],
                    )
                    .sort_by("frame_index")
                    .to_pydict()
                )
                actions = np.asarray(rows["action"], dtype=np.float32)
                name, offset, error = match_episode(actions, source["data"])
                demo = source["data"][name]
                state = np.asarray(demo["states"][offset], dtype=np.float64)
                observation = np.asarray(rows["observation.state"][0], dtype=np.float32)
                original_observation = np.concatenate(
                    (demo["obs/ee_states"][offset], demo["obs/gripper_states"][offset])
                )
                observation_error = float(
                    np.max(np.abs(observation - original_observation))
                )
                if observation_error > 1e-5:
                    raise ValueError(
                        f"Episode {index} source observation differs: {observation_error}"
                    )
                mapping[index] = {
                    "episode_index": index,
                    "split": split,
                    "source_demo": name,
                    "source_frame": offset,
                    "frames": len(actions),
                    "max_action_error": error,
                    "source_observation_error": observation_error,
                    "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                }
                states[index], observations[index] = state, observation
    for index in record["split"]["val"]:
        mapping[index]["matching_train_initial_states"] = [
            train
            for train in record["split"]["train"]
            if np.allclose(states[index][1:], states[train][1:], atol=1e-6, rtol=0)
        ]
    return mapping, states, observations


def main(argv=None):
    config = parse_args(Config, argv)
    record = json.loads(config.experiment.read_text())
    run = Path(record["config"]["output"])
    output = run / "validation-episodes"
    if output.exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    cache = Path(record["config"]["cache"])
    os.environ.update(runtime_environment(cache / "hf"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = str(cache / "libero-config")
    from huggingface_hub import hf_hub_download

    source_path = hf_hub_download(
        config.source_repo,
        config.source_file,
        repo_type="dataset",
        revision=config.source_revision,
        local_dir=config.source_cache,
    )
    mapping, states, observations = prepare_states(record, source_path)
    if {mapping[index]["source_demo"] for index in record["split"]["train"]} & {
        mapping[index]["source_demo"] for index in record["split"]["val"]
    }:
        raise ValueError("Source demonstrations overlap across splits")
    report = {
        "status": "running",
        "protocol": "held-out-demonstration-reset-with-regeneration-settle",
        "source": {
            "repo_id": config.source_repo,
            "revision": config.source_revision,
            "file": config.source_file,
            "sha256": hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        },
        "dataset_revision": record["dataset"]["revision"],
        "mapping": list(mapping.values()),
        "episodes": [],
        "num_steps_wait": 10,
        "max_steps": 300,
        "seed": config.seed,
    }
    write_json(output / "report.json", report)

    import torch
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import close_envs, preprocess_observation
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed

    set_seed(config.seed)
    checkpoint = latest_checkpoint(run)
    report["checkpoint"] = str(checkpoint)
    policy, pre, post = load_policy(checkpoint, config.device)
    env_config = LiberoEnv(
        task=record["config"]["suite"],
        task_ids=[record["config"]["task_id"]],
        observation_height=256,
        observation_width=256,
        episode_length=300,
        control_mode="relative",
    )
    env_pre, env_post = make_env_pre_post_processors(
        env_cfg=env_config, policy_cfg=policy.config
    )
    envs = make_env(env_config, n_envs=1, use_async_envs=False)
    env = envs[env_config.task][env_config.task_ids[0]]
    base = env.envs[0].unwrapped
    # Regeneration retains the original states[0], but obs[0] is after ten
    # zero-motion actions. Reproduce that preparation before policy actions.
    base.num_steps_wait = 10
    failed = True
    try:
        for number, index in enumerate(record["split"]["val"]):
            base._init_states = states[index][None, :]
            base.init_state_id = 0
            raw, _ = env.reset(seed=config.seed + number)
            processed = env_pre(preprocess_observation(raw))
            restored = processed["observation.state"].cpu().numpy()[0]
            error = float(np.max(np.abs(restored - observations[index])))
            if error > 0.003:
                raise ValueError(
                    f"Restored observation for episode {index} differs by {error}"
                )
            base.init_state_id = 0
            with torch.inference_mode():
                result = eval_policy(
                    env,
                    policy,
                    env_pre,
                    env_post,
                    pre,
                    post,
                    n_episodes=1,
                    max_episodes_rendered=1,
                    videos_dir=output / "videos" / f"episode_{index}",
                    start_seed=config.seed + number,
                )
            row = {
                **mapping[index],
                **result["per_episode"][0],
                "restored_observation_error": error,
                "video_paths": result["video_paths"],
            }
            report["episodes"].append(row)
            write_json(output / "report.json", report)
            print(
                f"Validation episode {index} ({row['source_demo']}): success={row['success']}",
                flush=True,
            )
        report["aggregated"] = {
            "n_episodes": len(report["episodes"]),
            "pc_success": 100 * np.mean([row["success"] for row in report["episodes"]]),
            "avg_sum_reward": np.mean(
                [row["sum_reward"] for row in report["episodes"]]
            ),
        }
        report["status"] = "complete"
        write_json(output / "report.json", report)
        if config.log_wandb:
            tracker = Tracker(
                run, WandbConfig(**record["config"]["wandb"]), "train", resume=True
            )
            try:
                tracker.log(
                    {"val/rollout": report["aggregated"]},
                    update=record["config"]["steps"],
                    reports=[output / "report.json"],
                    tables={"val/rollout/episodes": report["episodes"]},
                    videos={
                        f"val/episode_{row['episode_index']}": row["video_paths"][0]
                        for row in report["episodes"]
                    },
                )
            finally:
                tracker.finish()
        failed = False
        print(json.dumps(report["aggregated"], indent=2), flush=True)
    finally:
        close_envs(envs)
        if failed:
            report["status"] = "failed_or_interrupted"
            write_json(output / "report.json", report)


if __name__ == "__main__":
    main()
