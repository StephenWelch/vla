"""Retain every native attempt, with terminal states and quality evidence."""

import hashlib
import json
import time
from dataclasses import asdict

import numpy as np
from ocbench.mjwarp.controllers.block import BlockMjWarpController
from vla_tools.tracking import Tracker, write_json

from .config import ACTION, COMMIT, FIELDS
from .environment import Simulation

PLAN_FIELDS = (
    "rng_counter",
    "speed_dt",
    "target_block",
    "target_pos",
    "target_yaw",
    "key_count",
    "key_time",
    "key_xyz",
    "key_quat",
    "key_grasp",
    "key_stop",
    "key_tangent",
    "num_pick_retries",
    "is_mistake",
)


def quality(healthy, finite, peak, nonpad=0.001, penetration=0.003):
    return bool(healthy and finite and peak[0] <= nonpad and peak[1] <= penetration)


def rows(root):
    return [
        json.loads(p.read_text()) for p in sorted((root / "raw").glob("episode-*.json"))
    ]


def generate(config):
    root = config.output
    root.mkdir(parents=True, exist_ok=True)
    spec = json.loads(json.dumps(asdict(config), default=str)) | {
        "upstream_commit": COMMIT,
        "action_profile": ACTION,
    }
    path = root / "collection.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError(
            "Existing collection configuration differs; choose a new output"
        )
    write_json(path, spec)
    (root / "raw").mkdir(exist_ok=True)
    tracker = Tracker(
        root, config.wandb, "collection", spec, resume=(root / "tracking.json").exists()
    )
    completed = {r["episode_id"] for r in rows(root)}
    start = time.perf_counter()
    failed = True
    try:
        for offset in range(0, config.episodes, config.worlds):
            ids = list(range(offset, min(offset + config.worlds, config.episodes)))
            if set(ids) <= completed:
                continue
            # Re-run a partially saved batch with its original world count/seeds.
            sim = Simulation([config.seed + i for i in ids], config.task)
            try:
                controller = BlockMjWarpController(
                    sim.env,
                    np.array([config.oracle_seed + i for i in ids], dtype=np.uint32),
                    config.max_steps,
                )
                controller.reset()
                states = [sim.snapshot()]
                observations, actions, success_frames, depths = [], [], [], []
                plans = [[] for _ in ids]
                previous_rng = np.full(len(ids), -1)
                healthy = np.ones(len(ids), bool)
                finite = np.ones(len(ids), bool)
                peak = np.zeros((len(ids), 2))
                overflow_flags = np.zeros(len(ids), dtype=np.int32)
                for tick in range(config.max_steps):
                    active = ~controller.done.numpy().astype(bool)
                    observations.append(sim.state())
                    targets = controller.make_targets()
                    rng = controller.rng_counter.numpy()
                    changed = active & (rng != previous_rng)
                    if changed.any():
                        values = {
                            k: getattr(controller, k).numpy() for k in PLAN_FIELDS
                        }
                        for world in np.flatnonzero(changed):
                            plans[world].append(
                                {
                                    "frame": tick,
                                    **{k: v[world].tolist() for k, v in values.items()},
                                }
                            )
                        previous_rng = rng.copy()
                    action = sim.env.ee_target_arrays_to_joint_actions_gpu(*targets)
                    actions.append(action.numpy().copy())
                    sim.env.step_joint_actions_gpu(action, controller.done)
                    snapshot = sim.snapshot()  # MUST precede update_done/parking.
                    states.append(snapshot)
                    depth = sim.depth.numpy()
                    depths.append(depth.copy())
                    peak[active] = np.maximum(peak[active], depth[active])
                    healthy &= ~active | sim.env._gpu_healthy.numpy().astype(bool)
                    physics_ok, overflow = sim.physics_status()
                    finite &= ~active | physics_ok
                    overflow_flags[active] |= overflow[active]
                    success_frames.append(sim.env._gpu_success.numpy().astype(bool))
                    controller.update_done()
                    if controller.done.numpy().all():
                        break
                lengths = controller.episode_length.numpy()
                success = controller.episode_success.numpy().astype(bool)
                arrays = {k: np.stack([s[k] for s in states]) for k in FIELDS}
                for world, episode in enumerate(ids):
                    if episode in completed:
                        continue
                    n = int(lengths[world])
                    valid = quality(
                        healthy[world],
                        finite[world],
                        peak[world],
                        config.max_nonpad_penetration,
                        config.max_penetration,
                    )
                    name = f"episode-{episode:06d}"
                    payload = {f"sim/{k}": v[: n + 1, world] for k, v in arrays.items()}
                    payload.update(
                        state=np.asarray(observations)[:n, world],
                        action=np.asarray(actions)[:n, world],
                        success=np.asarray(success_frames)[:n, world],
                        contact_depth=np.asarray(depths)[:n, world],
                    )
                    temporary = root / "raw" / f"{name}.npz.tmp"
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, **payload)
                    temporary.replace(root / "raw" / f"{name}.npz")
                    row = {
                        "episode_id": episode,
                        "env_id": config.task,
                        "task_id": 2,
                        "seed": config.seed + episode,
                        "oracle_seed": config.oracle_seed + episode,
                        "length": n,
                        "fps": 50,
                        "archive": name + ".npz",
                        "native_success": bool(success[world]),
                        "native_healthy": bool(healthy[world]),
                        "physics_valid": bool(finite[world]),
                        "overflow_flags": int(overflow_flags[world]),
                        "physical_valid": valid,
                        "peak_penetration": peak[world].tolist(),
                        "audit_thresholds": [
                            config.max_nonpad_penetration,
                            config.max_penetration,
                        ],
                        "completion": "complete",
                        "termination_reason": "native_success"
                        if success[world]
                        else "unhealthy"
                        if not healthy[world]
                        else "horizon",
                        "dataset_split": "val" if episode % 5 == 0 else "train",
                        "action_profile": ACTION,
                        "upstream_commit": COMMIT,
                        "randomization": {
                            "profile": "native-full",
                            "initial_state_fingerprint": hashlib.sha256(
                                payload["sim/qpos"][0].tobytes()
                            ).hexdigest(),
                            "distribution_source": f"ocbench@{COMMIT}/ocbench/mjwarp/primitives/cube_kernels.py",
                            "speed_dt_uniform": [0.7, 1.5],
                            "tilt_magnitude_uniform_degrees": [0, 25],
                            "retry_noise_multiplier": 0.75,
                            "plans": plans[world],
                        },
                        "stable_stack": stable_stack(sim, payload),
                    }
                    write_json(root / "raw" / f"{name}.json", row)
                completed.update(ids)
                summary = summarize(rows(root)) | {
                    "seconds_this_invocation": time.perf_counter() - start
                }
                write_json(root / "status.json", {"status": "generating", **summary})
                tracker.log({"collection": summary})
            finally:
                sim.close()
        result = summarize(rows(root))
        write_json(root / "status.json", {"status": "generated", **result})
        failed = False
        return result
    except BaseException as error:
        write_json(
            root / "status.json",
            {
                "status": "failed_or_interrupted",
                "error": type(error).__name__,
                **summarize(rows(root)),
            },
        )
        raise
    finally:
        tracker.finish(failed)


def summarize(records):
    result = {
        "attempts": len(records),
        "native_successes": sum(r["native_success"] for r in records),
        "audited_successes": sum(
            r["physical_valid"] and r["native_success"] for r in records
        ),
        "valid_failures": sum(
            r["physical_valid"] and not r["native_success"] for r in records
        ),
        "invalid": sum(not r["physical_valid"] for r in records),
    }
    for label, subset in (
        ("all", records),
        (
            "successes",
            [r for r in records if r["physical_valid"] and r["native_success"]],
        ),
        (
            "failures",
            [r for r in records if r["physical_valid"] and not r["native_success"]],
        ),
    ):
        plans = [p for r in subset for p in r["randomization"]["plans"]]
        speeds = [p["speed_dt"] for p in plans]
        result[f"diversity/{label}"] = {
            "plans": len(plans),
            "retry_plans": sum(p["num_pick_retries"] > 0 for p in plans),
            "mistake_plans": sum(bool(p["is_mistake"]) for p in plans),
            "speed_dt_min": min(speeds, default=0),
            "speed_dt_max": max(speeds, default=0),
        }
    return result


def stable_stack(sim, payload):
    """Diagnostic only: never extend native episodes to obtain a stable hold."""
    import mujoco

    model = sim.host_model
    data = mujoco.MjData(model)
    reasons = set()
    qpos = payload["sim/qpos"]
    if not np.isfinite(qpos).all():
        return {"valid": False, "failures": ["nonfinite_state"], "window_seconds": 1}
    if len(qpos) < 51:
        reasons.add("insufficient_stability_window")
    positions = []
    geoms = [model.geom(f"object_{i}").id for i in range(2)]
    bodies = model.geom_bodyid[geoms]
    for q in qpos[-51:]:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        pos = data.xpos[bodies].copy()
        positions.append(pos)
        lower, upper = np.argsort(pos[:, 2])
        if data.qpos[sim.env._gripper_opening_joint_id] / 0.8 > 0.15:
            reasons.add("gripper_not_released")
        if (
            np.linalg.norm(pos[0, :2] - pos[1, :2]) > 0.012
            or abs(pos[upper, 2] - pos[lower, 2] - 0.04) > 0.004
        ):
            reasons.add("stack_alignment")
        if np.min(data.xmat[bodies].reshape(2, 3, 3)[:, 2, 2]) < np.cos(np.deg2rad(10)):
            reasons.add("tilted_cube")
        support = False
        for contact in data.contact:
            pair = set(map(int, contact.geom))
            if (
                pair == set(geoms)
                and contact.dist <= 0.001
                and abs(contact.frame[2]) >= 0.8
            ):
                support = True
            if (
                pair.intersection(geoms)
                and contact.dist <= 0
                and any(
                    model.body(model.geom_bodyid[g]).name.startswith("ur5e/")
                    for g in pair
                    if g >= 0
                )
            ):
                reasons.add("robot_still_touching_cube")
        if not support:
            reasons.add("missing_cube_support")
    if positions and np.max(np.ptp(positions, axis=0)) > 0.003:
        reasons.add("unstable_stack")
    return {"valid": not reasons, "failures": sorted(reasons), "window_seconds": 1}
