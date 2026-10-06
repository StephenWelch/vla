"""Restartable episode staging; no dataset writer is shared across GPU workers."""

import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import RandomizationConfig
from .contacts import contact_depths, contact_roles, update_contact_quality
from .environment import BatchEnvironment
from .io import episode_metadata, jsonable, versions, write_json
from .planner import SamplingMPC
from .randomization import episode_randomization, initial_state_fingerprint
from .skills import SkillPlan
from .tasks import cpu_snapshot, image_shape, make_env, reset_task, task_description


class EpisodeBuffer:
    def __init__(self, root, episode_id, metadata):
        self.root, self.episode_id, self.metadata = Path(root), episode_id, metadata
        self.frames, self.states = [], []

    def add(
        self,
        images,
        state,
        action,
        success,
        done,
        truncated,
        planner_seconds,
        annotation=None,
    ):
        self.frames.append(
            {
                **(images or {}),
                **(annotation or {}),
                "state": state,
                "action": action,
                "success": success,
                "done": done,
                "truncated": truncated,
                "planner_seconds": planner_seconds,
            }
        )

    def save(self, outcome, reason, terminal):
        archive_started = time.perf_counter()
        prefix = self.root / f"episode-{self.episode_id:06d}"
        arrays = (
            {
                key: np.asarray([frame[key] for frame in self.frames])
                for key in self.frames[0]
            }
            if self.frames
            else {}
        )
        if self.states:
            all_states = self.states + [terminal]
            arrays.update(
                {
                    f"sim/{key}": np.stack([s[key] for s in all_states])
                    for key in terminal
                }
            )
        with prefix.with_suffix(".npz.tmp").open("wb") as file:
            np.savez_compressed(file, **arrays)
        prefix.with_suffix(".npz.tmp").replace(prefix.with_suffix(".npz"))
        metadata = self.metadata | {
            "episode_id": self.episode_id,
            "length": len(self.frames),
            "outcome": outcome,
            "reason": reason,
            "archive": prefix.with_suffix(".npz").name,
            "archive_write_seconds": time.perf_counter() - archive_started,
        }
        write_json(prefix.with_suffix(".json"), metadata)
        return metadata


class ArchiveWriter:
    """Bound disk work and surface errors before acknowledging completion."""

    def __init__(self, workers=2, capacity=4, on_commit=None):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.pending = deque()
        self.capacity = capacity
        self.on_commit = on_commit
        self.wait_seconds = 0.0

    def commit(self, future):
        started = time.perf_counter()
        result = future.result()
        self.wait_seconds += time.perf_counter() - started
        if self.on_commit:
            self.on_commit(result)

    def submit(self, buffer, outcome, reason, terminal):
        while self.pending and self.pending[0].done():
            self.commit(self.pending.popleft())
        if len(self.pending) >= self.capacity:
            self.commit(self.pending.popleft())
        self.pending.append(self.pool.submit(buffer.save, outcome, reason, terminal))

    def close(self):
        self.pool.shutdown(wait=True)
        for future in self.pending:
            self.commit(future)


def generate(
    root,
    env_id,
    episodes,
    task_ids,
    seed,
    config,
    size=(480, 640),
    max_steps=None,
    progress=print,
    record_images=True,
    randomization=None,
    metrics=None,
    refill_slots=True,
    batched_cpu=True,
    settle_steps=0,
):
    started = time.perf_counter()
    timings = {
        "render_seconds": 0.0,
        "host_transfer_seconds": 0.0,
        "cpu_validation_seconds": 0.0,
        "planner_seconds": 0.0,
        "physics_seconds": 0.0,
    }
    active_ticks = total_ticks = 0
    planner_latencies, solve_latencies = [], []
    randomization = randomization or RandomizationConfig()
    joint_actions = config.joint_actions
    spline = config.backend == "spline"
    if spline:
        settle_steps = max(20, settle_steps)
    if joint_actions and (env_id != "cube-double-v0" or task_ids != [5]):
        raise ValueError("cuRobo pilot supports only cube-double-v0 task 5")
    image_shape(size)
    size = size if isinstance(size, int) else list(size)
    if episodes < 1 or not task_ids or (max_steps is not None and max_steps < 1):
        raise ValueError(
            "Need positive episode count, image size >=16, task IDs, and positive step limit"
        )
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run = {
        "format": "ogbench-rollouts-3" if joint_actions else "ogbench-rollouts-2",
        "env_id": env_id,
        "episodes": episodes,
        "task_ids": task_ids,
        "seed": seed,
        "planner": config.to_dict(),
        "randomization": {
            "schema_version": 1,
            "config": asdict(randomization),
            "seed_derivation": "SeedSequence([diversity_seed, episode_id, fixed_stream_id]); environment_seed=seed+scenario_id",
        },
        "image_size": size,
        "max_steps": max_steps,
        "record_images": record_images,
        "versions": versions(),
        "execution": {"refill_slots": refill_slots, "batched_cpu": batched_cpu},
    }
    if settle_steps:
        run["execution"]["settle_steps"] = settle_steps
    run = jsonable(run)
    run_path = root / "run.json"
    if run_path.exists():
        existing = json.loads(run_path.read_text())
        if {k: v for k, v in existing.items() if k != "rendering"} != run:
            raise ValueError(
                "Output belongs to a different run; choose a new directory"
            )
    else:
        write_json(run_path, run)
    pending = [
        i for i in range(episodes) if not (root / f"episode-{i:06d}.json").exists()
    ]
    if not pending:
        summary = root / "summary.json"
        return json.loads(summary.read_text()) if summary.exists() else summarize(root)
    first_scenario = pending[0] // randomization.variants_per_reset
    env = make_env(
        env_id, seed + first_scenario, task_ids[first_scenario % len(task_ids)], size
    )
    batch_size = min(config.episodes, len(pending))
    writer = ArchiveWriter(
        on_commit=(lambda row: metrics({"event": "episode", "episode": row}))
        if metrics
        else None
    )
    try:
        torch.cuda.reset_peak_memory_stats()
        execution = BatchEnvironment(env, batch_size, config)
        # Capture execution kernels without advancing any recorded reset state.
        saved = execution.snapshot()
        warm_action = (
            torch.cat((execution.qpos[:, execution.arm_q], execution.grip[:, None]), 1)
            if joint_actions
            else torch.zeros((batch_size, 5), device=execution.device)
        )
        execution.step(warm_action)
        execution.restore(saved)
        if record_images:
            execution.render_batch()
            run["rendering"] = execution.renderer.profile
            if (
                run_path.exists()
                and json.loads(run_path.read_text()).get("rendering", run["rendering"])
                != run["rendering"]
            ):
                raise ValueError("Recording rendering profile changed")
            write_json(run_path, run)
        roles = contact_roles(env.unwrapped._model)
        if joint_actions:
            from .actions import joint_profile
            from .curobo_planner import CuroboPlanner, CuroboSkill

            if spline:
                from .spline_planner import SplinePlanner, SplineSkill

                planner = SplinePlanner(execution, config, seed, root / "robot")
            else:
                planner = CuroboPlanner(execution, config, seed, root / "robot")
            profile = joint_profile(env.unwrapped, config.curobo)
        else:
            planner = SamplingMPC(execution, config, seed)
        initialization_seconds = time.perf_counter() - started
        free_bytes, total_bytes = torch.cuda.mem_get_info(execution.device)
        device_peak_used_bytes = total_bytes - free_bytes
        generation_started = time.perf_counter()
        limit = max_steps or env.spec.max_episode_steps
        pending = deque(pending)
        buffers, skills, snapshots = (
            [None] * batch_size,
            [None] * batch_size,
            [None] * batch_size,
        )
        ids = [-1] * batch_size
        active = np.zeros(batch_size, dtype=bool)
        elapsed = np.zeros(batch_size, dtype=int)
        settling = np.zeros(batch_size, dtype=int)

        def fill(world, episode_id):
            provenance = episode_randomization(seed, episode_id, randomization, config)
            streams = provenance["seeds"]
            task_id = task_ids[provenance["scenario_id"] % len(task_ids)]
            reset_task(
                env,
                streams["environment"],
                task_id,
                config.joint_target_noise,
                streams["joint_targets"],
            )
            snapshots[world] = cpu_snapshot(env)
            base = env.unwrapped
            provenance["factors"]["joint_targets"].update(
                actuator_ids=np.asarray(base._arm_actuator_ids).tolist(),
                actuator_bounds=base._model.actuator_ctrlrange[
                    base._arm_actuator_ids
                ].tolist(),
                applied_targets="sim/ctrl[frame+1, actuator_ids]",
                clamping="actuator limits after adding offsets when enabled",
            )
            provenance["factors"]["joint_targets"]["sample"] = snapshots[world][
                "joint_target_offset"
            ].tolist()
            provenance["initial_state"] = {
                key: snapshots[world][key]
                for key in (
                    "qpos",
                    "mocap_pos",
                    "buttons",
                    "button_goals",
                    "drawer_goal",
                    "window_goal",
                )
            }
            provenance["initial_state"]["cubes"] = []
            for index in range(getattr(base, "_num_cubes", 0)):
                address = int(base._model.joint(f"object_joint_{index}").qposadr[0])
                provenance["initial_state"]["cubes"].append(
                    {
                        "index": index,
                        "position": snapshots[world]["qpos"][
                            address : address + 3
                        ].tolist(),
                        "quaternion_wxyz": snapshots[world]["qpos"][
                            address + 3 : address + 7
                        ].tolist(),
                    }
                )
            instruction, goal = task_description(env)
            metadata = {
                "env_id": env_id,
                "task_id": task_id,
                "seed": streams["environment"],
                "randomization": provenance,
                "instruction": instruction,
                "goal": goal,
                "fps": 20,
                "image_size": size,
                "record_images": record_images,
                "rendering": run.get("rendering"),
                "versions": run["versions"],
                "planner": config.to_dict(),
                "joint_target_offset": snapshots[world]["joint_target_offset"].tolist(),
            }
            buffers[world] = EpisodeBuffer(root, episode_id, metadata)
            if joint_actions:
                metadata["action_profile"] = profile
                metadata["robot_hash"] = planner.robot_hash
            metadata["contact_quality"] = {
                "valid": True,
                "peak_nonpad_penetration": 0.0,
                "peak_penetration": 0.0,
                "max_nonpad_penetration": config.max_nonpad_penetration,
                "max_penetration": config.max_penetration,
                "scope": "GPU physics substeps/endpoints and CPU recorded states; robot/environment collision geometries only.",
            }
            skill_type = CuroboSkill if joint_actions else SkillPlan
            skills[world] = (
                SplineSkill(
                    env, streams["oracle"], randomization, provenance, config.spline
                )
                if spline
                else skill_type(env, streams["oracle"], randomization, provenance)
            )
            if spline:
                metadata["annotation_schema_version"] = 2
            ids[world] = episode_id
            elapsed[world] = 0
            settling[world] = 0
            active[world] = True
            if joint_actions:
                planner.reset(world, streams["planner"])
            else:
                planner.mean[world].zero_()
                planner.previous_action[world].zero_()
                planner.generators[world].manual_seed(streams["planner"])

        step = 0
        while pending or active.any():
            if refill_slots or not active.any():
                replaced = np.zeros(batch_size, dtype=bool)
                for world in np.flatnonzero(~active):
                    if not pending:
                        break
                    fill(int(world), pending.popleft())
                    replaced[world] = True
                for world in range(batch_size):
                    if snapshots[world] is None:
                        snapshots[world] = snapshots[0]
                if replaced.any():
                    execution.restore(
                        {
                            key: np.stack([state[key] for state in snapshots])
                            for key in snapshots[0]
                        },
                        world_mask=torch.as_tensor(replaced, device=execution.device),
                    )
            transfer_started = time.perf_counter()
            observations = execution.proprioception().cpu().numpy()
            host_before = execution.cpu_states() if batched_cpu else None
            timings["host_transfer_seconds"] += time.perf_counter() - transfer_started
            total_ticks += batch_size
            active_ticks += int(active.sum())
            elapsed[active] += 1
            # State columns 12:17 are effector xyz, yaw, and gripper opening.
            references = np.repeat(observations[:, None, 12:17], config.horizon, axis=1)
            objectives = [{"kind": "none"} for _ in range(batch_size)]
            images, before, cpu_depths = {}, {}, {}
            render_started = time.perf_counter()
            batch_images = execution.render_batch() if record_images else None
            timings["render_seconds"] += time.perf_counter() - render_started
            validation_started = time.perf_counter()
            for world in np.flatnonzero(active):
                state = (
                    {key: value[world].copy() for key, value in host_before.items()}
                    if host_before is not None
                    else None
                )
                before[world] = execution.sync_cpu(int(world), state)
                if elapsed[world] == 1:
                    buffers[world].metadata["randomization"][
                        "initial_state_fingerprint"
                    ] = initial_state_fingerprint(before[world])
                    buffers[world].metadata["randomization"]["fingerprint_method"] = (
                        "sha256_saved_sim_state_excluding_joint_target_offset"
                    )
                cpu_depths[world] = contact_depths(
                    env.unwrapped._model, env.unwrapped._data, roles
                )[:2]
                references[world] = skills[world].references(config.horizon)
                objectives[world] = skills[world].objective
                images[world] = (
                    {
                        view: pixels[world].copy()
                        for view, pixels in batch_images.items()
                    }
                    if record_images
                    else None
                )
            timings["cpu_validation_seconds"] += (
                time.perf_counter() - validation_started
            )
            action, stats = (
                planner.plan(references, objectives, skills, active)
                if joint_actions
                else planner.plan(references, objectives)
            )
            timings["planner_seconds"] += stats["seconds"]
            planner_latencies.append(stats["seconds"])
            if stats.get("planning_calls", 1):
                solve_latencies.append(stats["seconds"])
            if not joint_actions:
                action[torch.as_tensor(~active, device=execution.device)] = 0
            holding = torch.as_tensor(settling > 0, device=execution.device)
            if joint_actions:
                action[holding, :6] = execution.ctrl[holding][:, execution.arm_act]
                action[holding, 6] = 0
            else:
                action[holding] = 0
            physics_started = time.perf_counter()
            success, healthy = execution.step(action)
            torch.cuda.synchronize(execution.device)
            timings["physics_seconds"] += time.perf_counter() - physics_started
            if step % 20 == 0:
                free_bytes, total_bytes = torch.cuda.mem_get_info(execution.device)
                device_peak_used_bytes = max(
                    device_peak_used_bytes, total_bytes - free_bytes
                )
            transfer_started = time.perf_counter()
            host_after = execution.cpu_states() if batched_cpu else None
            successes, health = success.cpu().numpy(), healthy.cpu().numpy()
            actions = action.cpu().numpy()
            if joint_actions:
                actions = (
                    torch.cat(
                        (
                            execution.ctrl[:, execution.arm_act],
                            execution.ctrl[:, execution.gripper_act] / 255,
                        ),
                        1,
                    )
                    .cpu()
                    .numpy()
                )
            gpu_depths = execution.contact_depth.cpu().tolist()
            overflows = execution.overflow.cpu().numpy()
            timings["host_transfer_seconds"] += time.perf_counter() - transfer_started
            validation_started = time.perf_counter()
            for world in np.flatnonzero(active):
                invalid_plan = not stats["valid"][world]
                bad_physics = not health[world]
                quality = buffers[world].metadata["contact_quality"]
                bad_contact = not update_contact_quality(
                    quality, gpu_depths[world], cpu_depths[world]
                )
                skill = skills[world]
                skill_finished = (
                    skill.phase_done and skill.phase_index == len(skill.phases) - 1
                    if joint_actions
                    else skill.cursor + 1 >= len(skill.plan)
                )
                done = bool(
                    successes[world]
                    and not invalid_plan
                    and not bad_physics
                    and not bad_contact
                    and skill_finished
                )
                success_mismatch = False
                if done:
                    terminal = (
                        {key: value[world] for key, value in host_after.items()}
                        if host_after is not None
                        else None
                    )
                    execution.sync_cpu(int(world), terminal)
                    terminal_depths = contact_depths(
                        env.unwrapped._model, env.unwrapped._data, roles
                    )[:2]
                    bad_contact = not update_contact_quality(quality, terminal_depths)
                    env.unwrapped.post_step()
                    success_mismatch = not bool(env.unwrapped._success)
                    done = not success_mismatch and not bad_contact
                if spline:
                    # A scheduled completion is independent of native task success.
                    success_mismatch = False
                    done = bool(
                        skill.complete
                        and not (invalid_plan or bad_physics or bad_contact)
                    )
                if settle_steps and (done or settling[world]):
                    if not settling[world]:
                        buffers[world].metadata["native_success"] = bool(
                            successes[world]
                        )
                    settling[world] += 1
                    done = bool(done and settling[world] > settle_steps)
                truncated = elapsed[world] == limit and not done
                finished = (
                    done
                    or truncated
                    or invalid_plan
                    or bad_physics
                    or bad_contact
                    or success_mismatch
                )
                completed = bool(done)
                if spline and finished:
                    from .stack_audit import audit_stack

                    terminal = (
                        {key: value[world].copy() for key, value in host_after.items()}
                        if host_after is not None
                        else execution.cpu_state(int(world))
                    )
                    finite = bool(
                        np.isfinite(terminal["qpos"]).all()
                        and np.isfinite(terminal["qvel"]).all()
                    )
                    out_of_bounds = bad_physics and finite and not overflows[world]
                    truncated = bool(truncated or out_of_bounds)
                    metadata = buffers[world].metadata
                    metadata["native_success"] = bool(successes[world])
                    physical_valid = bool(
                        finite and not overflows[world] and not bad_contact
                    )
                    audit = {"valid": False, "failures": ["incomplete_attempt"]}
                    if completed:
                        states = buffers[world].states + [before[world], terminal]
                        audit = audit_stack(
                            env,
                            {
                                key: np.stack([s[key] for s in states])
                                for key in terminal
                            },
                        )
                    done = bool(completed and physical_valid and audit["valid"])
                    if completed:
                        skill.observed.append(
                            {
                                "event": "stable_stack",
                                "frame": int(elapsed[world]),
                                "observed": done,
                                "failure": None if done else audit["failures"],
                            }
                        )
                    metadata["stack_quality"] = audit
                    metadata["quality"] = {
                        "schema_version": 1,
                        "physical_valid": physical_valid,
                        "completed": completed,
                        "stable_success": done,
                        "planning_rejected": bool(invalid_plan),
                        "out_of_bounds": bool(out_of_bounds),
                    }
                    from .execution_ablation import motion_metrics

                    states = buffers[world].states + [before[world], terminal]
                    command = np.stack(
                        [s["ctrl"][env.unwrapped._arm_actuator_ids] for s in states]
                    )
                    q = execution.arm_q.cpu().numpy()
                    v = env.unwrapped._model.jnt_dofadr[env.unwrapped._arm_joint_ids]
                    metadata["motion"] = (
                        motion_metrics(
                            command,
                            np.stack([s["qpos"][q] for s in states]),
                            np.stack([s["qvel"][v] for s in states]),
                        )
                        if finite and np.isfinite(command).all()
                        else {"available": False}
                    )
                buffers[world].states.append(before[world])
                buffers[world].add(
                    images[world],
                    observations[world].copy(),
                    actions[world],
                    done,
                    finished,
                    truncated,
                    stats["seconds"],
                    skills[world].annotation(),
                )
                if not settling[world]:
                    skills[world].advance()
                if finished:
                    if done:
                        reason = "success"
                    elif success_mismatch:
                        reason = "success_mismatch"
                    elif bad_contact:
                        reason = "contact_violation"
                    elif invalid_plan:
                        reason = (
                            planner.phase_records[world][-1].get(
                                "failure", "planning_failure"
                            )
                            if joint_actions
                            else "invalid_candidates"
                        )
                    elif overflows[world]:
                        reason = "capacity_overflow"
                    elif bad_physics:
                        reason = (
                            "out_of_bounds"
                            if spline and out_of_bounds
                            else "numerical_failure"
                        )
                    elif spline and completed:
                        reason = "task_failure"
                    else:
                        reason = "timeout"
                    buffers[world].metadata["success_verified"] = done
                    writer.submit(
                        buffers[world],
                        "success" if done else "failure",
                        reason,
                        {key: value[world].copy() for key, value in host_after.items()}
                        if host_after is not None
                        else execution.cpu_state(int(world)),
                    )
                    progress(
                        f"episode {ids[world]}: {'success' if done else 'failure'} ({reason}), {len(buffers[world].frames)} frames; queued archive"
                    )
                    active[world] = False
            timings["cpu_validation_seconds"] += (
                time.perf_counter() - validation_started
            )
            # Keep unused tail slots stable; populated slots are untouched.
            if (~active).any():
                execution.restore(
                    {
                        key: np.stack([state[key] for state in snapshots])
                        for key in snapshots[0]
                    },
                    world_mask=torch.as_tensor(~active, device=execution.device),
                )
            if step % 20 == 0:
                if metrics:
                    metrics(
                        {
                            "event": "progress",
                            "step": step,
                            "planner_seconds": stats["seconds"],
                            "active_episodes": int(active.sum()),
                            "active_slot_utilization": active_ticks / total_ticks,
                            "archive_wait_seconds": writer.wait_seconds,
                            **timings,
                        }
                    )
                progress(
                    f"step {step}: MPC {stats['seconds']:.3f}s, {int(active.sum())} active episodes"
                )
            step += 1
    finally:
        try:
            writer.close()
        finally:
            if joint_actions and "planner" in locals():
                planner.close()
            env.close()
    result = summarize(root)
    result["performance"] = {
        **timings,
        "generation_seconds": time.perf_counter() - started,
        "initialization_seconds": initialization_seconds,
        "steady_generation_seconds": time.perf_counter() - generation_started,
        "archive_wait_seconds": writer.wait_seconds,
        "archive_write_seconds": sum(
            row.get("archive_write_seconds", 0) for row in episode_metadata(root)
        ),
        "active_slot_utilization": active_ticks / total_ticks if total_ticks else 0,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(execution.device),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(execution.device),
        "device_peak_used_bytes": device_peak_used_bytes,
        "planner_step_latency_p50_seconds": float(np.median(planner_latencies)),
        "planner_step_latency_p95_seconds": float(np.percentile(planner_latencies, 95)),
        "solve_latency_p50_seconds": float(np.median(solve_latencies))
        if solve_latencies
        else 0,
        "solve_calls": len(solve_latencies),
        "simulated_seconds": active_ticks / 20,
        "planning_seconds_per_simulated_second": timings["planner_seconds"]
        / (active_ticks / 20),
    }
    seconds = result["performance"]["steady_generation_seconds"]
    result["performance"]["validated_successes_per_minute"] = (
        result["validated_successes"] * 60 / seconds
    )
    result["performance"]["valid_failures_per_minute"] = (
        result["valid_failures"] * 60 / seconds
    )
    write_json(root / "summary.json", result)
    return result


def summarize(root):
    rows = episode_metadata(root)
    success = sum(row["outcome"] == "success" for row in rows)
    frames = sum(row["length"] for row in rows)
    return {
        "episodes": len(rows),
        "successes": success,
        "failures": len(rows) - success,
        "success_rate": success / len(rows) if rows else 0.0,
        "frames": frames,
        "validated_successes": sum(
            r.get("quality", {}).get("stable_success", False) for r in rows
        ),
        "valid_failures": len(episode_metadata(root, quality="valid-failure")),
        "planning_rejections": sum(
            r.get("quality", {}).get("planning_rejected", False) for r in rows
        ),
        "invalid_attempts": sum(
            r.get("quality", {}).get("physical_valid") is False for r in rows
        ),
        "strata": {
            name: {
                "attempts": len(selected),
                "validated_successes": sum(
                    r.get("quality", {}).get("stable_success", False) for r in selected
                ),
                "valid_failures": sum(
                    r.get("quality", {}).get("physical_valid") is True
                    and r.get("quality", {}).get("completed") is True
                    and r.get("quality", {}).get("stable_success") is False
                    for r in selected
                ),
            }
            for name in ("nominal", "moderate", "challenging")
            if (
                selected := [
                    r
                    for r in rows
                    if r.get("randomization", {})
                    .get("factors", {})
                    .get("spline", {})
                    .get("stratum")
                    == name
                ]
            )
        },
        "outcomes": [
            {
                "episode_id": r["episode_id"],
                "task_id": r["task_id"],
                "outcome": r["outcome"],
                "reason": r["reason"],
            }
            for r in rows
        ],
    }
