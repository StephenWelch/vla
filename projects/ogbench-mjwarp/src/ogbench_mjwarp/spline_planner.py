"""Plan once, execute by time, and observe contact outcomes independently."""

import time
from dataclasses import asdict, replace

import mujoco
import numpy as np
import torch

from .contacts import contact_depths, contact_roles
from .curobo_planner import CuroboPlanner
from .spline import (
    checked_fit,
    grasp_samples,
    matrix_pose,
    pose_matrix,
    transfer_events,
)


class SplineSkill:
    def __init__(self, env, seed, randomization, provenance, config):
        base = env.unwrapped
        self.env, self.provenance = env, provenance
        self.elapsed = self.phase_index = 0
        self.phase_done = self.complete = False
        self.requested_action = np.zeros(7, np.float32)
        self.objective = {"kind": "cube", "index": 0}
        self.events = provenance["skills"]
        self.initial_qpos = base._data.qpos.copy()
        self.objects = [
            base._data.qpos[int(base._model.joint(f"object_joint_{i}").qposadr[0]) :][
                :7
            ].copy()
            for i in range(2)
        ]
        goals = base._data.mocap_pos[base._cube_target_mocap_ids].copy()
        self.order = np.argsort(goals[:, 2]).tolist()
        self.goals = [np.r_[p, 1, 0, 0, 0] for p in goals]
        streams = {
            name: np.random.default_rng(provenance["seeds"][name])
            for name in ("grasp", "path", "timing")
        }
        sample = grasp_samples(
            config, streams["grasp"], streams["path"], streams["timing"]
        )
        # One stratum per attempt; both transfers share its motion parameters.
        self.sample = sample
        provenance["schema_version"] = 2
        provenance["factors"]["spline"] = sample
        for name in ("oracle", "cem", "cube_grasps", "path", "timing"):
            provenance["factors"][name]["enabled"] = False
        self.phases, self.transfers = [], []
        for transfer, index in enumerate(self.order):
            events = transfer_events(
                self.objects[index], self.goals[index], index, sample, 0, transfer
            )
            self.transfers.append(
                {
                    "start": len(self.phases),
                    "end": len(self.phases) + len(events),
                    "grasp": len(self.phases)
                    + next(
                        i for i, event in enumerate(events) if event.name == "grasp"
                    ),
                }
            )
            self.phases.extend(events)
        self.observed = []
        self.had_grasp = set()
        self.measured_grasps = {}
        self.last_observed_phase = -1
        self.arrivals = None
        self.failure = None
        self.events.append(
            {
                "kind": "stack",
                "start_frame": 0,
                "expected_events": [],
                "observed_events": self.observed,
                "measured_grasps": self.measured_grasps,
                "observation_method": {
                    "grasp": "cube lifted >5cm and center within 7.5cm of pinch",
                    "placement": "cube center within 2.5cm of native goal after release",
                    "drop": "previously lifted cube separates >7.5cm during transport",
                    "pose_convention": "world xyz + quaternion wxyz; grasp transform maps hand coordinates into object frame",
                },
            }
        )

    def select_candidate(self, transfer, candidate):
        index, group = self.order[transfer], self.transfers[transfer]
        self.phases[group["start"] : group["end"]] = transfer_events(
            self.objects[index],
            self.goals[index],
            index,
            self.sample,
            candidate,
            transfer,
        )

    def planned_objects(self, phase):
        result = [obj.copy() for obj in self.objects]
        for index in self.order[: self.phases[phase].transfer_index]:
            result[index] = self.goals[index].copy()
            result[index][:2] += self.sample["placement_offset_xy"][index]
        return result

    def references(self, horizon):
        event = self.phases[self.phase_index]
        rotation = pose_matrix(event.pose)[:3, :3]
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        self.objective = {"kind": "cube", "index": event.object_index}
        return np.repeat(
            np.array([[*event.pose[:3], yaw, event.gripper]], np.float32), horizon, 0
        )

    def annotation(self):
        return {
            "annotation/requested_action": self.requested_action.copy(),
            "annotation/skill_id": 0,
            "annotation/phase_id": self.phase_index,
            "annotation/route_id": self.provenance["variant_id"],
            "annotation/reference": self.references(1)[0],
            "annotation/target_pose": self.phases[self.phase_index].pose.astype(
                np.float32
            ),
            "annotation/task_error": bool(
                any(not event["observed"] for event in self.observed)
            ),
        }

    def advance(self):
        self.elapsed += 1
        self.events[0]["end_frame_exclusive"] = self.elapsed

    def observe(self, sim, world):
        """Task errors are sticky annotations, never a planning validity flag."""
        event = self.phases[self.phase_index]
        index = event.object_index
        obj = sim.qpos[world, sim.cube_q[index] : sim.cube_q[index] + 7].cpu().numpy()
        tool = np.eye(4)
        tool[:3, 3] = sim.effector[world].cpu().numpy()
        tool[:3, :3] = sim.site_rot[world, sim.pinch].reshape(3, 3).cpu().numpy()
        separated = np.linalg.norm(obj[:3] - tool[:3, 3]) > 0.075
        # Observe completed intervals on entry to the next, preserving onset frame.
        if self.phase_index != self.last_observed_phase:
            if event.name == "transport":
                lifted = obj[2] > self.objects[index][2] + 0.05 and not separated
                self.observed.append(
                    {
                        "event": "grasp",
                        "object": index,
                        "frame": self.elapsed,
                        "observed": bool(lifted),
                        "failure": None if lifted else "grasp_failed",
                    }
                )
                if lifted:
                    self.had_grasp.add(index)
                    self.measured_grasps[str(index)] = matrix_pose(
                        np.linalg.inv(pose_matrix(obj)) @ tool
                    )
            if event.name == "retreat":
                placed = np.linalg.norm(obj[:3] - self.goals[index][:3]) < 0.025
                self.observed.append(
                    {
                        "event": "placement",
                        "object": index,
                        "frame": self.elapsed,
                        "observed": bool(placed),
                        "failure": None if placed else "placement_failed",
                    }
                )
            self.last_observed_phase = self.phase_index
        if (
            event.carrying
            and event.name not in ("lift", "release")
            and index in self.had_grasp
            and separated
        ):
            self.observed.append(
                {
                    "event": "retain_grasp",
                    "object": index,
                    "frame": self.elapsed,
                    "observed": False,
                    "failure": "object_dropped",
                }
            )
            self.had_grasp.remove(index)


class SplinePlanner(CuroboPlanner):
    def __init__(self, execution, config, seed, directory):
        self.settings, self.full_config = config.spline, config
        cheap = replace(
            config.curobo,
            ik_seeds=config.spline.ik_seeds,
            trajectory_seeds=config.spline.trajectory_seeds,
            attempts=1,
        )
        super().__init__(execution, replace(config, curobo=cheap), seed, directory)
        self.programs = [None] * execution.worlds
        self.roles = contact_roles(execution.base._model)
        self.check_data = mujoco.MjData(execution.base._model)

    def reset(self, world, seed):
        super().reset(world, seed)
        self.programs[world] = None

    def scene(self, skills, needed, phase):
        """Planning-only predicted scene; never attaches anything in execution."""
        from curobo.types import Pose

        checker, sim = self.planner.scene_collision_checker, self.sim
        for world in np.flatnonzero(needed):
            skill, event = skills[world], skills[world].phases[phase]
            completed = event.transfer_index
            for index, obj in enumerate(skill.planned_objects(phase)):
                checker.update_obstacle_pose(
                    f"cube_{index}",
                    Pose(
                        position=sim.tensor(obj[:3])[None],
                        quaternion=sim.tensor(obj[3:])[None],
                    ),
                    int(world),
                )
                excluded = index == event.object_index and (
                    event.contact or event.carrying
                )
                if event.support_contact:
                    excluded = (
                        True  # Exact support geometry is checked after spline fitting.
                    )
                checker.enable_obstacle(f"cube_{index}", not excluded, int(world))
            checker.enable_obstacle("floor", not event.contact, int(world))
            self.params.link_spheres[world, self.attachment_indices, 3] = -100
            if event.carrying:
                selected = skill.sample["selected_candidates"][completed]
                transform = np.linalg.inv(
                    pose_matrix(skill.sample["candidates"][selected]["object_to_hand"])
                )
                centers = np.array(
                    [
                        [x, y, z]
                        for x in (-0.01, 0.01)
                        for y in (-0.01, 0.01)
                        for z in (-0.01, 0.01)
                    ]
                )
                centers = centers @ transform[:3, :3].T + transform[:3, 3]
                self.params.link_spheres[world, self.attachment_indices, :3] = (
                    sim.tensor(centers)
                )
                self.params.link_spheres[world, self.attachment_indices, 3] = 0.017321
        self.trajectory_params.link_spheres.copy_(self.params.link_spheres)

    def goal(self, poses):
        from curobo.types import GoalToolPose

        poses = self.sim.tensor(poses)
        return GoalToolPose(
            tool_frames=self.planner.tool_frames,
            position=poses[:, :3].reshape(-1, 1, 1, 1, 3),
            quaternion=poses[:, 3:].reshape(-1, 1, 1, 1, 4),
        )

    def validate_curve(
        self,
        skill,
        probe,
        times,
        arrivals,
        first_phase,
        initial_objects,
        *,
        relaxed=False,
        pose_errors=None,
    ):
        """Exact MuJoCo geometry, including interpolated samples and held-object pose.

        Open jaw geometry conservatively checks the hand during transport; real
        jaw closure and all physical contacts remain checked at execution substeps.
        """
        sim, data, model = self.sim, self.check_data, self.sim.base._model
        bounds = model.actuator_ctrlrange[sim.base._arm_actuator_ids]
        if np.any(probe < bounds[:, 0] - 1e-6) or np.any(probe > bounds[:, 1] + 1e-6):
            return False, "joint_limits"
        # Reject pose-infeasible proposals before the dense collision pass.
        for i, stamp in enumerate(arrivals):
            sample = np.argmin(np.abs(times - stamp))
            if abs(times[sample] - stamp) > 1e-8:
                return False, "missing_event_probe"
            data.qpos[:] = skill.initial_qpos
            data.qpos[sim.arm_q.cpu().numpy()] = probe[sample]
            mujoco.mj_kinematics(model, data)
            event = skill.phases[first_phase + i]
            target = pose_matrix(event.pose)
            position_error = float(
                np.linalg.norm(data.site_xpos[sim.pinch] - target[:3, 3])
            )
            rotation = data.site_xmat[sim.pinch].reshape(3, 3).T @ target[:3, :3]
            rotation_error = float(
                np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
            )
            guide = relaxed and not event.anchor
            if pose_errors is not None:
                pose_errors.append(
                    {
                        "position_m": position_error,
                        "rotation_radians": rotation_error,
                        "anchor": event.anchor,
                    }
                )
            if position_error > (
                max(0.003, self.settings.guide_position_tolerance) if guide else 0.003
            ):
                return False, "event_position_error"
            if rotation_error > (
                max(0.04, self.settings.guide_rotation_tolerance) if guide else 0.04
            ):
                return False, "event_orientation_error"
        for joints, stamp in zip(probe, times, strict=True):
            event = skill.phases[
                first_phase
                + min(np.searchsorted(arrivals, stamp, side="left"), len(arrivals) - 1)
            ]
            data.qpos[:] = skill.initial_qpos
            data.qpos[sim.arm_q.cpu().numpy()] = joints
            for index, obj in enumerate(initial_objects):
                q = sim.cube_q[index]
                data.qpos[q : q + 7] = obj
            mujoco.mj_forward(model, data)
            if event.carrying:
                transform = pose_matrix(
                    skill.sample["candidates"][
                        skill.sample["selected_candidates"][event.transfer_index]
                    ]["object_to_hand"]
                )
                tool = np.eye(4)
                tool[:3, :3] = data.site_xmat[sim.pinch].reshape(3, 3)
                tool[:3, 3] = data.site_xpos[sim.pinch]
                q = sim.cube_q[event.object_index]
                data.qpos[q : q + 7] = matrix_pose(tool @ np.linalg.inv(transform))
                mujoco.mj_forward(model, data)
            depth = contact_depths(model, data, self.roles)[:2]
            if (
                depth[0] > self.full_config.max_nonpad_penetration
                or depth[1] > self.full_config.max_penetration
            ):
                return False, "spline_collision"
            # Robot self-collision and held-object/environment penetration also
            # matter; internal gripper linkage contacts are part of its mechanism.
            for contact in data.contact:
                a, b = map(int, contact.geom)
                if min(a, b) < 0 or contact.dist >= -self.full_config.max_penetration:
                    continue
                r1, r2 = self.roles[a], self.roles[b]
                if (r1 == 1 and r2 in (1, 2, 3)) or (r2 == 1 and r1 in (1, 2, 3)):
                    return False, "self_collision"
                if (
                    event.carrying
                    and model.geom(f"object_{event.object_index}").id in (a, b)
                    and r1 == r2 == 0
                ):
                    return False, "held_object_collision"
        return True, None

    def build(self, skills, needed):
        from curobo.types import JointState

        sim, count = self.sim, self.sim.worlds
        current = sim.ctrl[:, sim.arm_act].clone()
        paths = [[] for _ in range(count)]
        durations = [[] for _ in range(count)]
        good = needed.copy()
        calls = 0
        for world in np.flatnonzero(needed):
            skills[world].sample["selected_candidates"] = []
        for phase in range(len(skills[np.flatnonzero(needed)[0]].phases)):
            if not good.any():
                break
            example = skills[np.flatnonzero(good)[0]]
            transfer = example.phases[phase].transfer_index
            group = example.transfers[transfer]
            grasp_phase = group["grasp"]
            if phase == group["start"]:
                best = np.full(count, np.inf)
                chosen = np.zeros(count, dtype=int)
                # Fixed-size IK batches reuse memory; no candidate physics worlds.
                for candidate in range(
                    max(
                        len(skills[w].sample["candidates"])
                        for w in np.flatnonzero(good)
                    )
                ):
                    for world in np.flatnonzero(good):
                        skill = skills[world]
                        c = min(candidate, len(skill.sample["candidates"]) - 1)
                        skill.select_candidate(transfer, c)
                    self.scene(skills, good, grasp_phase)
                    poses = np.array(
                        [
                            skills[w].phases[grasp_phase].pose
                            if good[w]
                            else np.r_[sim.effector[w].cpu().numpy(), 0, 1, 0, 0]
                            for w in range(count)
                        ]
                    )
                    with torch.enable_grad():
                        result = self.planner.ik_solver.solve_pose(
                            self.goal(poses),
                            JointState.from_position(
                                current, joint_names=self.planner.joint_names
                            ),
                        )
                    for world in np.flatnonzero(good):
                        if bool(result.success[world].any()):
                            solution = result.js_solution.position[world].reshape(
                                -1, 6
                            )[0]
                            skill = skills[world]
                            objects = skill.planned_objects(grasp_phase)
                            feasible, _ = self.validate_curve(
                                skill,
                                solution.cpu().numpy()[None],
                                np.array([0.0]),
                                np.array([0.0]),
                                grasp_phase,
                                objects,
                            )
                            if not feasible:
                                continue
                            score = float((solution - current[world]).square().sum())
                            if self.settings.grasp_selection == "random":
                                score = float(
                                    skill.sample["candidate_priority"][
                                        transfer, candidate
                                    ]
                                )
                            if score < best[world]:
                                best[world], chosen[world] = (
                                    score,
                                    min(
                                        candidate,
                                        len(skills[world].sample["candidates"]) - 1,
                                    ),
                                )
                for world in np.flatnonzero(good):
                    skill = skills[world]
                    skill.sample["selected_candidates"].append(int(chosen[world]))
                    if not np.isfinite(best[world]):
                        skill.failure, good[world] = "grasp_ik_rejected", False
                        continue
                    skill.select_candidate(transfer, chosen[world])
            if not good.any():
                break
            self.scene(skills, good, phase)
            dwell = bool(example.phases[phase].dwell)
            if not dwell:
                poses = np.array(
                    [
                        skills[w].phases[phase].pose
                        if good[w]
                        else np.r_[sim.effector[w].cpu().numpy(), 0, 1, 0, 0]
                        for w in range(count)
                    ]
                )
                with torch.enable_grad():
                    result = self.planner.plan_pose(
                        self.goal(poses),
                        JointState.from_position(
                            current, joint_names=self.planner.joint_names
                        ),
                        max_attempts=1,
                    )
                calls += 1
            for world in np.flatnonzero(good):
                skill, event = skills[world], skills[world].phases[phase]
                if dwell:
                    path = current[world].cpu().numpy()[None].repeat(2, 0)
                    duration = event.dwell
                elif result is None or not bool(result.success[world].any()):
                    skill.failure, good[world] = "planning_failure", False
                    continue
                else:
                    path = result.js_solution.position[world, 0].cpu().numpy().copy()
                    path[0] = current[world].cpu().numpy()
                    duration = (
                        max(
                            0.1,
                            (len(path) - 1)
                            * float(result.js_solution.dt.reshape(-1)[world]),
                        )
                        * event.duration_scale
                    )
                    current[world] = sim.tensor(path[-1])
                paths[world].append(path)
                durations[world].append(duration)
        for world in np.flatnonzero(needed):
            skill = skills[world]
            self.phase_records[world] = [
                {"failure": skill.failure or "spline_rejected"}
            ]
            if not good[world]:
                continue
            command = [np.r_[paths[world][0][0], 0.0]]
            arrivals, first = [], 0
            objects = [p.copy() for p in skill.objects]
            for last, event in enumerate(skill.phases):
                if not event.stop and last != len(skill.phases) - 1:
                    continue
                group = paths[world][first : last + 1]

                def event_speed(positions, tangents, skill=skill, first=first):
                    model, data = self.sim.base._model, self.check_data
                    ratios = []
                    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                    arm_q = self.sim.arm_q.cpu().numpy()
                    arm_v = self.sim.arm_v.cpu().numpy()
                    for i, (q, tangent) in enumerate(
                        zip(positions, tangents, strict=True)
                    ):
                        data.qpos[:] = skill.initial_qpos
                        data.qpos[arm_q] = q
                        mujoco.mj_kinematics(model, data)
                        mujoco.mj_comPos(model, data)
                        mujoco.mj_jacSite(model, data, jacp, jacr, self.sim.pinch)
                        constraint = skill.phases[first + i]
                        ratios.append(
                            max(
                                np.linalg.norm(jacp[:, arm_v] @ tangent)
                                / constraint.max_linear_speed
                                if constraint.max_linear_speed is not None
                                else 0,
                                np.linalg.norm(jacr[:, arm_v] @ tangent)
                                / constraint.max_angular_speed
                                if constraint.max_angular_speed is not None
                                else 0,
                            )
                        )
                    return ratios

                def validate(
                    fitted, radius, record, skill=skill, first=first, objects=objects
                ):
                    _, ends, (probe, times), _ = fitted
                    return self.validate_curve(
                        skill,
                        probe,
                        times,
                        ends,
                        first,
                        objects,
                        relaxed=radius > 0,
                        pose_errors=record.setdefault("event_pose_errors", []),
                    )

                fitted, attempts = checked_fit(
                    group,
                    durations[world][first : last + 1],
                    validate,
                    max_velocity=self.config.max_velocity,
                    max_acceleration=self.config.max_acceleration,
                    max_jerk=self.settings.max_jerk,
                    relaxation=self.settings.waypoint_relaxation,
                    anchors=[e.anchor for e in skill.phases[first : last + 1]],
                    retiming=self.settings.retiming,
                    execution_speed=skill.sample["execution_speed"],
                    event_speed=event_speed
                    if self.settings.retiming == "local"
                    else None,
                )
                skill.events[0].setdefault("smoothing", []).append(
                    {
                        "first_event": first,
                        "last_event": last,
                        "attempts": attempts,
                    }
                )
                if fitted is None:
                    skill.failure = attempts[-1]["reason"] or "spline_fit_rejected"
                    break
                samples, ends, _, scale = fitted
                start_time = (len(command) - 1) * 0.05
                arrivals.extend((start_time + ends).tolist())
                grips = np.full(len(samples), event.gripper)
                if event.dwell:
                    u = np.linspace(0, 1, len(samples))
                    weight = 10 * u**3 - 15 * u**4 + 6 * u**5
                    grips = command[-1][6] + weight * (event.gripper - command[-1][6])
                command.extend(np.column_stack((samples[1:], grips[1:])))
                for k, realized in enumerate(ends):
                    skill.events[0]["expected_events"].append(
                        asdict(skill.phases[first + k])
                        | {
                            "arrival_seconds": start_time + float(realized),
                            "time_scale": scale,
                            "segment_time_scale": float(
                                (realized - (ends[k - 1] if k else 0))
                                / durations[world][first + k]
                            ),
                        }
                    )
                if event.dwell and event.gripper == 0:
                    objects[event.object_index] = skill.goals[event.object_index].copy()
                    objects[event.object_index][:2] += skill.sample[
                        "placement_offset_xy"
                    ][event.object_index]
                first = last + 1
            if skill.failure:
                self.phase_records[world][-1]["failure"] = skill.failure
                continue
            skill.arrivals = np.asarray(arrivals)
            values = np.asarray(command)[:, :6]
            padded = np.vstack(
                (values[:1].repeat(3, 0), values, values[-1:].repeat(3, 0))
            )
            limits = (
                self.config.max_velocity,
                self.config.max_acceleration,
                self.settings.max_jerk,
            )
            if any(
                np.max(np.abs(np.diff(padded, n=n, axis=0))) / 0.05**n
                > limit * (1 + 1e-5)
                for n, limit in enumerate(limits, 1)
            ):
                skill.failure = "program_derivative_limits"
                self.phase_records[world][-1]["failure"] = skill.failure
                continue
            skill.events[0]["planned_frames"] = len(command) - 1
            self.programs[world] = sim.tensor(np.asarray(command[1:]))
        return calls

    @torch.no_grad()
    def plan(self, references, objectives, skills, active):
        started = time.perf_counter()
        needed = np.array(
            [
                bool(
                    active[w] and self.programs[w] is None and skills[w].failure is None
                )
                for w in range(self.sim.worlds)
            ]
        )
        calls = self.build(skills, needed) if needed.any() else 0
        action = torch.cat(
            (self.sim.ctrl[:, self.sim.arm_act].clone(), self.sim.grip[:, None]), 1
        )
        valid = np.ones(self.sim.worlds, dtype=bool)
        for world in np.flatnonzero(active):
            skill, program = skills[world], self.programs[world]
            if program is None:
                valid[world] = False
            else:
                cursor = min(skill.elapsed, len(program) - 1)
                action[world] = program[cursor]
                skill.phase_index = min(
                    int(
                        np.searchsorted(
                            skill.arrivals, skill.elapsed * 0.05, side="right"
                        )
                    ),
                    len(skill.phases) - 1,
                )
                skill.phase_done = skill.complete = skill.elapsed + 1 >= len(program)
                skill.observe(self.sim, world)
            skill.requested_action = action[world].cpu().numpy().copy()
        return action, {
            "valid": valid.tolist(),
            "cost": [0.0] * self.sim.worlds,
            "seconds": time.perf_counter() - started,
            "planning_calls": calls,
        }
