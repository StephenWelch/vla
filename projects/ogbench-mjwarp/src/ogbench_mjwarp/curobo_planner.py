"""Stacking pilot: batched cuRobo paths executed by physical joint actuators."""

import time

import numpy as np
import torch
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

from .curobo_model import robot_config
from .skills import SkillPlan


def resample_path(path, max_velocity, timestep=0.05):
    """Resample joint-space arc length without a mandatory tick per spline sample."""
    values = path.detach().cpu().numpy()
    distance = np.max(np.abs(np.diff(values, axis=0)), axis=1)
    length = np.concatenate(([0.0], np.cumsum(distance)))
    ticks = max(1, int(np.ceil(length[-1] / (max_velocity * timestep))))
    grid = np.linspace(0, length[-1], ticks + 1)[1:]
    result = np.column_stack([np.interp(grid, length, values[:, j]) for j in range(6)])
    return torch.as_tensor(result, device=path.device, dtype=path.dtype)


def timed_path(
    position,
    velocity,
    source_dt,
    max_velocity,
    max_acceleration,
    timestep=0.05,
    start=None,
):
    """Sample the planned position/velocity curve, slowing uniformly for actuator limits.

    Check discrete command derivatives including the start and final hold. Unlike
    arc-length resampling, this preserves the planner's relative segment timing.
    """
    values = position.detach().cpu().numpy().copy()
    derivatives = velocity.detach().cpu().numpy().copy()
    # Phases still terminate at rest. The final repeated command makes that
    # boundary explicit instead of relying on an error-triggered controller.
    derivatives[0] = derivatives[-1] = 0
    source_dt = float(source_dt)
    if source_dt <= 0 or not np.isfinite(source_dt):
        raise ValueError("cuRobo returned an invalid trajectory timestep")
    if (
        len(values) < 2
        or not np.isfinite(values).all()
        or not np.isfinite(derivatives).all()
    ):
        raise ValueError("cuRobo returned an invalid trajectory")
    duration = (len(values) - 1) * source_dt
    if start is not None:
        # Loaded joints can sit a little below their actuator targets. Changing
        # just the first knot creates an artificial sharp acceleration; taper
        # the command/measurement offset smoothly over the whole trajectory.
        u = np.linspace(0, 1, len(values))
        offset = start.detach().cpu().numpy() - values[0]
        weight = 1 - (10 * u**3 - 15 * u**4 + 6 * u**5)
        slope = -(30 * u**2 - 60 * u**3 + 30 * u**4) / duration
        values += weight[:, None] * offset
        derivatives += slope[:, None] * offset
    curve = CubicHermiteSpline(np.arange(len(values)) * source_dt, values, derivatives)
    probe = np.linspace(0, duration, 8 * (len(values) - 1) + 1)
    scale = (
        max(
            1.0,
            np.max(np.abs(curve(probe, 1))) / max_velocity,
            np.sqrt(np.max(np.abs(curve(probe, 2))) / max_acceleration),
        )
        * 1.01
    )
    for _ in range(40):
        ticks = max(1, int(np.ceil(duration * scale / timestep)))
        samples = curve(
            np.minimum(np.arange(1, ticks + 1) * timestep / scale, duration)
        )
        v = np.diff(np.vstack((values[:1], samples, samples[-1:])), axis=0) / timestep
        a = np.diff(np.vstack((np.zeros_like(v[:1]), v)), axis=0) / timestep
        ratio = max(
            np.max(np.abs(v)) / max_velocity,
            np.sqrt(np.max(np.abs(a)) / max_acceleration),
        )
        if ratio <= 1 + 1e-6:
            samples = np.vstack((samples, samples[-1:]))
            return torch.as_tensor(
                samples, device=position.device, dtype=position.dtype
            ), scale
        scale *= max(1.1, ratio * 1.01)
    raise ValueError(
        f"Could not retime cuRobo trajectory: dt={source_dt}, duration={duration}, scale={scale}, ratio={ratio}"
    )


class CuroboSkill(SkillPlan):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_index = 0
        self.phase_done = False
        self.complete = False

    def references(self, horizon):
        if self.plan is None or self.complete:
            self.build()
            self.phase_index = 0
            self.phase_done = self.complete = False
            self.phases = self.events[-1].get(
                "keyframes_before_workspace_clip",
                [
                    {
                        "name": "hold",
                        "position": self.plan[0, :3],
                        "yaw": self.plan[0, 3],
                        "gripper": self.plan[0, 4],
                        "time": 0,
                    }
                ],
            )[1:] or [
                {
                    "name": "hold",
                    "position": self.plan[0, :3],
                    "yaw": self.plan[0, 3],
                    "gripper": self.plan[0, 4],
                    "time": 0,
                }
            ]
        phase = self.phases[self.phase_index]
        target = np.array(
            [*phase["position"], phase["yaw"], phase["gripper"]], dtype=np.float32
        )
        target[:3] = np.clip(target[:3], *self.env.unwrapped._workspace_bounds)
        self.cursor = min(
            round(phase["time"] / self.env.unwrapped._control_timestep),
            len(self.plan) - 1,
        )
        return np.repeat(target[None], horizon, axis=0)

    def annotation(self):
        phase = self.phases[self.phase_index]
        return {
            "annotation/requested_action": self.requested_action,
            "annotation/skill_id": len(self.events) - 1,
            "annotation/phase_id": self.phase_index + 1,
            "annotation/route_id": self.provenance["variant_id"]
            if self.provenance
            else 0,
            "annotation/reference": np.array(
                [*phase["position"], phase["yaw"], phase["gripper"]], dtype=np.float32
            ),
        }

    def advance(self):
        self.elapsed += 1
        self.events[-1]["end_frame_exclusive"] = self.elapsed
        if self.phase_done:
            self.phase_index += 1
            self.phase_done = False
            if self.phase_index == len(self.phases):
                self.phase_index -= 1
                self.complete = True


class CuroboPlanner:
    def __init__(self, execution, config, seed, directory):
        from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg
        from curobo.types import JointState

        self.sim, self.config = execution, config.curobo
        sim, base, count = execution, execution.base, execution.worlds
        robot, self.grip_local, self.robot_hash = robot_config(
            base, directory, self.config
        )
        scene = {
            "cuboid": {
                "floor": {"dims": [4, 4, 0.1], "pose": [0, 0, -0.05, 1, 0, 0, 0]},
                **{
                    f"cube_{i}": {
                        "dims": [0.04] * 3,
                        "pose": [0.4, 0, 0.02, 1, 0, 0, 0],
                    }
                    for i in range(2)
                },
            }
        }
        cfg = MotionPlannerCfg.create(
            robot=robot,
            scene_model=[scene for _ in range(count)],
            multi_env=True,
            max_batch_size=count,
            num_ik_seeds=self.config.ik_seeds,
            num_trajopt_seeds=self.config.trajectory_seeds,
            random_seed=seed,
            position_tolerance=0.002,
            orientation_tolerance=0.03,
            optimizer_collision_activation_distance=0.001,
        )
        # The upstream 5000-point buffer wastes gigabytes at batch 32. The
        # optimizer's <=0.2s x81-point paths fit in 512 native 20Hz samples.
        cfg.trajopt_solver_config.interpolation_buffer_size = 512
        cfg.trajopt_solver_config.interpolation_dt = 0.05
        self.planner = BatchMotionPlanner(cfg)
        self.solve_attempts = 0
        solve_pose = self.planner.ik_solver.solve_pose

        def measured_solve(*args, **kwargs):
            self.solve_attempts += 1
            return solve_pose(*args, **kwargs)

        self.planner.ik_solver.solve_pose = measured_solve
        self.params = self.planner.ik_solver.kinematics.config.kinematics_config
        self.trajectory_params = (
            self.planner.trajopt_solver.kinematics.config.kinematics_config
        )
        self.grip_indices = self.params.get_sphere_index_from_link_name("gripper")
        self.attachment_indices = self.params.get_sphere_index_from_link_name(
            "attached_object"
        )
        self.params.link_spheres[:, self.attachment_indices, 3] = -100
        self.trajectory_params.link_spheres.copy_(self.params.link_spheres)
        self.paths = [None] * count
        self.keys = [None] * count
        self.cursors = np.zeros(count, dtype=int)
        self.ticks = np.zeros(count, dtype=int)
        self.phase_start = np.zeros(count, dtype=int)
        self.pickup_height = np.zeros(count)
        self.phase_records = [[] for _ in range(count)]
        self.wrist = int(base._model.body("ur5e/wrist_3_link").id)
        self.body_pos = execution.fields.get("xpos", None)
        import warp as wp

        self.body_pos = wp.to_torch(sim.data.xpos)
        self.body_rot = wp.to_torch(sim.data.xmat).reshape(count, -1, 3, 3)
        self.geom_pos = wp.to_torch(sim.data.geom_xpos)
        self.geom_rot = wp.to_torch(sim.data.geom_xmat).reshape(count, -1, 3, 3)
        current = JointState.from_position(
            sim.qpos[:, sim.arm_q], joint_names=self.planner.joint_names
        )
        fk = self.planner.ik_solver.kinematics.compute_kinematics(current)
        error = (
            (fk.tool_poses.position.reshape(count, 3) - sim.effector).abs().max().item()
        )
        if error > 1e-4:
            raise ValueError(f"cuRobo/MuJoCo tool position mismatch: {error}")
        self.fk_error = error
        quat = fk.tool_poses.quaternion.reshape(count, 4).detach().cpu().numpy()
        rotations = sim.tensor(
            Rotation.from_quat(np.column_stack((quat[:, 1:], quat[:, 0]))).as_matrix()
        )
        rotation_error = (
            (rotations - sim.site_rot[:, sim.pinch].reshape(count, 3, 3))
            .abs()
            .max()
            .item()
        )
        if rotation_error > 1e-4:
            raise ValueError(f"cuRobo/MuJoCo tool rotation mismatch: {rotation_error}")
        self.fk_rotation_error = rotation_error
        # Compile/capture this planner instance before generation is timed.
        from curobo.types import GoalToolPose

        warm_goal = GoalToolPose(
            tool_frames=self.planner.tool_frames,
            position=fk.tool_poses.position.reshape(count, 1, 1, 1, 3),
            quaternion=fk.tool_poses.quaternion.reshape(count, 1, 1, 1, 4),
        )
        with torch.enable_grad():
            self.planner.plan_pose(warm_goal, current, max_attempts=1)

    def reset(self, world, seed):
        self.paths[world] = self.keys[world] = None
        self.cursors[world] = self.ticks[world] = 0
        self.phase_records[world] = []
        self.sim.fields["joint_target_velocity"][world].zero_()
        self.params.link_spheres[world, self.attachment_indices, 3] = -100

    def update_scene(self, skills):
        from curobo.types import Pose

        sim, checker = self.sim, self.planner.scene_collision_checker
        pieces = []
        for geom, local in self.grip_local:
            values = sim.tensor(local)
            world = self.geom_pos[:, geom, None] + values[None, :, :3] @ self.geom_rot[
                :, geom
            ].transpose(-1, -2)
            centers = (world - self.body_pos[:, self.wrist, None]) @ self.body_rot[
                :, self.wrist
            ]
            pieces.append(
                torch.cat((centers, values[None, :, 3:].expand(sim.worlds, -1, -1)), -1)
            )
        self.params.link_spheres[:, self.grip_indices] = torch.cat(pieces, 1)
        for world, skill in enumerate(skills):
            if skill is None:
                continue
            phase = (
                skill.phases[skill.phase_index]["name"]
                if hasattr(skill, "phases")
                else "hold"
            )
            target = skill.objective.get("index", 0)
            carrying = phase in (
                "postpick",
                "clearance",
                "place",
                "place_start",
                "place_end",
            )
            contact = phase in (
                "pick_start",
                "pick_end",
                "postpick",
                "place_start",
                "place_end",
                "postplace",
            )
            for i, q in enumerate(sim.cube_q):
                state = sim.qpos[world, q : q + 7]
                checker.update_obstacle_pose(
                    f"cube_{i}",
                    Pose(position=state[:3][None], quaternion=state[3:][None]),
                    world,
                )
                checker.enable_obstacle(
                    f"cube_{i}", not (i == target and (contact or carrying)), world
                )
            checker.enable_obstacle("floor", not contact, world)
            if phase == "postplace":
                # Start the retreat from intentional support contact. Exact robot
                # contacts remain guarded throughout execution.
                checker.enable_obstacle(f"cube_{1 - target}", False, world)
            # Attached cube uses measured object-to-tool offset; it is never welded in physics.
            self.params.link_spheres[world, self.attachment_indices, 3] = -100
            if carrying:
                q = sim.cube_q[target]
                position = sim.qpos[world, q : q + 3]
                tool_rot = sim.site_rot[world, sim.pinch].reshape(3, 3)
                centers = sim.tensor(
                    [
                        [x, y, z]
                        for x in (-0.01, 0.01)
                        for y in (-0.01, 0.01)
                        for z in (-0.01, 0.01)
                    ]
                )
                cube_quat = sim.qpos[world, q + 3 : q + 7].detach().cpu().numpy()
                cube_rot = sim.tensor(
                    Rotation.from_quat([*cube_quat[1:], cube_quat[0]]).as_matrix()
                )
                centers = (
                    position + centers @ cube_rot.T - sim.effector[world]
                ) @ tool_rot
                self.params.link_spheres[world, self.attachment_indices, :3] = centers
                self.params.link_spheres[world, self.attachment_indices, 3] = 0.017321
                if phase in ("place_start", "place_end"):
                    # Final support contact is checked against the exact MuJoCo geometry.
                    checker.enable_obstacle(f"cube_{1 - target}", False, world)
        self.trajectory_params.link_spheres.copy_(self.params.link_spheres)

    @torch.no_grad()
    def plan(self, references, objectives, skills, active):
        from curobo.types import GoalToolPose, JointState

        started = time.perf_counter()
        sim, count = self.sim, self.sim.worlds
        targets = sim.tensor(references[:, 0])
        needed = np.zeros(count, dtype=bool)
        valid = np.ones(count, dtype=bool)
        for world in np.flatnonzero(active):
            skill = skills[world]
            key = (len(skill.events), skill.phase_index)
            if self.keys[world] != key:
                self.keys[world] = key
                self.paths[world] = None
                self.cursors[world] = self.ticks[world] = 0
                self.phase_start[world] = 0
                needed[world] = True
        calls = 0
        if needed.any():
            self.update_scene(skills)
            # Rows with cached paths hold their measured pose; fixed batch shapes stay warm.
            position = torch.where(
                sim.tensor(needed)[:, None].bool(), targets[:, :3], sim.effector
            )
            rotation = torch.stack(
                [
                    sim.tensor(
                        Rotation.from_rotvec([0, 0, float(targets[i, 3])]).as_matrix()
                    )
                    @ sim.down
                    for i in range(count)
                ]
            )
            rotation[~sim.tensor(needed).bool()] = sim.site_rot[
                ~sim.tensor(needed).bool(), sim.pinch
            ].reshape(-1, 3, 3)
            quats = Rotation.from_matrix(rotation.cpu().numpy()).as_quat()
            quat = sim.tensor(np.column_stack((quats[:, 3], quats[:, :3])))
            goal = GoalToolPose(
                tool_frames=self.planner.tool_frames,
                position=position.reshape(count, 1, 1, 1, 3),
                quaternion=quat.reshape(count, 1, 1, 1, 4),
            )
            # Until phase paths can be blended and checked jointly, preserve
            # the original rest-to-rest planning boundary convention.
            current = JointState.from_position(
                sim.qpos[:, sim.arm_q],
                joint_names=self.planner.joint_names,
            )
            # cuRobo's batched LM seed solver differentiates the pose cost.
            attempts_before = self.solve_attempts
            with torch.enable_grad():
                result = self.planner.plan_pose(
                    goal, current, max_attempts=self.config.attempts
                )
            calls = 1
            for world in np.flatnonzero(needed):
                solved = result is not None and bool(result.success[world].any())
                event = {
                    "phase": skills[world].phases[skills[world].phase_index]["name"],
                    "success": solved,
                    "max_attempts": self.config.attempts,
                    "batch_attempts": self.solve_attempts - attempts_before,
                }
                self.phase_records[world].append(event)
                if event["phase"] == "postpick":
                    cube = sim.cube_q[skills[world].objective["index"]]
                    self.pickup_height[world] = float(sim.qpos[world, cube + 2])
                skills[world].events[-1].setdefault("curobo", []).append(event)
                if not solved:
                    event["failure"] = "planning_failure"
                    valid[world] = False
                    continue
                # BatchMotionPlanner returns top-1 seeds already selected/ranked.
                path = result.js_solution.position[world, 0]
                if self.config.execution == "timed":
                    dt = result.js_solution.dt.reshape(-1)[world]
                    self.paths[world], scale = timed_path(
                        path,
                        result.js_solution.velocity[world, 0],
                        dt,
                        self.config.max_velocity,
                        self.config.max_acceleration,
                        sim.base._control_timestep,
                        start=sim.ctrl[world, sim.arm_act],
                    )
                    event["source_dt"] = float(dt)
                    event["time_scale"] = scale
                else:
                    self.paths[world] = resample_path(path, self.config.max_velocity)
                event["execution"] = self.config.execution
                event["control_ticks"] = len(self.paths[world])
        action = torch.cat((sim.qpos[:, sim.arm_q].clone(), sim.grip[:, None]), 1)
        for world in np.flatnonzero(active):
            self.ticks[world] += 1
            skill, path = skills[world], self.paths[world]
            if not valid[world] or path is None:
                valid[world] = False
                continue
            cursor = min(self.cursors[world], len(path) - 1)
            action[world, :6] = path[cursor]
            action[world, 6] = targets[world, 4]
            joint_error = (sim.qpos[world, sim.arm_q] - path[cursor]).abs().max().item()
            if (
                self.config.execution == "timed"
                or joint_error <= self.config.tracking_tolerance
            ):
                self.cursors[world] += 1
            final = self.cursors[world] >= len(path)
            phase = skill.phases[skill.phase_index]["name"]
            dwell = 12 if phase in ("pick_end", "place_end") else 2
            pose_error = (sim.effector[world] - targets[world, :3]).norm().item()
            self.phase_start[world] = (
                self.phase_start[world] + 1 if final and pose_error < 0.006 else 0
            )
            if self.phase_start[world] >= dwell:
                skill.phase_done = True
                if phase == "postpick":
                    cube = sim.cube_q[skill.objective["index"]]
                    if (
                        float(sim.qpos[world, cube + 2])
                        < self.pickup_height[world] + 0.05
                    ):
                        valid[world] = False
                        self.phase_records[world][-1]["failure"] = "grasp_failed"
            if phase in ("clearance", "place", "place_start"):
                cube = sim.cube_q[skill.objective["index"]]
                if (
                    sim.qpos[world, cube : cube + 3] - sim.effector[world]
                ).norm().item() > 0.065:
                    valid[world] = False
                    self.phase_records[world][-1]["failure"] = "object_dropped"
            if self.ticks[world] > len(path) + self.config.phase_timeout:
                valid[world] = False
                self.phase_records[world][-1]["failure"] = "phase_timeout"
        torch.cuda.synchronize(sim.device)
        requested = action.cpu().numpy()
        for world in np.flatnonzero(active):
            skills[world].requested_action = requested[world].copy()
        return action, {
            "valid": valid.tolist(),
            "cost": [0.0] * count,
            "seconds": time.perf_counter() - started,
            "planning_calls": calls,
        }

    def close(self):
        self.planner.destroy()
