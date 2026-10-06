"""Batched physics, controllers, and OGBench's non-physics task state."""

from functools import wraps

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from .contacts import contact_roles
from .ik import BatchedIK, rotation_z
from .tasks import (
    DATA_FIELDS,
    cpu_snapshot,
    puzzle_toggle,
    reset_task,
    restore_cpu,
    task_description,
)


@wp.kernel
def collect_contact_depth(
    count: wp.array(dtype=wp.int32),
    geom: wp.array(dtype=wp.vec2i),
    distance: wp.array(dtype=wp.float32),
    world: wp.array(dtype=wp.int32),
    roles: wp.array(dtype=wp.int32),
    peak: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    if i >= count[0]:
        return
    pair = geom[i]
    a, b = pair[0], pair[1]
    if a < 0 or b < 0:
        return
    r1, r2 = roles[a], roles[b]
    robot, other = 0, (-1)
    if r1 >= 1 and r1 <= 3 and (r2 == 0 or r2 == 4):
        robot = r1 + 0
        other = r2 + 0
    elif r2 >= 1 and r2 <= 3 and (r1 == 0 or r1 == 4):
        robot = r2 + 0
        other = r1 + 0
    if other < 0:
        return
    depth = wp.max(0.0, -distance[i])
    wp.atomic_max(peak, world[i], 1, depth)
    if robot != 3 and not (robot == 2 and other == 4):
        wp.atomic_max(peak, world[i], 0, depth)


def on_stream(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        caller = torch.cuda.current_stream(self.device)
        self.torch_stream.wait_stream(caller)
        with torch.cuda.stream(self.torch_stream):
            result = method(self, *args, **kwargs)
        caller.wait_stream(self.torch_stream)
        return result

    return call


class BatchEnvironment:
    def __init__(self, env, worlds, config, device="cuda:0"):
        if not torch.cuda.is_available():
            raise RuntimeError("MJWarp generation requires an NVIDIA CUDA GPU")
        self.torch_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(self.torch_stream):
            self._initialize(env, worlds, config, device)
        torch.cuda.current_stream(self.device).wait_stream(self.torch_stream)

    def _initialize(self, env, worlds, config, device):
        self.env = env
        self.base = env.unwrapped
        self.host_model = self.base._model
        self.worlds = worlds
        self.device = torch.device(device)
        self.config = config
        wp.config.quiet = True
        wp.init()
        self.warp_device = wp.get_device(str(self.device))
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        with wp.ScopedDevice(self.warp_device), wp.ScopedStream(self.stream):
            self.model = mjw.put_model(
                self.host_model, batch_sizes={"dof_damping": worlds}
            )
            self.data = mjw.make_data(
                self.host_model,
                nworld=worlds,
                nconmax=config.nconmax,
                njmax=config.njmax,
            )
        self.contact_depth = torch.zeros((worlds, 2), device=self.device)
        self.contact_peak = wp.from_torch(self.contact_depth)
        self.contact_roles = wp.array(
            contact_roles(self.host_model), dtype=wp.int32, device=self.warp_device
        )
        wp.load_module(module=__name__, device=self.warp_device)
        self.contact_inputs = [
            self.data.nacon,
            self.data.contact.geom,
            self.data.contact.dist,
            self.data.contact.worldid,
            self.contact_roles,
            self.contact_peak,
        ]
        self.fields = {
            name: wp.to_torch(getattr(self.data, name)) for name in DATA_FIELDS
        }
        self.fields["time"] = wp.to_torch(self.data.time)
        self.fields["dof_damping"] = wp.to_torch(self.model.dof_damping)
        self.qpos, self.qvel = self.fields["qpos"], self.fields["qvel"]
        self.ctrl = self.fields["ctrl"]
        self.site_pos = wp.to_torch(self.data.site_xpos)
        self.site_rot = wp.to_torch(self.data.site_xmat)
        self.overflow = wp.to_torch(self.data.overflow)
        self.arm_q = self.index(self.host_model.jnt_qposadr[self.base._arm_joint_ids])
        self.arm_v = self.index(self.host_model.jnt_dofadr[self.base._arm_joint_ids])
        self.arm_act = self.index(self.base._arm_actuator_ids)
        self.joint_target_bounds = self.tensor(
            self.host_model.actuator_ctrlrange[self.base._arm_actuator_ids].T
        )
        self.gripper_act = self.index(self.base._gripper_actuator_ids)
        self.gripper_q = int(
            self.host_model.jnt_qposadr[self.base._gripper_opening_joint_id]
        )
        self.gripper_v = int(
            self.host_model.jnt_dofadr[self.base._gripper_opening_joint_id]
        )
        self.pinch = self.base._pinch_site_id
        self.ik = BatchedIK(self.base._ik._model, self.device)
        self.pa_pos = self.tensor(self.base._T_pa.translation())
        self.pa_rot = self.tensor(self.base._T_pa.rotation().as_matrix())
        self.down = self.tensor(self.base._effector_down_rotation.as_matrix())
        self.bounds = self.tensor(self.base._workspace_bounds)
        self.scale = self.tensor(self.base.action_high)
        self.cube_q = [
            int(self.host_model.joint(f"object_joint_{i}").qposadr[0])
            for i in range(getattr(self.base, "_num_cubes", 0))
        ]
        self.cube_mocap = self.index(getattr(self.base, "_cube_target_mocap_ids", []))
        self.num_buttons = getattr(self.base, "_num_buttons", 0)
        self.button_q = self.index(
            [
                self.host_model.joint(f"buttonbox_joint_{i}").qposadr[0]
                for i in range(self.num_buttons)
            ]
        )
        self.button_sites = self.index(getattr(self.base, "_button_site_ids", []))
        self.fields["buttons"] = torch.zeros(
            (worlds, self.num_buttons), dtype=torch.int32, device=self.device
        )
        self.fields["button_goals"] = torch.zeros_like(self.fields["buttons"])
        self.fields["drawer_goal"] = torch.zeros(worlds, device=self.device)
        self.fields["window_goal"] = torch.zeros(worlds, device=self.device)
        self.fields["joint_target_offset"] = torch.zeros(
            (worlds, 6), device=self.device
        )
        self.toggle = torch.eye(self.num_buttons, dtype=torch.int32, device=self.device)
        self.is_puzzle = hasattr(self.base, "_num_rows")
        if self.is_puzzle:
            self.toggle = torch.as_tensor(
                puzzle_toggle(self.base._num_rows, self.base._num_cols),
                dtype=torch.int32,
                device=self.device,
            )
        self.is_scene = hasattr(self.base, "_target_drawer_pos")
        if self.is_scene:
            self.drawer_q = int(self.host_model.joint("drawer_slide").qposadr[0])
            self.window_q = int(self.host_model.joint("window_slide").qposadr[0])
            self.drawer_v = int(self.host_model.joint("drawer_slide").dofadr[0])
            self.window_v = int(self.host_model.joint("window_slide").dofadr[0])
        self.control_graph = None
        self.physics_graph = None
        self.static_action = torch.zeros(
            (worlds, 7 if config.joint_actions else 5), device=self.device
        )
        if config.joint_actions:
            self.fields["joint_target_velocity"] = torch.zeros(
                (worlds, 6), device=self.device
            )
        self.restore(
            {
                k: np.repeat(np.asarray(v)[None], worlds, axis=0)
                for k, v in cpu_snapshot(env).items()
            }
        )

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def reset(self, seeds, task_ids=None):
        """Reset worlds from upstream OGBench; return state and task metadata."""
        if isinstance(seeds, int):
            seeds = list(range(seeds, seeds + self.worlds))
        task_ids = [1] * self.worlds if task_ids is None else task_ids
        if len(seeds) != self.worlds or len(task_ids) != self.worlds:
            raise ValueError("Provide one seed and task ID per world")
        states, metadata = [], []
        for seed, task_id in zip(seeds, task_ids, strict=True):
            reset_task(self.env, seed, task_id, self.config.joint_target_noise)
            states.append(cpu_snapshot(self.env))
            instruction, goal = task_description(self.env)
            metadata.append(
                {
                    "seed": seed,
                    "task_id": task_id,
                    "instruction": instruction,
                    "goal": goal,
                }
            )
        self.restore({key: np.stack([s[key] for s in states]) for key in states[0]})
        return self.proprioception(), metadata

    def index(self, value):
        return torch.as_tensor(np.asarray(value), device=self.device, dtype=torch.long)

    @property
    def effector(self):
        return self.site_pos[:, self.pinch]

    @property
    def yaw(self):
        matrix = self.site_rot[:, self.pinch]
        return torch.atan2(matrix[:, 1, 0], matrix[:, 0, 0])

    @property
    def grip(self):
        return (self.qpos[:, self.gripper_q] / 0.8).clamp(0, 1)

    def cube_positions(self):
        if not self.cube_q:
            return torch.empty((self.worlds, 0, 3), device=self.device)
        return torch.stack([self.qpos[:, q : q + 3] for q in self.cube_q], 1)

    @on_stream
    def forward(self):
        with wp.ScopedDevice(self.warp_device), wp.ScopedStream(self.stream):
            mjw.forward(self.model, self.data)

    @on_stream
    def snapshot(self):
        return {key: value.clone() for key, value in self.fields.items()}

    @on_stream
    def restore(self, state, indices=None, world_mask=None):
        if indices is not None and world_mask is not None:
            raise ValueError("Choose source indices or a destination world mask")
        if "joint_target_offset" not in state:
            if world_mask is None:
                self.fields["joint_target_offset"].zero_()
            else:
                self.fields["joint_target_offset"][world_mask] = 0
        for key, value in state.items():
            target = self.fields[key]
            source = torch.as_tensor(value, device=self.device, dtype=target.dtype)
            if indices is not None:
                source = source[indices]
            if world_mask is None:
                target.copy_(source)
            else:
                target[world_mask] = source[world_mask]
        if world_mask is None:
            self.overflow.zero_()
        else:
            self.overflow[world_mask] = 0
        self.forward()

    def cpu_state(self, world):
        return {
            key: value[world].detach().cpu().numpy().copy()
            for key, value in self.fields.items()
        }

    @on_stream
    def cpu_states(self):
        """One host transfer per dtype, independent of the number of worlds."""
        groups = {}
        for key, value in self.fields.items():
            groups.setdefault(value.dtype, []).append((key, value))
        result = {}
        for fields in groups.values():
            packed = torch.cat(
                [value.reshape(self.worlds, -1) for _, value in fields], 1
            )
            host = packed.detach().cpu().numpy()
            offset = 0
            for key, value in fields:
                count = value.numel() // self.worlds
                result[key] = host[:, offset : offset + count].reshape(value.shape)
                offset += count
        return result

    def sync_cpu(self, world, state=None):
        state = self.cpu_state(world) if state is None else state
        restore_cpu(self.env, state)
        return state

    def _control(self, actions):
        if self.config.joint_actions:
            target = actions[:, :6].clamp(
                self.joint_target_bounds[0], self.joint_target_bounds[1]
            )
            previous = self.ctrl[:, self.arm_act]
            dt = self.base._control_timestep
            limits = self.config.curobo
            velocity = ((target - previous) / dt).clamp(
                -limits.max_velocity, limits.max_velocity
            )
            old_velocity = self.fields["joint_target_velocity"]
            velocity = torch.maximum(
                torch.minimum(velocity, old_velocity + limits.max_acceleration * dt),
                old_velocity - limits.max_acceleration * dt,
            )
            self.ctrl[:, self.arm_act] = (previous + velocity * dt).clamp(
                self.joint_target_bounds[0], self.joint_target_bounds[1]
            )
            old_velocity.copy_(velocity)
            self.ctrl[:, self.gripper_act] = 255 * actions[:, 6:7].clamp(0, 1)
            return
        action = actions.clamp(-1, 1) * self.scale
        pos = torch.maximum(
            torch.minimum(self.effector + action[:, :3], self.bounds[1]), self.bounds[0]
        )
        # Upstream wraps relative yaw through SO3 before computing the target.
        yaw = torch.atan2(
            (self.yaw + action[:, 3]).sin(), (self.yaw + action[:, 3]).cos()
        )
        rot = rotation_z(yaw) @ self.down
        attach_pos = pos + (
            rot @ self.pa_pos.expand(self.worlds, 3).unsqueeze(-1)
        ).squeeze(-1)
        attach_rot = rot @ self.pa_rot
        joints = self.ik.solve(self.qpos[:, self.arm_q], attach_pos, attach_rot)
        if self.config.joint_target_noise:
            joints = (joints + self.fields["joint_target_offset"]).clamp(
                self.joint_target_bounds[0], self.joint_target_bounds[1]
            )
        self.ctrl[:, self.arm_act] = joints
        grip = (self.grip + action[:, 4]).clamp(0, 1)
        self.ctrl[:, self.gripper_act] = 255 * grip[:, None]

    @on_stream
    def set_control(self, actions):
        if self.config.joint_actions:
            # Joint control is cheap; avoid advancing slew state during graph warmup.
            self._control(actions)
            return
        self.static_action.copy_(actions)
        if self.control_graph is None:
            # Warm cuSOLVER and allocator before capture. No physics is advanced.
            for _ in range(3):
                self._control(self.static_action)
            torch.cuda.synchronize(self.device)
            self.control_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.control_graph):
                self._control(self.static_action)
        self.control_graph.replay()

    def update_buttons(self, previous):
        if self.num_buttons:
            pressed = (previous > -0.02) & (self.qpos[:, self.button_q] <= -0.02)
            self.fields["buttons"].copy_(
                (self.fields["buttons"] + (pressed.float() @ self.toggle.float()).int())
                % self.base._num_button_states
            )
        if self.is_scene:
            self.fields["dof_damping"][:, self.drawer_v] = torch.where(
                self.fields["buttons"][:, 0] == 0, 1e6, 2.0
            )
            self.fields["dof_damping"][:, self.window_v] = torch.where(
                self.fields["buttons"][:, 1] == 0, 1e6, 2.0
            )

    @on_stream
    def step(self, actions):
        previous = self.qpos[:, self.button_q].clone()
        self.set_control(actions)
        self.contact_depth.zero_()
        with wp.ScopedDevice(self.warp_device), wp.ScopedStream(self.stream):
            if self.physics_graph is None:
                saved = self.snapshot()
                # Warm all kernels, then restore before capturing a full control tick.
                mjw.step(self.model, self.data)
                self.restore(saved)
                with wp.ScopedCapture(stream=self.stream) as capture:
                    for _ in range(self.base._n_steps):
                        mjw.step(self.model, self.data)
                        wp.launch(
                            collect_contact_depth,
                            dim=self.data.contact.dist.shape[0],
                            inputs=self.contact_inputs,
                            device=self.warp_device,
                            stream=self.stream,
                        )
                    mjw.kinematics(self.model, self.data)
                    # Include the integrated endpoint, not just solver contacts
                    # computed before each substep's position update.
                    mjw.collision(self.model, self.data)
                    wp.launch(
                        collect_contact_depth,
                        dim=self.data.contact.dist.shape[0],
                        inputs=self.contact_inputs,
                        device=self.warp_device,
                        stream=self.stream,
                    )
                self.physics_graph = capture.graph
                self.restore(saved)
                self.contact_depth.zero_()
            wp.capture_launch(self.physics_graph, stream=self.stream)
        self.update_buttons(previous)
        return self.success(), self.valid()

    def success(self):
        result = torch.ones(self.worlds, dtype=torch.bool, device=self.device)
        if self.cube_q:
            goals = self.fields["mocap_pos"][:, self.cube_mocap]
            result &= ((self.cube_positions() - goals).norm(dim=-1) <= 0.04).all(-1)
        if self.num_buttons:
            result &= (self.fields["buttons"] == self.fields["button_goals"]).all(-1)
        if self.is_scene:
            result &= (
                self.qpos[:, self.drawer_q] - self.fields["drawer_goal"]
            ).abs() <= 0.04
            result &= (
                self.qpos[:, self.window_q] - self.fields["window_goal"]
            ).abs() <= 0.04
        return result

    def valid(self):
        result = (
            self.qpos.isfinite().all(-1)
            & self.qvel.isfinite().all(-1)
            & (self.overflow == 0)
        )
        if self.cube_q:
            positions = self.cube_positions()
            result &= (
                (
                    (positions >= self.bounds[0] - 0.2)
                    & (positions <= self.bounds[1] + 0.2)
                )
                .all(-1)
                .all(-1)
            )
        return result

    def contact_valid(self):
        return (self.contact_depth[:, 0] <= self.config.max_nonpad_penetration) & (
            self.contact_depth[:, 1] <= self.config.max_penetration
        )

    def task_error(self):
        error = torch.zeros(self.worlds, device=self.device)
        if self.cube_q:
            error += (
                (self.cube_positions() - self.fields["mocap_pos"][:, self.cube_mocap])
                .square()
                .sum((-1, -2))
            )
        if self.num_buttons:
            error += (
                (self.fields["buttons"] != self.fields["button_goals"]).float().sum(-1)
            )
        if self.is_scene:
            error += (self.qpos[:, self.drawer_q] - self.fields["drawer_goal"]).square()
            error += (self.qpos[:, self.window_q] - self.fields["window_goal"]).square()
        return error

    def proprioception(self):
        return torch.cat(
            (
                self.qpos[:, self.arm_q],
                self.qvel[:, self.arm_v],
                self.effector,
                self.yaw[:, None],
                self.grip[:, None],
                self.qvel[:, self.gripper_v, None],
            ),
            -1,
        )

    @on_stream
    def render_batch(self):
        from .rendering import BatchRenderer

        if not hasattr(self, "renderer"):
            self.renderer = BatchRenderer(self)
        return self.renderer.render()

    def render(self, world, sync=True):
        return {key: value[world] for key, value in self.render_batch().items()}
