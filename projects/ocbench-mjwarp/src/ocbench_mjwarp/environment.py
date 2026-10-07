"""Native physics with observational contact diagnostics and batched cameras."""

import types

import mujoco_warp as mjw
import numpy as np
import ocbench
import torch
import warp as wp

from .config import ACTION, ARM_LIMITS, FIELDS, TASK
from .contacts import collect_contact_depth, contact_roles
from .rendering import BatchRenderer, image_shape


class Simulation:
    def __init__(self, seeds, task=TASK, audit=True, image_size=(480, 640)):
        self.task = task
        self.worlds = len(seeds)
        self.image_size = image_shape(image_size)
        self.env = ocbench.make(
            task,
            nworld=self.worlds,
            width=self.image_size[1],
            height=self.image_size[0],
        )
        self.env.reset(seeds=np.asarray(seeds, dtype=np.uint32))
        self.base = self.env.cpu_env
        self.host_model = self.base.model
        self.model, self.data = self.env.model, self.env.data
        # Keep convergence-budget evidence in overflow flags without printing
        # the same warning thousands of times. Capacity warnings stay enabled.
        self.model.opt.warn_overflow &= ~int(
            mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS
        )
        self.warp_device = self.data.qpos.device
        self.stream = wp.get_stream(self.warp_device)
        self.torch_stream = wp.stream_to_torch(self.stream)
        self.renderer = None
        if not np.allclose(self.base._joint_action_delta, ACTION["scales"]):
            raise ValueError("Upstream action scales changed")
        bounds = self.host_model.actuator_ctrlrange[self.base._arm_actuator_ids]
        if not np.allclose(
            bounds, np.column_stack((-np.asarray(ARM_LIMITS), ARM_LIMITS))
        ):
            raise ValueError("Upstream arm actuator bounds changed")
        if not np.isclose(self.base._control_timestep, 0.02):
            raise ValueError("Expected the full 50 Hz native environment")
        self.depth = wp.zeros((self.worlds, 2), dtype=float, device=self.warp_device)
        self.roles = wp.array(
            contact_roles(self.host_model), dtype=wp.int32, device=self.warp_device
        )
        if not {1, 2, 3} <= set(self.roles.numpy()):
            raise ValueError(
                "Cannot identify OCBench arm, gripper links and finger pads"
            )
        if audit:
            # Patch this instance only. Include diagnostics inside the captured
            # physics graph; do not alter controls, contacts or native outcomes.
            def sequence(env):
                self.depth.zero_()
                for _ in range(env.cpu_env._n_steps):
                    mjw.step(env.model, env.data)
                    self.collect_depth()
                mjw.rne_postconstraint(env.model, env.data)

            self.env._launch_mjwarp_control_step_sequence = types.MethodType(
                sequence, self.env
            )
            self.env._step_graph = None
        self.env._ensure_gpu_step_buffers()

    def reset_render(self, seeds):
        """Reset episode-dependent state without rebuilding the render context."""
        self.env.reset(seeds=np.asarray(seeds, dtype=np.uint32))
        if self.renderer is not None:
            with torch.cuda.stream(self.torch_stream):
                self.renderer.geom.copy_(self.tensor(self.host_model.geom_rgba))
                self.renderer.material.copy_(self.tensor(self.host_model.mat_rgba))

    def tensor(self, array):
        return torch.as_tensor(array, device=str(self.warp_device))

    def collect_depth(self):
        wp.launch(
            collect_contact_depth,
            dim=self.data.contact.dist.shape[0],
            inputs=[
                self.data.nacon,
                self.data.contact.geom,
                self.data.contact.dist,
                self.data.contact.worldid,
                self.roles,
                self.depth,
            ],
            device=self.warp_device,
        )

    def snapshot(self):
        return {name: getattr(self.data, name).numpy().copy() for name in FIELDS}

    def physics_status(self):
        overflow = self.data.overflow.numpy().copy()
        budget = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
        finite = np.isfinite(self.data.qpos.numpy()).all(1) & np.isfinite(
            self.data.qvel.numpy()
        ).all(1)
        return finite & ((overflow & ~budget) == 0), overflow

    def restore(self, states, forward=True):
        for name in FIELDS:
            dest = getattr(self.data, name)
            wp.copy(
                dest,
                wp.array(
                    np.asarray(states[name]), dtype=dest.dtype, device=self.warp_device
                ),
            )
        if forward:
            mjw.forward(self.model, self.data)
        else:
            mjw.kinematics(self.model, self.data)

    def state(self):
        q, v = self.data.qpos.numpy(), self.data.qvel.numpy()
        pos = self.data.site_xpos.numpy()[:, self.env._pinch_site_id]
        mat = self.data.site_xmat.numpy()[:, self.env._pinch_site_id].reshape(-1, 3, 3)
        yaw = np.arctan2(mat[:, 1, 0], mat[:, 0, 0])[:, None]
        grip = self.env._gripper_opening_joint_id
        return np.concatenate(
            [
                q[:, self.env._arm_qpos_ids],
                v[:, self.env._arm_dof_ids],
                pos,
                yaw,
                q[:, grip : grip + 1] / 0.8,
                v[:, grip : grip + 1] / 0.8,
            ],
            axis=1,
        ).astype(np.float32)

    def render(self):
        if self.renderer is None:
            self.renderer = BatchRenderer(self)
        return self.renderer.render()

    def close(self):
        # Release the instance-only diagnostic closure and renderer promptly
        # between batches instead of retaining a simulator reference cycle.
        self.env.__dict__.pop("_launch_mjwarp_control_step_sequence", None)
        self.renderer = None
        self.env.close()
