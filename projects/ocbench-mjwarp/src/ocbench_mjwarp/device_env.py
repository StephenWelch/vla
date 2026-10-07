"""OCBench execution with GPU actions, diagnostic accumulation and bounded history."""

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from .config import ABSOLUTE_ACTION, ABSOLUTE_GRIPPER_ACTION, FIELDS
from .lerobot_env import OCBenchVectorEnv


class DeviceEnvironment(OCBenchVectorEnv):
    reuse_simulation = True
    defer_observations = True

    def reset(self, **kwargs):
        super().reset(**kwargs)
        sim = self.sim
        self.views = {k: wp.to_torch(getattr(sim.data, k)) for k in FIELDS}
        with torch.cuda.stream(sim.torch_stream):
            self.gpu_peak = torch.zeros(
                (self.num_envs, 2), device=sim.warp_device.alias
            )
            self.gpu_finite = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.gpu_peak.device
            )
            self.gpu_valid = self.gpu_finite.clone()
            self.gpu_native = ~self.gpu_finite
            self.history_gpu = torch.zeros(
                (51, *self.views["qpos"].shape), device=self.gpu_peak.device
            )
            self.history_gpu[0].copy_(self.views["qpos"])
            self.frozen = {k: torch.empty_like(v) for k, v in self.views.items()}
        self.tick = 0
        return None, {}

    def state(self):
        sim, env = self.sim, self.sim.env
        # One compact read replaces transfers of full qpos/qvel/site arrays.
        # Keep NumPy atan2 for bit-exact agreement with the reference state.
        with torch.cuda.stream(sim.torch_stream):
            q, v = self.views["qpos"], self.views["qvel"]
            pos = wp.to_torch(sim.data.site_xpos)[:, env._pinch_site_id]
            mat = wp.to_torch(sim.data.site_xmat)[:, env._pinch_site_id].reshape(
                -1, 3, 3
            )
            grip = env._gripper_opening_joint_id
            packed = torch.cat(
                (
                    q[:, env._arm_qpos_ids],
                    v[:, env._arm_dof_ids],
                    pos,
                    mat[:, 1, 0, None],
                    mat[:, 0, 0, None],
                    q[:, grip : grip + 1] / 0.8,
                    v[:, grip : grip + 1] / 0.8,
                ),
                dim=1,
            )
        sim.torch_stream.synchronize()
        values = packed.cpu().numpy()
        return np.concatenate(
            (
                values[:, :15],
                np.arctan2(values[:, 15:16], values[:, 16:17]),
                values[:, 17:],
            ),
            axis=1,
        ).astype(np.float32)

    def advance(self, action):
        sim = self.sim
        if action.shape != (self.num_envs, 7) or not torch.isfinite(action).all():
            raise ValueError("Expected finite batched seven-dimensional actions")
        active = ~self.done
        had_done = self.done.any()
        sim.torch_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(sim.torch_stream), wp.ScopedStream(sim.stream):
            done = torch.as_tensor(self.done, device=self.gpu_peak.device)
            mask = wp.from_torch(done.to(torch.int32), dtype=wp.int32)
            if had_done:
                for key, value in self.views.items():
                    self.frozen[key].copy_(value)
            gpu_action = wp.from_torch(action.float().contiguous(), dtype=wp.float32)
            if self.config.action_profile in (ABSOLUTE_ACTION, ABSOLUTE_GRIPPER_ACTION):
                from .actions import step_absolute

                step_absolute(
                    sim.env,
                    gpu_action,
                    mask,
                    absolute_arm=self.config.action_profile == ABSOLUTE_ACTION,
                )
            else:
                sim.env.step_joint_actions_gpu(gpu_action, mask)
            self.tick += 1
            self.history_gpu[self.tick % 51].copy_(self.views["qpos"])
            self.gpu_peak = torch.where(
                done[:, None],
                self.gpu_peak,
                torch.maximum(self.gpu_peak, wp.to_torch(sim.depth)),
            )
            overflow = wp.to_torch(sim.data.overflow)
            budget = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
            finite = (
                torch.isfinite(self.views["qpos"]).all(1)
                & torch.isfinite(self.views["qvel"]).all(1)
                & ((overflow & ~budget) == 0)
            )
            healthy = wp.to_torch(sim.env._gpu_healthy).bool()
            self.gpu_finite &= done | finite
            self.gpu_valid &= done | (
                healthy
                & self.gpu_finite
                & (self.gpu_peak[:, 0] <= 0.001)
                & (self.gpu_peak[:, 1] <= 0.003)
            )
            self.gpu_native |= ~done & wp.to_torch(sim.env._gpu_success).bool()
            status = torch.stack(
                (self.gpu_native, self.gpu_valid, self.gpu_finite, healthy), dim=1
            )
        sim.torch_stream.synchronize()
        native, valid, finite, healthy = status.cpu().numpy().T
        self.steps += active
        terminated = active & (native | ~healthy | ~finite)
        truncated = active & (self.steps >= self.config.max_steps) & ~terminated
        finished = terminated | truncated
        if finished.any():
            from .collect import stable_stack

            length = min(self.tick + 1, 51)
            indices = [(self.tick - length + 1 + i) % 51 for i in range(length)]
            history = self.history_gpu[indices].cpu().numpy()
            peak = self.gpu_peak.cpu().numpy()
            for i in np.flatnonzero(finished):
                stable = stable_stack(sim, {"sim/qpos": history[:, i]})
                self.records.append(
                    {
                        "seed": int(self.seeds[i]),
                        "steps": int(self.steps[i]),
                        "task_success": bool(native[i]),
                        "contact_valid": bool((peak[i] <= [0.001, 0.003]).all()),
                        "physics_valid": bool(finite[i] and healthy[i]),
                        "success": bool(native[i] and valid[i]),
                        "truncated": bool(truncated[i]),
                        "peak_penetration": peak[i].tolist(),
                        "stable_stack": stable["valid"],
                        "stable_stack_failures": stable["failures"],
                    }
                )
        if had_done:
            with torch.cuda.stream(sim.torch_stream), wp.ScopedStream(sim.stream):
                for key, value in self.views.items():
                    value.copy_(
                        torch.where(
                            done.reshape(-1, *([1] * (value.ndim - 1))),
                            self.frozen[key],
                            value,
                        )
                    )
                # Match the reference forward pass after restoring frozen worlds.
                mjw.forward(sim.model, sim.data)
        self.done |= finished
        self.native, self.valid, self.finite = native, valid, finite
        return (native & valid).astype(float), terminated, truncated
