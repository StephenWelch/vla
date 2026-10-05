"""Batched UR5 differential IK, using the pinned OGBench controller model."""

import numpy as np
import torch


def rotation_z(yaw):
    c, s = yaw.cos(), yaw.sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(-1, 3, 3)


def matrix_rotvec(matrix):
    # Quaternion extraction handles the nearly-pi errors at episode starts.
    m = matrix
    candidates = torch.stack(
        (
            1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2],
            1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
            1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
            1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2],
        ),
        -1,
    )
    root = candidates.clamp_min(0).sqrt()
    quats = torch.stack(
        (
            torch.stack(
                (
                    root[:, 0] ** 2,
                    m[:, 2, 1] - m[:, 1, 2],
                    m[:, 0, 2] - m[:, 2, 0],
                    m[:, 1, 0] - m[:, 0, 1],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[:, 2, 1] - m[:, 1, 2],
                    root[:, 1] ** 2,
                    m[:, 1, 0] + m[:, 0, 1],
                    m[:, 0, 2] + m[:, 2, 0],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[:, 0, 2] - m[:, 2, 0],
                    m[:, 1, 0] + m[:, 0, 1],
                    root[:, 2] ** 2,
                    m[:, 2, 1] + m[:, 1, 2],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[:, 1, 0] - m[:, 0, 1],
                    m[:, 0, 2] + m[:, 2, 0],
                    m[:, 2, 1] + m[:, 1, 2],
                    root[:, 3] ** 2,
                ),
                -1,
            ),
        ),
        1,
    )
    quats = quats / (2 * root.clamp_min(1e-12).unsqueeze(-1))
    q = quats[torch.arange(len(m), device=m.device), root.argmax(-1)]
    q = torch.where(q[:, :1] < 0, -q, q)
    v = q[:, 1:]
    norm = v.norm(dim=-1, keepdim=True)
    return v * (2 * torch.atan2(norm, q[:, :1].clamp_min(0)) / norm.clamp_min(1e-12))


class BatchedIK:
    def __init__(self, model, device):
        import mujoco

        self.device = device
        self.dtype = torch.float64
        self.eye = torch.eye(3, dtype=self.dtype, device=device)
        self.eye6 = torch.eye(6, dtype=self.dtype, device=device)
        site = model.site("attachment_site").id
        chain = []
        body = int(model.site_bodyid[site])
        while body:
            chain.append(body)
            body = int(model.body_parentid[body])
        self.chain = []
        for body in reversed(chain):
            rot = np.empty(9)
            mujoco.mju_quat2Mat(rot, model.body_quat[body])
            joint = int(model.body_jntadr[body])
            if model.body_jntnum[body] not in (0, 1):
                raise ValueError("Expected one hinge per UR5 body")
            self.chain.append(
                (
                    self.tensor(model.body_pos[body]),
                    self.tensor(rot.reshape(3, 3)),
                    joint if model.body_jntnum[body] else -1,
                    self.tensor(model.jnt_axis[joint]) if joint >= 0 else None,
                    self.tensor(model.jnt_pos[joint]) if joint >= 0 else None,
                )
            )
        self.site_pos = self.tensor(model.site_pos[site])
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, model.site_quat[site])
        self.site_rot = self.tensor(rot.reshape(3, 3))
        self.qpos0 = self.tensor(model.qpos0)

    def tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def forward(self, q):
        n = len(q)
        pos = torch.zeros((n, 3), device=self.device, dtype=self.dtype)
        rot = self.eye.expand(n, 3, 3)
        pivots, axes = [], []
        for translation, rest, joint, axis, pivot in self.chain:
            pos = pos + (rot @ translation.expand(n, 3).unsqueeze(-1)).squeeze(-1)
            rot = rot @ rest
            if joint >= 0:
                world_axis = (rot @ axis.expand(n, 3).unsqueeze(-1)).squeeze(-1)
                world_pivot = pos + (rot @ pivot.expand(n, 3).unsqueeze(-1)).squeeze(-1)
                pivots.append(world_pivot)
                axes.append(world_axis)
                angle = q[:, joint] - self.qpos0[joint]
                x, y, z = axis.unbind()
                zero = x * 0
                skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(
                    3, 3
                )
                turn = (
                    self.eye
                    + angle.sin()[:, None, None] * skew
                    + (1 - angle.cos())[:, None, None] * (skew @ skew)
                )
                new_rot = rot @ turn
                pos = world_pivot - (
                    new_rot @ pivot.expand(n, 3).unsqueeze(-1)
                ).squeeze(-1)
                rot = new_rot
        site_pos = pos + (rot @ self.site_pos.expand(n, 3).unsqueeze(-1)).squeeze(-1)
        site_rot = rot @ self.site_rot
        axes = torch.stack(axes, -1)
        distance = site_pos.unsqueeze(-1) - torch.stack(pivots, -1)
        jacp = torch.linalg.cross(axes, distance, dim=1)
        return site_pos, site_rot, torch.cat((jacp, axes), 1)

    def solve(self, q, target_pos, target_rot):
        q = q.to(self.dtype).clone()
        target_pos, target_rot = target_pos.to(self.dtype), target_rot.to(self.dtype)
        for _ in range(20):
            pos, rot, jac = self.forward(q)
            err_pos = target_pos - pos
            err_rot = matrix_rotvec(target_rot @ rot.transpose(-1, -2))
            err = torch.cat((err_pos, err_rot), -1)
            solution, _ = torch.linalg.solve_ex(
                jac @ jac.transpose(-1, -2) + 1e-12 * self.eye6,
                err.unsqueeze(-1),
                check_errors=False,
            )
            delta = (jac.transpose(-1, -2) @ solution).squeeze(-1)
            delta *= 0.7853981633974483 / delta.abs().amax(-1, keepdim=True).clamp_min(
                0.7853981633974483
            )
            done = (err_pos.norm(dim=-1) <= 1e-4) & (err_rot.norm(dim=-1) <= 1e-4)
            q += torch.where(done[:, None], torch.zeros_like(delta), delta)
        return q.float()
