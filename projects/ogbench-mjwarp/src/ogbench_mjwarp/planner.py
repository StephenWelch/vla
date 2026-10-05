"""Cross-entropy MPC over independent episode/candidate MJWarp worlds."""

import time

import torch

from .environment import BatchEnvironment


class SamplingMPC:
    def __init__(self, execution, config, seed=0):
        self.execution, self.config = execution, config
        self.rollouts = BatchEnvironment(
            execution.env,
            execution.worlds * config.candidates,
            config,
            str(execution.device),
        )
        self.generators = [
            torch.Generator(device=execution.device).manual_seed(seed + i)
            for i in range(execution.worlds)
        ]
        self.source = torch.arange(
            execution.worlds, device=execution.device
        ).repeat_interleave(config.candidates)
        self.mean = torch.zeros(
            (execution.worlds, config.horizon, 5), device=execution.device
        )
        self.previous_action = torch.zeros(
            (execution.worlds, 5), device=execution.device
        )

    @torch.no_grad()
    def plan(self, references, objectives=None):
        start = time.perf_counter()
        c, batch, sim = self.config, self.execution.worlds, self.rollouts
        state = self.execution.snapshot()
        refs = torch.as_tensor(references, device=sim.device, dtype=torch.float32)
        flat_refs = refs.repeat_interleave(c.candidates, dim=0)
        stage_goals = self.stage_goals(objectives) if objectives is not None else None
        mean = self.mean.clone()
        std = torch.full_like(mean, c.noise)
        best_cost = torch.full((batch,), float("inf"), device=sim.device)
        best_action = torch.zeros((batch, 5), device=sim.device)
        best_offsets = torch.zeros_like(mean)
        elites = max(1, int(c.candidates * c.elite_fraction))
        for _ in range(c.iterations):
            noise = torch.stack(
                [
                    torch.randn(
                        (c.candidates, c.horizon, 5),
                        device=sim.device,
                        generator=generator,
                    )
                    for generator in self.generators
                ]
            )
            offsets = mean[:, None] + std[:, None] * noise
            # A grasp needs sustained closure even when contact arrests the jaws.
            # Keep skill gripper commands by default; enable search explicitly.
            offsets[..., 4] = mean[:, None, :, 4] + c.gripper_noise * noise[..., 4]
            offsets[:, 0] = 0  # Always evaluate the unmodified waypoint plan.
            if c.candidates > 2:
                offsets[:, 1] = mean
            sim.restore(state, self.source)
            cost = torch.zeros(sim.worlds, device=sim.device)
            valid = torch.ones(sim.worlds, device=sim.device, dtype=torch.bool)
            previous = self.previous_action.repeat_interleave(c.candidates, dim=0)
            first = None
            for tick in range(c.horizon):
                ref = flat_refs[:, tick]
                yaw_diff = torch.atan2(
                    (ref[:, 3] - sim.yaw).sin(), (ref[:, 3] - sim.yaw).cos()
                )
                base = (
                    torch.cat(
                        (
                            ref[:, :3] - sim.effector,
                            yaw_diff[:, None],
                            (ref[:, 4] - sim.grip)[:, None],
                        ),
                        -1,
                    )
                    / sim.scale
                )
                action = (base + offsets[:, :, tick].reshape(-1, 5)).clamp(-1, 1)
                if first is None:
                    first = action.reshape(batch, c.candidates, 5).clone()
                _, healthy = sim.step(action)
                valid &= healthy
                valid &= sim.contact_valid()
                cost += c.contact_weight * sim.contact_depth[:, 0]
                tracking = (sim.effector - ref[:, :3]).square().sum(-1)
                tracking += (
                    0.01
                    * torch.atan2(
                        (sim.yaw - ref[:, 3]).sin(), (sim.yaw - ref[:, 3]).cos()
                    ).square()
                )
                tracking += 0.005 * (sim.grip - ref[:, 4]).square()
                cost += (
                    c.tracking_weight * tracking
                    + c.action_weight * action.square().sum(-1)
                )
                cost += c.smoothness_weight * (action - previous).square().sum(-1)
                previous = action
            cost += c.task_weight * (
                self.stage_error(stage_goals)
                if stage_goals is not None
                else sim.task_error()
            )
            cost = torch.where(valid & cost.isfinite(), cost, float("inf")).reshape(
                batch, c.candidates
            )
            values, index = cost.topk(elites, largest=False, dim=1)
            selected = offsets.gather(
                1, index[:, :, None, None].expand(-1, -1, c.horizon, 5)
            )
            mean = selected.mean(1)
            std = selected.std(1, unbiased=False).clamp_min(c.min_std)
            improved = values[:, 0] < best_cost
            winner = index[:, 0]
            rows = torch.arange(batch, device=sim.device)
            best_action = torch.where(
                improved[:, None], first[rows, winner], best_action
            )
            best_offsets = torch.where(
                improved[:, None, None], offsets[rows, winner], best_offsets
            )
            best_cost = torch.minimum(best_cost, values[:, 0])
        self.mean = torch.cat(
            (best_offsets[:, 1:], torch.zeros_like(best_offsets[:, :1])), 1
        )
        self.previous_action.copy_(best_action)
        torch.cuda.synchronize(sim.device)
        return best_action, {
            "valid": best_cost.isfinite().cpu().tolist(),
            "cost": best_cost.cpu().tolist(),
            "seconds": time.perf_counter() - start,
        }

    def stage_goals(self, objectives):
        sim, count = self.rollouts, self.config.candidates
        kinds = {"none": -1, "cube": 0, "button": 1, "drawer": 2, "window": 3}
        kind = torch.tensor([kinds[o["kind"]] for o in objectives], device=sim.device)
        index = torch.tensor([o.get("index", 0) for o in objectives], device=sim.device)
        position = torch.tensor(
            [list(o["goal"]) if o["kind"] == "cube" else [0, 0, 0] for o in objectives],
            device=sim.device,
            dtype=torch.float32,
        )
        value = torch.tensor(
            [
                float(o["goal"]) if o["kind"] in ("drawer", "window") else 0
                for o in objectives
            ],
            device=sim.device,
        )
        buttons = torch.tensor(
            [list(o.get("buttons", [0] * sim.num_buttons)) for o in objectives],
            device=sim.device,
            dtype=torch.int32,
        )
        return {
            name: tensor.repeat_interleave(count, dim=0)
            for name, tensor in (
                ("kind", kind),
                ("index", index),
                ("position", position),
                ("value", value),
                ("buttons", buttons),
            )
        }

    def stage_error(self, goals):
        sim = self.rollouts
        kind = goals["kind"]
        error = torch.zeros(sim.worlds, device=sim.device)
        if sim.cube_q:
            positions = sim.cube_positions()[
                torch.arange(sim.worlds, device=sim.device),
                goals["index"].clamp(0, len(sim.cube_q) - 1),
            ]
            error += torch.where(
                kind == 0, (positions - goals["position"]).square().sum(-1), 0
            )
        if sim.num_buttons:
            error += torch.where(
                kind == 1,
                (sim.fields["buttons"] != goals["buttons"]).float().sum(-1),
                0,
            )
        if sim.is_scene:
            error += torch.where(
                kind == 2, (sim.qpos[:, sim.drawer_q] - goals["value"]).square(), 0
            )
            error += torch.where(
                kind == 3, (sim.qpos[:, sim.window_q] - goals["value"]).square(), 0
            )
        return torch.where(kind == -1, sim.task_error(), error)
