"""Task decomposition with upstream manipulation waypoint plans as MPC seeds."""

import numpy as np

from .config import RandomizationConfig
from .tasks import puzzle_toggle, solve_puzzle


class SkillPlan:
    def __init__(self, env, seed, randomization=None, provenance=None):
        self.env = env
        self.config = randomization or RandomizationConfig()
        self.provenance = provenance
        seeds = provenance["seeds"] if provenance else {}
        self.rng = np.random.RandomState(seeds.get("oracle", seed))
        self.streams = {
            name: np.random.default_rng(seeds.get(name, seed))
            for name in ("order", "grasp", "path", "timing")
        }
        self.events = provenance["skills"] if provenance else []
        self.choices = []
        self.elapsed = 0
        self.phase_times = np.array([0.0])
        self.phase_names = ["hold"]
        self.plan = None
        self.cursor = 0
        self.skill = None
        self.objective = {"kind": "none"}
        self.puzzle_pending = None

    def choose(self, candidates, reason):
        candidates = [int(i) for i in candidates]
        selected = (
            int(self.streams["order"].choice(candidates))
            if self.config.order and len(candidates) > 1
            else candidates[0]
        )
        self.choices.append(
            {"eligible": candidates, "selected": selected, "constraint": reason}
        )
        return selected

    def temporary_cube(self, index, positions):
        for xy in ((0.3, -0.25), (0.55, 0.25), (0.3, 0.25), (0.55, -0.25)):
            temporary = np.array([*xy, 0.02])
            if np.min(np.linalg.norm(positions - temporary, axis=1)) > 0.07:
                return "cube", index, temporary
        return None

    def choose_skill(self):
        e = self.env.unwrapped
        if hasattr(e, "_num_rows"):
            if not self.puzzle_pending:
                self.puzzle_pending = solve_puzzle(
                    e._cur_button_states,
                    e._target_button_states,
                    e._num_rows,
                    e._num_cols,
                )
            if self.puzzle_pending:
                index = self.choose(
                    self.puzzle_pending, "remaining GF(2) solution presses commute"
                )
                self.puzzle_pending.remove(index)
                return "button", index, None
            return None
        cube_positions = np.array(
            [
                e._data.qpos[e._model.joint(f"object_joint_{i}").qposadr[0] :][:3]
                for i in range(e._num_cubes)
            ]
        )
        goals = e._data.mocap_pos[e._cube_target_mocap_ids]
        unfinished = np.flatnonzero(
            np.linalg.norm(cube_positions - goals, axis=1) > 0.025
        )
        if hasattr(e, "_target_drawer_pos"):
            drawer, window = (
                e._data.joint("drawer_slide").qpos[0],
                e._data.joint("window_slide").qpos[0],
            )
            # Drawer-contained goals are specified at its final joint position.
            # Place in the open tray, then let closing carry the cube to its goal.
            placement_goals = goals.copy()
            in_drawer = goals[:, 1] < -0.3
            axis = e._data.xaxis[e._model.joint("drawer_slide").id]
            placement_goals[in_drawer] += axis * (drawer - e._target_drawer_pos + 0.01)
            unfinished = np.flatnonzero(
                np.linalg.norm(cube_positions - placement_goals, axis=1) > 0.025
            )
            # Unlock movable components before manipulating them.
            need_drawer = (
                abs(drawer - e._target_drawer_pos) > 0.015 or len(unfinished) > 0
            )
            need_window = abs(window - e._target_window_pos) > 0.015
            if need_drawer and e._cur_button_states[0] == 0:
                return (
                    "button",
                    self.choose([0], "unlock drawer before access or movement"),
                    1,
                )
            if need_window and e._cur_button_states[1] == 0:
                return "button", self.choose([1], "unlock window before movement"), 1
            if len(unfinished):
                i = self.choose(unfinished, "drawer access precedes placement")
                if (
                    goals[i, 1] < -0.3 or cube_positions[i, 1] < -0.3
                ) and drawer > -0.14:
                    return "drawer", 0, -0.16
                return "cube", i, placement_goals[i]
            if abs(drawer - e._target_drawer_pos) > 0.015:
                return (
                    "drawer",
                    self.choose([0], "finish cube placements before closing drawer"),
                    e._target_drawer_pos,
                )
            if abs(window - e._target_window_pos) > 0.015:
                return (
                    "window",
                    self.choose([0], "move window while unlocked"),
                    e._target_window_pos,
                )
            buttons = np.flatnonzero(e._cur_button_states != e._target_button_states)
            if len(buttons):
                i = self.choose(buttons, "restore final locks after joint movements")
                return "button", i, int(e._target_button_states[i])
        elif len(unfinished):
            # Clear topmost cubes first, then build target stacks bottom-up.
            top = [
                int(i)
                for i in unfinished
                if not any(
                    cube_positions[j, 2] > cube_positions[i, 2] + 0.02
                    and np.linalg.norm(cube_positions[j, :2] - cube_positions[i, :2])
                    < 0.035
                    for j in range(len(cube_positions))
                    if j != i
                )
            ]
            if top:
                lowest = min(goals[i, 2] for i in top)
                i = self.choose(
                    [j for j in top if goals[j, 2] <= lowest + 0.02],
                    "clear topmost cubes; build goals bottom-up",
                )
                supports = [
                    j
                    for j in unfinished
                    if goals[j, 2] < goals[i, 2] - 0.02
                    and np.linalg.norm(goals[j, :2] - goals[i, :2]) < 0.035
                ]
                if supports:
                    return self.temporary_cube(i, cube_positions)
                occupied = [
                    j
                    for j in range(len(goals))
                    if j != i and np.linalg.norm(cube_positions[j] - goals[i]) < 0.035
                ]
                if occupied:
                    # Temporary placement resolves swaps without overwriting goals.
                    j = occupied[0]
                    column = [
                        k
                        for k in range(len(goals))
                        if cube_positions[k, 2] >= cube_positions[j, 2]
                        and np.linalg.norm(
                            cube_positions[k, :2] - cube_positions[j, :2]
                        )
                        < 0.035
                    ]
                    j = max(column, key=lambda k: cube_positions[k, 2])
                    return self.temporary_cube(j, cube_positions)
                return "cube", i, goals[i]
        return None

    def build(self):
        from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle
        from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
        from ogbench.manipspace.oracles.plan.drawer_plan import DrawerPlanOracle
        from ogbench.manipspace.oracles.plan.window_plan import WindowPlanOracle

        e = self.env.unwrapped
        self.choices = []
        self.skill = self.choose_skill()
        info = e.compute_ob_info()
        if self.skill is None:
            self.objective = {"kind": "none"}
            self.plan = np.array(
                [
                    [
                        *info["proprio/effector_pos"],
                        info["proprio/effector_yaw"][0],
                        info["proprio/gripper_opening"][0],
                    ]
                ],
                dtype=np.float32,
            )
            self.events.append(
                {
                    "skill_id": len(self.events),
                    "kind": "none",
                    "start_frame": self.elapsed,
                    "choices": self.choices,
                    "phase_names": ["hold"],
                    "phase_times": [0.0],
                }
            )
            self.phase_names, self.phase_times = ["hold"], np.array([0.0])
            return
        kind, index, goal = self.skill
        self.objective = {"kind": kind, "index": index, "goal": goal}
        if kind == "cube":
            info.update(
                {
                    "privileged/target_block": index,
                    "privileged/target_block_pos": goal,
                    "privileged/target_block_yaw": np.array([0.0]),
                }
            )
        elif kind == "button":
            target = (
                (e._cur_button_states[index] + 1) % e._num_button_states
                if goal is None
                else goal
            )
            expected = e._cur_button_states.copy()
            if hasattr(e, "_num_rows"):
                expected = (
                    expected + puzzle_toggle(e._num_rows, e._num_cols)[index]
                ) % e._num_button_states
            else:
                expected[index] = target
            self.objective["buttons"] = expected
            info.update(
                {
                    "privileged/target_button": index,
                    "privileged/target_button_state": target,
                    "privileged/target_button_top_pos": e._data.site_xpos[
                        e._button_site_ids[index]
                    ].copy(),
                }
            )
        else:
            handle = info[f"privileged/{kind}_handle_pos"].copy()
            joint = e._data.joint(f"{kind}_slide").qpos[0]
            axis = e._data.xaxis[e._model.joint(f"{kind}_slide").id]
            handle += axis * (goal - joint)
            info[f"privileged/target_{kind}_handle_pos"] = handle
        oracle = {
            "cube": CubePlanOracle,
            "button": ButtonPlanOracle,
            "drawer": DrawerPlanOracle,
            "window": WindowPlanOracle,
        }[kind](env=self.env, noise=0, segment_dt=0.6)
        keyframes = oracle.compute_keyframes
        drawer_placement = (
            kind == "cube"
            and hasattr(e, "_target_drawer_pos")
            and e._data.mocap_pos[e._cube_target_mocap_ids[index], 1] < -0.3
        )
        event = {
            "skill_id": len(self.events),
            "kind": kind,
            "index": int(index),
            "goal": goal,
            "start_frame": self.elapsed,
            "choices": self.choices,
            "samples": {},
        }
        self.events.append(event)

        def describe(times, poses, grasps):
            return [
                {
                    "name": name,
                    "time": float(times[name]),
                    "position": poses[name].translation().tolist(),
                    "yaw": float(oracle.get_yaw(poses[name])),
                    "gripper": float(grasps[name]),
                }
                for name in times
            ]

        def finish_at_clearance(plan_input):
            times, poses, grasps = keyframes(plan_input)
            event["upstream_keyframes"] = describe(times, poses, grasps)
            if kind in ("drawer", "window"):
                # Establish the grasp before translating, and allow the handle to
                # follow without pushing it using the gripper linkage.
                durations = {
                    "grasp_end": 0.6,
                    "move": 1.2,
                    "release": 0.4,
                    "clearance": 0.5,
                }
                if kind == "window":
                    durations.update(approach=1.0, grasp_start=0.6)
                    poses["approach"] = oracle.above(poses["grasp_start"], 0.12)
                    poses["clearance"] = oracle.above(poses["release"], 0.12)
                previous = None
                original_times = times.copy()
                for name in times:
                    if previous is not None:
                        delta = durations.get(
                            name, times[name] - original_times[previous]
                        )
                        times[name] = times[previous] + delta
                    previous = name
            if drawer_placement:
                # Open the jaws across the tray's width, away from its front wall.
                # Cube yaw is unconstrained by OGBench's positional goal.
                for name in ("place", "place_start", "place_end", "postplace"):
                    poses[name] = oracle.to_pose(
                        pos=poses[name].translation(), yaw=np.pi / 2
                    )
            # Preserve release and clearance, then start the next manipulation.
            # Upstream's random final retreat consumes long-task step budgets.
            for frames in (times, poses, grasps):
                frames.pop("final")
            event["baseline_keyframes"] = describe(times, poses, grasps)
            if kind == "cube" and self.config.cube_grasps:
                symmetry = int(self.streams["grasp"].integers(4))
                yaw = oracle.get_yaw(poses["pick_start"]) + symmetry * np.pi / 2
                event["samples"]["grasp"] = {
                    "symmetry": symmetry,
                    "yaw": float(yaw),
                    "allowed_symmetries": [0, 1, 2, 3],
                    "drawer_placement_yaw": float(np.pi / 2)
                    if drawer_placement
                    else None,
                }
                for name in ("pick", "pick_start", "pick_end", "postpick"):
                    poses[name] = oracle.to_pose(poses[name].translation(), yaw)
            elif kind in ("drawer", "window") and self.config.handle_grasps:
                symmetry = int(self.streams["grasp"].integers(2))
                event["samples"]["grasp"] = {
                    "symmetry": symmetry,
                    "yaw_offset": float(symmetry * np.pi),
                    "allowed_symmetries": [0, 1],
                }
                for name in times:
                    if name != "initial":
                        poses[name] = oracle.to_pose(
                            poses[name].translation(),
                            oracle.get_yaw(poses[name]) + symmetry * np.pi,
                        )
            groups = {
                "cube": (("pick", "postpick"), ("clearance",), ("place", "postplace")),
                "button": (("press_start", "press_end"),),
                "drawer": (("approach",), ("clearance",)),
                "window": (("approach",), ("clearance",)),
            }[kind]
            if self.config.position_noise or self.config.yaw_noise:
                event["samples"]["path"] = []
                for names in groups:
                    offset = self.streams["path"].uniform(
                        -self.config.position_noise, self.config.position_noise, 3
                    )
                    offset[2] = abs(offset[2])
                    yaw_offset = float(
                        self.streams["path"].uniform(
                            -self.config.yaw_noise, self.config.yaw_noise
                        )
                    )
                    event["samples"]["path"].append(
                        {
                            "keyframes": list(names),
                            "position_offset": offset.tolist(),
                            "yaw_offset": yaw_offset,
                        }
                    )
                    for name in names:
                        poses[name] = oracle.to_pose(
                            poses[name].translation() + offset,
                            oracle.get_yaw(poses[name]) + yaw_offset,
                        )
            if (
                self.config.duration_scale_min != 1
                or self.config.duration_scale_max != 1
            ):
                original = times.copy()
                event["samples"]["timing"] = []
                previous = "initial"
                for name in list(times)[1:]:
                    nominal = original[name] - original[previous]
                    scale = float(
                        self.streams["timing"].uniform(
                            self.config.duration_scale_min,
                            self.config.duration_scale_max,
                        )
                    )
                    floor = (
                        nominal
                        if kind in ("drawer", "window")
                        or name in ("pick_end", "place_end", "press", "press_end")
                        else e._control_timestep
                    )
                    duration = max(floor, nominal * scale)
                    times[name] = times[previous] + duration
                    event["samples"]["timing"].append(
                        {
                            "segment": name,
                            "scale": scale,
                            "requested_seconds": nominal * scale,
                            "minimum_seconds": floor,
                            "applied_seconds": duration,
                        }
                    )
                    previous = name
            event["keyframes_before_workspace_clip"] = describe(times, poses, grasps)
            self.phase_names = list(times)
            self.phase_times = np.array(list(times.values()))
            event["phase_names"], event["phase_times"] = (
                self.phase_names,
                self.phase_times.tolist(),
            )
            return times, poses, grasps

        oracle.compute_keyframes = finish_at_clearance
        saved = np.random.get_state()
        try:
            np.random.set_state(self.rng.get_state())
            oracle.reset(None, info)
            self.rng.set_state(np.random.get_state())
        finally:
            np.random.set_state(saved)
        self.plan = oracle._plan.astype(np.float32)
        unclipped = self.plan[:, :3].copy()
        self.plan[:, :3] = np.clip(self.plan[:, :3], *e._workspace_bounds)
        event["workspace_bounds"] = np.asarray(e._workspace_bounds).tolist()
        event["workspace_clipped_frames"] = np.flatnonzero(
            np.any(self.plan[:, :3] != unclipped, axis=1)
        ).tolist()

    def annotation(self):
        tick = min(self.cursor + 1, len(self.plan) - 1)
        phase = int(
            np.clip(
                np.searchsorted(
                    self.phase_times,
                    tick * self.env.unwrapped._control_timestep,
                    side="right",
                )
                - 1,
                0,
                len(self.phase_names) - 1,
            )
        )
        return {
            "annotation/skill_id": len(self.events) - 1,
            "annotation/phase_id": phase,
            "annotation/route_id": self.provenance["variant_id"]
            if self.provenance
            else 0,
            "annotation/reference": self.plan[tick].copy(),
        }

    def references(self, horizon):
        if self.plan is None or self.cursor >= len(self.plan):
            self.build()
            self.cursor = 0
        indices = np.minimum(
            np.arange(self.cursor + 1, self.cursor + horizon + 1), len(self.plan) - 1
        )
        return self.plan[indices]

    def advance(self):
        self.cursor += 1
        self.elapsed += 1
        self.events[-1]["end_frame_exclusive"] = self.elapsed
