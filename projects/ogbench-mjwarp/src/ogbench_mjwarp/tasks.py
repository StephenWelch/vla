"""Task discovery, reset snapshots, and human-readable instructions."""

import copy

import numpy as np


def task_registry():
    import gymnasium as gym
    import ogbench.manipspace  # noqa: F401

    return {
        key: spec
        for key, spec in gym.envs.registry.items()
        if key.startswith(("cube-", "scene-", "puzzle-")) and "singletask" not in key
    }


def reset_task(env, seed, task_id, joint_target_noise=0.0, joint_target_seed=None):
    """Reset task RNGs and clear controls left by upstream goal-observation steps."""
    if not np.isfinite(joint_target_noise) or joint_target_noise < 0:
        raise ValueError("joint_target_noise must be finite and nonnegative")
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        env.reset(seed=seed, options={"task_id": task_id})
    finally:
        np.random.set_state(saved)
    base = env.unwrapped
    base._joint_target_offset = (
        np.random.default_rng(seed if joint_target_seed is None else joint_target_seed)
        .uniform(-joint_target_noise, joint_target_noise, 6)
        .astype(np.float32)
    )
    base._data.qacc_warmstart[:] = 0
    base.set_control(np.zeros(5, dtype=np.float32))
    base.pre_step()


def image_shape(size):
    """Image height/width; an integer retains the square-image shorthand."""
    shape = (size, size) if isinstance(size, int) else tuple(size)
    if len(shape) != 2 or any(not isinstance(v, int) or v < 16 for v in shape):
        raise ValueError("Image size must be an integer or (height, width), both >=16")
    return shape


def make_env(env_id, seed=0, task_id=1, size=(480, 640)):
    import gymnasium as gym
    import ogbench.manipspace  # noqa: F401

    if env_id not in task_registry():
        raise ValueError(f"Unknown manipulation environment {env_id}; use list-tasks")
    height, width = image_shape(size)
    env = gym.make(
        env_id,
        width=width,
        height=height,
        visualize_info=False,
        pixel_transparent_arm=False,
        terminate_at_goal=True,
    )
    # Add a camera before compilation without modifying upstream assets.
    base = env.unwrapped
    original = base.build_mjcf_model

    def add_camera(model):
        # OGBench's 7 cm near plane cuts close objects in the wrist camera.
        model.visual.map.znear = 0.005
        body = model.find("body", "ur5e/wrist_3_link")
        if body is None:
            raise RuntimeError("Pinned OGBench wrist body was not found")
        wrist = base._model.body("ur5e/wrist_3_link").id
        rotation = base._data.xmat[wrist].reshape(3, 3)
        pinch = base._pinch_site_id
        pinch_rot = base._data.site_xmat[pinch].reshape(3, 3)
        target = base._data.site_xpos[pinch] + pinch_rot @ np.array([0, 0, 0.03])
        position = base._data.site_xpos[pinch] + pinch_rot @ np.array([0.085, 0, -0.10])
        zaxis = position - target
        zaxis /= np.linalg.norm(zaxis)
        xaxis = pinch_rot[:, 1]
        yaxis = np.cross(zaxis, xaxis)
        body.add(
            "camera",
            name="wrist",
            pos=rotation.T @ (position - base._data.xpos[wrist]),
            xyaxes=np.concatenate((rotation.T @ xaxis, rotation.T @ yaxis)),
            fovy=75,
        )
        return model

    base.build_mjcf_model = lambda: add_camera(original())
    # Gym queries observation_space during make(), which may compile the model.
    if base._mjcf_model is not None:
        add_camera(base._mjcf_model)
        base.mark_dirty()
    original_close = base.close

    def close():
        if base._renderer is not None:
            base._renderer.close()
            base._renderer = None
        original_close()

    base.close = close
    try:
        reset_task(env, seed, task_id)
    except BaseException:
        env.close()
        raise
    return env


DATA_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc_warmstart",
    "mocap_pos",
    "mocap_quat",
    "eq_active",
    "qfrc_applied",
    "xfrc_applied",
)


def cpu_snapshot(env):
    e = env.unwrapped
    state = {key: getattr(e._data, key).copy() for key in DATA_FIELDS}
    state["time"] = np.asarray(e._data.time)
    state["dof_damping"] = e._model.dof_damping.copy()
    state["buttons"] = np.asarray(
        getattr(e, "_cur_button_states", []), dtype=np.int32
    ).copy()
    state["button_goals"] = np.asarray(
        getattr(e, "_target_button_states", []), dtype=np.int32
    ).copy()
    state["drawer_goal"] = np.asarray(getattr(e, "_target_drawer_pos", 0.0))
    state["window_goal"] = np.asarray(getattr(e, "_target_window_pos", 0.0))
    state["joint_target_offset"] = e._joint_target_offset.copy()
    return state


def restore_cpu(env, state):
    import mujoco

    e = env.unwrapped
    for key in DATA_FIELDS:
        getattr(e._data, key)[:] = state[key]
    e._data.time = float(state["time"])
    e._model.dof_damping[:] = state["dof_damping"]
    # Legacy recordings have no controller offsets.
    e._joint_target_offset = np.asarray(
        state.get("joint_target_offset", np.zeros(6)), dtype=np.float32
    ).copy()
    if len(state["buttons"]):
        e._cur_button_states = state["buttons"].copy()
        e._target_button_states = state["button_goals"].copy()
        if hasattr(e, "_target_drawer_pos"):
            e._target_drawer_pos = float(state["drawer_goal"])
            e._target_window_pos = float(state["window_goal"])
        e._apply_button_states()
    mujoco.mj_forward(e._model, e._data)
    e.pre_step()
    e._reset_next_step = False


def task_description(env):
    e = env.unwrapped
    goal = copy.deepcopy(e.cur_task_info)
    if hasattr(e, "_target_drawer_pos") and goal["task_name"] == "task1_open":
        return "Open the drawer and window. Leave the cube in place.", goal
    colors = ("red", "blue", "orange", "green", "yellow", "purple", "magenta", "gray")
    if goal["task_name"] == "task5_stack" and getattr(e, "_num_cubes", 0) == 2:
        heights = e._data.mocap_pos[e._cube_target_mocap_ids, 2]
        lower, upper = np.argsort(heights)
        return (
            f"Move the {colors[lower]} cube to the center of the workspace and stack the {colors[upper]} cube on it.",
            goal,
        )
    if hasattr(e, "_cube_target_mocap_ids"):
        targets = e._data.mocap_pos[e._cube_target_mocap_ids]
        parts = [
            f"Place the {colors[i]} cube at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) meters."
            for i, p in enumerate(targets)
        ]
    else:
        parts = []
    if hasattr(e, "_target_button_states"):
        parts.append(
            "Set the buttons in row-major order to "
            + ", ".join(str(int(x)) for x in e._target_button_states)
            + "."
        )
    if hasattr(e, "_target_drawer_pos"):
        parts.append(
            f"Set the drawer to {e._target_drawer_pos:.3f} meters and the window to {e._target_window_pos:.3f} meters."
        )
    return " ".join(parts), goal


def puzzle_toggle(rows, cols):
    """Each row gives the buttons toggled by pressing that row's button."""
    n = rows * cols
    toggle = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        x, y = divmod(i, cols)
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = x + dx, y + dy
            if 0 <= a < rows and 0 <= b < cols:
                toggle[i, a * cols + b] = 1
    return toggle


def solve_puzzle(current, goal, rows, cols):
    """Solve OGBench's binary neighbor-toggle puzzle over GF(2)."""
    n = rows * cols
    matrix = np.zeros((n, n + 1), dtype=np.uint8)
    matrix[:, :-1] = puzzle_toggle(rows, cols).T
    matrix[:, -1] = np.asarray(current, dtype=np.uint8) ^ np.asarray(
        goal, dtype=np.uint8
    )
    pivots = []
    r = 0
    for col in range(n):
        matches = np.flatnonzero(matrix[r:, col])
        if not len(matches):
            continue
        pivot = r + matches[0]
        matrix[[r, pivot]] = matrix[[pivot, r]]
        for other in range(n):
            if other != r and matrix[other, col]:
                matrix[other] ^= matrix[r]
        pivots.append(col)
        r += 1
        if r == n:
            break
    if np.any((matrix[:, :-1].sum(axis=1) == 0) & (matrix[:, -1] != 0)):
        raise ValueError("Puzzle goal is unreachable")
    answer = np.zeros(n, dtype=np.uint8)
    answer[pivots] = matrix[:r, -1]
    return np.flatnonzero(answer).tolist()
