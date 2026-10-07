"""Training action views and absolute position execution."""

import numpy as np
import warp as wp

from .config import ACTION, ARM_LIMITS


def absolute_targets(action, state, absolute_arm=False):
    """Reconstruct commanded gripper/arm targets, not measured next positions."""
    result = np.asarray(action, dtype=np.float32).copy()
    if absolute_arm:
        limits = np.asarray(ARM_LIMITS, dtype=np.float32)
        result[..., :6] = np.clip(
            np.asarray(state, dtype=np.float32)[..., :6]
            + np.clip(result[..., :6], -1, 1)
            * np.asarray(ACTION["scales"][:6], dtype=np.float32),
            -limits,
            limits,
        )
    result[..., 6] = np.clip(
        np.asarray(state)[..., 16] + 0.12 * np.clip(result[..., 6], -1, 1), 0, 1
    )
    return result


def prepare_training_views(
    datasets, stats, absolute_gripper, percentiles, absolute_arm=False
):
    """Transform per-frame targets before chunk lookup and fit only train frames."""
    import pyarrow as pa
    from lerobot.datasets.io_utils import hf_transform_to_torch

    for dataset in datasets.values():
        if absolute_gripper:
            table = dataset.hf_dataset.with_format(None)
            actions = absolute_targets(
                table["action"], table["observation.state"], absolute_arm
            )
            # Replace in-memory Arrow data; source parquet and videos stay immutable.
            arrow = table.data.table
            index = arrow.schema.get_field_index("action")
            arrow = arrow.set_column(
                index,
                arrow.schema.field(index),
                pa.array(actions.tolist(), type=arrow.schema.field(index).type),
            )
            from datasets import Dataset

            dataset.reader.hf_dataset = Dataset(arrow, info=table.info)
            dataset.reader.hf_dataset.set_transform(hf_transform_to_torch)
    train = datasets["train"].hf_dataset.with_format(None)
    for key in ("action", "observation.state"):
        values = np.asarray(train[key], dtype=np.float32)
        if key == "action" and absolute_gripper:
            for name, function in (
                ("mean", np.mean),
                ("std", np.std),
                ("min", np.min),
                ("max", np.max),
            ):
                stats[key][name] = function(values, axis=0)
        if percentiles:
            low, high = np.quantile(values, [0.01, 0.99], axis=0)
            # A constant feature must not amplify tiny errors by dividing by epsilon.
            high = np.where(high - low < 1e-6, low + 1, high)
            if key == "action" and absolute_gripper:
                low[6], high[6] = 0, 1
            stats[key].update(q01=low, q99=high)


@wp.kernel
def set_absolute_gripper(
    action: wp.array2d(dtype=float),
    done: wp.array(dtype=wp.int32),
    actuator_ids: wp.array(dtype=int),
    ctrl: wp.array2d(dtype=float),
    count: int,
):
    world = wp.tid()
    if done[world] == 0:
        for i in range(count):
            ctrl[world, actuator_ids[i]] = 255.0 * wp.clamp(action[world, 6], 0.0, 1.0)


@wp.kernel
def set_absolute_arm(
    action: wp.array2d(dtype=float),
    done: wp.array(dtype=int),
    ids: wp.array(dtype=int),
    low: wp.array(dtype=float),
    high: wp.array(dtype=float),
    ctrl: wp.array2d(dtype=float),
):
    world = wp.tid()
    if done[world] == 0:
        for joint in range(6):
            ctrl[world, ids[joint]] = wp.clamp(
                action[world, joint], low[joint], high[joint]
            )


def step_absolute(env, action, done, absolute_arm=False):
    """Apply direct position targets to selected actuators, then native physics."""
    env._before_step_joint_actions_gpu()
    env.set_joint_action_control_gpu(action, done)
    if absolute_arm:
        wp.launch(
            set_absolute_arm,
            dim=env._nworld,
            inputs=[
                action,
                done,
                env._gpu_arm_actuator_ids,
                env._gpu_arm_ctrl_low,
                env._gpu_arm_ctrl_high,
                env._data.ctrl,
            ],
            device=env._data.qpos.device,
        )
    wp.launch(
        set_absolute_gripper,
        dim=env._nworld,
        inputs=[
            action,
            done,
            env._gpu_gripper_actuator_ids,
            env._data.ctrl,
            len(env._gripper_actuator_ids),
        ],
        device=env._data.qpos.device,
    )
    env.advance_physics_gpu()
    env.compute_success_gpu()
