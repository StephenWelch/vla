from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from ocbench_mjwarp.collect import stable_stack


@pytest.mark.parametrize("half_size", [0.02, 0.03])
def test_stable_stack_uses_model_dimensions(half_size):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body name="ur5e/gripper" pos="2 0 1"><joint name="grip" type="slide"/><geom size=".01"/></body>
      <body pos="0 0 {half_size}"><freejoint/><geom name="object_0" type="box" size="{half_size} {half_size} {half_size}"/></body>
      <body pos="0 0 {3 * half_size - 0.0001}"><freejoint/><geom name="object_1" type="box" size="{half_size} {half_size} {half_size}"/></body>
    </worldbody></mujoco>''')
    sim = SimpleNamespace(
        host_model=model, env=SimpleNamespace(_gripper_opening_joint_id=0)
    )
    q = np.repeat(model.qpos0[None], 51, axis=0)
    assert stable_stack(sim, {"sim/qpos": q})["valid"]
    upper = model.jnt_qposadr[2]
    gap = q.copy()
    gap[:, upper + 2] += 0.01
    result = stable_stack(sim, {"sim/qpos": gap})
    assert "stack_alignment" in result["failures"]
    assert "missing_cube_support" in result["failures"]
    held = q.copy()
    held[:, 0] = 0.5
    assert "gripper_not_released" in stable_stack(sim, {"sim/qpos": held})["failures"]
    moving = q.copy()
    moving[0, upper] += 0.005
    assert "unstable_stack" in stable_stack(sim, {"sim/qpos": moving})["failures"]
    assert (
        "insufficient_stability_window"
        in stable_stack(sim, {"sim/qpos": q[:2]})["failures"]
    )
