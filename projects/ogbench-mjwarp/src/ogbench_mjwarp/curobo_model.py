"""Derive cuRobo kinematics and conservative collision spheres from MuJoCo."""

import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def spheres_for_geom(model, geom, spacing=0.015):
    """Cover physical primitive/mesh bounds; centers are in the geom frame."""
    kind, size = int(model.geom_type[geom]), model.geom_size[geom]
    if kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        radius, half = size[:2]
        z = np.linspace(-half, half, max(2, int(2 * half / spacing) + 1))
        radius = float(np.hypot(radius, (z[1] - z[0]) / 2))
        return np.array([[0, 0, value, radius] for value in z])
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.array([[0, 0, 0, size[0]]])
    center = np.zeros(3)
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start : start + count]
        lower, upper = vertices.min(0), vertices.max(0)
        center, size = (lower + upper) / 2, (upper - lower) / 2
    elif kind != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError(f"Unsupported robot collision geom {kind}")
    counts = np.maximum(1, np.ceil(2 * size / spacing).astype(int))
    cell = 2 * size / counts
    points = (
        np.stack(
            np.meshgrid(
                *[
                    np.linspace(-extent + step / 2, extent - step / 2, count)
                    for extent, step, count in zip(size, cell, counts, strict=True)
                ],
                indexing="ij",
            ),
            -1,
        ).reshape(-1, 3)
        + center
    )
    return np.column_stack((points, np.full(len(points), np.linalg.norm(cell) / 2)))


def robot_config(base, directory, config):
    """Export the exact six-joint tree and cache its generated URDF by content."""
    model, data = base._model, base._data
    names = [model.joint(int(i)).name for i in base._arm_joint_ids]
    arm_bodies = {int(model.jnt_bodyid[i]): int(i) for i in base._arm_joint_ids}
    wrist = int(model.body("ur5e/wrist_3_link").id)
    bodies = [i for i in range(1, wrist + 1) if model.body(i).name.startswith("ur5e/")]
    link = lambda i: "world" if i == 0 else model.body(i).name.replace("/", "_")
    root = ET.Element("robot", name="ogbench_ur5e")
    ET.SubElement(root, "link", name="world")
    spheres = {}
    for body in bodies:
        ET.SubElement(root, "link", name=link(body))
        joint_id = arm_bodies.get(body)
        joint = ET.SubElement(
            root,
            "joint",
            name=names[list(base._arm_joint_ids).index(joint_id)]
            if joint_id is not None
            else f"fixed_{body}",
            type="revolute" if joint_id is not None else "fixed",
        )
        ET.SubElement(joint, "parent", link=link(int(model.body_parentid[body])))
        ET.SubElement(joint, "child", link=link(body))
        quat = model.body_quat[body]
        rpy = Rotation.from_quat([*quat[1:], quat[0]]).as_euler("xyz")
        ET.SubElement(
            joint,
            "origin",
            xyz=" ".join(map(str, model.body_pos[body])),
            rpy=" ".join(map(str, rpy)),
        )
        if joint_id is not None:
            if (
                np.linalg.norm(model.jnt_pos[joint_id]) > 1e-9
                or model.qpos0[model.jnt_qposadr[joint_id]] != 0
            ):
                raise ValueError(
                    "URDF exporter requires the pinned zero-pivot UR5e joints"
                )
            ET.SubElement(
                joint, "axis", xyz=" ".join(map(str, model.jnt_axis[joint_id]))
            )
            bounds = model.actuator_ctrlrange[
                base._arm_actuator_ids[list(base._arm_joint_ids).index(joint_id)]
            ]
            ET.SubElement(
                joint,
                "limit",
                lower=str(bounds[0]),
                upper=str(bounds[1]),
                velocity=str(config.max_velocity),
                effort="150",
            )
        if body in arm_bodies:
            points = []
            for geom in np.flatnonzero(
                (model.geom_bodyid == body) & (model.geom_contype != 0)
            ):
                values = spheres_for_geom(model, geom)
                q = model.geom_quat[geom]
                values[:, :3] = (
                    Rotation.from_quat([*q[1:], q[0]]).apply(values[:, :3])
                    + model.geom_pos[geom]
                )
                points.extend(
                    {"center": v[:3].tolist(), "radius": float(v[3])} for v in values
                )
            spheres[link(body)] = points
    # Pinch frame is fixed to the gripper base, independent of jaw opening.
    wrist_rot = data.xmat[wrist].reshape(3, 3)
    tool_pos = wrist_rot.T @ (data.site_xpos[base._pinch_site_id] - data.xpos[wrist])
    tool_rot = wrist_rot.T @ data.site_xmat[base._pinch_site_id].reshape(3, 3)
    for name, position, rotation in [
        ("pinch", tool_pos, tool_rot),
        ("gripper", np.zeros(3), np.eye(3)),
    ]:
        ET.SubElement(root, "link", name=name)
        joint = ET.SubElement(root, "joint", name=f"fixed_{name}", type="fixed")
        ET.SubElement(joint, "parent", link=link(wrist))
        ET.SubElement(joint, "child", link=name)
        ET.SubElement(
            joint,
            "origin",
            xyz=" ".join(map(str, position)),
            rpy=" ".join(map(str, Rotation.from_matrix(rotation).as_euler("xyz"))),
        )
    # Gripper spheres are updated from actual jaw geometry before each planning call.
    grip = []
    grip_local = []
    for geom in range(model.ngeom):
        if model.geom_contype[geom] and model.body(
            int(model.geom_bodyid[geom])
        ).name.startswith("ur5e/robotiq/"):
            values = spheres_for_geom(model, geom)
            grip_local.append((geom, values.copy()))
            positions = (
                data.geom_xpos[geom]
                + values[:, :3] @ data.geom_xmat[geom].reshape(3, 3).T
                - data.xpos[wrist]
            ) @ wrist_rot
            grip.extend(
                {"center": p.tolist(), "radius": float(v[3])}
                for p, v in zip(positions, values, strict=True)
            )
    spheres["gripper"] = grip
    raw = ET.tostring(root)
    digest = hashlib.sha256(raw).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"ur5e-{digest[:16]}.urdf"
    if not path.exists():
        path.write_bytes(raw)
    collision_links = list(spheres) + ["attached_object"]
    ignore = {
        name: [link(int(model.body_parentid[body]))]
        for body, name in ((i, link(i)) for i in arm_bodies)
    }
    # Wrist/gripper/held cube belong to one assembly; all other self collisions stay enabled.
    assembly = [link(wrist), "gripper", "attached_object"]
    for name in assembly:
        ignore.setdefault(name, []).extend(other for other in assembly if other != name)
    # The fixed gripper housing covers overlap the immediately neighboring wrist
    # joint by 4.4 mm. This is joint-assembly clearance, not a moving link collision.
    ignore["gripper"].append("ur5e_wrist_2_link")
    cfg = {
        "robot_cfg": {
            "kinematics": {
                "format_version": 2.0,
                "urdf_path": str(path.resolve()),
                "base_link": "world",
                "tool_frames": ["pinch"],
                "collision_link_names": collision_links,
                "collision_spheres": spheres,
                "collision_sphere_buffer": 0.0,
                "self_collision_ignore": ignore,
                "self_collision_buffer": {},
                "extra_collision_spheres": {"attached_object": 8},
                "extra_links": {
                    "attached_object": {
                        "parent_link_name": "pinch",
                        "link_name": "attached_object",
                        "joint_name": "attach_joint",
                        "joint_type": "FIXED",
                        "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
                    }
                },
                "cspace": {
                    "joint_names": names,
                    "default_joint_position": data.qpos[
                        model.jnt_qposadr[base._arm_joint_ids]
                    ].tolist(),
                    "cspace_distance_weight": [1] * 6,
                    "null_space_weight": [1] * 6,
                    "max_acceleration": config.max_acceleration,
                    "max_jerk": 50.0,
                },
            }
        }
    }
    return cfg, grip_local, digest
