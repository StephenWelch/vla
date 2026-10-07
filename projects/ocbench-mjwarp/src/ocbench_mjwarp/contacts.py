"""Shared robot/environment contact diagnostics."""

import numpy as np
import warp as wp


def contact_roles(model):
    """Environment=0, arm=1, gripper linkage=2, finger pad=3, button=4."""
    roles = np.zeros(model.ngeom, dtype=np.int32)
    for geom in range(model.ngeom):
        body = model.body(model.geom_bodyid[geom]).name
        if body.startswith("ur5e/robotiq/"):
            roles[geom] = 3 if body.endswith(("/left_pad", "/right_pad")) else 2
        elif body.startswith("ur5e/"):
            roles[geom] = 1
        elif "button" in body:
            roles[geom] = 4
    return roles


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
