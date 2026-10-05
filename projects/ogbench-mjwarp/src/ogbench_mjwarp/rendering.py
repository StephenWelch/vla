"""Shared MJWarp cameras and a strict image provenance contract."""

import copy
import hashlib
import importlib.metadata
import json
import weakref
from pathlib import Path

import mujoco_warp as mjw
import numpy as np
import warp as wp

from .tasks import image_shape


def rendering_profile(model, size):
    # Geometry, assets, and optics are static; colors and damping are task state.
    digest = hashlib.sha256()
    for name in (
        "names",
        "body_parentid",
        "body_pos",
        "body_quat",
        "jnt_type",
        "jnt_pos",
        "jnt_axis",
        "geom_type",
        "geom_group",
        "geom_bodyid",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_matid",
        "geom_dataid",
        "mesh_vert",
        "mesh_face",
        "mesh_texcoord",
        "tex_data",
        "mat_texid",
        "mat_texrepeat",
        "cam_bodyid",
        "cam_pos",
        "cam_quat",
        "cam_fovy",
        "light_pos",
        "light_dir",
    ):
        value = getattr(model, name)
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
            # Wrist optics are derived by frame transforms; ignore roundoff.
            value = np.round(value, decimals=10)
            value[value == 0] = 0  # Canonicalize signed zero.
        digest.update(name.encode())
        digest.update(
            value if isinstance(value, bytes) else np.asarray(value).tobytes()
        )
    identity = digest.hexdigest()
    rgba = model.geom_rgba.copy()
    default = np.all(rgba == [0.5, 0.5, 0.5, 1.0], axis=1)
    material = (model.geom_matid >= 0) & default
    rgba[material] = model.mat_rgba[model.geom_matid[material]]
    return {
        "backend": "mujoco-warp",
        "revision": 2,
        "visibility": "groups-0-1-2-nonzero-alpha",
        "hidden_geoms": np.flatnonzero(rgba[:, 3] == 0).tolist(),
        "clip_meters": [
            float(model.vis.map.znear * model.stat.extent),
            float(model.vis.map.zfar * model.stat.extent),
        ],
        "cameras": ["front", "ur5e/wrist"],
        "resolution": list(image_shape(size)),
        "model_sha256": identity,
        "settings": {
            "use_shadows": False,
            "use_textures": True,
            "use_ambient_lighting": True,
            "samples_per_pixel": 1,
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mujoco", "mujoco-warp", "warp-lang")
        },
    }


def dataset_profile(root, checkpoint=None):
    """Reject old OGBench images while allowing unrelated LeRobot datasets."""
    path = Path(root) / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    if not manifest.get("format", "").startswith("ogbench-mjwarp-"):
        return None
    if manifest["format"] != "ogbench-mjwarp-2" or not manifest.get("rendering"):
        raise ValueError("Legacy OGBench dataset: collect a fresh MJWarp v2 dataset")
    profile = manifest["rendering"]
    if profile.get("revision") != 2:
        raise ValueError(
            "OGBench images use a faulty renderer; collect a fresh dataset"
        )
    if checkpoint is not None:
        saved = Path(checkpoint) / "rendering.json"
        if not saved.exists() or json.loads(saved.read_text()) != profile:
            raise ValueError("Checkpoint and dataset rendering profiles differ")
    return profile


class BatchRenderer:
    def __init__(self, sim):
        self.sim = weakref.proxy(sim)
        model = sim.host_model
        self.height, self.width = (
            int(sim.base._render_height),
            int(sim.base._render_width),
        )
        self.profile = rendering_profile(model, (self.height, self.width))
        self.geom = sim.tensor(np.repeat(model.geom_rgba[None], sim.worlds, axis=0))
        self.material = sim.tensor(np.repeat(model.mat_rgba[None], sim.worlds, axis=0))
        sim.model.geom_rgba = wp.from_torch(self.geom, dtype=wp.vec4)
        sim.model.mat_rgba = wp.from_torch(self.material, dtype=wp.vec4)
        self.button_ids, self.button_indices = [], []
        for button, ids in enumerate(getattr(sim.base, "_button_geom_ids_list", [])):
            self.button_ids.extend(ids)
            self.button_indices.extend([button] * len(ids))
        if self.button_ids:
            self.colors = sim.tensor(
                np.stack(
                    [
                        sim.base._colors["red"],
                        sim.base._colors["blue" if sim.is_puzzle else "white"],
                    ]
                )
            )
        self.handle_ids = (
            [model.material(name).id for name in ("drawer_handle", "window_handle")]
            if sim.is_scene
            else []
        )
        with wp.ScopedDevice(sim.warp_device), wp.ScopedStream(sim.stream):
            # MJWarp ray tracing ignores alpha: exclude invisible goal markers
            # using a render-only copy, leaving the physics model untouched.
            render_model = copy.copy(model)
            render_model.geom_group[self.profile["hidden_geoms"]] = 5
            self.context = mjw.create_render_context(
                render_model,
                nworld=sim.worlds,
                cam_res=(self.width, self.height),
                cam_active=self.profile["cameras"],
                render_rgb=True,
                **self.profile["settings"],
            )

    def render(self):
        sim = self.sim
        if self.button_ids:
            colors = self.colors[sim.fields["buttons"].long()]
            self.geom[:, self.button_ids] = colors[:, self.button_indices]
            if self.handle_ids:
                self.material[:, self.handle_ids] = colors[:, :2]
        with wp.ScopedDevice(sim.warp_device), wp.ScopedStream(sim.stream):
            mjw.camlight(sim.model, sim.data)
            # render() traces against cached bounds; it does not update them.
            mjw.refit_bvh(sim.model, sim.data, self.context)
            mjw.render(sim.model, sim.data, self.context)
        # One transfer for both cameras; packed pixels are B,G,R,A bytes.
        sim.torch_stream.synchronize()
        packed = self.context.rgb_data.numpy().reshape(
            sim.worlds, 2, self.height, self.width
        )
        rgb = np.stack([(packed >> shift).astype(np.uint8) for shift in (16, 8, 0)], -1)
        return {view: rgb[:, i] for i, view in enumerate(("front", "wrist"))}
