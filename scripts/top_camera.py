"""A close top-down camera over the Square workcell, for the BEV query planner (paper 2).

robosuite's own 'birdview' looks down from 3 m: at 84 px the table is a third of the frame and
the nut a few pixels. This moves that camera to 0.9 m above the table centre (fovy 45: ~0.75 m
of table across 84 px, ~9 mm per pixel, the nut ~11 px). NutAssembly hard-resets, i.e. reloads
the model on every reset, so `apply` must run after each reset; it re-renders the observation.

`matrices` gives each camera's 3x4 world -> pixel projection (robosuite camera_utils), which the
BEV spatial cross-attention uses to place table-plane reference points in each image. Its pixel
rows already match sil_step.frames' upright (flipud) images; checked by overlaying the projected
nut on both views.
"""
from __future__ import annotations

import numpy as np

CAMERA = 'birdview'
POS = (0.0, 0.0, 1.72)            # table top is at z = 0.82
FOVY = 45.0


def apply(env) -> dict:
    """Move the top camera on a freshly reset env; returns the re-rendered observation."""
    m = env.sim.model
    i = m.camera_name2id(CAMERA)
    m.cam_pos[i] = POS
    m.cam_fovy[i] = FOVY
    env.sim.forward()
    return env._get_observations(force_update=True)


def matrices(env, cams, size: int) -> dict[str, np.ndarray]:
    """{camera: [3, 4]} world -> (col, row, 1) * depth for upright size x size images."""
    from robosuite.utils.camera_utils import get_camera_transform_matrix
    return {c: get_camera_transform_matrix(env.sim, c, size, size)[:3] for c in cams}


def project(P: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """[..., 3] world points -> [..., 2] (col, row) pixels under a [3, 4] projection."""
    h = np.concatenate([xyz, np.ones(xyz.shape[:-1] + (1,))], -1) @ P.T
    return h[..., :2] / h[..., 2:3]
