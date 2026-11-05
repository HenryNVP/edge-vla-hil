#!/usr/bin/env python3
"""Interp A: appearance shifts for robosuite Lift, applied to the sim's model at run time.

Render-only (physics untouched), so the same state can be rendered with and without the shift:
the policy's action on the unshifted render is an exact, free label for repairing the shifted
policy (interp_collect --selflabel cf). Spec: ';'-separated items
  light=s            scale the scene light's and the headlight's diffuse by s
  table=r,g,b        tint the table top material (multiplies its texture)
  cube=r,g,b         tint the cube material
  walls=r,g,b        tint the walls material (background)
  cam=dx,dy,dz,yaw   move the agentview camera (m) and turn it about world z (deg)
  wcam=dx,dy,dz,yaw  the same for the wrist camera, in the hand's frame
  roll=deg / wroll=deg   turn the agentview / wrist camera about its viewing axis
  fov=deg / wfov=deg     set the agentview / wrist camera's vertical field of view
  cubealpha=a        cube opacity in every camera (0: invisible; physics unchanged)
robosuite rebuilds the model on every reset (hard_reset), so call on() again after each reset.

    python3 scripts/interp_nuisance.py --spec 'cube=0.2,0.4,1' --out outputs/interp/nuisance/cube.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MATERIALS = {'table': 'table_ceramic', 'cube': 'cube_redwood_mat', 'walls': 'walls_mat'}


def parse(spec: str) -> list[tuple[str, np.ndarray]]:
    out = []
    for item in filter(None, (s.strip() for s in spec.split(';'))):
        key, val = item.split('=')
        out.append((key.strip(), np.array([float(v) for v in val.split(',')])))
    return out


def yaw_quat(q: np.ndarray, deg: float) -> np.ndarray:
    """wxyz quaternion q turned by deg about world z: rz * q."""
    h = np.radians(deg) / 2
    c, s = np.cos(h), np.sin(h)
    w, x, y, z = q
    return np.array([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w])


def roll_quat(q: np.ndarray, deg: float) -> np.ndarray:
    """wxyz quaternion q turned by deg about its own z (a camera's viewing axis): q * rz."""
    h = np.radians(deg) / 2
    c, s = np.cos(h), np.sin(h)
    w, x, y, z = q
    return np.array([w * c - z * s, x * c + y * s, y * c - x * s, z * c + w * s])


class Nuisance:
    """on(env) applies the spec (remembering what it overwrote), off(env) restores it."""

    def __init__(self, spec: str) -> None:
        self.items, self.saved = parse(spec), []

    def on(self, env) -> None:
        import mujoco
        m = env.sim.model._model
        self.saved = []
        for key, v in self.items:
            if key == 'light':
                self.saved += [('light_diffuse', None, m.light_diffuse.copy()),
                               ('headlight', None, m.vis.headlight.diffuse.copy())]
                m.light_diffuse[:] *= v[0]
                m.vis.headlight.diffuse[:] = m.vis.headlight.diffuse * v[0]
            elif key in MATERIALS:
                i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, MATERIALS[key])
                self.saved.append(('mat_rgba', i, m.mat_rgba[i].copy()))
                m.mat_rgba[i, :3] = v
            elif key == 'cubealpha':                    # 0 = the cube vanishes from every camera
                i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, MATERIALS['cube'])
                g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'cube_g0')
                self.saved += [('mat_rgba', i, m.mat_rgba[i].copy()), ('geom_rgba', g, m.geom_rgba[g].copy())]
                m.mat_rgba[i, 3] = v[0]
                m.geom_rgba[g, 3] = min(m.geom_rgba[g, 3], v[0])
            elif key in ('roll', 'wroll', 'fov', 'wfov'):
                name = 'agentview' if key in ('roll', 'fov') else 'robot0_eye_in_hand'
                i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
                if key.endswith('fov'):
                    self.saved.append(('cam_fovy', i, m.cam_fovy[i].copy()))
                    m.cam_fovy[i] = v[0]
                else:                                   # about the camera's own viewing axis
                    self.saved.append(('cam_quat', i, m.cam_quat[i].copy()))
                    m.cam_quat[i] = roll_quat(m.cam_quat[i], v[0])
            elif key in ('cam', 'wcam'):
                name = 'agentview' if key == 'cam' else 'robot0_eye_in_hand'
                i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
                self.saved += [('cam_pos', i, m.cam_pos[i].copy()), ('cam_quat', i, m.cam_quat[i].copy())]
                m.cam_pos[i] += v[:3]
                m.cam_quat[i] = yaw_quat(m.cam_quat[i], v[3] if len(v) > 3 else 0.0)
            else:
                raise ValueError(f'unknown nuisance {key!r}')
        env.sim.forward()

    def off(self, env) -> None:
        m = env.sim.model._model
        for name, i, v in reversed(self.saved):
            if name == 'headlight':
                m.vis.headlight.diffuse[:] = v
            elif i is None:
                getattr(m, name)[:] = v
            else:
                getattr(m, name)[i] = v
        self.saved = []
        env.sim.forward()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--spec', required=True)
    p.add_argument('--out', required=True, help='png: agentview and wrist, unshifted | shifted')
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from dp_anchor_probe import observe
    from interp_gainmap import TASKS, make_env
    env, _ = make_env(TASKS['lift'])
    n = Nuisance(args.spec)
    views = [observe(env)]
    n.on(env)
    views.append(observe(env))
    n.off(env)
    restored = observe(env)
    assert np.array_equal(restored['agentview_image'], views[0]['agentview_image']), 'off() must restore'
    fig, axes = plt.subplots(2, 2, figsize=(4, 4))
    for c, o in enumerate(views):
        for r, cam in enumerate(('agentview_image', 'robot0_eye_in_hand_image')):
            axes[r, c].imshow(np.flipud(o[cam]))
            axes[r, c].axis('off')
    axes[0, 0].set_title('original', fontsize=8)
    axes[0, 1].set_title(args.spec, fontsize=8)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    env.close()


if __name__ == '__main__':
    main()
