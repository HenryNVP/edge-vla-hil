#!/usr/bin/env python3
"""Interp gain map: DP's instantaneous response to where the nut is, as a control surface.

Operating points: each scene is rolled out synchronously with seeded noise until the arm is
within each --radii of the nut's handle (gripper still open); the sim is frozen there. The nut
is then moved over a --grid of (dx, dy) offsets, one chunk is predicted per offset with the SAME
noise as the unmoved baseline (common random numbers), and the sim is restored. Both history
frames show the moved scene (as in dp_anchor_probe), so the policy never sees the nut jump.

Response: the chunk's last open-gripper target xy minus the baseline's (mm). Since the handle
moves with the nut, perfect tracking is response = offset; the local gain is the 2x2 Jacobian
d(response)/d(offset), by central differences over the grid.

    python3 scripts/interp_gainmap.py --out outputs/interp/gainmap.npz --scenes 10
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sil_step  # noqa: E402
from dp_anchor_probe import observe, predict, set_object  # noqa: E402
from fuzzy_gate import _data_arrays  # noqa: E402
from oracle_square import handle_pos  # noqa: E402

# per task: env, DP checkpoint, object keys and joint, the demos' xy range of the object at
# reset (robosuite placement ranges), and the grasp point the response is measured at
TASKS = {
    'square': dict(env='NutAssemblySquare', ckpt='dp_square_ph_image_cnn.ckpt',
                   pos='SquareNut_pos', quat='SquareNut_quat', joint='SquareNut_joint0',
                   lo=(-0.115, 0.11), hi=(-0.11, 0.225),
                   grasp=lambda o: handle_pos(o['SquareNut_pos'], o['SquareNut_quat'])[:2]),
    'lift': dict(env='Lift', ckpt='dp_lift_ph_image_cnn.ckpt', pos='cube_pos', quat='cube_quat',
                 joint='cube_joint0', lo=(-0.03, -0.03), hi=(0.03, 0.03),
                 grasp=lambda o: np.asarray(o['cube_pos'][:2])),
}


def target(chunk: np.ndarray) -> np.ndarray:
    """The last open-gripper target of a chunk (xy); the whole chunk's last entry if it closes
    from the start."""
    open_ = np.flatnonzero(chunk[:, 6] <= 0)
    return chunk[open_[-1] if len(open_) else -1, :2]


def jacobians(resp: np.ndarray, step: float) -> np.ndarray:
    """resp[i, j] = response (2,) at offset (grid[i], grid[j]) = (dx, dy) -> J[i, j] (2x2) on
    the interior, by central differences; columns are d/ddx, d/ddy."""
    jx = (resp[2:, 1:-1] - resp[:-2, 1:-1]) / (2 * step)
    jy = (resp[1:-1, 2:] - resp[1:-1, :-2]) / (2 * step)
    return np.stack([jx, jy], axis=-1)


def operating_points(env, obs, policy, radii, scene: int, task=TASKS['square']):
    """Roll one scene out synchronously with seeded noise and yield (obs, radius, tick) each
    time the eef first comes within a radius of the grasp point (xy), gripper still open.
    The caller may probe the sim at a yield but must leave it as it found it."""
    hist = deque([sil_step.frames(obs)] * policy.n_obs_steps, maxlen=policy.n_obs_steps)
    radii = sorted(radii, reverse=True)
    chunk, i, ri = None, 0, 0
    for t in range(400):
        gap = np.linalg.norm(task['grasp'](obs) - obs['robot0_eef_pos'][:2])
        while ri < len(radii) and gap < radii[ri]:
            if chunk is not None and chunk[min(i, len(chunk) - 1), 6] > 0:
                return                               # already closing: nothing left to probe
            yield obs, radii[ri], t
            ri += 1
        if ri == len(radii):
            return
        if chunk is None or i >= len(chunk):
            chunk, i = predict(policy, hist, 10_000 + scene * 100 + t), 0
        obs, *_ = env.step(chunk[i])
        i += 1
        hist.append(sil_step.frames(obs))


def make_env(task=TASKS['square']):
    from evh_plant.env_factory import EnvSpec, build_env
    built = build_env(EnvSpec(env_name=task['env'], action_hz=sil_step.CONTROL_HZ,
                              max_episode_s=21.0, seed=0, absolute_actions=True))
    return built.env, built.obs


def collect(args) -> dict:
    from evh_controller.policy import make_policy
    task = TASKS[args.task]
    weights = args.weights or os.path.join(sil_step._REPO, 'checkpoints', task['ckpt'])
    policy = make_policy(args.backend, weights, denoise_steps=args.denoise_steps)
    env, obs = make_env(task)
    grid = np.linspace(-args.extent, args.extent, args.grid)
    radii = [float(r) for r in args.radii.split(',')]
    out = {k: [] for k in ('resp', 'scene', 'radius', 'nut_xy', 'eef_xy', 'tick')}
    for scene in range(args.scenes):
        probed = []
        for o, r, t in operating_points(env, obs, policy, radii, scene, task):
            out['resp'].append(probe(env, policy, o, grid, 1000 + scene, task))
            out['scene'].append(scene)
            out['radius'].append(r)
            out['nut_xy'].append(np.asarray(o[task['pos']][:2]))
            out['eef_xy'].append(np.asarray(o['robot0_eef_pos'][:2]))
            out['tick'].append(t)
            probed.append(r)
        print(f'scene {scene}: probed radii {probed}', flush=True)
        obs = env.reset()
    env.close()
    return {**{k: np.array(v) for k, v in out.items()}, 'grid': grid, 'task': args.task}


def probe(env, policy, o, grid, seed: int, task=TASKS['square']) -> np.ndarray:
    """Response (mm) of the chunk's grasp target to each nut offset; every mjData array is
    restored after each (fuzzy_gate's bitwise restore), so the warm-up carries on unchanged."""
    d = env.sim.data
    saved = _data_arrays(d)
    xy0 = np.array(o[task['pos']][:2], float)

    def chunk_at(dx, dy):
        set_object(env, task['joint'], xy0 + [dx, dy])
        cur = sil_step.frames(observe(env))
        c = predict(policy, deque([cur] * policy.n_obs_steps), seed)
        for k, v in saved.items():
            cur = getattr(d, k)
            if cur.shape == v.shape:        # contact-sized (efc_*) arrays are rebuilt each step
                cur[...] = v
        return target(c)

    base = chunk_at(0.0, 0.0)
    resp = np.array([[(chunk_at(dx, dy) - base) * 1e3 for dy in grid] for dx in grid])
    observe(env)
    return resp


def report(d: dict) -> None:
    grid_mm = d['grid'] * 1e3
    step = grid_mm[1] - grid_mm[0]
    for r in sorted(set(d['radius']), reverse=True):
        J = np.array([jacobians(x, step) for x in d['resp'][d['radius'] == r]])
        Jm = np.median(J.reshape(-1, 2, 2), axis=0)
        sv = np.linalg.svd(J.reshape(-1, 2, 2), compute_uv=False)
        print(f'radius {r * 100:.0f} cm, {len(J)} scenes: median J = '
              f'[[{Jm[0, 0]:.2f} {Jm[0, 1]:.2f}] [{Jm[1, 0]:.2f} {Jm[1, 1]:.2f}]]  '
              f'(rows: response x, y; cols: d/dx, d/dy; identity = follows)  '
              f'gains: max {np.median(sv[:, 0]):.2f}, min {np.median(sv[:, 1]):.2f}')
        # split the interior grid points by whether the moved object is inside the demos' range
        task = TASKS[str(d.get('task', 'square'))]
        g = grid_mm[1:-1] / 1e3
        pos = d['nut_xy'][d['radius'] == r][:, None, None, :] + np.stack(
            np.meshgrid(g, g, indexing='ij'), -1)[None]
        inside = np.all((pos >= task['lo']) & (pos <= task['hi']), axis=-1)
        for name, m in (('inside', inside), ('outside', ~inside)):
            if m.any():
                Js = J[m]
                print(f'    {name:7s} demo range: {m.sum():4d} points, gain xx '
                      f'{np.median(Js[:, 0, 0]):.2f}, yy {np.median(Js[:, 1, 1]):.2f}, min sv '
                      f'{np.median(np.linalg.svd(Js, compute_uv=False)[:, 1]):.2f}')


def plot(d: dict, path: str) -> None:
    """Gain ellipses: at each interior offset, the median J maps a circle of radius R to an
    ellipse; a dashed circle of radius R is perfect tracking (J = I)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, quiet, accent, rule = '#0b0b0b', '#52514e', '#2a78d6', '#d9d8d4'
    grid_mm = d['grid'] * 1e3
    step = grid_mm[1] - grid_mm[0]
    R = 0.4 * step
    radii = sorted(set(d['radius']), reverse=True)
    fig, axes = plt.subplots(1, len(radii), figsize=(4.2 * len(radii), 4.4), sharey=True)
    th = np.linspace(0, 2 * np.pi, 64)
    circle = np.stack([np.cos(th), np.sin(th)])
    for ax, r in zip(np.atleast_1d(axes), radii):
        J = np.median(np.array([jacobians(x, step) for x in d['resp'][d['radius'] == r]]), 0)
        ax.axvspan(-2.5, 2.5, color=rule, lw=0)   # demos' x range (std 1.4 mm)
        for i, gx in enumerate(grid_mm[1:-1]):
            for j, gy in enumerate(grid_mm[1:-1]):
                ax.plot(gx + R * circle[0], gy + R * circle[1], ls=(0, (2, 2)), lw=1, color=quiet)
                e = J[i, j] @ circle * R
                ax.fill(gx + e[0], gy + e[1], color=accent, alpha=0.25, lw=0)
                ax.plot(gx + e[0], gy + e[1], color=accent, lw=2)
        ax.set_title(f'arm {r * 100:.0f} cm from the handle', color=ink, fontsize=11)
        ax.set_xlabel('nut moved in x (mm)', color=quiet)
        ax.set_aspect('equal')
        lim = grid_mm[-2] + step * 0.7
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.tick_params(colors=quiet, labelsize=9)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(rule)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel('nut moved in y (mm)', color=quiet)
    fig.suptitle('One chunk follows a moved nut in y, hardly in x (the axis the demos never vary)',
                 color=ink, fontsize=12, x=0.02, y=0.99, ha='left')
    fig.text(0.02, 0.925, 'Blue: median 2x2 gain over 20 scenes (image of the dashed circle); '
             'dashed: perfect tracking; grey band: demos\' x range', color=quiet, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=200, facecolor='#fcfcfb', bbox_inches='tight')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', required=True)
    p.add_argument('--task', default='square', choices=sorted(TASKS))
    p.add_argument('--backend', default='dp', help='make_policy backend (dp, act)')
    p.add_argument('--weights', default='', help="default: the task's DP checkpoint")
    p.add_argument('--denoise-steps', type=int, default=16)
    p.add_argument('--scenes', type=int, default=10)
    p.add_argument('--grid', type=int, default=7)
    p.add_argument('--extent', type=float, default=0.06, help='m, each side')
    p.add_argument('--radii', default='0.08,0.03', help='m, eef to grasp point, xy')
    p.add_argument('--report-only', action='store_true')
    p.add_argument('--plot', default='', help='png path for the gain-ellipse figure')
    args = p.parse_args()
    if not args.report_only:
        np.savez_compressed(args.out, **collect(args))
    d = dict(np.load(args.out))
    report(d)
    if args.plot:
        plot(d, args.plot)


if __name__ == '__main__':
    main()
