#!/usr/bin/env python3
"""Interp: the paper's numbers with 95% confidence intervals, from saved sweeps and rollouts.

  gains     action gain (gain map) and seen gain (interp_perceive's out-of-fold decode) inside /
            outside the demos' range, and the action's inside - outside drop; bootstrap over
            scenes (the decode is fitted once, so its own training variance is not included)
  swap      interp_swap's 2x2 head-state x feature-direction table, the direction effect (dir
            in - dir out, averaged over head rows) and the head effect; bootstrap over scenes
  rollouts  per object shift (Square DP; Lift DP / ACT when lift_<backend>_shift_* exist):
            success (Wilson), and where the gripper first closes: the fraction of
            the shift it followed (1 = at the real handle, 0 = where the unshifted handle would be),
            the share of episodes nearer that "ghost" target than the real one, and the cross-axis
            drift (mm); bootstrap over episodes

    python3 scripts/interp_ci.py | tee outputs/interp/ci.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interp_gainmap import TASKS, handle_pos, inside_demo_range, jacobians  # noqa: E402
from interp_perceive import seen  # noqa: E402
from interp_swap import block_gains, directions  # noqa: E402
from plot_sweep import wilson  # noqa: E402

OUT = 'outputs/interp'
ROWS = []                                   # every printed number, for --json

SWEEPS = [('square DP', 'perceive_square_dp', 'gainmap', 'swap_square_dp'),
          ('lift DP', 'perceive_lift_dp', 'gainmap_lift_dp', 'swap_lift_dp'),
          ('lift ACT', 'perceive_lift_act', 'gainmap_lift_act', '')]
# single-layer weight edits of the base arm (interp_edit, plan §11): rollout dirs
# lift_act_edit_<name>_shift_*, reported as 'lift ACT edit <name>'
EDIT_ARMS = ('x_head_r2_b1', 'x_head_r2_b1_unsh', 'y_head_r2_b1', 'x_head_r1_b1', 'x_head_r2_b1_n10',
             'x_head_r2_b1_n160', 'x_head_r2_b1_s1', 'x_cf_head_n160', 'x_dec_ff2_r4_b1',
             'transplant2', 'transplant3', 'relu_m20', 'relu_m20_f5', 'x_onpolicy_head',
             'x_onpolicy_head_r2', 'y_onpolicy_head', 'y_onpolicy_head_r2', 'sl_onpolicy_head', 'slfull_onpolicy_head', 'slfull_b3', 'slfull_b10')
# the arms re-run on fresh scenes (env seed 3) after selecting edits on seed 0: the headline numbers;
# rollout dirs lift_act_<name>_s3_shift_*, reported as 'lift ACT <name> s3'
S3_ARMS = ('base', 'xaug', 'yaug', 'edit_x_head_r2_b1_unsh', 'edit_slfull_b3', 'edit_x_head_r2_b1',
           'edit_y_head_r2_b1', 'edit_x_onpolicy_head_r2', 'edit_y_onpolicy_head_r2')
# Lift ACT retrained on demos replayed with the cube shifted along one axis (interp_augment)
AXIS_MAPS = [('lift ACT', 'gainmap_lift_act'), ('lift ACT base', 'gainmap_lift_act_base'),
             ('lift ACT x-aug', 'gainmap_lift_act_x'), ('lift ACT y-aug', 'gainmap_lift_act_y')]


def boot(groups: np.ndarray, stat, n: int = 2000, seed: int = 0):
    """stat(indices) -> array; (estimate, 2.5%, 97.5%) resampling whole groups with replacement."""
    ids = np.unique(groups)
    by = [np.flatnonzero(groups == g) for g in ids]
    rng = np.random.default_rng(seed)
    reps = np.array([stat(np.concatenate([by[k] for k in rng.integers(len(ids), size=len(ids))]))
                     for _ in range(n)])
    return stat(np.arange(len(groups))), *np.nanpercentile(reps, [2.5, 97.5], axis=0)


def fmt(est, lo, hi, i: int, f: str = '5.2f') -> str:
    return f'{est[i]:{f}} [{lo[i]:{f}}, {hi[i]:{f}}]'


def follow(close: np.ndarray, nut0: np.ndarray, shift: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Fraction of the shift the close point followed, per axis (nan on unshifted axes)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(shift != 0, (close - (nut0 - shift) - o) / shift, np.nan)


def gains(name: str, perceive: str, action: str) -> None:
    d, a = (dict(np.load(f'{OUT}/{f}.npz')) for f in (perceive, action))
    step = (d['grid'][1] - d['grid'][0]) * 1e3
    for r in sorted(set(d['radius']), reverse=True):
        m = d['radius'] == r
        Ja = np.array([jacobians(x, step) for x in a['resp'][a['radius'] == r]])
        Js = np.array([jacobians(x, step) for x in seen(d, r)])
        inside = inside_demo_range(d, r)

        def stat(idx):
            v = [np.median(J[idx][s(inside[idx])][:, k, k]) for J in (Ja, Js)
                 for s in (lambda x: x, lambda x: ~x) for k in (0, 1)]
            return np.array(v + [v[0] - v[2], v[1] - v[3]])
        e = boot(d['scene'][m], stat)
        print(f'{name}, radius {r * 100:.0f} cm ({m.sum()} points)')
        for label, i in (('action inside', 0), ('action outside', 2), ('action drop in-out', 8),
                         ('seen inside', 4), ('seen outside', 6)):
            print(f'    {label:20s} x {fmt(*e, i)}   y {fmt(*e, i + 1)}')
            record('gains', name, r, label, e, i)


def record(section, name, r, label, e, i) -> None:
    for k, ax in ((i, 'x'), (i + 1, 'y')):
        ROWS.append(dict(section=section, policy=name, radius=float(r), label=label, axis=ax,
                         est=float(e[0][k]), lo=float(e[1][k]), hi=float(e[2][k])))


def axis_parts(a: dict, r: float):
    """Lift gain map at radius r -> (J (P, n, n, 2, 2), per-axis in-demo-range mask (P, n, n, 2))."""
    step = (a['grid'][1] - a['grid'][0]) * 1e3
    lo, hi = np.array(TASKS['lift']['lo']), np.array(TASKS['lift']['hi'])
    m = a['radius'] == r
    J = np.array([jacobians(x, step) for x in a['resp'][m]])
    g = a['grid'][1:-1]
    pos = a['nut_xy'][m][:, None, None, :] + np.stack(np.meshgrid(g, g, indexing='ij'), -1)
    return J, (pos >= lo) & (pos <= hi)


def axis_stat(J: np.ndarray, inr: np.ndarray) -> np.ndarray:
    """[in x, in y, out x, out y, drop x, drop y]: median own-axis gain with both axes in range
    vs out of range along that axis only (the other in range)."""
    both, v = inr.all(-1), []
    for k in (0, 1):
        out_k = ~inr[..., k] & inr[..., 1 - k]
        v += [np.median(J[both][:, k, k]), np.median(J[out_k][:, k, k])]
    return np.array([v[0], v[2], v[1], v[3], v[0] - v[1], v[2] - v[3]])


def axis_gains(name: str, action: str) -> None:
    """Lift: each axis's gain in the original demo range vs with the cube out of it along that
    axis only (the other axis in range), and the in - out drop; bootstrap over scenes."""
    a = dict(np.load(f'{OUT}/{action}.npz'))
    for r in sorted(set(a['radius']), reverse=True):
        m = a['radius'] == r
        J, inr = axis_parts(a, r)
        e = boot(a['scene'][m], lambda idx: axis_stat(J[idx], inr[idx]))
        print(f'{name} per-axis gain, radius {r * 100:.0f} cm')
        for label, i in (('in range', 0), ('out along own axis', 2), ('drop', 4)):
            print(f'    {label:20s} x {fmt(*e, i)}   y {fmt(*e, i + 1)}')
            record('axis', name, r, label, e, i)


def swap(name: str, perceive: str, swapf: str) -> None:
    d, s = dict(np.load(f'{OUT}/{perceive}.npz')), dict(np.load(f'{OUT}/{swapf}.npz'))
    V = directions(d['feat'].astype(np.float64), (d['grid'][1] - d['grid'][0]) * 1e3)
    for r in sorted(set(d['radius']), reverse=True):
        m = d['radius'] == r
        J, Vm, inside = s['J'][m].astype(np.float64), V[m], inside_demo_range(d, r)

        def stat(idx):
            b = block_gains(J[idx], Vm[idx], inside[idx])
            cells = [b[h, k] for h in ('in', 'out') for k in ('in', 'out')]
            direction = (b['in', 'in'] + b['out', 'in'] - b['in', 'out'] - b['out', 'out']) / 2
            head = (b['in', 'in'] + b['in', 'out'] - b['out', 'in'] - b['out', 'out']) / 2
            return np.concatenate(cells + [direction, head])
        e = boot(d['scene'][m], stat)
        print(f'{name} swap, radius {r * 100:.0f} cm')
        for label, i in (('head in, dir in', 0), ('head in, dir out', 2), ('head out, dir in', 4),
                         ('head out, dir out', 6), ('direction effect', 8), ('head effect', 10)):
            print(f'    {label:20s} x {fmt(*e, i)}   y {fmt(*e, i + 1)}')
            record('swap', name, r, label, e, i)


def rollouts(dirs: str, name: str = 'square DP') -> None:
    rows = []
    for f in (f for p in dirs.split(',') for dd in sorted(glob.glob(p)) if os.path.isdir(dd)
              for f in sorted(glob.glob(os.path.join(dd, 'ep*.npz')))):
        e = np.load(f)
        s = np.zeros(3)
        if 'shift' in e:
            s[:len(e['shift'])] = e['shift']
        c = np.flatnonzero(e['commands'][:, 6] > 0)
        # Square: the gripper grasps the handle, which swings with the nut's random yaw
        grasp = (handle_pos(e['nut_pos'][0], e['nut_quat'][0]) if str(e.get('task', 'square'))
                 == 'square' else e['nut_pos'][0])
        rows.append((s, bool(e['success']), grasp[:2],
                     e['eef_pos'][c[0], :2] if len(c) else np.full(2, np.nan)))
    S = np.array([r[0] for r in rows])
    ok = np.array([r[1] for r in rows])
    nut0, close = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
    closed = ~np.isnan(close[:, 0])
    base = (np.abs(S).sum(1) == 0) & closed
    o = np.median(close[base] - nut0[base], 0)
    print(f'{name} rollouts: {len(rows)} episodes, {(~closed).sum()} never closed; unshifted '
          f'success {ok[np.abs(S).sum(1) == 0].mean():.2f} (n {(np.abs(S).sum(1) == 0).sum()})')
    print('  shift (m)      n  success [Wilson]     followed [boot]      nearer ghost  '
          'followed succ / fail  cross-axis mm')
    for s in np.unique(S[(S[:, 2] == 0) & (np.abs(S[:, :2]).sum(1) > 0)], axis=0):
        m = np.all(S == s, axis=1)
        lo, hi = wilson(int(ok[m].sum()), int(m.sum()))
        k = int(np.argmax(np.abs(s[:2])))
        mc = m & closed
        fol = follow(close[mc], nut0[mc], s[:2], o)[:, k]
        e = boot(np.arange(mc.sum()), lambda i: np.array([np.median(fol[i])]))
        real, ghost = nut0[mc] + o, nut0[mc] - s[:2] + o
        nearer = (np.linalg.norm(close[mc] - ghost, axis=1)
                  < np.linalg.norm(close[mc] - real, axis=1)).mean()
        okc = ok[mc]
        sf = ' / '.join(f'{np.median(fol[x]):5.2f}' if x.any() else '  nan' for x in (okc, ~okc))
        cross = np.median((close[mc] - real)[:, 1 - k]) * 1e3
        ROWS.append(dict(section='rollouts', policy=name, shift=s[:2].tolist(), n=int(m.sum()),
                         success=float(ok[m].mean()), success_lo=lo, success_hi=hi,
                         followed=float(e[0][0]), followed_lo=float(e[1][0]),
                         followed_hi=float(e[2][0]), nearer_ghost=float(nearer),
                         cross_mm=float(cross)))
        print(f'  {s[0]:+.2f} {s[1]:+.2f}  {m.sum():4d}  {ok[m].mean():.2f} [{lo:.2f}, {hi:.2f}]'
              f'   {fmt(*e, 0)}   {nearer:5.2f}         {sf}        {cross:+6.1f}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dirs', default=f'{OUT}/dp_square,{OUT}/shift_*')
    p.add_argument('--json', default=f'{OUT}/ci.json')
    args = p.parse_args()
    for name, perceive, action, swapf in SWEEPS:
        gains(name, perceive, action)
        if swapf:
            swap(name, perceive, swapf)
    for name, action in AXIS_MAPS:
        if os.path.exists(f'{OUT}/{action}.npz'):
            axis_gains(name, action)
    rollouts(args.dirs)
    for b in ('dp', 'act', 'act_base', 'act_x', 'act_y'):
        if glob.glob(f'{OUT}/lift_{b}_shift_*/ep*.npz'):
            rollouts(f'{OUT}/lift_{b}_shift_*', 'lift ' + b.upper().replace('_', ' ')
                     .replace('BASE', 'base').replace(' X', ' x-aug').replace(' Y', ' y-aug'))
    for n in EDIT_ARMS:
        if glob.glob(f'{OUT}/lift_act_edit_{n}_shift_*/ep*.npz'):
            rollouts(f'{OUT}/lift_act_edit_{n}_shift_*', f'lift ACT edit {n}')
    for n in S3_ARMS:
        if glob.glob(f'{OUT}/lift_act_{n}_s3_shift_*/ep*.npz'):
            rollouts(f'{OUT}/lift_act_{n}_s3_shift_*', f'lift ACT {n} s3')
    with open(args.json, 'w') as f:
        json.dump(ROWS, f, indent=1)


if __name__ == '__main__':
    main()
