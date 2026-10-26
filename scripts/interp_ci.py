#!/usr/bin/env python3
"""Interp: the paper's numbers with 95% confidence intervals, from saved sweeps and rollouts.

  gains     action gain (gain map) and seen gain (interp_perceive's out-of-fold decode) inside /
            outside the demos' range, and the action's inside - outside drop; bootstrap over
            scenes (the decode is fitted once, so its own training variance is not included)
  swap      interp_swap's 2x2 head-state x feature-direction table, the direction effect (dir
            in - dir out, averaged over head rows) and the head effect; bootstrap over scenes
  rollouts  per nut shift: success (Wilson), and where the gripper first closes: the fraction of
            the shift it followed (1 = at the real handle, 0 = where the unshifted handle would be),
            the share of episodes nearer that "ghost" target than the real one, and the cross-axis
            drift (mm); bootstrap over episodes

    python3 scripts/interp_ci.py | tee outputs/interp/ci.txt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interp_gainmap import handle_pos, inside_demo_range, jacobians  # noqa: E402
from interp_perceive import seen  # noqa: E402
from interp_swap import block_gains, directions  # noqa: E402
from plot_sweep import wilson  # noqa: E402

OUT = 'outputs/interp'
SWEEPS = [('square DP', 'perceive_square_dp', 'gainmap', 'swap_square_dp'),
          ('lift DP', 'perceive_lift_dp', 'gainmap_lift_dp', 'swap_lift_dp'),
          ('lift ACT', 'perceive_lift_act', 'gainmap_lift_act', '')]


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


def rollouts(dirs: str) -> None:
    rows = []
    for f in (f for p in dirs.split(',') for dd in sorted(glob.glob(p)) if os.path.isdir(dd)
              for f in sorted(glob.glob(os.path.join(dd, 'ep*.npz')))):
        e = np.load(f)
        s = np.zeros(3)
        if 'shift' in e:
            s[:len(e['shift'])] = e['shift']
        c = np.flatnonzero(e['commands'][:, 6] > 0)
        # the gripper grasps the handle, which swings with the nut's random yaw
        rows.append((s, bool(e['success']), handle_pos(e['nut_pos'][0], e['nut_quat'][0])[:2],
                     e['eef_pos'][c[0], :2] if len(c) else np.full(2, np.nan)))
    S = np.array([r[0] for r in rows])
    ok = np.array([r[1] for r in rows])
    nut0, close = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
    closed = ~np.isnan(close[:, 0])
    base = (np.abs(S).sum(1) == 0) & closed
    o = np.median(close[base] - nut0[base], 0)
    print(f'rollouts: {len(rows)} episodes, {(~closed).sum()} never closed; unshifted '
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
        print(f'  {s[0]:+.2f} {s[1]:+.2f}  {m.sum():4d}  {ok[m].mean():.2f} [{lo:.2f}, {hi:.2f}]'
              f'   {fmt(*e, 0)}   {nearer:5.2f}         {sf}        {cross:+6.1f}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dirs', default=f'{OUT}/dp_square,{OUT}/shift_*')
    args = p.parse_args()
    for name, perceive, action, swapf in SWEEPS:
        gains(name, perceive, action)
        if swapf:
            swap(name, perceive, swapf)
    rollouts(args.dirs)


if __name__ == '__main__':
    main()
