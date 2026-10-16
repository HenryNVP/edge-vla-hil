#!/usr/bin/env python3
"""Interp H2 / world shift: the nut moved beyond the demos' range (interp_collect --shift).

Square's demos barely vary the nut in x (std 1.4 mm), so any x shift is out of distribution. Per
shift, paired by scene with the unshifted run, each as a slope against the shift (1 = follows the
true nut, 0 = stays where the training data had it):
  seen    - the shift decoded from the t = 0 features by a ridge probe, fit across all shifts
            with whole scenes held out (a demo-fit probe cannot read x: the demos never vary it);
            proprio is identical across shifts at t = 0, so only the images can carry it;
  nn dist - distance to the nearest demo frame in image-feature space (how far out of
            distribution the scene looks);
  action  - where the arm closed (eef x at the first close command).
Plus success and the failure classes.

    python3 scripts/interp_shift.py outputs/interp outputs/interp/dp_square_demos.npz
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

import interp_probe as ip


def load(path):
    return {int(os.path.basename(f)[2:6]): dict(np.load(f))
            for f in sorted(glob.glob(os.path.join(path, 'ep*.npz')))}


def first_close_x(e):
    c = e['commands'][:, 6] > 0
    return float(e['eef_pos'][int(np.argmax(c)), 0]) if c.any() else np.nan


def main(root: str, demo_path: str) -> None:
    D = dict(np.load(demo_path))
    img = ip.BLOCKS['images']
    mu, sd = D['feats'][:, img].mean(0), D['feats'][:, img].std(0) + 1e-6
    Dz = (D['feats'][:, img] - mu) / sd
    Dz2 = (Dz ** 2).sum(1)

    def summary(e):
        X = e['feats'][:1]
        z = (X[:, img] - mu) / sd
        nn = np.argmin((z ** 2).sum(1)[:, None] + Dz2[None] - 2 * z @ Dz.T, axis=1)
        d2 = (z ** 2).sum(1) + Dz2[nn] - 2 * (z * Dz[nn]).sum(1)
        return {'nn': float(np.median(np.sqrt(np.maximum(d2, 0)))),
                'action': first_close_x(e) * 1e3}

    base_eps = load(os.path.join(root, 'dp_square'))
    paths = sorted(glob.glob(os.path.join(root, 'shift_x*')), key=lambda p: float(p.split('x')[-1]))
    runs = [(float(p.split('shift_x')[-1]) * 1e3, load(p)) for p in paths]
    scenes = sorted(set.intersection(*[set(e) for _, e in runs]))
    # the paired decode: every shift (and 0) of the same scenes, folds by scene
    sets = [(0.0, base_eps)] + runs
    X = np.array([eps[k]['feats'][0, img] for _, eps in sets for k in scenes])
    Y = np.array([[dx] for dx, _ in sets for _k in scenes])
    G = np.array([k for _ in sets for k in scenes])
    seen = ip.group_cv(X, Y, G)[:, 0]
    base = {k: summary(e) for k, e in base_eps.items()}
    print(f'scenes {scenes[0]}..{scenes[-1]}; unshifted nn dist {np.median([b["nn"] for b in base.values()]):.2f}')
    print(f'{"dx mm":>6s} {"n":>3s} {"succ":>5s} {"seen":>6s} {"action":>7s} {"nn dist":>8s}'
          f'  classes   (seen/action: slopes vs the true shift, medians)')
    for i, (dx, eps) in enumerate(runs, start=1):
        rows = [(summary(eps[k]), base[k]) for k in scenes]
        cls = [str(eps[k]['cls']) for k in scenes]
        counts = {c: cls.count(c) for c in sorted(set(cls)) if c != 'success'}
        act = np.nanmedian([(s['action'] - b['action']) / dx for s, b in rows])
        print(f'{dx:6.0f} {len(scenes):3d} {cls.count("success") / len(cls):5.2f}'
              f' {np.median(seen[i * len(scenes):(i + 1) * len(scenes)]) / dx:6.2f} {act:7.2f}'
              f' {np.median([s["nn"] for s, _ in rows]):8.2f}  {counts}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
