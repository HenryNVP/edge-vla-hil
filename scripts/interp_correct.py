#!/usr/bin/env python3
"""Interp: can the policy's own features decode its action mistake? (offline, one chunk)

On a perception sweep (interp_perceive) and the gain map of the same task/backend (same points,
same grid), the mistake at each offset is what the chunk failed to follow: offset - response
(mm, xy). A ridge probe decodes it from the paired feature change, group CV by scene, and the
corrected response response + decoded mistake gives a corrected gain, inside / outside the
demos' range like the original. Two training sets:

  all      folds over every grid point (the corrector has seen offsets like the test ones)
  inside   trained on in-range offsets only, tested everywhere (must extrapolate the mistake
           to where the action gain collapses; the realistic case if labels come from
           in-coverage perturbations)

Paired (feature change against the unmoved scene), so this bounds what a deployed corrector
working from absolute features could do.

    python3 scripts/interp_correct.py --perceive outputs/interp/perceive_lift_dp.npz \
        --action outputs/interp/gainmap_lift_dp.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_probe as ip  # noqa: E402
from interp_gainmap import inside_demo_range, jacobians  # noqa: E402


def full_mask(inner: np.ndarray) -> np.ndarray:
    """Interior inside-range mask (P, G-2, G-2) -> whole grid (P, G, G); edges count as outside
    unless their interior neighbour is inside (only used to pick training rows)."""
    P, n, _ = inner.shape
    m = np.zeros((P, n + 2, n + 2), bool)
    m[:, 1:-1, 1:-1] = inner
    return m


def gains(J, inside):
    return ' '.join(f'{name} xx {np.median(J[m][:, 0, 0]):5.2f} yy {np.median(J[m][:, 1, 1]):5.2f}'
                    for name, m in (('inside', inside), ('outside', ~inside)) if m.any())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--perceive', required=True)
    p.add_argument('--action', required=True)
    args = p.parse_args()
    d, a = dict(np.load(args.perceive)), dict(np.load(args.action))
    assert np.allclose(d['nut_xy'], a['nut_xy']) and np.allclose(d['grid'], a['grid']), \
        'perception and action sweeps must share operating points and grid'
    g = d['grid'] * 1e3
    step = g[1] - g[0]
    off = np.stack(np.meshgrid(g, g, indexing='ij'), -1)            # (G, G, 2) mm
    for r in sorted(set(d['radius']), reverse=True):
        m = d['radius'] == r
        F, resp = d['feat'][m], a['resp'][a['radius'] == r]           # (P,G,G,D), (P,G,G,2)
        P, G = resp.shape[:2]
        err = off[None] - resp                                       # the mistake
        X, Y = F.reshape(P * G * G, -1), err.reshape(-1, 2)
        groups = np.repeat(d['scene'][m], G * G)
        inside = inside_demo_range(a, r)
        print(f'radius {r * 100:.0f} cm, {P} points; mistake median |e| {np.median(np.linalg.norm(Y, axis=1)):.1f} mm')
        print(f'    uncorrected      {gains(np.array([jacobians(x, step) for x in resp]), inside)}')
        train_in = full_mask(inside).reshape(-1)
        for name, rows in (('all', np.ones(len(Y), bool)), ('inside', train_in)):
            pred = np.zeros_like(Y)
            ids = np.unique(groups)
            np.random.default_rng(0).shuffle(ids)
            for fold in np.array_split(ids, 5):
                test = np.isin(groups, fold)
                pred[test] = ip.FIT(X[~test & rows], Y[~test & rows])(X[test])
            res = np.linalg.norm(Y - pred, axis=1)
            r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
            fixed = resp + pred.reshape(P, G, G, 2)
            Jc = np.array([jacobians(x, step) for x in fixed])
            print(f'    train {name:6s}  R2 {r2:.2f}, residual mistake median {np.median(res):.1f} mm;'
                  f' corrected {gains(Jc, inside)}')


if __name__ == '__main__':
    main()
