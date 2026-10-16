#!/usr/bin/env python3
"""Interp: does the demos' feature coverage predict the action gain?

Hypothesis: the action head only responds to feature changes the demos varied. Two measures
per interior grid point of a perception sweep (interp_perceive), from the policy's features
alone, no sim ground truth:

  out-of-span   fraction of the local feature change per axis (d features / d offset, central
                differences) outside the top-k PCA subspace of the demos' features
  novelty       distance from the moved scene's features to the nearest demo frame, over the
                demos' own typical nearest-other-demo distance (1 = as close as a demo)

With --local N, out-of-span is instead taken against a local subspace at each moved scene:
the nearest frame of each of the N nearest demos, centred, top --local-k PCA directions (how
the demos vary ACROSS episodes at this point of the task, not over a whole episode).

Each is set against the action gain (the gain map's Jacobian diagonal) at the same point:
Spearman rank correlation, and median gain per quartile of the measure.

    python3 scripts/interp_subspace.py --demos outputs/interp/dp_square_demos.npz \
        --perceive outputs/interp/perceive_square_dp.npz --action outputs/interp/gainmap.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interp_gainmap import inside_demo_range, jacobians  # noqa: E402


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = (np.argsort(np.argsort(v)).astype(float) for v in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


def out_of_span(V: np.ndarray, v: np.ndarray) -> np.ndarray:
    """|v - V V^T v| / |v| for rows of v; V (D, k) orthonormal."""
    return np.linalg.norm(v - (v @ V) @ V.T, axis=-1) / (np.linalg.norm(v, axis=-1) + 1e-12)


def nearest(X: np.ndarray, Q: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Distance from each row of Q to its nearest row of X."""
    x2 = (X ** 2).sum(1)
    return np.concatenate([np.sqrt(np.maximum(
        (q ** 2).sum(1)[:, None] - 2 * q @ X.T + x2[None], 0).min(1))
        for q in np.array_split(Q, max(1, len(Q) // chunk))])


def local_basis(X: np.ndarray, ids: np.ndarray, g: np.ndarray, n: int, k: int) -> np.ndarray:
    """(D, k): top-k directions of the nearest frame of each of the n demos nearest to g."""
    dist = np.linalg.norm(X - g, axis=1)
    order = np.lexsort((dist, ids))                         # by demo, then distance
    first = order[np.r_[True, ids[order][1:] != ids[order][:-1]]]
    near = first[np.argsort(dist[first])[:n]]
    Z = X[near] - X[near].mean(0)
    return np.linalg.svd(Z, full_matrices=False)[2][:k].T


def demo_scale(demos: dict, n: int = 2000, seed: int = 0) -> float:
    """Median distance from a demo frame to the nearest frame of ANOTHER demo."""
    X, ids = demos['feats'], demos['demo']
    pick = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return float(np.median([nearest(X[ids != ids[i]], X[i:i + 1])[0] for i in pick]))


def table(name: str, measure: np.ndarray, gain: np.ndarray) -> None:
    q = np.quantile(measure, [0, .25, .5, .75, 1])
    bins = np.clip(np.searchsorted(q[1:-1], measure, side='right'), 0, 3)
    cells = ' '.join(f'{np.median(gain[bins == b]):5.2f}' for b in range(4))
    print(f'    {name:24s} spearman {spearman(measure, gain):+.2f}   gain by quartile '
          f'(low -> high): {cells}   quartile edges {np.round(q, 2).tolist()}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--demos', required=True)
    p.add_argument('--perceive', required=True)
    p.add_argument('--action', required=True)
    p.add_argument('--var', default='0.9,0.99', help='PCA variance kept, one k per value')
    p.add_argument('--local', type=int, default=0, help='N nearest demos for a local subspace')
    p.add_argument('--local-k', default='3,8', help='local subspace sizes')
    args = p.parse_args()
    demos, d, a = (dict(np.load(f)) for f in (args.demos, args.perceive, args.action))
    assert np.allclose(d['nut_xy'], a['nut_xy']) and np.allclose(d['grid'], a['grid']), \
        'perception and action sweeps must share operating points and grid'

    X = demos['feats'].astype(np.float64)
    mu = X.mean(0)
    _u, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    ks = [int(np.searchsorted(cum, float(v))) + 1 for v in args.var.split(',')]
    scale = demo_scale(demos)
    print(f'{len(X)} demo frames, D = {X.shape[1]}; k = {ks} for variance {args.var}; '
          f'demo nearest-other-demo distance {scale:.3f}')

    step = (d['grid'][1] - d['grid'][0]) * 1e3
    for r in sorted(set(d['radius']), reverse=True):
        m = d['radius'] == r
        F = d['feat'][m].astype(np.float64)                         # (P, G, G, D) changes
        vx = (F[:, 2:, 1:-1] - F[:, :-2, 1:-1]) / (2 * step)        # d features / d mm
        vy = (F[:, 1:-1, 2:] - F[:, 1:-1, :-2]) / (2 * step)
        Ja = np.array([jacobians(x, step) for x in a['resp'][a['radius'] == r]])
        gx, gy = Ja[..., 0, 0], Ja[..., 1, 1]
        moved = d['base'][m][:, None, None] + F[:, 1:-1, 1:-1]
        nov = nearest(X, moved.reshape(-1, X.shape[1])).reshape(gx.shape) / scale
        inside = inside_demo_range(a, r)
        print(f'\nradius {r * 100:.0f} cm, {m.sum()} points')
        if args.local:
            flat = moved.reshape(-1, X.shape[1])
            for k in (int(v) for v in args.local_k.split(',')):
                B = [local_basis(X, demos['demo'], g, args.local, k) for g in flat]
                ox = np.array([out_of_span(b, v) for b, v in zip(B, vx.reshape(len(B), -1))])
                oy = np.array([out_of_span(b, v) for b, v in zip(B, vy.reshape(len(B), -1))])
                ox, oy = ox.reshape(gx.shape), oy.reshape(gy.shape)
                print(f'  local {args.local} demos, k = {k}: out-of-span median x {np.median(ox):.2f}'
                      f' y {np.median(oy):.2f}; inside demo range '
                      f'{np.median(np.r_[ox[inside], oy[inside]]):.2f}, outside '
                      f'{np.median(np.r_[ox[~inside], oy[~inside]]):.2f}')
                table('out-of-span, x', ox.ravel(), gx.ravel())
                table('out-of-span, y', oy.ravel(), gy.ravel())
                table('out-of-span, both axes', np.r_[ox.ravel(), oy.ravel()],
                      np.r_[gx.ravel(), gy.ravel()])
            continue
        for k in ks:
            V = vt[:k].T
            ox, oy = out_of_span(V, vx), out_of_span(V, vy)
            print(f'  k = {k}: out-of-span median x {np.median(ox):.2f} y {np.median(oy):.2f}; '
                  f'inside demo range {np.median(np.r_[ox[inside], oy[inside]]):.2f}, '
                  f'outside {np.median(np.r_[ox[~inside], oy[~inside]]):.2f}')
            table('out-of-span, x', ox.ravel(), gx.ravel())
            table('out-of-span, y', oy.ravel(), gy.ravel())
            table('out-of-span, both axes', np.r_[ox.ravel(), oy.ravel()],
                  np.r_[gx.ravel(), gy.ravel()])
        print(f'  novelty median: inside demo range {np.median(nov[inside]):.2f}, '
              f'outside {np.median(nov[~inside]):.2f}')
        table('novelty vs x gain', nov.ravel(), gx.ravel())
        table('novelty vs y gain', nov.ravel(), gy.ravel())


if __name__ == '__main__':
    main()
