#!/usr/bin/env python3
"""Interp: is the action-gain collapse in the feature DIRECTIONS or in the head's STATE?

The gain at a state factors, to first order, as J_head(g) @ V: the action head's Jacobian of the
grasp target w.r.t. global_cond at the moved scene's features g, times V = d global_cond / d
offset (the feature direction the object's move produces, central differences on a perception
sweep). Swapping the two between in-range and out-of-range states of the SAME operating point
(same arm state and noise) gives a 2x2 table per axis:

                    direction from in-range     direction from out-of-range
    head in-range   ~ the in-range gain          new directions, healthy head
    head out        healthy directions, moved    ~ the collapsed gain
                    head

If the collapse follows the direction column, the head ignores the directions out-of-range moves
produce; if it follows the head row, the head itself goes flat out there. Run on DP through the
fixed-noise denoiser replica (interp_sensitivity.Denoiser; seed 1000 + scene as in the sweeps).
Checks first that the replica reproduces the gain map's response from the stored features, and
that the linear self-gain J @ V tracks the gain map's finite-difference gain.

    python3 scripts/interp_swap.py --perceive outputs/interp/perceive_square_dp.npz \
        --action outputs/interp/gainmap.npz --out outputs/interp/swap_square_dp.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sil_step  # noqa: E402
from interp_gainmap import TASKS, inside_demo_range, jacobians  # noqa: E402


def directions(F: np.ndarray, step: float) -> np.ndarray:
    """(P, G, G, D) feature changes -> (P, G-2, G-2, D, 2) d features / d mm, interior."""
    vx = (F[:, 2:, 1:-1] - F[:, :-2, 1:-1]) / (2 * step)
    vy = (F[:, 1:-1, 2:] - F[:, 1:-1, :-2]) / (2 * step)
    return np.stack([vx, vy], -1)


def block_gains(J: np.ndarray, V: np.ndarray, inside: np.ndarray) -> dict:
    """J (P, n, n, 2, D), V (P, n, n, D, 2), inside (P, n, n) -> {(head, dir): (2,) median
    diagonal gain of J_a @ V_b over all same-point pairs, a from the head set, b from the dir set}."""
    out = {}
    for hn, hm in (('in', inside), ('out', ~inside)):
        for dn, dm in (('in', inside), ('out', ~inside)):
            g = [np.einsum('aod,bdo->abo', J[p][hm[p]], V[p][dm[p]]).reshape(-1, 2)
                 for p in range(len(J)) if hm[p].any() and dm[p].any()]
            out[hn, dn] = np.median(np.concatenate(g), 0)
    return out


def collect(args) -> dict:
    import torch
    from evh_controller.policy import make_policy
    from interp_sensitivity import Denoiser
    d = dict(np.load(args.perceive))
    weights = args.weights or os.path.join(sil_step._REPO, 'checkpoints',
                                           TASKS[str(d['task'])]['ckpt'])
    policy = make_policy('dp', weights, denoise_steps=16)
    dev, dt = policy._policy.device, policy._policy.dtype
    P, G = d['feat'].shape[:2]
    resp, J = np.zeros((P, G, G, 2)), np.zeros((P, G - 2, G - 2, 2, d['feat'].shape[-1]))
    for p in range(P):
        den = Denoiser(policy, 1000 + int(d['scene'][p]))

        def tgt(g):
            den.idx = None                          # each state picks its own grasp entry
            return den.target(g)
        gs = torch.tensor(d['base'][p] + d['feat'][p], device=dev, dtype=dt)   # (G, G, D)
        with torch.no_grad():
            t0 = tgt(torch.tensor(d['base'][p], device=dev, dtype=dt)).cpu().numpy()
            resp[p] = [[tgt(gs[i, j]).cpu().numpy() - t0 for j in range(G)] for i in range(G)]
        for i in range(1, G - 1):
            for j in range(1, G - 1):
                J[p, i - 1, j - 1] = torch.autograd.functional.jacobian(
                    tgt, gs[i, j]).detach().cpu().numpy()
        print(f'point {p + 1}/{P}', flush=True)
    return {'resp': resp, 'J': J.astype(np.float32)}


def report(s: dict, d: dict, a: dict) -> None:
    step = (d['grid'][1] - d['grid'][0]) * 1e3
    V = directions(d['feat'].astype(np.float64), step)
    for r in sorted(set(d['radius']), reverse=True):
        m = d['radius'] == r
        inside = inside_demo_range(d, r)
        err = np.linalg.norm(s['resp'][m] - a['resp'][a['radius'] == r], axis=-1)
        Jfd = np.array([jacobians(x, step) for x in a['resp'][a['radius'] == r]])
        Jm, Vm = s['J'][m].astype(np.float64), V[m]
        lin = np.einsum('pijod,pijdo->pijo', Jm, Vm)
        print(f'radius {r * 100:.0f} cm, {m.sum()} points, {inside.sum()} in-range / '
              f'{(~inside).sum()} out-of-range states')
        print(f'  replica vs gain map response: median |diff| {np.median(err):.2f} mm, '
              f'p95 {np.percentile(err, 95):.2f} mm')
        for name, msk in (('in-range', inside), ('out-of-range', ~inside)):
            print(f'  {name:12s} gain map x {np.median(Jfd[msk][:, 0, 0]):5.2f} y '
                  f'{np.median(Jfd[msk][:, 1, 1]):5.2f} | linear J@V x {np.median(lin[msk][:, 0]):5.2f}'
                  f' y {np.median(lin[msk][:, 1]):5.2f}')
        cos = [np.abs(np.einsum('adk,bdk->abk', Vm[p][inside[p]], Vm[p][~inside[p]])
                      / np.linalg.norm(Vm[p][inside[p]], axis=1)[:, None]
                      / np.linalg.norm(Vm[p][~inside[p]], axis=1)[None]).reshape(-1, 2)
               for p in range(len(Vm)) if inside[p].any() and (~inside[p]).any()]
        c = np.median(np.concatenate(cos), 0)
        print(f'  |cos| in-range vs out-of-range feature direction: x {c[0]:.2f} y {c[1]:.2f}')
        b = block_gains(Jm, Vm, inside)
        print('  swap (median gain x / y)   dir in-range    dir out-of-range')
        for h in ('in', 'out'):
            print(f'    head {h + "-range":10s}       {b[h, "in"][0]:5.2f} / {b[h, "in"][1]:5.2f}'
                  f'    {b[h, "out"][0]:5.2f} / {b[h, "out"][1]:5.2f}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--perceive', required=True)
    p.add_argument('--action', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--weights', default='', help="default: the task's DP checkpoint")
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    d, a = dict(np.load(args.perceive)), dict(np.load(args.action))
    assert np.allclose(d['nut_xy'], a['nut_xy']) and np.allclose(d['grid'], a['grid']), \
        'perception and action sweeps must share operating points and grid'
    if not args.report_only:
        np.savez_compressed(args.out, **collect(args))
    report(dict(np.load(args.out)), d, a)


if __name__ == '__main__':
    main()
