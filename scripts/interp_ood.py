#!/usr/bin/env python3
"""Interp: does DP Square "know" it is out of distribution, and how far?

On recorded rollouts (interp_collect: unshifted and nut-shifted episodes, per-tick global_cond),
three OOD signals per episode from the first frame's features, demos only, no labels:

  novelty   nearest-demo-frame distance / the demos' nearest-other-demo distance
  off-span  fraction of (g - local demo mean) outside the local cross-demo subspace
            (interp_subspace.local_basis: nearest frame of each of the 50 nearest demos, k = 8)
  spread    DP's own uncertainty: std over --seeds noise draws of the chunk's grasp target (mm),
            from the stored global_cond through the denoiser replica (interp_sensitivity)

Each is scored as a failure predictor (AUROC; 0.5 = chance), over all episodes, over the
shifted ones, and over the unshifted ones (natural failures). "How far": a ridge probe trained
on demo frames only (first --demo-ticks of each demo) decodes the nut's xy at t = 0; its error
and slope against the true nut position per shift say whether drift is readable without
out-of-coverage labels.

    python3 scripts/interp_ood.py --out outputs/interp/ood.npz        # in the container (spread)
    python3 scripts/interp_ood.py --out outputs/interp/ood.npz --report-only
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_probe as ip  # noqa: E402
import sil_step  # noqa: E402
from interp_subspace import demo_scale, local_basis, nearest, out_of_span  # noqa: E402

DIRS = 'outputs/interp/dp_square,outputs/interp/shift_*'


def auroc(score: np.ndarray, fail: np.ndarray) -> float:
    """P(score of a failure > score of a success), ties counted half (Mann-Whitney)."""
    pos, neg = score[fail], score[~fail]
    if not len(pos) or not len(neg):
        return float('nan')
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def load(pattern: str) -> dict:
    eps = [f for p in pattern.split(',') for d in sorted(glob.glob(p)) if os.path.isdir(d)
           for f in sorted(glob.glob(os.path.join(d, 'ep*.npz')))]
    rows = {k: [] for k in ('g0', 'success', 'cls', 'shift', 'nut0', 'path')}
    for f in eps:
        e = np.load(f)
        rows['g0'].append(e['feats'][0])
        rows['success'].append(bool(e['success']))
        rows['cls'].append(str(e['cls']))
        s = np.zeros(3)
        if 'shift' in e:
            s[:len(e['shift'])] = e['shift']
        rows['shift'].append(s)
        rows['nut0'].append(e['nut_pos'][0, :2])
        rows['path'].append(f)
    return {k: np.array(v) for k, v in rows.items()}


def spread(g0: np.ndarray, weights: str, seeds: int) -> np.ndarray:
    import torch
    from evh_controller.policy import make_policy
    from interp_sensitivity import Denoiser
    policy = make_policy('dp', weights, denoise_steps=16)
    dev, dt = policy._policy.device, policy._policy.dtype
    out = []
    with torch.no_grad():
        for g in g0:
            gt = torch.tensor(g, device=dev, dtype=dt)
            t = np.array([Denoiser(policy, s).target(gt).cpu().numpy() for s in range(seeds)])
            out.append(float(np.linalg.norm(t.std(0))))
    return np.array(out)


def collect(args) -> dict:
    r = load(args.dirs)
    demos = dict(np.load(args.demos))
    X = demos['feats'].astype(np.float64)
    scale = demo_scale(demos)
    g0 = r['g0'].astype(np.float64)
    r['novelty'] = nearest(X, g0) / scale
    off = []
    for g in g0:
        B, mu = local_basis(X, demos['demo'], g, 50, 8, with_mean=True)
        off.append(out_of_span(B, (g - mu)[None])[0])
    r['offspan'] = np.array(off)
    r['spread'] = spread(r['g0'], args.weights, args.seeds)
    early = demos['t'] < args.demo_ticks
    r['nut0_hat'] = ip.FIT(X[early], demos['nut_pos'][early, :2] * 1e3)(g0)
    return r


def report(r: dict) -> None:
    fail = ~r['success']
    shifted = np.abs(r['shift']).sum(1) > 0
    print(f'{len(fail)} episodes, {fail.sum()} failures ({fail[shifted].sum()}/{shifted.sum()} '
          f'shifted, {fail[~shifted].sum()}/{(~shifted).sum()} unshifted)')
    print(f'{"signal":10s} {"AUROC all":>10s} {"shifted":>8s} {"unshifted":>10s}   '
          'median: success / failure')
    for k in ('novelty', 'offspan', 'spread'):
        s = r[k]
        print(f'{k:10s} {auroc(s, fail):10.2f} {auroc(s[shifted], fail[shifted]):8.2f} '
              f'{auroc(s[~shifted], fail[~shifted]):10.2f}   '
              f'{np.median(s[~fail]):.2f} / {np.median(s[fail]):.2f}')
    print('\nper shift (dx, dy m; dyaw deg): success, medians of novelty / off-span / spread mm,'
          ' demo-trained decode of nut xy at t=0: error mm, decoded shift vs true')
    base = np.median(r['nut0_hat'][~shifted] - r['nut0'][~shifted] * 1e3, 0)   # probe bias
    for s in np.unique(r['shift'], axis=0):
        m = np.all(r['shift'] == s, axis=1)
        err = r['nut0_hat'][m] - r['nut0'][m] * 1e3
        dec = np.median(err - base, 0) + s[:2] * 1e3                     # decoded shift (mm)
        print(f'  {s[0]:+.2f} {s[1]:+.2f} {s[2]:+4.0f}  n {m.sum():3d}  success '
              f'{r["success"][m].mean():.2f}  {np.median(r["novelty"][m]):.2f} / '
              f'{np.median(r["offspan"][m]):.2f} / {np.median(r["spread"][m]):5.1f}   err '
              f'{np.median(np.linalg.norm(err, axis=1)):5.1f}  decoded shift '
              f'[{dec[0]:+5.1f} {dec[1]:+5.1f}] vs [{s[0] * 1e3:+5.1f} {s[1] * 1e3:+5.1f}]')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', required=True)
    p.add_argument('--dirs', default=DIRS)
    p.add_argument('--demos', default='outputs/interp/dp_square_demos.npz')
    p.add_argument('--weights', default=os.path.join(sil_step._REPO, 'checkpoints',
                                                     'dp_square_ph_image_cnn.ckpt'))
    p.add_argument('--seeds', type=int, default=8)
    p.add_argument('--demo-ticks', type=int, default=16)
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    if not args.report_only:
        np.savez_compressed(args.out, **collect(args))
    report(dict(np.load(args.out)))


if __name__ == '__main__':
    main()
