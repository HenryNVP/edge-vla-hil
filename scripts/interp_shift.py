#!/usr/bin/env python3
"""Interp H2 / world shift: does DP follow a moved nut? (runs from interp_collect --shift)

Each `shift_*` directory holds the same scenes as `dp_square` with the nut moved and/or turned
at reset (the shift is stored in every episode). Everything is measured at the nut's HANDLE, the
grasp target, so moves and turns share one scale: per episode the target displacement is
handle(shifted, t = 0) - handle(base, t = 0) in xy, and each quantity below is its projection
on that displacement over its length squared (1 = follows the true nut, 0 = stays where the
unshifted scene had it):
  seen   - the handle xy decoded from the t = 0 image features by a ridge probe fit across every
           run, whole scenes held out (proprio is identical across shifts at t = 0);
  reach  - the commanded target at fractions of the time to the first close, and the eef at
           the close itself;
  nn     - distance to the nearest demo frame in image-feature space (how unfamiliar it looks).
Scenes are split by whether the shifted nut stays inside the demos' xy range (demos vary y over
11-22 cm and yaw over the full circle, x hardly at all).

    python3 scripts/interp_shift.py outputs/interp outputs/interp/dp_square_demos.npz
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

import interp_probe as ip
from oracle_square import handle_pos

PROGRESS = (0.25, 0.5, 0.75)


def load(path):
    return {int(os.path.basename(f)[2:6]): dict(np.load(f))
            for f in sorted(glob.glob(os.path.join(path, 'ep*.npz')))}


def handle0(e):
    return handle_pos(e['nut_pos'][0], e['nut_quat'][0])[:2]


def t_close(e):
    c = e['commands'][:, 6] > 0
    return int(np.argmax(c)) if c.any() else None


def follow(delta, disp):
    """Fraction of the target displacement `disp` that `delta` covers."""
    return float(delta @ disp / (disp @ disp))


def main(root: str, demo_path: str) -> None:
    D = dict(np.load(demo_path))
    first = D['t'] == 0
    lo, hi = D['nut_pos'][first, :2].min(0), D['nut_pos'][first, :2].max(0)
    img = ip.BLOCKS['images']
    mu, sd = D['feats'][:, img].mean(0), D['feats'][:, img].std(0) + 1e-6
    Dz = (D['feats'][:, img] - mu) / sd
    Dz2 = (Dz ** 2).sum(1)

    def nn_dist(e):
        z = (e['feats'][0, img] - mu) / sd
        return float(np.sqrt(max(np.min(z @ z + Dz2 - 2 * Dz @ z), 0)))

    base = load(os.path.join(root, 'dp_square'))
    runs = [(p, load(p)) for p in sorted(glob.glob(os.path.join(root, 'shift_*')))
            if os.path.isdir(p)]
    scenes = sorted(set.intersection(*[set(e) for _, e in runs]))
    sets = [base] + [eps for _, eps in runs]
    X = np.array([eps[k]['feats'][0, img] for eps in sets for k in scenes])
    Y = np.array([handle0(eps[k]) for eps in sets for k in scenes]) * 1e3
    G = np.array([k for _ in sets for k in scenes])
    pred = ip.group_cv(X, Y, G).reshape(len(sets), len(scenes), 2)
    print(f'scenes {scenes[0]}..{scenes[-1]}; unshifted success '
          f'{np.mean([bool(base[k]["success"]) for k in scenes]):.2f}, nn '
          f'{np.median([nn_dist(base[k]) for k in scenes]):.2f}')
    print(f'{"shift":>16s} {"range":>5s} {"n":>3s} {"succ":>5s} {"seen":>5s} '
          + ' '.join(f'{"r@" + str(f):>6s}' for f in PROGRESS)
          + f' {"r@close":>7s} {"nn":>5s}  failures')
    for i, (path, eps) in enumerate(runs, start=1):
        rows = []
        for j, k in enumerate(scenes):
            e, b = eps[k], base[k]
            disp = (handle0(e) - handle0(b)) * 1e3
            inside = bool(np.all((e['nut_pos'][0, :2] >= lo) & (e['nut_pos'][0, :2] <= hi)))
            r = {'in': inside, 'success': bool(e['success']), 'cls': str(e['cls']),
                 'seen': follow(pred[i, j] - pred[0, j], disp), 'nn': nn_dist(e)}
            te, tb = t_close(e), t_close(b)
            if te is not None and tb is not None:
                for f in PROGRESS:
                    d = e['commands'][int(f * te), :2] - b['commands'][int(f * tb), :2]
                    r[f] = follow(d * 1e3, disp)
                r['close'] = follow((e['eef_pos'][te, :2] - b['eef_pos'][tb, :2]) * 1e3, disp)
            rows.append(r)
        label = os.path.basename(path)[len('shift_'):]
        for name, part in (('in', [r for r in rows if r['in']]),
                           ('out', [r for r in rows if not r['in']])):
            if not part:
                continue
            med = lambda key: np.nanmedian([r.get(key, np.nan) for r in part])
            cls = [r['cls'] for r in part if not r['success']]
            print(f'{label:>16s} {name:>5s} {len(part):3d} {np.mean([r["success"] for r in part]):5.2f}'
                  f' {med("seen"):5.2f} ' + ' '.join(f'{med(f):6.2f}' for f in PROGRESS)
                  + f' {med("close"):7.2f} {med("nn"):5.2f}  '
                  + str({c: cls.count(c) for c in sorted(set(cls))}))


if __name__ == '__main__':
    main(*sys.argv[1:3])
