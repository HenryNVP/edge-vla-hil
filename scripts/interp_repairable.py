#!/usr/bin/env python3
"""Interp A: under a shift, is the task variable still linearly readable at ACT's head input?

The prediction under test: a last-layer repair (interp_edit --onpolicy) can only fix a shift if
the head input still carries what the action needs in a form one linear map can use. From
recorded on-policy episodes (interp_collect --teacher / --selflabel: per tick 'dec' = the head
inputs, 16 rows x 512, plus the cube and gripper poses) the cube's xy relative to the gripper
(mm), or with --absolute the cube's table position before it is touched, is ridge-decoded from
the 16 rows concatenated, every --stride-th tick:
  joint     one readout fit on unshifted + shifted episodes (5-fold CV by episode); R² on the
            shifted rows: is the information there for a single linear map?
  transfer  readout fit on the unshifted episodes only, applied to the shifted ones: does the
            code stay where the head already reads it?
R² is 1 - SSE / SST per axis, averaged over x and y.

    python3 scripts/interp_repairable.py --base 'outputs/interp/nuis/cf1_none' \
        --shift light='outputs/interp/nuis/cf1_light03' --shift cube='outputs/interp/nuis/cf1_cubeblue'
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_probe as ip  # noqa: E402


def load(pattern: str, stride: int, first_group: int = 0, absolute: bool = False):
    """(X rows, Y xy mm, episode groups) from every ep*.npz under pattern. Y is the cube minus
    the gripper, or with `absolute` the cube's own table position on ticks before it is touched
    (it has not moved yet), which the gripper's motion cannot give away."""
    X, Y, G = [], [], []
    files = sorted(f for d in sorted(glob.glob(pattern)) for f in glob.glob(f'{d}/ep*.npz'))
    for g, f in enumerate(files):
        e = np.load(f)
        T = len(e['dec'] if 'dec' in e else e['feats'])          # ACT head inputs / DP global_cond
        idx = np.arange(0, T, stride)
        if absolute:
            q = e['nut_pos'][:T] - e['nut_pos'][0]   # it settles a little in z at reset
            moved = (np.linalg.norm(q[:, :2], axis=1) > 0.003) | (q[:, 2] > 0.005)
            first = np.argmax(moved) if moved.any() else T
            idx = idx[idx < first]
            if not len(idx):
                continue
        X.append((e['dec'] if 'dec' in e else e['feats'])[idx].reshape(len(idx), -1).astype(np.float32))
        Y.append((e['nut_pos'][idx, :2] if absolute else
                  e['nut_pos'][idx, :2] - e['eef_pos'][idx, :2]) * 1e3)
        G.append(np.full(len(idx), first_group + g))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)))


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(((pred - y) ** 2).sum(1).mean()))


def scores(base, shifted) -> dict:
    """R² and xy RMSE (mm): 'joint' / 'transfer' on the shifted rows, 'base' = joint on the
    unshifted rows."""
    Xb, Yb, Gb = base
    Xs, Ys, Gs = shifted
    X, Y, G = np.vstack([Xb, Xs]), np.vstack([Yb, Ys]), np.r_[Gb, Gs + Gb.max() + 1]
    pred = ip.group_cv(X, Y, G)
    n = len(Xb)
    tr = ip.ridge_fit(Xb, Yb)(Xs)
    return {'joint': r2(pred[n:], Ys), 'base': r2(pred[:n], Yb), 'transfer': r2(tr, Ys),
            'joint_mm': rmse(pred[n:], Ys), 'base_mm': rmse(pred[:n], Yb), 'transfer_mm': rmse(tr, Ys)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--base', required=True, help='glob of unshifted recorded episode dirs')
    p.add_argument('--shift', action='append', default=[], help='name=glob of shifted dirs')
    p.add_argument('--stride', type=int, default=4, help='use every n-th tick')
    p.add_argument('--absolute', action='store_true',
                   help="the cube's table position before it is touched, not cube - gripper")
    args = p.parse_args()
    base = load(args.base, args.stride, absolute=args.absolute)
    print(f'{"shift":14s} {"joint R² / mm":>15s} {"transfer R² / mm":>17s} {"unshifted R² / mm":>18s}  rows')
    for item in args.shift:
        name, pattern = item.split('=', 1)
        shifted = load(pattern, args.stride, absolute=args.absolute)
        s = scores(base, shifted)
        print(f'{name:14s} {s["joint"]:7.2f} / {s["joint_mm"]:5.1f} {s["transfer"]:9.2f} / '
              f'{s["transfer_mm"]:5.1f} {s["base"]:10.2f} / {s["base_mm"]:5.1f}  {len(shifted[0])}',
              flush=True)


if __name__ == '__main__':
    main()
