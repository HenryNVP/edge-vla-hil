#!/usr/bin/env python3
"""Interp kill test (docs/interp-plan.md, step 0): can the nut pose be read out of DP's
global_cond, and where does the readout sit when the policy fails?

Inputs: interp_collect.py episodes (rollouts) and interp_embed_demos.py (training frames).
Reports, for each target and feature block:
  * rollout CV  - ridge probe, 5-fold grouped by episode (no tick of a test episode in training);
  * demo->roll  - probe fit on training frames, tested on rollouts (the encoder's belief as
                  calibrated on its own training distribution);
  * shuffled    - the CV with labels permuted across episodes (chance level).
Then, before the first close of each episode: belief error for successes vs failures, and the
lookup-table test (H2): is the belief closer to the nearest training frame's truth than to the
episode's own truth?

    python3 scripts/interp_probe.py outputs/interp/dp_square outputs/interp/dp_square_demos.npz
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

STEP = 137                                  # per obs step: agentview 64 | wrist 64 | proprio 9
BLOCKS = {'all': np.arange(2 * STEP),
          'images': np.r_[0:128, STEP:STEP + 128],
          'agentview': np.r_[0:64, STEP:STEP + 64],
          'wrist': np.r_[64:128, STEP + 64:STEP + 128],
          'proprio': np.r_[128:STEP, STEP + 128:2 * STEP]}
PRE_CLOSE_TICKS = 10


def ridge_fit(X, Y, lam: float = 1.0):
    """Standardise X on the training rows, then closed-form ridge with an unpenalised bias."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    W = np.linalg.solve(Z.T @ Z + lam * len(Z) * 1e-3 * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean(0)))
    b = Y.mean(0)
    return lambda Xn: ((Xn - mu) / sd) @ W + b


def mlp_fit(X, Y, hidden: int = 256, epochs: int = 60, seed: int = 0):
    """Two-hidden-layer MLP probe (torch, CPU), same interface as ridge_fit."""
    import torch
    torch.manual_seed(seed)
    mu, sd, ym, ys = X.mean(0), X.std(0) + 1e-6, Y.mean(0), Y.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor((Y - ym) / ys, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xt)).split(256):
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(Xt[idx]), Yt[idx]).backward()
            opt.step()
    net.eval()

    def predict(Xn):
        with torch.no_grad():
            return net(torch.tensor((Xn - mu) / sd, dtype=torch.float32)).numpy() * ys + ym
    return predict


FIT = ridge_fit       # --mlp swaps in mlp_fit for every probe


def group_cv(X, Y, groups, k: int = 5, seed: int = 0):
    """Out-of-fold predictions, folds made of whole groups (episodes)."""
    ids = np.unique(groups)
    np.random.default_rng(seed).shuffle(ids)
    pred = np.zeros_like(Y, dtype=float)
    for fold in np.array_split(ids, k):
        test = np.isin(groups, fold)
        pred[test] = FIT(X[~test], Y[~test])(X[test])
    return pred


def yaw(q):
    """Nut yaw from robosuite xyzw quaternions (the nut stays upright on the table)."""
    q = np.asarray(q)
    return np.arctan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                      1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))


def targets(d) -> dict:
    """mm for positions; (cos, sin) for yaw so the probe stays linear."""
    a = yaw(d['nut_quat'])
    return {'nut_rel_eef': (d['nut_pos'] - d['eef_pos']) * 1e3,
            'nut_xy': d['nut_pos'][:, :2] * 1e3,
            'nut_yaw': np.c_[np.cos(a), np.sin(a)]}


def err(name, pred, Y):
    if name == 'nut_yaw':
        d = np.angle(np.exp(1j * (np.arctan2(pred[:, 1], pred[:, 0]) - np.arctan2(Y[:, 1], Y[:, 0]))))
        return np.degrees(np.abs(d))
    return np.linalg.norm(pred - Y, axis=1)


def load_rollouts(path):
    eps = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(path, 'ep*.npz')))]
    rows = {k: [] for k in ('feats', 'nut_pos', 'nut_quat', 'eef_pos', 'ep', 't')}
    for i, e in enumerate(eps):
        n = len(e['feats'])
        for k in ('nut_pos', 'nut_quat', 'eef_pos'):
            rows[k].append(e[k][:n])           # state at the tick the features were taken
        rows['feats'].append(e['feats'])
        rows['ep'].append(np.full(n, i))
        rows['t'].append(np.arange(n))
    return eps, {k: np.concatenate(v) for k, v in rows.items()}


def main(roll_dir: str, demo_path: str) -> None:
    eps, R = load_rollouts(roll_dir)
    D = dict(np.load(demo_path))
    cls = [str(e['cls']) for e in eps]
    print(f'{len(eps)} episodes, {len(R["t"])} ticks; classes:',
          {c: cls.count(c) for c in sorted(set(cls))})
    TR, TD = targets(R), targets(D)

    print(f'\n{"target":12s} {"block":10s} {"CV med":>8s} {"CV p90":>8s} {"demo->roll":>10s}'
          f' {"shuffled":>9s}   (mm, yaw in deg)')
    for name in TR:
        for block, cols in BLOCKS.items():
            if FIT is mlp_fit and block not in ('all', 'images', 'proprio'):
                continue                      # MLP fits are slow; the two blocks that decide
            Xr, Xd = R['feats'][:, cols], D['feats'][:, cols]
            cv = err(name, group_cv(Xr, TR[name], R['ep']), TR[name])
            dr = err(name, FIT(Xd, TD[name])(Xr), TR[name])
            # shuffled control: labels rolled by half the data, so each tick gets another
            # episode's label (episodes are contiguous)
            Ysh = np.roll(TR[name], len(Xr) // 2, axis=0)
            sh = err(name, group_cv(Xr, Ysh, R['ep']), Ysh)
            print(f'{name:12s} {block:10s} {np.median(cv):8.1f} {np.percentile(cv, 90):8.1f}'
                  f' {np.median(dr):10.1f} {np.median(sh):9.1f}')

    # ---- before the first close: belief error by outcome, and the lookup-table test (H2)
    probe = FIT(D['feats'], TD['nut_rel_eef'])
    img = BLOCKS['images']
    mu, sd = D['feats'][:, img].mean(0), D['feats'][:, img].std(0) + 1e-6
    Dz = (D['feats'][:, img] - mu) / sd
    Dz2 = (Dz ** 2).sum(1)
    print(f'\nlast {PRE_CLOSE_TICKS} ticks before the first close (nut_rel_eef, mm, medians)')
    print(f'{"class":11s} {"n":>3s} {"|b-true|":>9s} {"|b-nn|":>8s} {"|true-nn|":>10s}'
          f' {"nn closer":>10s} {"nn dist":>8s}')
    by = {}
    for e, c in zip(eps, cls):
        closing = e['commands'][:, 6] > 0
        if not closing.any():
            by.setdefault(c, []).append(None)
            continue
        t1 = int(np.argmax(closing))
        ts = np.arange(max(t1 - PRE_CLOSE_TICKS, 0), t1 + 1)
        X = e['feats'][ts]
        b = probe(X)
        true = (e['nut_pos'][ts] - e['eef_pos'][ts]) * 1e3
        z = (X[:, img] - mu) / sd
        d2 = (z ** 2).sum(1)[:, None] + Dz2[None] - 2 * z @ Dz.T
        nn = np.argmin(d2, axis=1)
        nn_rel = TD['nut_rel_eef'][nn]
        dist = np.sqrt(np.maximum(d2[np.arange(len(nn)), nn], 0))
        by.setdefault(c, []).append((err('', b, true), err('', b, nn_rel), err('', true, nn_rel),
                                     dist))
    for c, vals in sorted(by.items()):
        v = [x for x in vals if x is not None]
        if not v:
            print(f'{c:11s} {len(vals):3d}  (never closed)')
            continue
        bt, bn, tn, dd = (np.concatenate([x[i] for x in v]) for i in range(4))
        print(f'{c:11s} {len(vals):3d} {np.median(bt):9.1f} {np.median(bn):8.1f}'
              f' {np.median(tn):10.1f} {np.mean(bn < bt):10.2f} {np.median(dd):8.2f}'
              + (f'  ({len(vals) - len(v)} never closed)' if len(v) < len(vals) else ''))


if __name__ == '__main__':
    if '--mlp' in sys.argv:
        sys.argv.remove('--mlp')
        FIT = mlp_fit
    main(*sys.argv[1:3])
