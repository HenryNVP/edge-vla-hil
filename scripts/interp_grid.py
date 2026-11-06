#!/usr/bin/env python3
"""Interp A across policies and tasks: success before and after the readout repair, per shift,
with Wilson 95% intervals, from the fresh-scene eval dirs (env seed 3) the pipelines write:
<prefix>eval3_base_<shift>, <prefix>eval3_rep_<shift>, <prefix>eval3_rep_<shift>_none.
Writes outputs/interp/grid.json and figs/fig_grid.{pdf,png}: one panel per policy and task,
shifts along x, base (hollow) and repaired (filled), the invisible-object control last.

    python3 scripts/interp_grid.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_sweep import wilson  # noqa: E402

PANELS = (('ACT, Lift', 'outputs/interp/nuis/', ''), ('Diffusion Policy, Lift', 'outputs/interp/nuis/', 'dp_'),
          ('SmolVLA, Lift', 'outputs/interp/remote2/', ''), ('ACT, Square', 'outputs/interp/nuis/', 'sq_'),
          ('Diffusion Policy, Square', 'outputs/interp/nuis/', 'dpsq_'),
          ('ACT, ToolHang', 'outputs/interp/nuis/', 'th_'))
SHIFTS = (('bothcam', 'turn + roll'), ('darkredwroll', 'red, dim + roll'), ('bothflip', 'upside down'),
          ('bothzoom', 'zoomed'), ('faint', 'faint object'), ('invisible', 'invisible object'))


def rate(d: str):
    """(successes, episodes) of an eval dir, or None if it has no episodes."""
    fs = sorted(glob.glob(f'{d}/ep*.npz'))
    if not fs:
        return None
    return int(sum(bool(np.load(f)['success']) for f in fs)), len(fs)


def collect(panels=PANELS, shifts=SHIFTS) -> list:
    rows = []
    for name, root, pre in panels:
        for tag, label in shifts:
            r = {k: rate(f'{root}{pre}eval3_{k}_{tag}') for k in ('base', 'rep')}
            r['rep_none'] = rate(f'{root}{pre}eval3_rep_{tag}_none')
            if r['base'] is None or r['rep'] is None:
                continue
            row = dict(panel=name, shift=tag, label=label)
            for k, v in r.items():
                if v is not None:
                    lo, hi = wilson(*v)
                    row[k] = dict(k=v[0], n=v[1], p=v[0] / v[1], lo=lo, hi=hi)
            rows.append(row)
    return rows


def figure(rows: list, out: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from interp_figs import BLUE, DOUBLE, save, style
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7})
    panels = [p for p, _, _ in PANELS if any(r['panel'] == p for r in rows)]
    fig, axes = plt.subplots(1, len(panels), figsize=(DOUBLE, 2.3), sharey=True, constrained_layout=True)
    for ax, p in zip(np.atleast_1d(axes), panels):
        rs = [r for r in rows if r['panel'] == p]
        for i, r in enumerate(rs):
            for k, color, fill, dx in (('base', '0.45', False, -0.12), ('rep', BLUE, True, 0.12)):
                v = r[k]
                ax.errorbar(i + dx, v['p'], yerr=[[max(0.0, v['p'] - v['lo'])], [max(0.0, v['hi'] - v['p'])]], fmt='o',
                            color=color, ms=3.5, lw=0.8, capsize=1.5, markeredgewidth=0.8,
                            markerfacecolor=color if fill else 'none', zorder=3)
        ax.set_xticks(range(len(rs)), [r['label'] for r in rs], rotation=60, ha='right', fontsize=6.5)
        ax.set_xlim(-0.6, len(rs) - 0.4)
        ax.set_title(p, fontsize=7.5)
        style(ax)
    np.atleast_1d(axes)[0].set_ylabel('success')
    np.atleast_1d(axes)[0].set_ylim(-0.03, 1.03)
    from interp_figs import legend_handle
    fig.legend(handles=[legend_handle('0.45', 'base', inside=False), legend_handle(BLUE, 'readout refit')],
               loc='outside lower center', ncol=2, frameon=False, fontsize=7)
    save(fig, 'fig_grid', out)


def fig_depth(out: str, root: str = 'outputs/interp/nuis/') -> None:
    """Square, both cameras shifted: success under the shift and unshifted over DAgger rounds of
    the full refit (left), and the refit depth on the same four rounds of labels (right)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from interp_figs import BLUE, SINGLE, save, style, legend_handle
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7})
    rounds = [('0', 'sq_eval3_base_bothcam', 'sq_eval3_base_none')] + [
        (str(k), f'sq_depth_full{"" if k == 1 else f"_r{k}"}_eval3_bothcam',
         f'sq_depth_full{"" if k == 1 else f"_r{k}"}_eval3_none') for k in range(1, 6)]
    depth = [('last layer\n5K', 'sq_depth_head_r4'), ('rank 8, all\n457K', 'sq_depth_all8_r4'),
             ('full\n51.6M', 'sq_depth_full_r4')]
    fig, (a, b) = plt.subplots(1, 2, figsize=(SINGLE * 2, 2.1), sharey=True,
                               constrained_layout=True)
    for j, (name, color, fill) in enumerate((('under the shift', BLUE, True), ('unshifted', '0.45', False))):
        pts = [(k, rate(root + (sh if j == 0 else un))) for k, sh, un in rounds]
        pts = [(int(k), v) for k, v in pts if v]
        for x, v in pts:
            p_ = v[0] / v[1]
            lo, hi = wilson(*v)
            a.errorbar(x, p_, yerr=[[max(0, p_ - lo)], [max(0, hi - p_)]], fmt='o', color=color, ms=3.5,
                       lw=0.8, capsize=1.5, markerfacecolor=color if fill else 'none', zorder=3)
        a.plot([x for x, _ in pts], [v[0] / v[1] for _, v in pts], color=color, lw=0.8, zorder=2)
        for i, (lab, n) in enumerate(depth):
            v = rate(root + n + ('_eval3_bothcam' if j == 0 else '_eval3_none'))
            if v:
                p_ = v[0] / v[1]
                lo, hi = wilson(*v)
                b.errorbar(i + (0.1 if j else -0.1), p_, yerr=[[max(0, p_ - lo)], [max(0, hi - p_)]],
                           fmt='o', color=color, ms=3.5, lw=0.8, capsize=1.5,
                           markerfacecolor=color if fill else 'none', zorder=3)
    a.set_xlabel('DAgger round (full refit)')
    a.set_ylabel('success')
    a.set_ylim(-0.03, 1.03)
    a.set_title('rounds of on-policy labels')
    b.set_xticks(range(len(depth)), [d for d, _ in depth])
    b.set_xlim(-0.5, len(depth) - 0.5)
    b.set_title('refit depth, same 4 rounds')
    for ax in (a, b):
        style(ax)
    fig.legend(handles=[legend_handle(BLUE, 'under the shift'), legend_handle('0.45', 'unshifted', inside=False)],
               loc='outside lower center', ncol=2, frameon=False, fontsize=7)
    save(fig, 'fig_depth', out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--json', default='outputs/interp/grid.json')
    p.add_argument('--figs', default='outputs/interp/figs')
    args = p.parse_args()
    rows = collect()
    for r in rows:
        print(f"{r['panel']:26s} {r['shift']:13s} base {r['base']['k']:3d}/{r['base']['n']:<3d} "
              f"repaired {r['rep']['k']:3d}/{r['rep']['n']:<3d}"
              + (f" unshifted {r['rep_none']['k']}/{r['rep_none']['n']}" if 'rep_none' in r else ''))
    with open(args.json, 'w') as f:
        json.dump(rows, f, indent=1)
    if rows:
        figure(rows, args.figs)
    if glob.glob('outputs/interp/nuis/sq_depth_full_r4_eval3_bothcam/ep*.npz'):
        fig_depth(args.figs)


if __name__ == '__main__':
    main()
