#!/usr/bin/env python3
"""Interp: the paper's four figures from the saved confidence intervals and gain maps.

  fig_seen_vs_action  seen vs action gain, inside vs outside the demos' range
  fig_swap            the 2x2 head-state x feature-direction swap table
  fig_closepoint      per nut shift: success, and how much of the shift the first gripper
                      close followed (1 = at the real handle, 0 = at the demo position)
  fig_gain_space      local gain over the Lift workspace (DP, ACT), demo range outlined

Every gains/swap number is plotted at the SMALLER probe radius (square DP 3 cm, lift 2 cm).
Whiskers are always the 95% CIs recorded in ci.json (bootstrap, Wilson for success rates),
drawn asymmetric; two measures never share a panel pair on a second y-axis, they get two
panels.

    python3 scripts/interp_figs.py --data outputs/interp --out outputs/interp/figs
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interp_gainmap import TASKS, jacobians  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

BLUE, ORANGE, GREEN = '#0072B2', '#E69F00', '#009E73'    # categorical, fixed order
POLICIES = ('square DP', 'lift DP', 'lift ACT')          # gains section, fig 1
SWAP_POLICIES = ('lift DP', 'square DP')                 # fig 2
SPACE_POLICIES = ('lift DP', 'lift ACT')                 # fig 4: same task, same workspace
RADIUS = {'square DP': 0.03, 'lift DP': 0.02, 'lift ACT': 0.02}    # m, the smaller radius
TASK = {'square DP': 'square', 'lift DP': 'lift', 'lift ACT': 'lift'}
GAINMAP = {'square DP': 'gainmap.npz', 'lift DP': 'gainmap_lift_dp.npz',
           'lift ACT': 'gainmap_lift_act.npz'}
SINGLE, DOUBLE = 3.4, 7.0                                # column widths, in


def pick(rows: list, **kw) -> dict:
    """The one ci.json row matching every field (floats compared with tolerance)."""
    for r in rows:
        if all(np.isclose(r[k], v) if isinstance(v, float) else r[k] == v
               for k, v in kw.items()):
            return r
    raise KeyError(f'no ci.json row for {kw}')


def dot(ax, x: float, row: dict, color: str, inside: bool, ms: float = 3.5) -> None:
    """One estimate as a dot with asymmetric 95% CI whiskers; filled marker = inside."""
    ax.errorbar([x], [row['est']],
                yerr=[[max(0.0, row['est'] - row['lo'])], [max(0.0, row['hi'] - row['est'])]],
                fmt='o', color=color, markersize=ms, markeredgewidth=0.8, lw=0.9, capsize=2,
                capthick=0.9, markerfacecolor=color if inside else 'none', zorder=3)


def style(ax) -> None:
    """Light y-only grid, no top/right spines."""
    ax.grid(axis='y', color='0.87', lw=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, pad=2)


def legend_handle(color: str, label: str, inside: bool = True) -> Line2D:
    return Line2D([], [], ls='none', marker='o', ms=3.5, color=color, markeredgewidth=0.8,
                  markerfacecolor=color if inside else 'none', label=label)


def save(fig, name: str, out: str) -> None:
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=300)
    plt.close(fig)
    print(f'[fig] {out}/{name}.pdf, .png')


def fig_seen_vs_action(rows: list, out: str) -> None:
    """Seen vs action gain per policy and axis; colour = seen/action, fill = inside/outside."""
    labels = ('seen inside', 'seen outside', 'action inside', 'action outside')
    kind = {'seen': BLUE, 'action': ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), sharey=True, constrained_layout=True)
    shown = [r for r in rows if r['label'] in labels]
    for ax, axis in zip(axes, ('x', 'y')):
        for gi, pol in enumerate(POLICIES):
            for li, lab in enumerate(labels):
                row = pick(shown, policy=pol, radius=RADIUS[pol], label=lab, axis=axis)
                dot(ax, gi + 0.2 * (li - 1.5), row, kind[lab.split()[0]],
                    inside=lab.endswith('inside'))
        ax.axhline(1.0, ls='--', lw=0.8, color='0.45', zorder=2)
        ax.text(2.45, 1.01, 'follows the object', va='bottom', color='0.25')
        ax.set_xlim(-0.55, 3.15)
        ax.set_xticks(range(len(POLICIES)), POLICIES)
        ax.set_title(f'{axis} axis')
        style(ax)
    axes[0].set_ylim(min(0.0, min(r['lo'] for r in shown) - 0.06),
                     max(1.16, max(r['hi'] for r in shown) + 0.08))
    axes[0].set_ylabel('gain (mm / mm)')
    handles = [legend_handle(c, f'{k} {"inside" if inside else "outside"}', inside)
               for k, c in kind.items() for inside in (True, False)]
    fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False,
               columnspacing=1.2, handletextpad=0.4)
    save(fig, 'fig_seen_vs_action', out)


def fig_swap(rows: list, out: str) -> None:
    """The 2x2 swap table: x = feature direction, colour = head state; rows = gain axis."""
    heads = (('head in', BLUE, 'head in-range'), ('head out', ORANGE, 'head out-of-range'))
    fig, axes = plt.subplots(2, 2, figsize=(SINGLE, 2.7), sharex=True, sharey='row',
                             constrained_layout=True)
    for ri, axis in enumerate(('x', 'y')):
        for ci, pol in enumerate(SWAP_POLICIES):
            ax = axes[ri, ci]
            for hi, (head, color, _) in enumerate(heads):
                for di, dirn in enumerate(('dir in', 'dir out')):
                    row = pick(rows, policy=pol, radius=RADIUS[pol], axis=axis,
                               label=f'{head}, {dirn}')
                    dot(ax, di + 0.18 * (hi - 0.5), row, color, inside=True)
            ax.set_xlim(-0.4, 1.4)
            ax.set_xticks((0, 1), ('in-range', 'out-of-range'))
            ax.margins(y=0.2)
            style(ax)
        axes[ri, 0].set_ylabel(f'{axis} gain')
    for ci, pol in enumerate(SWAP_POLICIES):
        axes[0, ci].set_title(pol)
        axes[1, ci].set_xlabel('feature direction from')
    fig.align_ylabels(axes[:, 0])
    fig.legend(handles=[legend_handle(c, lab) for _, c, lab in heads],
               loc='outside lower center', ncol=2, frameon=False)
    save(fig, 'fig_swap', out)


def fig_closepoint(rows: list, out: str) -> None:
    """Success and followed-fraction vs nut shift; blue = x shift, orange = y shift."""
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 2.6), sharex=True, height_ratios=(1, 1.2),
                             constrained_layout=True)
    series = []
    for i, (name, color) in enumerate((('x shift', BLUE), ('y shift', ORANGE))):
        pts = sorted((r['shift'][i] * 100.0, r) for r in rows if r['shift'][i] != 0)
        if pts:
            series.append((name, color, pts))
    for name, color, pts in series:
        xs = [p[0] for p in pts]
        for ax, (est, lo, hi) in zip(axes, (('success', 'success_lo', 'success_hi'),
                                            ('followed', 'followed_lo', 'followed_hi'))):
            ys = [p[1][est] for p in pts]
            ax.errorbar(xs, ys, yerr=[[max(0.0, y - p[1][lo]) for y, p in zip(ys, pts)],
                                      [max(0.0, p[1][hi] - y) for y, p in zip(ys, pts)]],
                        fmt='o-', color=color, ms=3, lw=0.8, markeredgewidth=0.8,
                        capsize=2, capthick=0.9, zorder=3)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel('success')
    xs_all = [p[0] for _, _, pts in series for p in pts]
    tx = max(xs_all) + 0.02 * (max(xs_all) - min(xs_all))
    for y, lab, va in ((1.0, 'real handle', 'bottom'), (0.0, 'demo position', 'top')):
        axes[1].axhline(y, ls='--', lw=0.8, color='0.45', zorder=2)
        axes[1].text(tx, y + (0.015 if va == 'bottom' else -0.015), lab, ha='right', va=va,
                     color='0.25')
    axes[1].set_ylim(min(0.0, min(r['followed_lo'] for r in rows) - 0.2),
                     max(1.0, max(r['followed_hi'] for r in rows) + 0.2))
    axes[1].set_ylabel('shift followed')
    fig.align_ylabels(axes)
    axes[1].set_xlabel('nut shift (cm)')
    for ax in axes:
        style(ax)
    fig.legend(handles=[Line2D([], [], ls='-', lw=0.8, marker='o', ms=3, color=c, label=n)
                        for n, c, _ in series],
               loc='outside lower center', ncol=2, frameon=False)
    save(fig, 'fig_closepoint', out)


def fig_gain_space(data: str, out: str) -> None:
    """Local gain over the Lift workspace for DP and ACT: colour by gain at each probed position."""
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 2.1), sharex=True, sharey=True,
                             constrained_layout=True)
    norm = Normalize(0.0, 1.2, clip=True)
    for ci, pol in enumerate(SPACE_POLICIES):
        d = np.load(os.path.join(data, GAINMAP[pol]))
        m = np.isclose(d['radius'], RADIUS[pol])
        step = (d['grid'][1] - d['grid'][0]) * 1e3
        J = np.array([jacobians(r, step) for r in d['resp'][m]])        # (P, G-2, G-2, 2, 2)
        g = d['grid'][1:-1]
        xy = d['nut_xy'][m][:, None, None, :] + np.stack(np.meshgrid(g, g, indexing='ij'), -1)
        lo, hi = TASKS[TASK[pol]]['lo'], TASKS[TASK[pol]]['hi']
        for k, axis in enumerate(('x', 'y')):                           # J[...,k,k]: k gain
            ax = axes[2 * k + ci]
            sc = ax.scatter(xy[..., 0] * 100, xy[..., 1] * 100, c=J[..., k, k].ravel(),
                            cmap='Blues', norm=norm, s=5, linewidths=0)
            ax.add_patch(Rectangle((lo[0] * 100, lo[1] * 100), (hi[0] - lo[0]) * 100,
                                   (hi[1] - lo[1]) * 100, fill=False, edgecolor='k',
                                   lw=0.8, zorder=3))
            ax.set_aspect('equal')
            ax.set_title(f'{pol}, {axis} gain')
            ax.set_xlabel('object x (cm)')
            style(ax)
    axes[0].set_ylabel('object y (cm)')
    fig.colorbar(sc, ax=axes.tolist(), shrink=0.8, pad=0.01, label='gain (mm / mm)')
    save(fig, 'fig_gain_space', out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data', default='outputs/interp', help='dir with ci.json and the gain maps')
    p.add_argument('--out', default='outputs/interp/figs', help='dir for the pdf/png figures')
    args = p.parse_args(argv)
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                         'legend.frameon': False})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.data, 'ci.json')) as fh:
        rows = json.load(fh)
    fig_seen_vs_action([r for r in rows if r['section'] == 'gains'], args.out)
    fig_swap([r for r in rows if r['section'] == 'swap'], args.out)
    fig_closepoint([r for r in rows if r['section'] == 'rollouts'
                    and r.get('policy', 'square DP') == 'square DP'], args.out)
    fig_gain_space(args.data, args.out)


if __name__ == '__main__':
    main()
