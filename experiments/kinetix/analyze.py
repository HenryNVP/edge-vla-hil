"""Summarise a Kinetix sweep: success over (delay, world speed) per strategy, and where each wins.

Reads the CSV written by runner.py (one row per level per configuration) and reports, for each
world speed alpha and delay d:

  * each strategy's solve rate averaged over the 12 levels, at a FIXED execution horizon s and at
    its BEST s (chosen per (method, alpha, d) on the mean over levels -- an oracle tuning that
    favours every method equally; say so when reporting it);
  * which strategy wins, and by how much.

Figure: one panel per world speed, solve rate vs d/H for sync / naive / RTC at best s, the
published envelope (d <= H/2) shaded.

Plain Python + matplotlib, runs outside the Kinetix environment:
    python3 experiments/kinetix/analyze.py outputs/kinetix/sweep_v1.csv
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

H = 8
METHODS = ('sync', 'naive', 'rtc')
COLORS = {'sync': '#2a78d6', 'naive': '#eb6834', 'rtc': '#b98100'}


def load(path: str) -> dict:
    """{(method, alpha, s, d): mean solve over levels}."""
    acc = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r['method'], float(r['world_speed']), int(r['execute_horizon']), int(r['delay']))
            acc[key].append(float(r['solve_rate']))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def best_over_s(table: dict) -> dict:
    """{(method, alpha, d): (best solve, its s)}."""
    best = {}
    for (m, a, s, d), v in table.items():
        k = (m, a, d)
        if k not in best or v > best[k][0]:
            best[k] = (v, s)
    return best


def main(path: str) -> None:
    table = load(path)
    best = best_over_s(table)
    alphas = sorted({a for _, a, _ in best})
    delays = sorted({d for _, _, d in best})
    for a in alphas:
        print(f'\nworld speed {a:g}   (solve rate at best s; s in brackets)')
        print(f"{'d':>3} {'d/H':>5} " + ''.join(f'{m:>13}' for m in METHODS) + '   winner')
        for d in delays:
            cells = {m: best.get((m, a, d)) for m in METHODS}
            row = ''.join(f'{v:>8.3f} [{s}]' if (v := cells[m]) and (s := cells[m][1]) and
                          (v := cells[m][0]) is not None else f"{'-':>13}" for m in METHODS)
            present = {m: c[0] for m, c in cells.items() if c}
            win = max(present, key=present.get) if present else '-'
            margin = sorted(present.values())[-1] - sorted(present.values())[-2] if len(
                present) > 1 else 0
            print(f'{d:>3} {d / H:>5.2f} {row}   {win} (+{margin:.2f})')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, len(alphas), figsize=(2.6 * len(alphas), 2.8), sharey=True)
    axes = axes if len(alphas) > 1 else [axes]
    for ax, a in zip(axes, alphas):
        ax.axvspan(0, 0.5, color='0.92', zorder=0)
        for m in METHODS:
            xs = [d / H for d in delays if (m, a, d) in best]
            ys = [best[(m, a, d)][0] for d in delays if (m, a, d) in best]
            ax.plot(xs, ys, marker='o', markersize=3, linewidth=1.6, color=COLORS[m], label=m)
        ax.set_title(f'world speed {a:g}', fontsize=9)
        ax.set_xlabel('delay / chunk (d/H)', fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel('solve rate (12 levels)', fontsize=8)
    axes[0].text(0.02, 0.02, 'published\nrange', fontsize=6, color='0.4')
    axes[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = path.rsplit('.', 1)[0] + '_map.png'
    fig.savefig(out, dpi=160)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'outputs/kinetix/sweep_v1.csv')
