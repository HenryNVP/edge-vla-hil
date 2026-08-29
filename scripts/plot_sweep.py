#!/usr/bin/env python3
"""Turn a sweep CSV into the headline figure: success rate vs network degradation.

Two things this does that a bare plot of the CSV would not, both because the raw rows invite a
wrong reading:

  * Error bars are Wilson score intervals, not +/- sqrt(p(1-p)/n). At the n per cell a sweep
    realistically reaches (20-50), the normal approximation puts the bar above 100% for any cell
    near ceiling — which is most of the interesting ones — and understates uncertainty exactly
    where the curves separate. Wilson stays inside [0, 1] and is honest at small n.

  * Rows for strategies that are behaving IDENTICALLY can be pooled with --pool. `rtc` and
    `network_aware` differ only in their delay forecast, and when the delay distribution is
    light-tailed `ceil(p95) == max`, so the two executors compute the same freeze horizon and
    their rows are replicates of one condition. Pooling them is then legitimate and doubles n;
    pooling them when the forecasts DO differ would be silently averaging two methods, so this
    is opt-in and never the default.

Usage:
    python3 scripts/plot_sweep.py outputs/sweep.csv -o outputs/sweep.png --pool rtc,network_aware
"""
from __future__ import annotations

import argparse
import collections
import csv
import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves at small n and near 0/1, unlike the normal approximation."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load(path: str, pool: list[str], axis: str) -> dict:
    """{(label, reactive): {x: (successes, trials)}} with pooled strategies merged."""
    out: dict = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row.get('truncated', 'False') == 'True':
                continue      # short cell: smaller n than the cells it would be plotted against
            name = '+'.join(pool) if row['strategy'] in pool else row['strategy']
            trials = int(row['trials'])
            cell = out[(name, row['reactive'] == 'True')][float(row[axis])]
            cell[0] += round(float(row['success_rate']) * trials)
            cell[1] += trials
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('csv')
    p.add_argument('-o', '--out', default='sweep.png')
    p.add_argument('--axis', default='latency_ms', help='CSV column to put on the x axis')
    p.add_argument('--pool', default='',
                   help='comma-separated strategies to merge as replicates of one condition; '
                        'only valid when they are provably behaving identically')
    p.add_argument('--title', default='Task success vs network degradation')
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pool = [s.strip() for s in args.pool.split(',') if s.strip()]
    series = load(args.csv, pool, args.axis)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    for (name, reactive), cells in sorted(series.items()):
        xs = sorted(cells)
        ys = [cells[x][0] / cells[x][1] for x in xs]
        lo = [ys[i] - wilson(*cells[x])[0] for i, x in enumerate(xs)]
        hi = [wilson(*cells[x])[1] - ys[i] for i, x in enumerate(xs)]
        n = min(cells[x][1] for x in xs)
        ax.errorbar(xs, ys, yerr=[lo, hi], marker='o', capsize=3, linewidth=1.8,
                    markersize=5, alpha=0.9,
                    linestyle='-' if reactive else '--',
                    label=f'{name} · reactive {"on" if reactive else "off"} (n≥{n})')

    ax.set_xlabel(args.axis.replace('_', ' '))
    ax.set_ylabel('task success rate')
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_title(args.title, fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(args.out)
    print(f'[plot] {args.out}  ({len(series)} series, error bars = Wilson 95%)')


if __name__ == '__main__':
    main()
