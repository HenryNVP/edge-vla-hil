#!/usr/bin/env python3
"""Time-to-success curves: what a longer deadline reveals that success rate hides.

Success rate is one point on a completion-time distribution, read at the episode deadline. Under
delay, every failed episode in the static sweeps was a timeout, so what matters is whether an
execution strategy is SLOW (its curve keeps rising past the deadline) or BROKEN (its curve is flat
and stays flat). One deadline cannot tell those apart; E6 ran the same channel at two, and this
draws both on one axis.

The estimator is Kaplan-Meier, as in PhAIL (arXiv 2605.29710), with a timeout treated as right-
censored at the deadline, never as a completion at it. In these sweeps every episode in a cell
shares one deadline, so Kaplan-Meier reduces to the plain fraction completed by t, for t below the
deadline. It is implemented in full anyway, because pooling cells with different deadlines (20 s
and 40 s) is exactly the case where the simple fraction goes wrong.

RMST(tau) -- restricted mean survival time, the area under the not-yet-finished curve up to tau --
is the single number that summarises a curve without choosing a threshold. tau must not exceed the
shortest deadline in the pool, since nothing is observed beyond it.

The first episode of every cell is excluded: its duration is documented as unreliable (it
includes the policy's warm-up).

`wall_s` is WALL-CLOCK time, but the deadline is enforced in SIMULATED time (robosuite's horizon),
and the two disagree by up to ~2.5 s: E6's 20 s timeouts log 17.5-23.3 s, and a few successes log
past their deadline (20.3 s, 41.2 s). So completion times carry that much measurement noise. Here
a timeout is censored at the deadline and a success logged past it is clamped to the deadline.
Sweeps recorded after 2026-09-28 carry `sim_s` (the plant's env.timestep / action_hz), which is
used when present.

Run from the repo root:
    python3 scripts/analyze_survival.py    # E6 table + outputs/figures/e6_survival.png

(outputs/paper1 is written by the sweep containers as root, so figures go to outputs/figures.)
"""
from __future__ import annotations

import csv
from collections import defaultdict

DATA = 'outputs/paper1'
FIGS = 'outputs/figures'
STRATS = ('synchronous', 'naive_async', 'temporal_ensemble', 'rtc')


def load(path: str, deadline: float, drop_first: bool = True) -> dict[str, list[tuple]]:
    """Episodes per strategy as (time, completed, deadline), from one *.episodes.csv."""
    cells: dict[str, list[tuple]] = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if drop_first and int(r['scene']) == 0:
                continue
            strat = r['condition'].split('_reactive=')[0].removeprefix('strat=')
            done = bool(int(r['success']))
            # prefer the plant's simulated duration where the file has it; otherwise a completion
            # logged past the deadline happened inside it in simulated time (module docstring),
            # so it is clamped to the deadline, not dropped or extended
            sim = float(r.get('sim_s') or 'nan')
            t = sim if sim == sim else min(float(r['wall_s']), deadline)
            cells[strat].append((t if done else deadline, done, deadline))
    return cells


def km(episodes: list[tuple]) -> list[tuple[float, float]]:
    """Kaplan-Meier: [(t, S(t))] at each completion time, S = fraction not yet finished.

    `episodes` are (time, completed, deadline); a non-completion is censored at `time`. At a tie,
    completions are counted before censorings, the standard convention.
    """
    order = sorted(episodes, key=lambda e: (e[0], not e[1]))
    at_risk, s, curve = len(order), 1.0, [(0.0, 1.0)]
    i = 0
    while i < len(order):
        t = order[i][0]
        events = censored = 0
        while i < len(order) and order[i][0] == t:
            if order[i][1]:
                events += 1
            else:
                censored += 1
            i += 1
        if events:
            s *= 1.0 - events / at_risk
            curve.append((t, s))
        at_risk -= events + censored
    return curve


def survival_at(curve: list[tuple[float, float]], t: float) -> float:
    """S(t) from a step curve (right-continuous)."""
    s = 1.0
    for ti, si in curve:
        if ti > t:
            break
        s = si
    return s


def rmst(episodes: list[tuple], tau: float) -> float:
    """Restricted mean time-to-success up to tau: the area under S(t) on [0, tau]."""
    if tau > min(e[2] for e in episodes) + 1e-9:
        raise ValueError(f'tau={tau} exceeds the shortest deadline; nothing is observed past it')
    curve = km(episodes)
    area, prev_t, prev_s = 0.0, 0.0, 1.0
    for t, s in curve[1:]:
        if t >= tau:
            break
        area += prev_s * (t - prev_t)
        prev_t, prev_s = t, s
    return area + prev_s * (tau - prev_t)


def main() -> None:
    runs = {20.0: load(f'{DATA}/e6_deadline_h20.episodes.csv', 20.0),
            40.0: load(f'{DATA}/e6_deadline_h40.episodes.csv', 40.0)}

    print('E6: 1600 ms action-path delay, buffered executor. Fraction finished by t.')
    print(f"{'strategy':19s}{'deadline':>9}{'n':>4}{'by 10s':>8}{'by 20s':>8}{'by 30s':>8}"
          f"{'by 40s':>8}{'RMST(dl)':>10}")
    for strat in STRATS:
        for dl, cells in runs.items():
            eps = cells.get(strat)
            if not eps:
                continue
            curve = km(eps)
            done = [f"{1 - survival_at(curve, t):>8.2f}" if t <= dl else f"{'-':>8}"
                    for t in (10, 20, 30, 40)]
            print(f"{strat:19s}{dl:>8.0f}s{len(eps):>4}{''.join(done)}{rmst(eps, dl):>10.1f}")
    print('Consistency: the 40 s run read at 20 s must reproduce the 20 s run, or the two cells'
          ' are not one distribution.')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    import os
    os.makedirs(FIGS, exist_ok=True)
    colors = dict(zip(STRATS, ('#2a78d6', '#eb6834', '#1baf7a', '#b98100')))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for strat in STRATS:
        eps = runs[40.0].get(strat)
        if not eps:
            continue
        curve = km(eps)
        ts = [0.0] + [t for t, _ in curve[1:]] + [40.0]
        ys = [0.0] + [1 - s for _, s in curve[1:]] + [1 - curve[-1][1]]
        n_done = sum(1 for e in eps if e[1])
        ax.step(ts, ys, where='post', color=colors[strat], linewidth=1.8,
                label=f'{strat}: {n_done}/{len(eps)} finished')
    ax.text(20.3, 0.93, 'original 20 s deadline', fontsize=7, color='0.4')
    ax.axvline(20, color='0.5', linewidth=0.8)
    ax.set_xlabel('time since episode start (s)')
    ax.set_ylabel('fraction of episodes finished')
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlim(0, 42)
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.set_title('1600 ms round trip, 40 s deadline: slow (blocking, naive async) vs broken (TE, RTC)',
                 fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f'{FIGS}/e6_survival.png', dpi=160)
    print(f'wrote {FIGS}/e6_survival.png')


if __name__ == '__main__':
    main()
