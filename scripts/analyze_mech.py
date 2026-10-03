"""Summarise mechanism-testbed sweeps (scripts/oracle_lift.py): success and completion time per cell.

Completion time is the Kaplan-Meier median with timeouts censored (paper 1's protocol), in
control ticks; '-' when fewer than half the episodes succeed. Usage:

    python3 scripts/analyze_mech.py outputs/mech/m0_baselines.csv outputs/mech/m13_mechanisms.csv
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict


def km_median(times_events: list[tuple[int, bool]]) -> float | None:
    """Kaplan-Meier median time-to-success; events are successes, timeouts are censored."""
    n = len(times_events)
    surv, at_risk = 1.0, n
    for t in sorted({t for t, _ in times_events}):
        events = sum(1 for tt, e in times_events if tt == t and e)
        if events:
            surv *= 1.0 - events / at_risk
            if surv <= 0.5:
                return float(t)
        at_risk -= sum(1 for tt, _ in times_events if tt == t)
    return None


def load(paths) -> dict:
    cells = defaultdict(list)
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                key = (r.get('tracker', 'privileged'), float(r['speed']), r['strategy'],
                       r['delay'])
                cells[key].append((int(r['ticks']), r['success'] == '1', int(r['requests'])))
    return cells


def main() -> None:
    cells = load(sys.argv[1:])
    groups = sorted({(k[0], k[1]) for k in cells})
    delays = sorted({k[3] for k in cells},
                    key=lambda d: (0, int(d), '') if d.isdigit() else (1, 0, d))
    strategies = sorted({k[2] for k in cells})
    for tr, v in groups:
        print(f'\ntracker {tr}, cube speed {v} m/s   (success/n, KM median ticks)')
        print(f'{"strategy":22s}' + ''.join(f'{"d=" + str(d):>16s}' for d in delays))
        for s in strategies:
            row = f'{s:22s}'
            for d in delays:
                eps = cells.get((tr, v, s, d))
                if not eps:
                    row += f'{"":>16s}'
                    continue
                k = sum(e for _, e, _ in eps)
                med = km_median([(t, e) for t, e, _ in eps])
                row += f'{f"{k}/{len(eps)} " + ("-" if med is None else f"{med:.0f}"):>16s}'
            print(row)


if __name__ == '__main__':
    main()
