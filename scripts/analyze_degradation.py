#!/usr/bin/env python3
"""Completion time as the primary outcome; success rate is its censored shadow.

Every failure in 7170 episodes (excluding each cell's first, whose duration the recorder documents
as unreliable) is a horizon timeout. Nothing fails at the task; things fail to FINISH. So the
outcome is completion time, censored at the 20 s horizon, and "success rate" is one point on that
distribution. That distinction is what separates degrading gracefully from degrading catastrophically, and the
three signatures it exposes are the point of the analysis:

  synchronous        command rate falls, completion time grows as the duty cycle predicts (within
                     9%), the plateau holds (0.90 -> 0.97 -> 0.76). It converts delay into TIME.
  naive_async        command rate holds at ~19 Hz, completion time grows MORE than predicted
                     (+32 to +67%), the plateau falls. It pays twice: stale actions cost recovery
                     time and completion.
  TE / RTC           command rate falls but completion time does not move (median ~8.5 s at every
                     delay); the plateau collapses instead (0.83 -> 0.62 -> 0.00). They refuse to
                     slow down and fail rather than wait.

So the duty-cycle law is specific to blocking execution. Reported as a rate alone this looks like
"synchronous is slow"; reported as a distribution it is the only strategy that degrades gracefully.

Run from the repo root; reads outputs/paper1/*.episodes.csv. Pure stdlib.
"""
import csv
import re
import statistics
from collections import defaultdict

PAT = re.compile(r'^strat=(?P<strat>.+?)_reactive=(?P<reactive>\w+)_place=(?P<place>\w+)'
                 r'_exec=(?P<exec>\w+)_jit=(?P<jit>[^_]+)_loss=(?P<loss>[^_]+)_lat=[\d.]+'
                 r'_(?:latency|drop)=(?P<swept>.+)$')

def load(files):
    cells = defaultdict(list)     # key -> [(scene, success, wall_s)]
    for f in files:
        for r in csv.DictReader(open(f'outputs/paper1/{f}.episodes.csv')):
            d = PAT.match(r['condition']).groupdict()
            if int(r['scene']) == 0:      # documented as an upper bound, not a measurement
                continue
            cells[(d['place'], d['exec'], d['strat'], d['reactive'] == 'True',
                   d['jit'], d['loss'], d['swept'])].append(
                       (int(r['scene']), int(r['success']), float(r['wall_s'])))
    return cells

def completed_by(eps, t):
    """Fraction of episodes finishing within t seconds (a censored survival point)."""
    return sum(1 for _, s, w in eps if s and w <= t) / len(eps)

cells = load(['e1', 'e3', 'e3ext'])
STRATS = ('synchronous', 'naive_async', 'temporal_ensemble', 'rtc')
TS = (8, 10, 12, 15, 20)

print('=== 1. Buffered execution: the distribution SHIFTS (graceful)')
print('    action-path delay, reactive layer off where available, else on')
print(f"{'strategy':19s}{'ms':>6}" + ''.join(f'{f"<={t}s":>8}' for t in TS) + f"{'med':>7}")
for strat in STRATS:
    for lat in ('0', '200', '400', '800', '1600'):
        eps = (cells.get(('act', 'robot', strat, False, 'gaussian:0', 'iid:0:100', lat))
               or cells.get(('act', 'robot', strat, True, 'gaussian:0', 'iid:0:100', lat)))
        if not eps:
            continue
        done = [w for _, s, w in eps if s]
        med = f'{statistics.median(done):.1f}' if done else '-'
        print(f"{strat:19s}{lat:>6}" + ''.join(f'{completed_by(eps, t):>8.2f}' for t in TS)
              + f"{med:>7}")

print('\n=== 2. Streamed execution: the distribution TRUNCATES (catastrophic)')
print(f"{'strategy':19s}{'ms':>6}" + ''.join(f'{f"<={t}s":>8}' for t in TS) + f"{'med':>7}")
for strat in ('synchronous', 'rtc'):
    for lat in ('0', '200', '400', '800'):
        eps = (cells.get(('act', 'policy', strat, False, 'gaussian:0', 'iid:0:100', lat))
               or cells.get(('act', 'policy', strat, True, 'gaussian:0', 'iid:0:100', lat)))
        if not eps:
            continue
        done = [w for _, s, w in eps if s]
        med = f'{statistics.median(done):.1f}' if done else '-'
        print(f"{strat:19s}{lat:>6}" + ''.join(f'{completed_by(eps, t):>8.2f}' for t in TS)
              + f"{med:>7}")

print('\n=== 3. The two degradation modes, as numbers')
print('    "plateau" = fraction completing by the 20 s horizon; "median" = time among completers')
print(f"{'exec':9s}{'strategy':19s}{'median 0->max delay':>22}{'plateau 0->max delay':>24}")
for ex in ('robot', 'policy'):
    for strat in STRATS:
        row = []
        for lat in ('0', '200', '400', '800'):
            eps = (cells.get(('act', ex, strat, False, 'gaussian:0', 'iid:0:100', lat))
                   or cells.get(('act', ex, strat, True, 'gaussian:0', 'iid:0:100', lat)))
            if not eps:
                row.append(None)
                continue
            done = [w for _, s, w in eps if s]
            row.append((statistics.median(done) if done else None, completed_by(eps, 20)))
        if any(r is None for r in row):
            continue
        meds = ' -> '.join('n/a' if r[0] is None else f'{r[0]:.1f}' for r in row)
        plat = ' -> '.join(f'{r[1]:.2f}' for r in row)
        print(f"{'buffered' if ex == 'robot' else 'streamed':9s}{strat:19s}{meds:>22}{plat:>24}")

print('\n=== 4. Does the duty cycle predict completion TIME, not just command rate?')
print('    predicted time ratio = command rate at 0 ms / command rate at this delay')
rates = {}
for r in csv.DictReader(open('outputs/paper1/e1.csv')):
    if r['placement'] == 'act' and r['executor'] == 'robot':
        rates[(r['strategy'], f"{float(r['latency_ms']):.0f}")] = float(r['wp_rx_hz'])
print(f"{'strategy':19s}{'ms':>6}{'cmd Hz':>9}{'predicted':>11}{'measured':>10}{'err':>8}")
for strat in STRATS:
    base_eps = (cells.get(('act', 'robot', strat, False, 'gaussian:0', 'iid:0:100', '0'))
                or cells.get(('act', 'robot', strat, True, 'gaussian:0', 'iid:0:100', '0')))
    base_med = statistics.median([w for _, s, w in base_eps if s])
    r0 = rates.get((strat, '0'))
    for lat in ('200', '400', '800'):
        eps = (cells.get(('act', 'robot', strat, False, 'gaussian:0', 'iid:0:100', lat))
               or cells.get(('act', 'robot', strat, True, 'gaussian:0', 'iid:0:100', lat)))
        done = [w for _, s, w in eps if s] if eps else []
        rd = rates.get((strat, lat))
        if not done or not rd or not r0:
            continue
        pred, meas = r0 / rd, statistics.median(done) / base_med
        print(f"{strat:19s}{lat:>6}{rd:>9.1f}{pred:>11.2f}{meas:>10.2f}{100*(meas-pred)/pred:>7.0f}%")
