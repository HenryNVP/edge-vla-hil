#!/usr/bin/env python3
"""The second term in the degradation law: commanded path length.

The duty-cycle law `duty = k/(k+d)` predicts command rate and completion time to within 4% on both
the delay and the chunk-horizon axis, but it does not predict success below duty ~0.45, and it cannot
explain why blocking and naive_async change places between the two axes. This finds the missing
quantity in a column the recorder was already writing.

`wp_step_mm` is the mean distance between consecutive commanded waypoints. Times the command rate,
times the median completion time, it is the total distance the arm was told to travel to do a job
whose real distance the task fixes. Anything above that is wandering, and it separates the two
strategies that merely get slower:

  blocking      1557 mm, CV 5.7% over 12 cells spanning duty 0.29-0.88, d = 2-34, k = 4-15 and both
                deadlines. INVARIANT: delay buys time and nothing else.
  naive_async   2717 mm, CV 20.9%, up to 2.2x the task distance. Variance ratio 41 (F, df 8 and 11,
                0.01 critical value ~5.4), so this is a difference in kind, not a noisy gap.

Which resolves the reversal. At d = 10, shortening the chunk moves the two costs in opposite
directions: blocking's time grows 11.9 -> 19.7 s at constant path, while naive_async's path falls
3229 -> 2093 mm at constant rate, because a shorter chunk holds fewer stale actions to execute before
a fresh one lands. One strategy degrades while the other improves, in the same cells.

  Blocking pays in TIME at constant PATH.  naive_async pays in PATH at constant RATE.

Both resolve to seconds -- which is why E6's longer deadline rescued both -- but the duty cycle only
captures the first.

Two limits, which bound the claim rather than break it: `wp_step_mm` is a per-cell mean, so these are
12 and 9 cell-level observations rather than 360 episodes; and medians are over successful episodes
only, so the longest-delay naive_async cell is survivorship-biased (2646 mm, below the d = 18 cell).
The trend in d is therefore suggestive (slope +16 mm/step, t = 0.71); the difference in LEVEL and in
VARIANCE is what the statistics support. Recording per-episode path length would fix both.

Run from the repo root; reads outputs/paper1/*.csv and *.episodes.csv. Pure stdlib.
"""
import csv
import glob
import math
import statistics
from collections import defaultdict

TASK_DISTANCE_MM = 1557.0    # blocking's invariant, i.e. the task's own distance plus its overhead


def commanded_path_mm(rate_hz: float, step_mm: float, seconds: float) -> float:
    """Total commanded travel: waypoints per second x mm per waypoint x seconds.

    Kept separate from the loading so the arithmetic is testable without a sweep on disk. All three
    factors are per-cell means, so the result is a cell-level estimate and not a per-episode one.
    """
    return rate_hz * step_mm * seconds


def load_cells(pattern: str = 'outputs/paper1/*.csv'):
    """Buffered act-path blocking and naive_async cells on a clean channel, with their medians.

    Only `executor=robot, placement=act` with no jitter and no loss, reactive off: the comparable set
    across E1/E3/E3ext (delay axis), E5 (horizon axis) and E6 (deadline axis). Cells whose columns
    predate a factor read their default, which is why `max_chunk_actions` falls back to the
    checkpoint's 15 and `max_episode_s` to 20.
    """
    episodes = defaultdict(list)
    for path in sorted(glob.glob(pattern.replace('.csv', '.episodes.csv'))):
        src = path.split('/')[-1][:-len('.episodes.csv')]
        for row in csv.DictReader(open(path)):
            if int(row['scene']) == 0:        # documented as an upper bound, not a measurement
                continue
            episodes[(src, row['condition'])].append((int(row['success']), float(row['wall_s'])))

    cells = []
    for path in sorted(glob.glob(pattern)):
        if 'episodes' in path:
            continue
        src = path.split('/')[-1][:-4]
        for row in csv.DictReader(open(path)):
            if row.get('executor') != 'robot' or row.get('placement') != 'act':
                continue
            if float(row.get('jitter_ms') or 0) or float(row.get('drop_prob') or 0):
                continue
            if row['strategy'] not in ('synchronous', 'naive_async'):
                continue
            if (row.get('reactive') or '').lower() not in ('false', '0'):
                continue
            d = float(row.get('d_chunk_steps_p50') or 0)
            if d <= 0:
                continue
            done = [w for ok, w in episodes.get((src, row['condition']), []) if ok]
            if not done:                      # a cell where nothing finished has no path to measure
                continue
            k = int(row.get('max_chunk_actions') or 0) or 15
            med = statistics.median(done)
            rate = float(row.get('wp_rx_hz') or 0)
            step = float(row.get('wp_step_mm') or 'nan')
            cells.append(dict(src=src, strat=row['strategy'], k=k, d=d, duty=k / (k + d),
                              deadline=float(row.get('max_episode_s') or 20.0),
                              success=float(row['success_rate']), median_s=med,
                              rate_hz=rate, step_mm=step,
                              path_mm=commanded_path_mm(rate, step, med)))
    return cells


def ols(xs, ys):
    """Slope, intercept and a t statistic for it. Two-variable fit, so scipy is not worth the import."""
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    se = (sum(r * r for r in resid) / (n - 2) / sxx) ** 0.5
    # An exact fit has no residual and so no standard error. That is a degenerate input rather than
    # infinite confidence, but returning inf keeps the caller's comparisons ordered instead of
    # raising ZeroDivisionError on a synthetic or single-slope dataset.
    return intercept, slope, math.inf if se == 0 else slope / se


def main():
    cells = load_cells()
    for strat in ('synchronous', 'naive_async'):
        cs = sorted((c for c in cells if c['strat'] == strat), key=lambda c: -c['duty'])
        print(f'--- {strat}')
        print(f"{'cell':22s}{'k':>3}{'d':>4}{'duty':>6}{'dl':>4}"
              f"{'succ':>6}{'med':>7}{'mm/wp':>7}{'path':>7}")
        for c in cs:
            print(f"{c['src']:22s}{c['k']:>3}{c['d']:>4.0f}{c['duty']:>6.3f}{c['deadline']:>4.0f}"
                  f"{c['success']:>6.2f}{c['median_s']:>7.1f}{c['step_mm']:>7.2f}{c['path_mm']:>7.0f}")
        paths = [c['path_mm'] for c in cs]
        _, slope, t = ols([c['d'] for c in cs], paths)
        cv = 100 * statistics.stdev(paths) / statistics.mean(paths)
        print(f"{'':22s}path {statistics.mean(paths):.0f} mm, CV {cv:.1f}%, "
              f"slope in d {slope:+.1f} mm/step (t={t:+.2f})\n")

    var = {s: statistics.variance([c['path_mm'] for c in cells if c['strat'] == s])
           for s in ('synchronous', 'naive_async')}
    print(f"variance ratio naive/blocking = {var['naive_async'] / var['synchronous']:.0f}")


if __name__ == '__main__':
    main()
