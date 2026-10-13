#!/usr/bin/env python3
"""Check that the condition each results file claims is the condition that actually ran.

Every silent failure this project has had produced a plausible, complete-looking CSV: a relay
parameter that landed on the wrong node, columns that drifted mid-sweep, a timebase that ran
simulated time at 80%, an out-of-memory kill that stopped a sweep a third of the way through.
None of them raised. This script re-derives, from what each run MEASURED, whether it matches what
it was ASKED for, and fails loudly when it does not. It reads four formats, told apart by their
columns:

  testbed summary (benchmark.py)  every cell complete and untruncated; on a clean channel the
                                  measured delay is the injected one (action path: d_act = latency;
                                  observation path: d_obs = latency + the ~10 ms tick phase); on
                                  the robot executor the round trip in ticks is 2 + latency / 50
  per-episode logs (*.episodes)   every condition has the expected number of episodes, scenes
                                  0..n-1 once each; a success is never logged past its deadline
  Kinetix (runner.py)             all 12 levels present per configuration; the idle fraction is
                                  the one the schedule implies (sync: d / (d + s); time-aligned
                                  methods: 1 - coverage), which fails if the executor changed
  LIBERO (adapter.py)             every task has the same number of episodes; no episode ran past
                                  its deadline or stepped the world more often than it ticked

plus every file's manifest (scripts/run_manifest.py): present, commit recorded, tree clean, and
any logs passed with --logs: no out-of-memory, kill or unexplained traceback.

    python3 scripts/check_conditions.py outputs/paper1/e1.csv outputs/paper1/e1.episodes.csv \
        outputs/kinetix/sweep_v1.csv outputs/libero/grid2_sync_k10.csv --logs outputs/kinetix/*.log

Exit status 1 if anything FAILs; WARNs are for a human to read, not to gate on.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

PASS, WARN, FAIL = 'PASS', 'WARN', 'FAIL'
TICK_MS = 50.0
KINETIX_H = 8
KINETIX_LEVELS = 12


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, status: str, path: str, check: str, detail: str = '') -> None:
        self.rows.append((status, path, check, detail))

    @property
    def failed(self) -> bool:
        return any(r[0] == FAIL for r in self.rows)

    def print(self) -> None:
        for status, path, check, detail in self.rows:
            print(f'{status:4s}  {os.path.basename(path):34s}  {check:34s}  {detail}')
        c = Counter(r[0] for r in self.rows)
        print(f'\n{c[PASS]} passed, {c[WARN]} warnings, {c[FAIL]} failed')


def read(path: str) -> list[dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def kind_of(rows: list[dict]) -> str:
    cols = set(rows[0]) if rows else set()
    if {'placement', 'latency_ms', 'd_act_ms_p50'} <= cols:
        return 'testbed'
    if {'level', 'solve_rate', 'idle_frac', 'world_speed'} <= cols:
        return 'kinetix'
    if {'condition', 'task', 'scene', 'success'} <= cols:
        return 'libero'
    if {'condition', 'scene', 'success'} <= cols:
        return 'episodes'
    return 'unknown'


# ------------------------------------------------------------------ testbed summary
def check_testbed(path: str, rows: list[dict], rep: Report, expect_trials: int) -> None:
    bad_trials = [r['condition'] for r in rows if int(float(r['trials'])) < expect_trials]
    truncated = [r['condition'] for r in rows if r.get('truncated', 'False') == 'True']
    rep.add(FAIL if bad_trials else PASS, path, f'every cell has {expect_trials} trials',
            f'{len(bad_trials)} short: {bad_trials[:2]}' if bad_trials else f'{len(rows)} cells')
    rep.add(FAIL if truncated else PASS, path, 'no truncated cell',
            f'{truncated[:2]}' if truncated else '')

    delay_bad, chunk_bad, checked = [], [], 0
    for r in rows:
        clean = (float(r.get('jitter_ms') or 0) == 0 and float(r.get('drop_prob') or 0) == 0)
        if not clean:
            continue
        lat = float(r['latency_ms'])
        place = r['placement']
        checked += 1
        if place in ('act', 'both'):
            if abs(float(r['d_act_ms_p50']) - lat) > max(5.0, 0.02 * lat):
                delay_bad.append(f"{r['condition']}: d_act {float(r['d_act_ms_p50']):.0f} vs {lat:.0f}")
        if place in ('obs', 'both'):
            if abs(float(r['d_obs_ms_p50']) - (lat + 10.0)) > max(15.0, 0.05 * lat):
                delay_bad.append(f"{r['condition']}: d_obs {float(r['d_obs_ms_p50']):.0f} vs {lat + 10:.0f}")
        if r['executor'] == 'robot' and place in ('act', 'obs'):
            want = 2.0 + lat / TICK_MS
            if abs(float(r['d_chunk_steps_p50']) - want) > 1.0:
                chunk_bad.append(f"{r['condition']}: {r['d_chunk_steps_p50']} vs {want:.0f}")
    rep.add(FAIL if delay_bad else PASS, path, 'measured delay = injected',
            '; '.join(delay_bad[:2]) if delay_bad else f'{checked} clean-channel cells')
    rep.add(FAIL if chunk_bad else PASS, path, 'round trip = 2 + latency/50 ticks',
            '; '.join(chunk_bad[:2]) if chunk_bad else '')


# ------------------------------------------------------------------ per-episode logs
def check_episodes(path: str, rows: list[dict], rep: Report, expect: int | None) -> None:
    by = defaultdict(list)
    for r in rows:
        by[r['condition']].append(int(r['scene']))
    counts = {c: len(s) for c, s in by.items()}
    want = expect or max(counts.values())
    short = {c: n for c, n in counts.items() if n < want}
    rep.add(FAIL if short else PASS, path, f'every condition has {want} episodes',
            f'{len(short)} short, e.g. {next(iter(short.items()))}' if short
            else f'{len(counts)} conditions')
    per = defaultdict(list)                    # a fleet log numbers scenes per robot
    for r in rows:
        per[(r['condition'], r.get('robot', ''))].append(int(r['scene']))
    dup = [c for (c, _), s in per.items() if len(s) != len(set(s)) or sorted(s) != list(range(len(s)))]
    rep.add(FAIL if dup else PASS, path, 'scenes 0..n-1, each once',
            f'{dup[:2]}' if dup else '')
    if 'sim_s' not in rows[0]:
        rep.add(WARN, path, 'simulated time recorded',
                'wall_s only: completion times carry reset time (+-2.5 s)')
        return
    late = []
    for r in rows:
        m = re.search(r'deadline=([\d.]+)', r['condition'])
        if r['success'] == '1' and m and r.get('sim_s'):
            limit = float(m.group(1))
            limit_s = limit / 20.0 if limit > 100 else limit   # ticks or seconds
            if float(r['sim_s']) > limit_s + 1e-6:
                late.append(r['condition'])
    rep.add(FAIL if late else PASS, path, 'no success past its deadline',
            f'{len(late)} rows' if late else '')


# ------------------------------------------------------------------ Kinetix
def kinetix_idle(method: str, d: int, s: int, H: int = KINETIX_H) -> float:
    """Steady-state idle fraction the schedule implies (executor.cycle_plan)."""
    if method == 'sync':
        return d / (d + s)
    P = max(s, d)
    from_old = max(0, min(d, H - P))          # ticks k < d whose shifted old entry exists
    from_new = max(0, min(P, H) - d)          # ticks d <= k < P whose new entry exists
    return 1.0 - (from_old + from_new) / P


def check_kinetix(path: str, rows: list[dict], rep: Report) -> None:
    by = defaultdict(list)
    for r in rows:
        key = (r['method'], int(r['delay']), int(r['execute_horizon']), float(r['world_speed']),
               int(r['deadline']), int(r['num_evals']), int(r['seed']))
        by[key].append(r)
    short = [k for k, v in by.items() if len({r['level'] for r in v}) < KINETIX_LEVELS]
    rep.add(FAIL if short else PASS, path, f'all {KINETIX_LEVELS} levels per config',
            f'{len(short)} incomplete, e.g. {short[0]}' if short else f'{len(by)} configs')
    off = []
    for (m, d, s, a, dl, _n, _seed), v in by.items():
        if m not in ('sync', 'naive', 'rtc', 'hard'):
            continue
        measured = sum(float(r['idle_frac']) for r in v) / len(v)
        want = kinetix_idle(m, d, s)
        # the first chunk is free and the last cycle is cut by the deadline: allow for both
        tol = 0.06 + (d + s) / dl
        if abs(measured - want) > tol:
            off.append(f'{m} d={d} s={s}: {measured:.2f} vs {want:.2f}')
    rep.add(FAIL if off else PASS, path, 'idle fraction = schedule',
            '; '.join(off[:3]) if off else '')


# ------------------------------------------------------------------ LIBERO
def check_libero(path: str, rows: list[dict], rep: Report) -> None:
    by = defaultdict(Counter)
    for r in rows:
        by[r['condition']][r['task']] += 1
    uneven = [c for c, t in by.items() if len(set(t.values())) > 1]
    rep.add(WARN if uneven else PASS, path, 'same episodes per task',
            f'{len(uneven)} conditions uneven (still running?)' if uneven else
            f'{len(by)} conditions')
    over = []
    for r in rows:
        m = re.search(r'deadline=(\d+)', r['condition'])
        ticks = round(float(r['sim_s']) * 20)
        if m and ticks > int(m.group(1)):
            over.append(r['condition'])
        if int(r['env_steps']) > ticks:
            over.append(r['condition'] + ' (world stepped more than ticked)')
    rep.add(FAIL if over else PASS, path, 'within deadline, steps <= ticks',
            f'{over[:2]}' if over else '')


# ------------------------------------------------------------------ manifest, logs
def check_manifest(path: str, rep: Report) -> None:
    mpath = os.path.splitext(path)[0] + '.manifest.json'
    if path.endswith('.episodes.csv'):
        mpath = path[:-len('.episodes.csv')] + '.manifest.json'
    if not os.path.exists(mpath):
        rep.add(WARN, path, 'provenance manifest',
                'none: recorded before manifests existed, or by a runner without one')
        return
    with open(mpath) as f:
        runs = json.load(f)
    no_commit = [e['time'] for e in runs if not e.get('repo', {}).get('commit')]
    dirty = [e['time'] for e in runs if e.get('repo', {}).get('dirty')]
    rep.add(FAIL if no_commit else PASS, path, 'manifest records the commit',
            f'{len(runs)} invocation(s)')
    rep.add(WARN if dirty else PASS, path, 'run from a clean tree',
            f'{len(dirty)} invocation(s) had uncommitted changes' if dirty else '')


BAD_LOG = re.compile(r'RESOURCE_EXHAUSTED|Out of memory|Killed|MemoryError|CUDA out of memory')


def check_logs(paths: list[str], rep: Report) -> None:
    for p in paths:
        with open(p, errors='replace') as f:
            text = f.read()
        fatal = BAD_LOG.findall(text)
        # tracebacks from the EGL renderer at teardown are known noise; anything else is not
        blocks = text.split('Traceback (most recent call last):')[1:]
        other = [b for b in blocks if 'EGL' not in b[:4000] and 'OpenGL' not in b[:4000]]
        status = FAIL if fatal else (WARN if other else PASS)
        detail = (f'{len(fatal)} fatal ({fatal[0]})' if fatal else
                  f'{len(other)} non-renderer traceback(s)' if other else '')
        rep.add(status, p, 'log clean', detail)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('files', nargs='+')
    p.add_argument('--logs', nargs='*', default=[])
    p.add_argument('--expect-trials', type=int, default=30)
    p.add_argument('--expect-episodes', type=int, default=None)
    args = p.parse_args(argv)
    rep = Report()
    for path in args.files:
        rows = read(path)
        if not rows:
            rep.add(FAIL, path, 'has rows', 'empty')
            continue
        kind = kind_of(rows)
        if kind == 'testbed':
            check_testbed(path, rows, rep, args.expect_trials)
        elif kind == 'kinetix':
            check_kinetix(path, rows, rep)
        elif kind == 'libero':
            check_libero(path, rows, rep)
        elif kind == 'episodes':
            check_episodes(path, rows, rep, args.expect_episodes)
        else:
            rep.add(FAIL, path, 'known format', f'columns {list(rows[0])[:5]}')
            continue
        check_manifest(path, rep)
    check_logs(args.logs, rep)
    rep.print()
    return 1 if rep.failed else 0


if __name__ == '__main__':
    sys.exit(main())
