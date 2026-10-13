"""Tests for the condition checker — pure Python, synthetic CSVs, no experiment data.

Each test builds the failure a real run once produced silently and checks that the checker calls
it out; a checker that passes everything would look exactly like one that works.
"""
import csv
import json

import pytest

from check_conditions import kinetix_idle, main

TESTBED_COLS = ['condition', 'strategy', 'latency_ms', 'jitter_ms', 'drop_prob', 'placement',
                'executor', 'trials', 'truncated', 'd_obs_ms_p50', 'd_act_ms_p50',
                'd_chunk_steps_p50']


def _write(path, cols, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    return str(path)


def _cell(lat=400.0, place='act', executor='robot', trials=30, d_act=None, chunk=None, **kw):
    row = dict(condition=f'c{lat}{place}', strategy='synchronous', latency_ms=lat, jitter_ms=0,
               drop_prob=0, placement=place, executor=executor, trials=trials, truncated='False',
               d_obs_ms_p50=10.0, d_act_ms_p50=lat + 1 if d_act is None else d_act,
               d_chunk_steps_p50=2 + lat / 50 if chunk is None else chunk)
    row.update(kw)
    return row


def test_a_clean_testbed_file_passes(tmp_path, capsys):
    p = _write(tmp_path / 'e.csv', TESTBED_COLS, [_cell(0), _cell(400), _cell(800)])
    assert main([p]) == 0


def test_a_delay_that_never_reached_the_relay_fails(tmp_path):
    """The 2026-09 class of bug: a parameter lands on the wrong node, the cell reports 0 ms."""
    p = _write(tmp_path / 'e.csv', TESTBED_COLS, [_cell(400, d_act=0.6)])
    assert main([p]) == 1


def test_a_short_or_truncated_cell_fails(tmp_path):
    p = _write(tmp_path / 'e.csv', TESTBED_COLS, [_cell(400, trials=20)])
    assert main([p]) == 1
    p = _write(tmp_path / 'f.csv', TESTBED_COLS, [_cell(400, truncated='True')])
    assert main([p]) == 1


def test_a_round_trip_off_the_injected_latency_fails(tmp_path):
    p = _write(tmp_path / 'e.csv', TESTBED_COLS, [_cell(400, chunk=2.0)])
    assert main([p]) == 1


def test_a_fleet_log_numbers_scenes_per_robot(tmp_path):
    rows = [dict(condition='a', robot=i, scene=k, success=1, wall_s=9.0) for i in range(2) for k in range(3)]
    p = _write(tmp_path / 'x.episodes.csv', ['condition', 'robot', 'scene', 'success', 'wall_s'], rows)
    assert main([p]) == 0
    rows.append(dict(condition='a', robot=1, scene=2, success=1, wall_s=9.0))
    p = _write(tmp_path / 'y.episodes.csv', ['condition', 'robot', 'scene', 'success', 'wall_s'], rows)
    assert main([p]) == 1


def test_a_condition_with_missing_episodes_fails(tmp_path):
    rows = [dict(condition='a', scene=k, success=1, wall_s=9.0) for k in range(30)]
    rows += [dict(condition='b', scene=k, success=1, wall_s=9.0) for k in range(20)]
    p = _write(tmp_path / 'x.episodes.csv', ['condition', 'scene', 'success', 'wall_s'], rows)
    assert main([p]) == 1


def test_a_success_logged_past_its_deadline_fails(tmp_path):
    rows = [dict(condition='strat=s_deadline=20', scene=0, success=1, wall_s=21, sim_s=20.5)]
    p = _write(tmp_path / 'x.csv', ['condition', 'scene', 'success', 'wall_s', 'sim_s'], rows)
    assert main([p, '--expect-episodes', '1']) == 1


@pytest.mark.parametrize('method,d,s,want', [
    ('sync', 4, 4, 0.5),        # wait 4, play 4
    ('rtc', 4, 4, 0.0),         # the published envelope: fully covered
    ('naive', 6, 4, 2 / 3),     # past H/2 with H = 8: 2 of every 6 ticks act
    ('rtc', 8, 2, 1.0),         # d = H: every chunk arrives wholly in the past
])
def test_the_kinetix_idle_fraction_follows_the_schedule(method, d, s, want):
    assert kinetix_idle(method, d, s) == pytest.approx(want)


def test_a_kinetix_config_missing_levels_fails(tmp_path):
    cols = ['method', 'delay', 'execute_horizon', 'world_speed', 'deadline', 'free_first_chunk',
            'num_evals', 'seed', 'level', 'solve_rate', 'idle_frac', 'median_end_tick']
    rows = [dict(method='sync', delay=4, execute_horizon=4, world_speed=1, deadline=256,
                 free_first_chunk=True, num_evals=512, seed=0, level=f'l{i}', solve_rate=0.5,
                 idle_frac=0.5, median_end_tick=100) for i in range(11)]
    p = _write(tmp_path / 'k.csv', cols, rows)
    assert main([p]) == 1


def test_an_idle_fraction_off_the_schedule_fails(tmp_path):
    cols = ['method', 'delay', 'execute_horizon', 'world_speed', 'deadline', 'free_first_chunk',
            'num_evals', 'seed', 'level', 'solve_rate', 'idle_frac', 'median_end_tick']
    rows = [dict(method='sync', delay=4, execute_horizon=4, world_speed=1, deadline=256,
                 free_first_chunk=True, num_evals=512, seed=0, level=f'l{i}', solve_rate=0.5,
                 idle_frac=0.05, median_end_tick=100) for i in range(12)]
    p = _write(tmp_path / 'k.csv', cols, rows)
    assert main([p]) == 1


def test_a_log_with_an_out_of_memory_kill_fails(tmp_path):
    rows = [_cell(0)]
    p = _write(tmp_path / 'e.csv', TESTBED_COLS, rows)
    log = tmp_path / 'run.log'
    log.write_text('ok\nXlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory\n')
    assert main([p, '--logs', str(log)]) == 1


def test_renderer_teardown_tracebacks_are_not_failures(tmp_path):
    p = _write(tmp_path / 'e.csv', TESTBED_COLS, [_cell(0)])
    log = tmp_path / 'run.log'
    log.write_text('Traceback (most recent call last):\n  File "OpenGL/error.py"\nEGLError\n')
    assert main([p, '--logs', str(log)]) == 0


def test_a_dirty_or_commitless_manifest_is_reported(tmp_path, capsys):
    p = _write(tmp_path / 'e.csv', TESTBED_COLS, [_cell(0)])
    (tmp_path / 'e.manifest.json').write_text(json.dumps(
        [{'time': 't', 'repo': {'commit': None, 'dirty': ['a.py']}}]))
    assert main([p]) == 1
    out = capsys.readouterr().out
    assert 'uncommitted' in out
