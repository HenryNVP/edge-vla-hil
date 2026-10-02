"""Tests for the mechanism testbed's scripted planner and conveyor — numpy only, no robosuite.

What is pinned: the planner never moves faster than its speed limit, walks the phases in order
from wherever the observation puts it, aims at the cube position it was GIVEN (so a stale chunk
is stale), and the conveyor stays inside its range.
"""
import numpy as np
import pytest

from oracle_lift import (APPROACH, CLOSE, DESCEND, GRIP_CLOSE, GRIP_OPEN, LIFT, OracleConfig,
                         OraclePolicy, advance, is_held, plan_chunk, start_phase)

CFG = OracleConfig(chunk_size=15)
ROT = np.array([np.pi, 0.0, 0.0])
OPEN = [0.04, -0.04]
CUBE = np.array([0.0, 0.0, 0.82])


def test_chunk_shape_and_speed_limit():
    eef = np.array([-0.1, 0.05, 1.0])
    c = plan_chunk(eef, CUBE, OPEN, ROT, CFG)
    assert c.shape == (15, 7)
    steps = np.linalg.norm(np.diff(np.vstack([eef, c[:, :3]]), axis=0), axis=1)
    assert steps.max() <= CFG.step_m + 1e-6
    assert np.allclose(c[:, 3:6], ROT)


def test_phases_in_order_from_hover():
    hover = CUBE + [0, 0, CFG.hover_m]
    assert start_phase(hover, CUBE, OPEN, CFG) == DESCEND
    c = plan_chunk(hover, CUBE, OPEN, ROT, OracleConfig(chunk_size=40))
    grip = c[:, 6]
    first_close = int(np.argmax(grip == GRIP_CLOSE))
    assert (grip[:first_close] == GRIP_OPEN).all() and (grip[first_close:] == GRIP_CLOSE).all()
    # the gripper closes at grasp height and the arm only rises after close_ticks
    assert abs(c[first_close, 2] - CUBE[2]) <= CFG.z_tol
    assert c[-1, 2] > c[first_close, 2]


def test_start_phase_cases():
    assert start_phase(CUBE + [0.2, 0, 0.1], CUBE, OPEN, CFG) == APPROACH
    assert start_phase(CUBE, CUBE, OPEN, CFG) == CLOSE
    assert start_phase(CUBE, CUBE, [0.02, -0.02], CFG) == LIFT     # fingers on the cube


def test_plan_aims_at_the_given_cube_not_the_true_one():
    eef = CUBE + [0.0, 0.0, CFG.hover_m]
    stale = plan_chunk(eef, CUBE, OPEN, ROT, OracleConfig(chunk_size=30))
    moved = plan_chunk(eef, CUBE + [0, 0.05, 0], OPEN, ROT, OracleConfig(chunk_size=30))
    assert np.allclose(stale[-1, :2], CUBE[:2], atol=CFG.xy_tol)
    assert not np.allclose(stale[:, :3], moved[:, :3])


def test_is_held():
    assert is_held(CUBE, CUBE, [0.02, -0.02], CFG)
    assert not is_held(CUBE + [0, 0, 0.1], CUBE, [0.02, -0.02], CFG)
    assert not is_held(CUBE, CUBE, OPEN, CFG)


@pytest.mark.parametrize('pos,direction,expect', [
    (0.0, 1.0, (0.1, 1.0)),
    (0.95, 1.0, (0.95, -1.0)),       # overshoot by 0.05 reflects back to 0.95
    (-0.95, -1.0, (-0.95, 1.0)),
])
def test_advance_reflects_at_the_ends(pos, direction, expect):
    p, d = advance(pos, direction, 0.1, -1.0, 1.0)
    assert p == pytest.approx(expect[0]) and d == expect[1]


def test_policy_reads_the_appended_cube():
    quat = np.array([1.0, 0.0, 0.0, 0.0])             # any unit quaternion
    s = np.concatenate([CUBE + [0, 0, CFG.hover_m], quat, OPEN, CUBE]).astype(np.float32)
    chunk = OraclePolicy(CFG).predict({'proprio': s[None]})
    assert chunk.shape == (15, 7) and np.isfinite(chunk).all()


# ------------------------------------------------------------------ tracker model
from oracle_lift import TrackerModel, TrackerSpec, cube_tracker, make_frames  # noqa: E402


def _track(spec, n=400, seed=0):
    tr = TrackerModel(spec)
    tr.reset(seed)
    return [tr.update(np.array([0.0, 0.001 * t, 0.8])) for t in range(n)]


def test_default_tracker_is_privileged():
    out = _track(TrackerSpec(), n=5)
    assert np.allclose(out[3], [0.0, 0.003, 0.8])


def test_latency_reports_an_old_position_and_none_at_first():
    out = _track(TrackerSpec(latency=4), n=20)
    assert all(o is None for o in out[:4])
    assert np.allclose(out[10], [0.0, 0.006, 0.8])


def test_dropout_fraction_and_bursts():
    spec = TrackerSpec(dropout=0.3, burst=5)
    tr = TrackerModel(spec)
    tr.reset(0)
    misses = [tr._measure(np.zeros(3)) is None for _ in range(20000)]
    assert np.mean(misses) == pytest.approx(0.3, abs=0.03)
    runs, n = [], 0
    for m in misses:
        if m:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    assert np.mean(runs) == pytest.approx(5, rel=0.15)


def test_raw_tracker_holds_the_last_fix_through_a_dropout():
    out = _track(TrackerSpec(dropout=0.5, burst=10), n=200)
    assert all(o is not None for o in out[1:])


def test_frames_carry_the_tracked_position_for_anchoring():
    obs = {'robot0_eef_pos': np.zeros(3), 'robot0_eef_quat': np.array([0, 0, 0, 1.0]),
           'robot0_gripper_qpos': np.array(OPEN), 'cube_pos': CUBE}
    p = make_frames(TrackerModel(TrackerSpec(latency=2)))(obs)[2]
    assert cube_tracker({'proprio': p[None]}) is None           # no fix yet
    p = make_frames()(obs)[2]
    assert np.allclose(cube_tracker({'proprio': p[None]}), CUBE)
