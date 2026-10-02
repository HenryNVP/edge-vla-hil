"""Tests for the Square mechanism testbed's planner and roll — numpy only, no robosuite.

Pinned: angle wrapping and the two turns (across the handle; to the peg modulo 90 degrees); the
planner's speed limits and phase order; the rolled observation keeps a held nut held through a
short just-in-time queue (the bug that made continuation loop between lowering and re-lifting);
and the divergence check fires on a missed grasp but not on a nut already on the peg.
"""
import numpy as np
import pytest

import oracle_square as sq

CFG = sq.SquareConfig(chunk_size=15)
DOWN = np.array([1.0, 0.0, 0.0, 0.0])          # EE pointing down (roughly the reset pose)


def nut_quat(yaw):
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


@pytest.mark.parametrize('a,period,expect', [
    (3 * np.pi / 2, 2 * np.pi, -np.pi / 2), (0.8, np.pi / 2, 0.8 - np.pi / 2), (-0.2, np.pi, -0.2)])
def test_wrap(a, period, expect):
    assert sq.wrap(a, period) == pytest.approx(expect)


def test_seat_turn_is_the_nearest_quarter_turn():
    assert sq.seat_turn(0.1) == pytest.approx(-0.1)
    assert sq.seat_turn(np.pi / 2 - 0.1) == pytest.approx(0.1)
    assert abs(sq.seat_turn(1.31)) <= np.pi / 4


def test_handle_and_grasp_turn():
    q = nut_quat(np.pi / 2)
    assert np.allclose(sq.handle_pos([0, 0, 0.83], q), [0.0, 0.054, 0.83], atol=1e-9)
    turn = sq.grasp_turn(DOWN, q)
    assert abs(turn) <= np.pi / 2
    # after turning, the closing axis is along the nut's y axis (mod pi)
    fy = sq.finger_yaw(DOWN) + turn
    assert abs(sq.wrap(fy - (np.pi / 2 + np.pi / 2), np.pi)) < 1e-9


def test_plan_respects_speed_limits_and_orientation_steps():
    eef = np.array([-0.09, 0.02, 1.02])
    c = sq.plan_chunk(eef, DOWN, [0.04, -0.04], np.array([-0.11, 0.15, 0.83]), nut_quat(-1.3), CFG)
    steps = np.linalg.norm(np.diff(np.vstack([eef, c[:, :3]]), axis=0), axis=1)
    assert c.shape == (15, 7) and steps.max() <= CFG.step_m + 1e-6
    assert (c[:, 6] == -1.0).all()                        # still approaching: gripper open


def test_held_nut_is_carried_toward_the_peg():
    nut = np.array([0.0, 0.10, 1.02])
    q = nut_quat(0.0)
    eef = sq.handle_pos(nut, q)
    assert sq.start_phase(eef, DOWN, [0.016, -0.016], nut, q, CFG) == sq.TRANSPORT
    c = sq.plan_chunk(eef, DOWN, [0.016, -0.016], nut, q, sq.SquareConfig(chunk_size=80))
    opened = np.flatnonzero(c[:, 6] < 0)
    assert len(opened) and (c[:opened[0], 6] == 1.0).all()  # holds until it lets go once
    # it lets go only after carrying the nut over the peg and lowering it
    assert c[opened[0], 2] < c[0, 2] - 0.1
    assert c[opened[0], 0] > eef[0] + 0.15


def _held_obs(nut=np.array([0.2, 0.1, 1.0]), yaw=0.0):
    q = nut_quat(yaw)
    eef = sq.handle_pos(nut, q)
    s = np.concatenate([eef, DOWN, [0.016, -0.016], nut, q, nut, [1.0]]).astype(np.float32)
    return {'proprio': s[None]}, eef


def test_roll_keeps_a_held_nut_held_through_a_short_queue():
    obs, eef = _held_obs()
    queued = np.zeros((2, 7), np.float32)                  # fewer than close_ticks entries
    queued[:, :3] = eef + [0.0, 0.0, -0.01]
    queued[:, 3:6] = sq.quat_to_axisangle(sq.eef_to_control_quat(DOWN))
    queued[:, 6] = 1.0
    rolled = sq.square_roll(obs, queued)['proprio'][-1]
    e, eq, g, n, nq = sq.unpack(rolled)
    assert sq.is_held(e, g, n, nq)                         # predicted still in hand
    assert np.allclose(n, obs['proprio'][-1][9:12] + [0.0, 0.0, -0.01], atol=1e-4)


def test_divergence_fires_on_a_missed_grasp_only():
    recent = np.zeros((12, 7), np.float32)
    recent[:, 6] = 1.0                                     # closed for 12 ticks
    nut = np.array([-0.1, 0.15, 0.83])
    s = np.concatenate([nut + [0, 0, 0.05], DOWN, [0.0, 0.0], nut, nut_quat(0), nut, [1.0]])
    assert sq.square_diverged({'proprio': s[None].astype(np.float32)}, recent)
    obs, _ = _held_obs()
    assert not sq.square_diverged(obs, recent)            # holding it: fine
    on = np.concatenate([[0.23, 0.05, 0.9], DOWN, [0.0, 0.0], sq.PEG[:2].tolist() + [0.83],
                         nut_quat(0), [0.23, 0.1, 0.83], [1.0]])
    assert not sq.square_diverged({'proprio': on[None].astype(np.float32)}, recent)
