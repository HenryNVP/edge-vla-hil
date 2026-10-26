"""Tests for the Square mechanism testbed's planner and roll — numpy only, no robosuite.

Pinned: angle wrapping and the two turns (across the handle; to the peg modulo 90 degrees); the
planner's speed limits and phase order; the rolled observation keeps a held nut held through a
short just-in-time queue (the bug that made continuation loop between lowering and re-lifting);
and the divergence check fires on a missed grasp but not on a nut already on the peg.

The per-nut generalization pins: the specs follow nut_goal (obs keys, handle offset, peg, peg
top), the placement yaw rule (square modulo 90 degrees, round unconstrained), the round nut's
reach-limited seat choice, a full carry-to-peg2 plan, and the re-target on a mid-episode goal
switch (open, then approach the new nut).
"""
import numpy as np
import pytest

import nut_goal as ng
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


def test_a_gripper_closed_on_nothing_reopens_instead_of_closing_again():
    nut = np.array([-0.1, 0.15, 0.83])
    q = nut_quat(0.0)
    at_handle = sq.handle_pos(nut, q)
    turn = sq.grasp_turn(DOWN, q)
    eq = sq.ol_quat_mul(sq.axisangle_z(turn), DOWN)       # EE turned across the handle
    assert sq.start_phase(at_handle, eq, [0.04, -0.04], nut, q, CFG) == sq.CLOSE
    assert sq.start_phase(at_handle, eq, [0.0005, -0.0005], nut, q, CFG) == sq.APPROACH
    c = sq.plan_chunk(at_handle, eq, [0.0005, -0.0005], nut, q, CFG)
    assert c[0, 6] == -1.0                                 # opens first


def test_gate_slack_admits_a_close_a_fixed_tolerance_refuses():
    nut = np.array([-0.1, 0.15, 0.83])
    q = nut_quat(0.0)
    eq = sq.ol_quat_mul(sq.axisangle_z(sq.grasp_turn(DOWN, q)), DOWN)
    eef = sq.handle_pos(nut, q) + [0.015, 0.0, CFG.grasp_dz]       # 15 mm off in x
    s = np.concatenate([eef, eq, [0.04, -0.04], nut, q, nut, [1.0]]).astype(np.float32)
    obs = {'proprio': s[None]}
    assert not sq.square_gate(obs, None, CFG)
    assert sq.square_gate(obs, None, CFG, slack=0.02)


# ------------------------------------------------------------- per-nut generalization
def test_specs_follow_nut_goal():
    assert [s.name for s in sq.NUT_SPECS] == list(ng.NUTS)      # spec i is the nut goal i names
    assert sq.spec_for(0) is sq.SQUARE and sq.spec_for(1) is sq.ROUND
    assert np.allclose(sq.SQUARE.peg_xy, ng.PEGS_XY[0])
    assert np.allclose(sq.ROUND.peg_xy, ng.PEGS_XY[1])
    assert sq.SQUARE.handle_local[0] == pytest.approx(0.054)    # square-nut.xml handle_site
    assert sq.ROUND.handle_local[0] == pytest.approx(0.06)      # round-nut.xml handle_site
    assert sq.ROUND.peg_top == sq.SQUARE.peg_top == pytest.approx(0.95)   # both posts end at 0.95


def test_placement_yaw_rule_per_nut():
    for yaw in (0.0, 0.4, 1.2, -2.0, np.pi):
        assert sq.yaw_err(yaw, sq.ROUND) == 0.0    # round hole clears the round peg at any yaw
    for yaw in (0.0, np.pi / 2, -np.pi, 3 * np.pi / 2):
        assert sq.yaw_err(yaw, sq.SQUARE) <= CFG.yaw_tol
    assert sq.yaw_err(0.3, sq.SQUARE) == pytest.approx(0.3)


def test_round_grasp_is_biased_below_the_handle_site():
    assert sq.SQUARE.grasp_dz(CFG) == CFG.grasp_dz           # square grasp is unchanged
    assert sq.ROUND.grasp_dz(CFG) == pytest.approx(CFG.grasp_dz - 0.006)
    nut, q = np.array([-0.1, -0.15, 0.83]), nut_quat(0.3)
    at_handle = sq.handle_pos(nut, q, spec=sq.ROUND)
    eq = sq.ol_quat_mul(sq.axisangle_z(sq.grasp_turn(DOWN, q)), DOWN)
    eef = at_handle + [0.0, 0.0, 0.004]                      # 4 mm above the handle site:
    assert sq.start_phase(eef, eq, [0.04, -0.04], nut, q, CFG) == sq.CLOSE       # square: in window
    assert sq.start_phase(eef, eq, [0.04, -0.04], nut, q, CFG, spec=sq.ROUND) == sq.DESCEND
    assert sq.start_phase(eef - [0.0, 0.0, 0.008], eq, [0.04, -0.04], nut, q, CFG,
                          spec=sq.ROUND) == sq.CLOSE         # round: window sits 6 mm lower


def test_round_seat_keeps_a_zero_turn_when_already_within_reach():
    rel = np.array([0.06, 0.0])                    # handle toward the robot: EE lands nearer base
    dyaw, target = sq.seat_choice(1.0, rel, spec=sq.ROUND)
    assert dyaw == 0.0
    assert np.allclose(target, sq.ROUND.peg_xy - rel)


def test_round_seat_takes_the_least_turn_that_restores_reach():
    rel = np.array([-0.06, 0.0])                   # handle away: EE target ~0.86 m from the base
    assert np.linalg.norm(sq.ROUND.peg_xy - rel - sq.ROBOT_BASE_XY) > CFG.reach_max
    dyaw, target = sq.seat_choice(0.0, rel, spec=sq.ROUND)
    assert 0.0 < dyaw < np.pi                      # some turn, never the full 180 degrees
    assert np.linalg.norm(target - sq.ROBOT_BASE_XY) <= CFG.reach_max + 1e-9
    # minimality: any smaller turn is still out of reach
    less = sq.ROUND.peg_xy - (sq.rz(dyaw - 0.01) @ np.array([rel[0], rel[1], 0.0]))[:2]
    assert np.linalg.norm(less - sq.ROBOT_BASE_XY) > CFG.reach_max


def test_plan_carries_a_held_round_nut_to_peg2():
    nut = np.array([0.0, -0.10, 1.02])
    q = nut_quat(0.7)
    eef = sq.handle_pos(nut, q, spec=sq.ROUND)
    assert eef[0] == pytest.approx(0.06 * np.cos(0.7))          # the round handle is 6 cm out
    assert sq.start_phase(eef, DOWN, [0.016, -0.016], nut, q, CFG, spec=sq.ROUND) == sq.TRANSPORT
    c = sq.plan_chunk(eef, DOWN, [0.016, -0.016], nut, q, sq.SquareConfig(chunk_size=200),
                      spec=sq.ROUND)
    opened = np.flatnonzero(c[:, 6] < 0)
    assert len(opened) and (c[:opened[0], 6] == 1.0).all()       # holds until it lets go once
    assert c[opened[0], 2] < c[0, 2] - 0.1                       # ...after lowering onto the peg
    # the nut (EE + the rotated grasp offset) sits on peg2 when the gripper lets go
    eq_end = sq.ctrl_to_eef_quat(sq.axisangle_to_quat(c[opened[0], 3:6]))
    turn = sq.wrap(sq.finger_yaw(eq_end) - sq.finger_yaw(DOWN))
    nut_end = c[opened[0], :3] + sq.rz(turn) @ (nut - eef)
    assert np.linalg.norm(nut_end[:2] - sq.ROUND.peg_xy) <= 2 * CFG.align_tol
    assert nut_end[2] == pytest.approx(CFG.seat_z, abs=3 * CFG.z_tol)


def test_retarget_opens_and_approaches_the_new_nut():
    nut = np.array([-0.1, 0.15, 0.83])             # the square nut, being grasped...
    q = nut_quat(0.0)
    eef = sq.handle_pos(nut, q)
    eq = sq.ol_quat_mul(sq.axisangle_z(sq.grasp_turn(DOWN, q)), DOWN)
    round_nut, rq = np.array([-0.1, -0.15, 0.83]), nut_quat(1.1)   # ...when the goal flips
    c = sq.plan_chunk(eef, eq, [0.016, -0.016], round_nut, rq, CFG, spec=sq.ROUND)
    assert (c[:, 6] == -1.0).all()                  # not held on it: let go and open first
    # and the chunk heads for the round nut's handle hover, not the square's it just left
    c2 = sq.plan_chunk(eef, eq, [0.04, -0.04], round_nut, rq, sq.SquareConfig(chunk_size=80),
                       spec=sq.ROUND)
    end = c2[-1, :2]
    assert np.linalg.norm(end - sq.handle_pos(round_nut, rq, spec=sq.ROUND)[:2]) <= 0.01
    assert np.linalg.norm(end - sq.handle_pos(nut, q)[:2]) > 0.1


def test_frames_follow_the_spec_and_the_goal():
    obs = {'robot0_eef_pos': np.zeros(3), 'robot0_eef_quat': DOWN,
           'robot0_gripper_qpos': [0.04, -0.04],
           'SquareNut_pos': np.arange(1.0, 4.0), 'SquareNut_quat': np.arange(4.0, 8.0),
           'RoundNut_pos': np.arange(8.0, 11.0), 'RoundNut_quat': np.arange(11.0, 15.0),
           'goal': 1, 'goal_nut_pos': np.arange(8.0, 11.0),
           'goal_nut_quat': np.arange(11.0, 15.0)}
    assert np.allclose(sq.make_frames(None)(obs)[2][9:16], np.arange(1.0, 8.0))    # square default
    assert np.allclose(sq.make_frames(None, spec=sq.ROUND)(obs)[2][9:16], np.arange(8.0, 15.0))
    assert np.allclose(sq.make_frames(None, spec=None)(obs)[2][9:16], np.arange(8.0, 15.0))
