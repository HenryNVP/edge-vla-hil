"""Tests for the DP mechanism testbed's glue — numpy only, no robosuite or torch.

Pinned: the proprio roll writes the queued plan's last targets into the history (newest last),
in the eef-quat frame (the controller-frame offset undone, invariant 7), and closes the fingers
only once the plan closes them; the tracked object rides outside the policy's keys.
"""
import numpy as np

import dp_mech
from evh_plant.env_factory import eef_to_control_quat, quat_to_axisangle

DOWN = np.array([1.0, 0.0, 0.0, 0.0])


def _obs(width=0.04):
    s = np.tile(np.concatenate([[0.0, 0.0, 1.0], DOWN, [width, -width]]), (2, 1))
    return {'proprio': s.astype(np.float32), 'object': np.zeros(11, np.float32)}


def _queue(n, grip=-1.0):
    q = np.zeros((n, 7), np.float32)
    q[:, 0] = np.arange(1, n + 1) * 0.01
    q[:, 2] = 1.0
    q[:, 3:6] = quat_to_axisangle(eef_to_control_quat(DOWN))
    q[:, 6] = grip
    return q


def test_roll_writes_the_last_targets_newest_last():
    rolled = dp_mech.proprio_roll(_obs(), _queue(5))['proprio']
    assert np.allclose(rolled[:, 0], [0.04, 0.05])
    assert np.allclose(np.abs(rolled[-1, 3:7] @ DOWN), 1.0, atol=1e-5)   # same orientation
    assert np.allclose(rolled[-1, 7:9], [0.04, -0.04])


def test_roll_closes_the_fingers_only_when_the_plan_does():
    rolled = dp_mech.proprio_roll(_obs(), _queue(3, grip=1.0), closed_qpos=0.01)['proprio']
    assert np.allclose(rolled[-1, 7:9], [0.01, -0.01])
    held = dp_mech.proprio_roll(_obs(width=0.012), _queue(3, grip=1.0))['proprio']
    assert np.allclose(held[-1, 7:9], [0.012, -0.012])        # already closed: keep the measurement


def test_short_queue_rolls_only_the_newest_frame_and_empty_is_a_no_op():
    obs = _obs()
    rolled = dp_mech.proprio_roll(obs, _queue(1))['proprio']
    assert np.allclose(rolled[0], obs['proprio'][0]) and rolled[1, 0] == np.float32(0.01)
    assert dp_mech.proprio_roll(obs, _queue(0)) is obs


def test_object_tracker_reads_the_extra_not_the_proprio():
    obs = _obs()
    assert dp_mech.object_tracker(obs) is None
    obs['object'][7:11] = [0.1, 0.2, 0.8, 1.0]
    assert np.allclose(dp_mech.object_tracker(obs), [0.1, 0.2, 0.8])
