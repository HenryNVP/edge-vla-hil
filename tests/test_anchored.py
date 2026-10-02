"""Tests for object anchoring — pure Python, no ROS, no robosuite.

Pinned: a chunk's open-gripper actions are shifted by exactly how far the tracked object moved
since THAT chunk's request (not the latest request's); closed-gripper actions are never shifted;
no tracker fix means no shift; and the never-skip join underneath is unchanged.
"""
import numpy as np
import pytest

from evh_controller.chunk_executor import NoSkipAsyncExecutor
from evh_controller.chunk_executor.anchored import AnchoredExecutor
from evh_controller.inference_worker import Arrival

H, A = 6, 7


class Policy:
    chunk_size = H
    action_dim = A
    absolute_actions = True
    guided_resampling = False

    def __init__(self, grip=-1.0):
        self.grip = grip

    def chunk(self):
        c = np.zeros((H, A), np.float32)
        c[:, 0] = np.arange(H) * 0.01
        c[:, 6] = self.grip
        return c


class Worker:
    def __init__(self, policy, d):
        self.policy, self.d, self._p = policy, d, None

    def try_request(self, obs, t_issue, epoch, prefix=None, weights=None):
        if self._p is not None:
            return False
        self._p = [self.d, Arrival(self.policy.chunk(), t_issue, epoch, 0.0)]
        return True

    def poll(self):
        if self._p is None:
            return None
        self._p[0] -= 1
        if self._p[0] > 0:
            return None
        a, self._p = self._p[1], None
        return a


def _obj(y):
    return {'obj': np.array([0.0, y, 0.0])}


def _make(policy, d, anchor=lambda o: o.get('obj')):
    return AnchoredExecutor(Worker(policy, d), policy, anchor_fn=anchor)


def test_open_gripper_actions_follow_the_object_since_their_request():
    p = Policy(grip=-1.0)
    ex = _make(p, d=3)
    ys = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]          # object moves 1 cm per tick
    out = [ex.step(_obj(y), t) for t, y in enumerate(ys)]
    assert out[0] is None and out[1] is None and out[2] is None   # waiting for the first chunk
    # chunk requested at t=0 (object at y=0) and played from t=3 on: shift = y(now) - 0
    for t in range(3, 6):
        assert out[t][1] == pytest.approx(ys[t])


def test_closed_gripper_actions_are_not_shifted():
    p = Policy(grip=1.0)
    ex = _make(p, d=1)
    out = [ex.step(_obj(0.01 * t), t) for t in range(4)]
    assert all(a is None or a[1] == 0.0 for a in out)


def test_no_tracker_fix_means_no_shift():
    p = Policy(grip=-1.0)
    ex = _make(p, d=1, anchor=lambda o: None)
    out = [ex.step(_obj(0.01 * t), t) for t in range(4)]
    assert all(a is None or a[1] == 0.0 for a in out)


def test_static_object_reproduces_the_noskip_queue():
    for d in (1, 4, 9):
        p = Policy()
        anchored, plain = _make(p, d), NoSkipAsyncExecutor(Worker(p, d), p)
        for t in range(30):
            a, b = anchored.step(_obj(0.0), t), plain.step(_obj(0.0), t)
            assert (a is None) == (b is None)
            if a is not None:
                assert np.allclose(a, b)


def test_refuses_delta_policies():
    p = Policy()
    p.absolute_actions = False
    with pytest.raises(ValueError, match='deltas'):
        _make(p, d=1)
