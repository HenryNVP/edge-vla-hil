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


class GraspPolicy(Policy):
    """Two open entries, then closed: an approach that ends in a grasp."""

    def chunk(self):
        c = super().chunk()
        c[:, 6] = -1.0
        c[2:, 6] = 1.0
        return c


# The grasp tests use anchored WAITING, which plays each chunk from entry 0 to its end; with the
# no-skip queue at d = 1 an obs-independent chunk would be re-joined at entry 1 every tick.
def _wait(policy, d=1, **kw):
    return AnchoredExecutor(Worker(policy, d), policy, anchor_fn=lambda o: o.get('obj'),
                            rule='waiting', **kw)


def test_closed_actions_keep_the_shift_frozen_at_the_close():
    p = GraspPolicy()
    ex = _wait(p)
    ys = [0.01 * t for t in range(6)]                  # object keeps moving 1 cm per tick
    out = [ex.step(_obj(y), t) for t, y in enumerate(ys)]
    # chunk requested at t=0 (y=0), played from t=1: open at t=1,2, close from t=3 (y=0.03)
    assert out[1][1] == pytest.approx(0.01) and out[2][1] == pytest.approx(0.02)
    assert out[3][1] == pytest.approx(0.03)
    assert out[4][1] == pytest.approx(0.03) and out[5][1] == pytest.approx(0.03)   # frozen


def test_a_chunk_planned_after_the_close_is_not_shifted():
    from evh_controller.chunk_executor.anchored import anchor_shift
    a = np.array([0, 0, 0, 0, 0, 0, 1.0], np.float32)  # a closed action
    close = (5, np.array([0.0, 0.03, 0.0]))
    assert anchor_shift(a, (7, np.zeros(3)), np.array([0, 0.1, 0]), close) is a
    shifted = anchor_shift(a, (2, np.zeros(3)), np.array([0, 0.1, 0]), close)
    assert shifted[1] == pytest.approx(0.03)


def test_gate_holds_the_close_and_pauses_the_chunk():
    p = GraspPolicy()
    ok = {'v': False}
    ex = _wait(p, gate_fn=lambda o, a: ok['v'])
    out = [ex.step(_obj(0.0), t) for t in range(8)]
    grips = [None if a is None else a[6] for a in out]
    assert -1.0 in grips and 1.0 not in grips          # never closes while the gate is shut
    assert ex.gate_holds > 0
    ok['v'] = True
    later = [ex.step(_obj(0.0), t) for t in range(8, 12)]
    assert later[0][6] == 1.0                          # closes as soon as the gate opens


def test_gate_gives_way_after_the_timeout():
    p = GraspPolicy()
    ex = _wait(p, gate_fn=lambda o, a: False, gate_timeout=5)
    grips = [a[6] for a in (ex.step(_obj(0.0), t) for t in range(20)) if a is not None]
    assert 1.0 in grips


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


class LiftPolicy(Policy):
    """Two open entries, a close at z = 0, then a lift 1 cm per entry."""

    def chunk(self):
        c = super().chunk()
        c[:, 0] = 0.0
        c[:, 6] = -1.0
        c[2:, 6] = 1.0
        c[3:, 2] = 0.01 * np.arange(1, H - 2)
        return c


def test_lift_gate_holds_the_close_pose_and_pauses_the_chunk():
    p = LiftPolicy()
    ok = {'v': False}
    ex = _wait(p, lift_fn=lambda o, n: ok['v'])
    out = [ex.step(_obj(0.0), t) for t in range(8)]
    closed = [a for a in out if a is not None and a[6] == 1.0]
    assert closed and all(a[2] == 0.0 for a in closed)  # closed, but never rises
    assert ex.lift.holds > 0
    ok['v'] = True
    later = [ex.step(_obj(0.0), t) for t in range(8, 10)]
    assert later[0][2] == pytest.approx(0.01)            # the chunk resumes at its first lift entry
