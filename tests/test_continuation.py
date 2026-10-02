"""Tests for continuation planning — pure Python, no ROS.

The policy labels each action with its chunk and entry (x = 100 * chunk + entry), so a test sees
exactly what was played. Pinned: chunks are appended and played from entry 0 (nothing dropped,
nothing replayed); requests carry the rolled-forward observation; requests go out just in time;
and steady-state plan progress (on a line planner) reaches the one-slot ceiling min(1, H/d)
where the no-skip queue gets H/(2d).
"""
import numpy as np
import pytest

from evh_controller.chunk_executor import NoSkipAsyncExecutor
from evh_controller.chunk_executor.continuation import ContinuationExecutor
from evh_controller.inference_worker import Arrival

H, A = 8, 7


class Policy:
    chunk_size = H
    action_dim = A
    absolute_actions = True
    guided_resampling = False

    def __init__(self):
        self.calls = 0
        self.seen = []

    def chunk(self, obs):
        self.calls += 1
        self.seen.append(obs)
        c = np.zeros((H, A), np.float32)
        c[:, 0] = 100 * self.calls + np.arange(H)
        c[:, 6] = -1.0
        return c


class Worker:
    def __init__(self, policy, d):
        self.policy, self.d, self._p = policy, d, None

    def try_request(self, obs, t_issue, epoch, prefix=None, weights=None):
        if self._p is not None:
            return False
        self._p = [self.d, Arrival(self.policy.chunk(obs), t_issue, epoch, 0.0)]
        return True

    def poll(self):
        if self._p is None:
            return None
        self._p[0] -= 1
        if self._p[0] > 0:
            return None
        a, self._p = self._p[1], None
        return a


def _played(ex, steps):
    out = []
    for t in range(steps):
        a = ex.step({'t': t}, t)
        out.append(None if a is None else int(a[0]))
    return out


def test_chunks_are_appended_whole_and_in_order():
    p = Policy()
    ex = ContinuationExecutor(Worker(p, 5), p, roll_fn=None)
    seq = [x for x in _played(ex, 80) if x is not None]
    # every chunk appears from entry 0 to H-1, chunks in order, no entry twice
    for i, x in enumerate(seq):
        assert x == 100 * (i // H + 1) + i % H


def test_request_carries_the_rolled_observation():
    p = Policy()
    rolled = []

    def roll(obs, queued):
        rolled.append(len(queued))
        return {**obs, 'end': None if len(queued) == 0 else int(queued[-1][0])}

    ex = ContinuationExecutor(Worker(p, 3), p, roll_fn=roll)
    _played(ex, 30)
    assert rolled[0] == 0                     # bootstrap: nothing queued yet
    later = [o for o in p.seen[1:] if o.get('end') is not None]
    assert later and all(o['end'] % 100 == H - 1 for o in later)   # planned from a chunk's end


class LinePolicy(Policy):
    """Plans x_obs + V * (1..H): a perfect planner on a line, so progress per tick is measurable.
    Each chunk depends on the observed state, which is what makes a stale state cost progress."""
    V = 0.01

    def chunk(self, obs):
        c = super().chunk(obs)
        c[:, 0] = float(obs['x']) + self.V * np.arange(1, H + 1)
        return c


def _progress(make, d, T=800, warm=200):
    """Steady-state plan progress per tick, as a fraction of the planner's speed, on an arm that
    reaches each absolute target in one tick (a static world, perfect tracking)."""
    p = LinePolicy()
    ex = make(Worker(p, d), p)
    if hasattr(ex, '_delays'):
        ex._delays = [d] * 10                 # a link whose delay is already known
    x, xs = 0.0, []
    for t in range(T):
        a = ex.step({'x': x}, t)
        if a is not None:
            x = float(a[0])
        xs.append(x)
    return (xs[-1] - xs[warm]) / (LinePolicy.V * (T - 1 - warm))


def _roll(obs, queued):
    return obs if len(queued) == 0 else {**obs, 'x': float(queued[-1][0])}


@pytest.mark.parametrize('d', [2, 6, 8, 12, 16])
def test_rolled_continuation_reaches_the_one_slot_ceiling(d):
    rate = _progress(lambda w, p: ContinuationExecutor(w, p, roll_fn=_roll), d)
    assert rate == pytest.approx(min(1.0, H / d), abs=0.03)


@pytest.mark.parametrize('d', [5, 6, 8, 12, 16])
def test_noskip_queue_progresses_at_half_the_chunk_per_round_trip(d):
    # the analytic contrast in the module docstring, pinned on the repo's own executor
    rate = _progress(lambda w, p: NoSkipAsyncExecutor(w, p), d)
    assert rate == pytest.approx(min(1.0, H / (2 * d)), abs=0.03)


def test_without_the_roll_continuation_replans_ground_already_covered():
    rolled = _progress(lambda w, p: ContinuationExecutor(w, p, roll_fn=_roll), 6)
    unrolled = _progress(lambda w, p: ContinuationExecutor(w, p, roll_fn=None), 6)
    assert unrolled < 0.5 * rolled


def test_just_in_time_queue_stays_bounded_when_delay_is_short():
    p = Policy()
    ex = ContinuationExecutor(Worker(p, 2), p, roll_fn=None)
    for t in range(200):
        ex.step({}, t)
        assert len(ex._queue) <= H + ex.forecast_delay() + ex.lead


def test_refuses_delta_policies():
    p = Policy()
    p.absolute_actions = False
    with pytest.raises(ValueError, match='deltas'):
        ContinuationExecutor(Worker(p, 1), p, roll_fn=None)


def test_divergence_flushes_the_queue_and_the_chunk_in_flight():
    p = Policy()
    state = {'fired_at': None, 'in_flight': None}

    def diverged(obs, recent):
        # fire once, at the first tick from t = 10 with a request in flight
        if state['fired_at'] is None and obs['t'] >= 10 and ex._pending_t is not None:
            state['fired_at'], state['in_flight'] = obs['t'], p.calls
            return True
        return False

    ex = ContinuationExecutor(Worker(p, 3), p, roll_fn=None, diverged_fn=diverged, cooldown=1)
    played = _played(ex, 60)
    t0 = state['fired_at']
    assert ex.flushes == 1 and t0 is not None
    assert played[t0] is None                 # queue dropped at the flush tick
    chunks_played = {x // 100 for x in played if x is not None}
    assert state['in_flight'] not in chunks_played   # the chunk in flight was thrown away
    after = [x for x in played[t0:] if x is not None]
    assert after and after[0] % 100 == 0      # play resumes at a fresh chunk's entry 0


def test_divergence_check_respects_the_cooldown():
    p = Policy()
    ex = ContinuationExecutor(Worker(p, 2), p, roll_fn=None,
                              diverged_fn=lambda o, r: True, cooldown=10)
    _played(ex, 100)
    assert 5 <= ex.flushes <= 10              # at most one flush per cooldown window
