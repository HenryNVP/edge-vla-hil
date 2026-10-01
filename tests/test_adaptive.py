"""Tests for the coverage-aware executor — pure Python, no ROS.

The policy here labels every action with its chunk and entry (value = 100 * chunk + entry), so a
test can see exactly which entry each tick played: whether a chunk was joined where the actions
actually left off (no skip), and whether waiting replays a chunk from its start.
"""
import numpy as np
import pytest

from evh_controller.chunk_executor import (
    CoverageAwareExecutor,
    NoSkipRTCExecutor,
    SwitchOnlyExecutor,
    make_executor,
)
from evh_controller.chunk_executor.adaptive import ALIGNED, WAITING, expected_rates
from evh_controller.inference_worker import Arrival

H, A = 8, 7


class IndexedPolicy:
    chunk_size = H
    action_dim = A
    guided_resampling = True

    def __init__(self):
        self.calls = 0

    def next_chunk(self):
        self.calls += 1
        c = np.zeros((H, A), np.float32)
        c[:, 0] = 100 * self.calls + np.arange(H)
        return c


class DelayWorker:
    """Request k arrives `delays[k]` polls after it was made (the last value repeats)."""

    def __init__(self, policy, delays):
        self.policy, self.delays = policy, list(delays)
        self._pending = None
        self.n = 0

    def try_request(self, obs, t_issue, epoch, prefix=None, weights=None):
        if self._pending is not None:
            return False
        d = self.delays[min(self.n, len(self.delays) - 1)]
        self.n += 1
        self._pending = [max(1, d), Arrival(self.policy.next_chunk(), t_issue, epoch, 0.001)]
        return True

    def poll(self):
        if self._pending is None:
            return None
        self._pending[0] -= 1
        if self._pending[0] > 0:
            return None
        a, self._pending = self._pending[1], None
        return a


def _run(ex, steps):
    out = []
    for t in range(steps):
        a = ex.step({}, t)
        out.append(None if a is None else int(a[0]))
    return out


def _make(cls=CoverageAwareExecutor, delays=(2,), **kw):
    policy = IndexedPolicy()
    worker = DelayWorker(policy, delays)
    return cls(worker, policy, **kw), policy


# ------------------------------------------------------------------ the estimator
def test_with_a_full_waiting_chunk_the_modes_cross_at_the_golden_ratio():
    """coverage (H - d)/d = duty H/(H + d)  <=>  d = H(sqrt5 - 1)/2 = 0.618 H."""
    below, above = int(0.618 * 64) - 1, int(0.618 * 64) + 2
    a, w = expected_rates([below], 64, 64)
    assert a > w
    a, w = expected_rates([above], 64, 64)
    assert w > a


def test_a_heavy_tail_lowers_the_time_aligned_rate_before_the_median_moves():
    light = [2] * 20
    heavy = [2] * 16 + [12] * 4                 # same median, one stall in five
    assert expected_rates(heavy, H, H)[0] < expected_rates(light, H, H)[0]


def test_a_short_waiting_chunk_makes_waiting_less_attractive():
    """A policy that tolerates only s_w open-loop steps pays the delay more often."""
    assert expected_rates([5], H, 2)[1] < expected_rates([5], H, H)[1]


# ------------------------------------------------------------------ execution
def test_short_delay_stays_time_aligned_and_never_idles_after_start():
    ex, _ = _make(delays=(2,))
    out = _run(ex, 60)
    first = next(i for i, v in enumerate(out) if v is not None)
    assert all(v is not None for v in out[first:])
    assert set(ex.mode_log) == {ALIGNED}


def test_long_delay_switches_to_waiting_and_plays_each_chunk_from_its_start():
    ex, _ = _make(delays=(7,))
    out = _run(ex, 120)
    assert ex.mode_log[-1] == WAITING
    # once waiting, every chunk's first played entry is entry 0
    firsts = [v for prev, v in zip(out, out[1:]) if prev is None and v is not None]
    assert all(v % 100 == 0 for v in firsts[-3:])


def test_starved_time_aligned_joins_where_the_actions_left_off():
    """Forced time-aligned at d = 6 > H/2: the chunk in hand runs out before the next lands.
    Starvation-aware alignment joins at the count of actions actually played; RTC's join skips
    to entry d, dropping the entries meant for the starved ticks."""
    noskip, _ = _make(NoSkipRTCExecutor, delays=(6,))
    skip, _ = _make(CoverageAwareExecutor, delays=(6,), rule='aligned', reindex=False)
    joins = {}
    for label, ex in (('noskip', noskip), ('skip', skip)):
        played = [v for v in _run(ex, 80) if v is not None]
        # the entry at which each newly arrived chunk was joined
        joins[label] = [b % 100 for a, b in zip(played, played[1:]) if b // 100 != a // 100]
    assert joins['skip'] and all(j == 6 for j in joins['skip']), joins['skip']
    # joined at the actions actually played: 6 when the chunk in hand covered the whole wait,
    # fewer when it ran out first (starved), never later than what was played
    assert joins['noskip'] and all(j <= 6 for j in joins['noskip']), joins['noskip']
    assert min(joins['noskip']) < 6, 'some waits were starved and joined early'


def test_varying_delay_goes_waiting_in_a_stall_and_back_after_it():
    delays = [2] * 12 + [7] * 12 + [1] * 40
    ex, _ = _make(delays=delays, delay_buffer=6)
    _run(ex, 400)
    log = ex.mode_log
    assert log[0] == ALIGNED
    assert WAITING in log
    assert log[-1] == ALIGNED
    assert log.index(WAITING) < len(log) - 1 - log[::-1].index(WAITING) or log.count(WAITING) == 1


def test_the_half_rule_ablation_switches_on_the_forecast():
    ex, _ = _make(delays=(5,), rule='half')
    _run(ex, 80)
    assert ex.mode_log[-1] == WAITING           # forecast 5 > H/2 = 4
    ex, _ = _make(delays=(3,), rule='half')
    _run(ex, 80)
    assert set(ex.mode_log) == {ALIGNED}


def test_switch_only_is_the_switch_with_rtc_joins():
    ex, _ = _make(SwitchOnlyExecutor, delays=(2,))
    assert ex.reindex is False and ex.rule == 'rate'


@pytest.mark.parametrize('name', ['coverage_aware', 'rtc_noskip', 'switch_only',
                                  'noskip_async'])
def test_registered_and_resettable(name):
    policy = IndexedPolicy()
    ex = make_executor(name, DelayWorker(policy, (3,)), policy)
    _run(ex, 30)
    ex.reset()
    assert ex._chunk is None and ex._played == 0
    assert _run(ex, 30)[-1] is not None


class RecordingWorker(DelayWorker):
    def __init__(self, *a):
        super().__init__(*a)
        self.prefixes = []

    def try_request(self, obs, t_issue, epoch, prefix=None, weights=None):
        ok = super().try_request(obs, t_issue, epoch, prefix, weights)
        if ok:
            self.prefixes.append(prefix)
        return ok


def test_the_deployed_default_overlaps_unguided_and_never_skips():
    from evh_controller.chunk_executor import NoSkipAsyncExecutor
    policy = IndexedPolicy()
    worker = RecordingWorker(policy, (6,))
    ex = NoSkipAsyncExecutor(worker, policy)
    played = [v for v in _run(ex, 80) if v is not None]
    assert all(p is None for p in worker.prefixes), 'no guidance prefix is ever sent'
    joins = [b % 100 for a, b in zip(played, played[1:]) if b // 100 != a // 100]
    assert joins and all(j <= 6 for j in joins) and min(joins) < 6
    assert set(ex.mode_log) == {ALIGNED}
