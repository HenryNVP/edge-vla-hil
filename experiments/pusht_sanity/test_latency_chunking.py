"""Pure-numpy tests for the delayed-chunk scheduler + strategies. No gym/torch needed.

Validates the *mechanics* of the latency model and the strategy splices. The empirical
degradation curve (does RTC beat naive under latency?) is what run_pusht.py produces with the real
policy — that is the sanity check itself; here we just guard the machinery it relies on.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from latency_chunking import (  # noqa: E402
    DelayModel, PointMassEnv, make_mock_policy, make_strategy, run_episode,
    RTCFreeze, NaiveAsync,
)

STRATEGIES = ['synchronous', 'naive_async', 'temporal_ensemble', 'rtc_freeze']


def _run(strategy_name, **delay_kw):
    return run_episode(
        PointMassEnv(), make_mock_policy(), make_strategy(strategy_name),
        DelayModel(**delay_kw), max_steps=300)


# --------------------------------------------------------------- machinery
@pytest.mark.parametrize('name', STRATEGIES)
def test_runs_and_returns_finite_metrics(name):
    r = _run(name, latency_steps=3, jitter_steps=1, seed=0)
    assert r.steps > 0
    assert np.isfinite(r.total_reward) and np.isfinite(r.max_reward)
    assert 0.0 <= r.max_reward <= 1.0 + 1e-9


def test_make_strategy_rejects_unknown():
    with pytest.raises(ValueError):
        make_strategy('telekinesis')


def test_zero_latency_all_succeed_on_toy_task():
    # With no delay the toy reach task is trivial for every strategy.
    for name in STRATEGIES:
        r = _run(name, latency_steps=0)
        assert r.success, f'{name} failed at zero latency'


def test_delay_model_is_seeded():
    a = DelayModel(latency_steps=5, jitter_steps=2, seed=42)
    b = DelayModel(latency_steps=5, jitter_steps=2, seed=42)
    assert [a.sample_delay() for _ in range(20)] == [b.sample_delay() for _ in range(20)]


def test_delay_never_negative():
    d = DelayModel(latency_steps=1, jitter_steps=10, seed=1)
    assert all(d.sample_delay() >= 0 for _ in range(200))


def test_drop_prob_one_means_one_replan():
    # Everything dropped after the first attempt -> at most the initial replan succeeds (0 here,
    # since the very first issue is also subject to the drop). Either way it must not hang.
    r = _run('naive_async', latency_steps=2, drop_prob=1.0)
    assert r.replans == 0
    assert r.steps == 300   # runs to max without ever getting a chunk


# --------------------------------------------------------------- splices
def test_rtc_freeze_skips_overlapped_prefix():
    s = RTCFreeze()
    s.reset()
    chunk = np.arange(16).reshape(16, 1).astype(float)
    s.on_arrival(executed_since_issue=4, new_chunk=chunk)
    assert s.idx == 4                      # frozen prefix skipped, time-aligned
    assert s.act(None)[0] == 4.0


def test_naive_async_starts_new_chunk_at_zero():
    s = NaiveAsync()
    s.reset()
    chunk = np.arange(16).reshape(16, 1).astype(float)
    s.on_arrival(executed_since_issue=4, new_chunk=chunk)
    assert s.idx == 0                       # ignores elapsed time -> discontinuity
    assert s.act(None)[0] == 0.0


def test_rtc_and_naive_identical_at_zero_delay():
    # When d=0, freeze has nothing to skip -> the two strategies must behave identically.
    rtc = _run('rtc_freeze', latency_steps=0, seed=3)
    naive = _run('naive_async', latency_steps=0, seed=3)
    assert rtc.success == naive.success
    assert rtc.steps == naive.steps


def test_synchronous_pauses_under_latency():
    # The synchronous strategy must spend control steps waiting when delay > 0.
    r = _run('synchronous', latency_steps=5, seed=0)
    assert r.paused_steps > 0
    r0 = _run('synchronous', latency_steps=0, seed=0)
    assert r0.paused_steps == 0
