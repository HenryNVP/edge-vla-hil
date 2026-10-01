"""Tests for `schedule.py`, pure numpy plus the repo's executors (no JAX):

    python -m pytest -q experiments/kinetix/test_schedule.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schedule import RING, from_cycle_plan, simulate  # noqa: E402

H, T = 8, 256


def test_noskip_async_inside_the_envelope_never_idles():
    sch = simulate('noskip_async', 2, 2, H, T)
    assert sch.idle_frac == 0
    assert set(sch.joins()[1:]) == {2}            # joined at the delay, like time-aligned


def test_noskip_async_past_half_alternates_joins_and_never_skips():
    """d = 6, H = 8: a chunk joined at 6 has 2 left, starves for 4, so the next is joined at 2;
    that one has 6 left, covers the whole wait, and the next is joined at 6 again. Coverage is
    H / 2d = 2/3 against time-aligned (H - d)/d = 1/3, and no entry is ever skipped."""
    sch = simulate('noskip_async', 6, 2, H, T)
    joins = sch.joins()[2:]
    assert set(joins) == {2, 6}
    assert sch.idle_frac == pytest.approx(1 - H / 12, abs=0.02)
    # within each chunk the played entries are consecutive from the join to the switch
    for s in range(RING):
        e = sch.entry[sch.slot == s]
        assert np.all(np.diff(e)[np.diff(e) > 0] == 1)


def test_noskip_async_from_d_equals_h_plays_every_chunk_whole_from_its_start():
    """At d >= H the chunk in hand runs out before the next lands and the next is entirely in the
    past by the count of actions played, so it is played from entry 0: stale, but nothing
    skipped. At d = H that is back to back (no idle); past it, d - H idle ticks per chunk."""
    sch = simulate('noskip_async', 8, 2, H, T)
    assert set(sch.joins()) == {0} and sch.idle_frac == 0
    sch = simulate('noskip_async', 12, 2, H, T)
    assert set(sch.joins()) == {0}
    assert sch.idle_frac == pytest.approx(4 / 12, abs=0.02)


def test_requests_are_spaced_by_the_execution_horizon():
    sch = simulate('noskip_async', 2, 4, H, T)
    gaps = np.diff(np.flatnonzero(sch.issue >= 0))
    assert set(gaps[1:]) == {4}


def test_guided_strategies_are_refused():
    with pytest.raises(ValueError):
        simulate('rtc_noskip', 4, 2, H, 32)


@pytest.mark.parametrize('method,d,s', [('sync', 2, 2), ('sync', 8, 8), ('naive', 2, 4),
                                        ('naive', 6, 2), ('naive', 12, 8)])
def test_unrolled_cycle_plan_matches_the_timing_model(method, d, s):
    sch = from_cycle_plan(method, d, s, H, T)
    assert len(sch.slot) == T
    if method == 'sync':
        assert sch.idle_frac == pytest.approx(d / (d + s), abs=0.05)
    else:
        P = max(s, d)
        cover = min(1.0, max(H - d, 0) / P) if d > s else 1.0
        assert sch.idle_frac == pytest.approx(1 - cover, abs=0.05)
