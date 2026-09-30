"""Tests for the time-to-success analysis — pure Python, no ROS, no experiment data.

The claim these curves carry is "slow, not broken": a timeout is censored, never a completion at
the deadline, and pooling cells with different deadlines must not treat the short one's timeouts as
failures at the long one's horizon.
"""
import pytest

from analyze_survival import km, rmst, survival_at


def _ep(t, done, deadline=20.0):
    return (t if done else deadline, done, deadline)


def test_with_one_shared_deadline_km_is_the_fraction_finished():
    eps = [_ep(8.0, True), _ep(12.0, True), _ep(18.0, True), _ep(0, False)]
    curve = km(eps)
    assert survival_at(curve, 10) == pytest.approx(0.75)
    assert survival_at(curve, 15) == pytest.approx(0.50)
    assert survival_at(curve, 19.9) == pytest.approx(0.25)


def test_a_timeout_never_becomes_a_completion():
    eps = [_ep(5.0, True), _ep(0, False), _ep(0, False)]
    assert survival_at(km(eps), 20.0) == pytest.approx(2 / 3)


def test_mixed_deadlines_follow_the_textbook_estimator():
    """One episode censored at 20 s, then completions at 25 and 30 s among the 40 s runs:
    after the censoring only two remain at risk, so each completion removes half of S."""
    eps = [(10.0, True, 40.0), (20.0, False, 20.0), (25.0, True, 40.0), (30.0, True, 40.0)]
    curve = km(eps)
    assert survival_at(curve, 10) == pytest.approx(0.75)
    assert survival_at(curve, 25) == pytest.approx(0.375)
    assert survival_at(curve, 30) == pytest.approx(0.0)


def test_rmst_bounds():
    assert rmst([_ep(0, False)] * 3, 20.0) == pytest.approx(20.0)     # nobody finishes
    assert rmst([_ep(5.0, True)] * 3, 20.0) == pytest.approx(5.0)     # everyone at 5 s
    assert rmst([_ep(5.0, True), _ep(0, False)], 20.0) == pytest.approx(12.5)


def test_rmst_refuses_a_horizon_past_the_shortest_deadline():
    with pytest.raises(ValueError):
        rmst([_ep(5.0, True, 20.0), (30.0, True, 40.0)], 30.0)
