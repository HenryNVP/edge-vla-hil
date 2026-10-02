"""Tests for the mechanism-sweep summary: the Kaplan-Meier median with censored timeouts."""
import pytest

from analyze_mech import km_median


def test_all_successes_is_the_ordinary_median():
    assert km_median([(t, True) for t in (10, 20, 30, 40, 50)]) == 30


def test_censoring_moves_the_median_later_not_earlier():
    # 3 of 6 succeed early; the 3 timeouts at 100 are censored, not counted as 100-tick successes
    assert km_median([(10, True), (20, True), (30, True), (100, False), (100, False),
                      (100, False)]) == 30
    assert km_median([(10, True), (20, True), (100, False), (100, False), (100, False)]) is None


def test_none_when_fewer_than_half_succeed():
    assert km_median([(5, True), (100, False), (100, False)]) is None
