"""Tests for the alpha-beta object filter — numpy only.

Pinned: it converges on a constant-velocity target; with delay compensation a late measurement
stream gives the CURRENT position, not the old one; it coasts through a dropout; and it smooths
noise instead of passing it through.
"""
import numpy as np
import pytest

from evh_reactive.object_filter import AlphaBetaFilter

V = np.array([0.0, 0.001, 0.0])          # 1 mm per tick


def _feed(f, n, latency=0, noise=0.0, drop=lambda t: False, seed=0):
    rng = np.random.default_rng(seed)
    out = None
    for t in range(n):
        seen = t - latency
        z = None if seen < 0 or drop(t) else V * seen + rng.normal(0, noise, 3)
        out = f.update(z)
    return out, V * (n - 1)


def test_none_before_the_first_measurement():
    assert AlphaBetaFilter().update(None) is None


def test_tracks_constant_velocity():
    est, truth = _feed(AlphaBetaFilter(), 200)
    assert np.allclose(est, truth, atol=1e-4)


def test_delay_compensation_reports_the_current_position():
    compensated, truth = _feed(AlphaBetaFilter(latency=5), 200, latency=5)
    raw, _ = _feed(AlphaBetaFilter(latency=0), 200, latency=5)
    assert np.allclose(compensated, truth, atol=2e-4)
    assert np.linalg.norm(raw - truth) == pytest.approx(5e-3, rel=0.05)   # 5 ticks behind


def test_coasts_through_a_dropout():
    est, truth = _feed(AlphaBetaFilter(), 200, drop=lambda t: 180 <= t < 200)
    assert np.allclose(est, truth, atol=5e-4)


def test_stops_extrapolating_after_a_long_occlusion():
    f = AlphaBetaFilter(max_coast=10)
    _feed(f, 100)
    stuck = [f.update(None) for _ in range(30)]
    assert np.allclose(stuck[-1], stuck[-2])


def test_smooths_noise():
    errs_f, errs_raw = [], []
    for seed in range(20):
        est, truth = _feed(AlphaBetaFilter(alpha=0.3, beta=0.02), 300, noise=0.005, seed=seed)
        errs_f.append(np.linalg.norm(est - truth))
        errs_raw.append(np.linalg.norm(np.random.default_rng(seed).normal(0, 0.005, 3)))
    assert np.mean(errs_f) < 0.7 * np.mean(errs_raw)


def test_rejects_bad_gains():
    with pytest.raises(ValueError):
        AlphaBetaFilter(alpha=0.0)
