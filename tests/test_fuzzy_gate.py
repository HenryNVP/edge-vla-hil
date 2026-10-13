"""Tests for fuzzy_gate.py's robot-side pieces (features, TS inference) — no torch, no robosuite.

Pinned: the feature definitions against oracle_square's handle geometry, and the zero-order TS
output as the firing-weighted mean of the rule consequents (so it is bounded by them).
"""
import itertools

import numpy as np

import fuzzy_gate as fg
import oracle_square as sq


def test_features_match_handle_geometry():
    nut, nq = np.array([0.1, 0.2, 0.83]), np.array([0, 0, 0, 1.0])
    h = sq.handle_pos(nut, nq)
    eef = h + [0.003, -0.004, 0.01]
    eq = np.array([1.0, 0, 0, 0])
    f = fg.gate_features(eef, eq, nut, nq, eef + [0, 0, 0.02])
    assert np.isclose(f[0], 0.005, atol=1e-6) and np.isclose(f[1], 0.01, atol=1e-6)
    assert np.isclose(f[2], 0.02, atol=1e-6) and np.isclose(f[3], abs(sq.grasp_turn(eq, nq)))


def test_ts_output_is_weighted_mean_of_consequents(tmp_path):
    c = np.array([[0.0, 1.0]] * 4)
    s = np.ones((4, 2))
    q = np.linspace(-2, 3, 16)
    np.savez(tmp_path / 'g.npz', centers=c, widths=s, consequents=q, threshold=0.5)
    g = fg.FuzzyGate(str(tmp_path / 'g.npz'))
    f = np.array([0.3, 0.7, 0.0, 1.0]) * fg.SCALE
    x = f / fg.SCALE
    mu = np.exp(-0.5 * (x[:, None] - c) ** 2)
    w = np.array([np.prod([mu[i, r[i]] for i in range(4)])
                  for r in itertools.product(range(2), repeat=4)])
    assert np.isclose(g.logit(f), w @ q / w.sum())
    assert q.min() <= g.logit(f * 50) <= q.max()
    assert g(f) == (g.logit(f) > 0.5)
