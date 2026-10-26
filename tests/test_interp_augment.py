import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from interp_augment import draws, shifted_targets  # noqa: E402


def test_shifted_targets_ramp_and_axis():
    a = np.zeros((20, 7))
    s = shifted_targets(a, np.array([0.05, 0.0]), ramp=10)
    assert np.allclose(s[0, :2], 0) and np.allclose(s[5, :2], [0.025, 0])
    assert np.allclose(s[10:, :2], [0.05, 0]) and np.allclose(s[:, 2:], 0)


def test_draws_range_and_reproducible():
    m = draws(50, 3, 0.02, 0.06, seed=0)
    assert m.shape == (50, 3) and np.all((np.abs(m) >= 0.02) & (np.abs(m) <= 0.06))
    assert (m > 0).any() and (m < 0).any() and np.array_equal(m, draws(50, 3, 0.02, 0.06, 0))
