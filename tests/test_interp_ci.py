import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from interp_ci import boot, follow  # noqa: E402


def test_follow_real_ghost_unshifted():
    nut0, o, s = np.array([0.04, 0.2]), np.array([0.01, 0.0]), np.array([0.04, 0.0])
    assert np.isclose(follow(nut0 + o, nut0, s, o)[0], 1)
    assert np.isclose(follow(nut0 - s + o, nut0, s, o)[0], 0)
    assert np.isnan(follow(nut0 + o, nut0, s, o)[1])


def test_boot_brackets_and_resamples_groups():
    x = np.repeat(np.arange(10.0), 3)                 # 10 groups of 3 identical values
    est, lo, hi = boot(np.repeat(np.arange(10), 3), lambda i: np.array([x[i].mean()]), n=500)
    assert np.isclose(est[0], 4.5) and lo[0] < 4.5 < hi[0]
    const = boot(np.arange(5), lambda i: np.array([1.0]), n=50)
    assert all(np.isclose(v[0], 1) for v in const)
