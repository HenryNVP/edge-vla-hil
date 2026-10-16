import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import interp_probe as ip  # noqa: E402


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6)) * [1, 10, 0.1, 1, 1, 1]
    Y = X @ rng.normal(size=(6, 2)) + 3.0
    assert np.abs(ip.ridge_fit(X, Y)(X) - Y).max() < 0.05


def test_group_cv_holds_out_whole_groups():
    # each group's label is a constant X cannot explain: a leak would predict it exactly
    groups = np.repeat(np.arange(10), 20)
    X = np.random.default_rng(1).normal(size=(200, 3))
    Y = groups[:, None].astype(float) * 100
    assert np.abs(ip.group_cv(X, Y, groups) - Y).mean() > 100


def test_yaw_of_z_rotation():
    a = np.array([0.3, -2.0, 3.0])
    q = np.c_[np.zeros(3), np.zeros(3), np.sin(a / 2), np.cos(a / 2)]   # xyzw
    assert np.allclose(ip.yaw(q), a)
