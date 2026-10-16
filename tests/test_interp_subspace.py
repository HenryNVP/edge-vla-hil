import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from interp_subspace import nearest, out_of_span, spearman  # noqa: E402


def test_out_of_span():
    V = np.eye(3)[:, :1]                       # span of e0
    v = np.array([[2.0, 0, 0], [0, 3.0, 0], [1.0, 1.0, 0]])
    assert np.allclose(out_of_span(V, v), [0, 1, np.sqrt(0.5)])


def test_nearest_and_spearman():
    X = np.array([[0.0, 0], [10, 0]])
    assert np.allclose(nearest(X, np.array([[1.0, 0], [10, 2]])), [1, 2])
    a = np.arange(10.0)
    assert np.isclose(spearman(a, a ** 3), 1) and np.isclose(spearman(a, -a), -1)


def test_local_basis_uses_cross_demo_variation():
    from interp_subspace import local_basis
    # demo i has frames at (i, 0) and far away at (i, 100): near g = 0 they vary along e0
    X = np.array([[i, d] for i in range(5) for d in (0.0, 100.0)])
    ids = np.repeat(np.arange(5), 2)
    B = local_basis(X, ids, np.zeros(2), n=5, k=1)
    assert np.allclose(abs(B[:, 0]), [1, 0])
