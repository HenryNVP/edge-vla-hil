import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from interp_gainmap import jacobians, target  # noqa: E402


def test_jacobians_recover_a_linear_response():
    grid = np.linspace(-60, 60, 7)
    A = np.array([[0.3, 0.1], [-0.2, 0.9]])               # response = A @ (dx, dy)
    resp = np.einsum('ab,ijb->ija', A, np.stack(np.meshgrid(grid, grid, indexing='ij'), -1))
    J = jacobians(resp, grid[1] - grid[0])
    assert J.shape == (5, 5, 2, 2) and np.allclose(J, A)


def test_target_is_last_open_entry():
    c = np.zeros((5, 7))
    c[:, 0] = np.arange(5)
    c[3:, 6] = 1.0                                        # closes from entry 3
    assert target(c)[0] == 2
