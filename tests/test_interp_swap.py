import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from interp_swap import block_gains, directions  # noqa: E402


def test_directions_linear_field():
    g = np.arange(5.0)
    F = (g[:, None, None] * np.array([1.0, 0.0]) + g[None, :, None] * np.array([0.0, 2.0]))
    V = directions(F[None], 1.0)                       # (1, 3, 3, 2, 2)
    assert np.allclose(V[0, 1, 1], [[1, 0], [0, 2]])


def test_block_gains_head_vs_direction():
    D = 6
    inside = np.zeros((1, 2, 2), bool)
    inside[0, 0] = True
    e = np.eye(D)
    # in-range directions use features 0/1, out-of-range ones 2/3; every head reads only 0/1
    V = np.zeros((1, 2, 2, D, 2))
    V[0][inside[0]] = np.stack([e[0], e[1]], -1)
    V[0][~inside[0]] = np.stack([e[2], e[3]], -1)
    J = np.broadcast_to(np.stack([e[0], e[1]]), (1, 2, 2, 2, D))
    b = block_gains(J, V, inside)
    assert np.allclose(b['in', 'in'], 1) and np.allclose(b['out', 'in'], 1)
    assert np.allclose(b['in', 'out'], 0) and np.allclose(b['out', 'out'], 0)
