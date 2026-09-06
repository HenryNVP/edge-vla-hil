"""Tests for the shared rotation conversions (pure numpy).

The property that motivated this module is the last one here: absolute orientation targets for a
downward-facing gripper sit ON the pi wrap, where axis-angle flips sign between neighbouring
rotations. 6D does not, which is why training targets and checkpoint heads carry 6D and this
module converts only at the boundary.
"""
import numpy as np
import pytest

from evh_controller.rotation import (
    abs7_to_abs10,
    abs10_to_abs7,
    axisangle_to_matrix,
    matrix_to_axisangle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


@pytest.mark.parametrize('theta', [0.0, 0.3, 1.5, np.pi - 1e-3, np.pi])
def test_matrix_axisangle_roundtrip_including_the_pi_branch(theta):
    aa = matrix_to_axisangle(_rot_x(theta))
    assert np.isclose(np.linalg.norm(aa), theta, atol=1e-6)
    assert np.allclose(axisangle_to_matrix(aa), _rot_x(theta), atol=1e-6)


def test_matrix_6d_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(10):
        R = axisangle_to_matrix(rng.uniform(-np.pi, np.pi, 3))
        assert np.allclose(rotation_6d_to_matrix(matrix_to_rotation_6d(R)), R, atol=1e-9)


def test_6d_to_matrix_gram_schmidts_a_non_orthogonal_input():
    """The network's raw 6D output is not orthonormal; the conversion must still give a rotation."""
    d6 = np.array([2.0, 0.0, 0.0, 0.3, 1.7, 0.0])
    R = rotation_6d_to_matrix(d6)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_abs_chunk_roundtrip_preserves_position_and_gripper():
    rng = np.random.default_rng(2)
    chunk7 = np.concatenate([rng.uniform(-0.3, 0.3, (5, 3)),
                             rng.uniform(-1.5, 1.5, (5, 3)),
                             rng.choice([-1.0, 1.0], (5, 1))], axis=1)
    back = abs10_to_abs7(abs7_to_abs10(chunk7))
    assert back.shape == (5, 7)
    assert np.allclose(back[:, :3], chunk7[:, :3], atol=1e-6)
    assert np.allclose(back[:, 6], chunk7[:, 6], atol=1e-6)
    for i in range(5):
        assert np.allclose(axisangle_to_matrix(back[i, 3:6]),
                           axisangle_to_matrix(chunk7[i, 3:6]), atol=1e-6)


def test_6d_is_continuous_where_axis_angle_flips_sign():
    """Two nearly identical downward gripper poses: axis-angle jumps ~2pi, 6D barely moves.

    Measured on the robomimic Lift demos, 100% of absolute orientation targets live in this
    regime and 48% of frames flip sign — which is what makes axis-angle unlearnable there.
    """
    a = axisangle_to_matrix(np.array([np.pi - 1e-3, 0.0, 0.0]))
    b = axisangle_to_matrix(np.array([-(np.pi - 1e-3), 0.0, 0.0]) * -1.0)   # same rotation
    near = axisangle_to_matrix(np.array([-(np.pi - 1e-3), 1e-4, 0.0]))

    aa_gap = np.linalg.norm(matrix_to_axisangle(a) - np.array([-(np.pi - 1e-3), 1e-4, 0.0]))
    d6_gap = np.linalg.norm(matrix_to_rotation_6d(a) - matrix_to_rotation_6d(near))

    assert np.allclose(a, b, atol=1e-9)
    assert aa_gap > 6.0          # ~2pi apart in the representation the policy would regress
    assert d6_gap < 1e-2         # the same pair, continuous in 6D
