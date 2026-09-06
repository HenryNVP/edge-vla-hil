"""Pure-numpy tests for the DP backend's abs-action rotation conversion (no torch/ROS).

The published robomimic image checkpoints emit rotation_6d; the backend must invert it to
axis-angle exactly like the repo's pytorch3d RotationTransformer, including near-pi rotations
(a downward-facing gripper IS a ~pi rotation, so the degenerate branch is the common case).
"""
import numpy as np

from evh_controller.dp_repo_policy import DiffusionPolicyRepoBackend
from evh_controller.rotation import (
    matrix_to_axisangle as _matrix_to_axisangle,
)
from evh_controller.rotation import (
    rotation_6d_to_matrix as _rotation_6d_to_matrix,
)
from evh_reactive.transforms import axisangle_to_quat


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _to_6d(R):
    return np.concatenate([R[0], R[1]])   # pytorch3d: first two rows


def test_rotation_6d_roundtrip():
    for R in (_rot_x(0.3), _rot_z(-1.2), _rot_x(0.5) @ _rot_z(0.7)):
        R2 = _rotation_6d_to_matrix(_to_6d(R))
        assert np.allclose(R2, R, atol=1e-9)


def test_matrix_to_axisangle_small_and_large():
    for theta in (0.0, 0.3, 1.5, np.pi - 1e-3, np.pi):
        aa = _matrix_to_axisangle(_rot_x(theta))
        assert np.isclose(np.linalg.norm(aa), theta, atol=1e-6), f'theta={theta}'
        if theta > 0:
            assert np.allclose(aa / np.linalg.norm(aa), [1, 0, 0], atol=1e-6)


def test_axisangle_consistent_with_reactive_transforms():
    """Plant OSC and reactive layer must agree with the backend on the encoding."""
    rng = np.random.default_rng(1)
    for _ in range(10):
        aa = rng.uniform(-1.5, 1.5, size=3)
        R = _rotation_6d_to_matrix(_to_6d_from_aa(aa))
        aa2 = _matrix_to_axisangle(R)
        assert np.allclose(aa2, aa, atol=1e-8)
        # quat built from the recovered axis-angle matches the original rotation
        q = axisangle_to_quat(aa2)
        assert np.isclose(abs(q[3]), np.cos(np.linalg.norm(aa) / 2.0), atol=1e-8)


def _to_6d_from_aa(aa):
    q = axisangle_to_quat(aa)
    x, y, z, w = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return _to_6d(R)


def test_undo_abs_transform_shape_and_gripper():
    chunk10 = np.zeros((4, 10), dtype=np.float32)
    chunk10[:, 0] = 0.1                                # x position
    chunk10[:, 3:9] = _to_6d(np.eye(3))                # identity rotation
    chunk10[:, 9] = -1.0                               # gripper open
    out = DiffusionPolicyRepoBackend._undo_abs_transform(chunk10)
    assert out.shape == (4, 7)
    assert np.allclose(out[:, 0], 0.1)
    assert np.allclose(out[:, 3:6], 0.0, atol=1e-9)
    assert np.allclose(out[:, 6], -1.0)
