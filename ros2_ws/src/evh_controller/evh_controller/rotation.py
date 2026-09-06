"""Rotation conversions shared by the policy backends (pure numpy, no ROS/torch).

Two conventions meet here. The project's action contract is 7-dim
`[pos(3), axis-angle(3), gripper]`; the abs-action checkpoints — the DP robomimic ones, and any
ACT trained on an absolute-action dataset — carry rotation as pytorch3d **6D**: the first two
ROWS of the rotation matrix, with the third recovered by Gram-Schmidt.

The 6D detour is not decoration. Measured on the robomimic Lift demos, every absolute
orientation target sits within 0.12 rad of the pi wrap (the gripper points down for the whole
task), where axis-angle is discontinuous: 48% of frames flip sign and consecutive frames jump by
up to 2 pi with no motion behind it. Regressing that target directly teaches a policy to average
two antipodal representations. 6D is continuous everywhere, which is why a network predicts it
and this module converts at the boundary — never the other way round.

Ported from the diffusion_policy repo's pytorch3d RotationTransformer, in numpy so the Jetson
image needs neither torch nor pytorch3d to run an exported policy.
"""
from __future__ import annotations

import numpy as np


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """pytorch3d convention: d6 = first two ROWS of R; Gram-Schmidt the third."""
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2p = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2p / np.linalg.norm(a2p, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def matrix_to_axisangle(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle, robust near 0 and pi (via quaternion, xyzw)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 0.0)) * 2.0
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    if q[3] < 0.0:
        q = -q
    v = np.linalg.norm(q[:3])
    if v < 1e-12:
        return np.zeros(3)
    return (q[:3] / v) * (2.0 * np.arctan2(v, q[3]))


def axisangle_to_matrix(v: np.ndarray) -> np.ndarray:
    """Axis-angle (axis * angle) -> rotation matrix. Rodrigues; identity at zero rotation."""
    v = np.asarray(v, dtype=np.float64)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def matrix_to_rotation_6d(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> pytorch3d 6D: the first two ROWS, matching rotation_6d_to_matrix."""
    return np.asarray(R, dtype=np.float64)[..., :2, :].reshape(*np.shape(R)[:-2], 6)


def abs10_to_abs7(chunk10: np.ndarray) -> np.ndarray:
    """[H, 10] abs [pos, rot_6d, gripper] -> the graph's 7-dim [pos, axis-angle, gripper]."""
    chunk10 = np.asarray(chunk10, dtype=np.float64)
    out = np.empty((chunk10.shape[0], 7), dtype=np.float32)
    out[:, :3] = chunk10[:, :3]
    R = rotation_6d_to_matrix(chunk10[:, 3:9])
    for i in range(chunk10.shape[0]):
        out[i, 3:6] = matrix_to_axisangle(R[i])
    out[:, 6] = chunk10[:, 9]
    return out


def abs7_to_abs10(chunk7: np.ndarray) -> np.ndarray:
    """Inverse of `abs10_to_abs7`: back into a checkpoint's native 10-dim action space.

    Needed wherever something computed in the 7-dim contract has to re-enter the model's own
    space — RTC guidance for the DP backend, and converting a demonstration into training
    targets. Guiding or training in the wrong space looks perfectly well-formed and is nonsense.
    """
    chunk7 = np.asarray(chunk7, dtype=np.float64)
    out = np.empty((chunk7.shape[0], 10), dtype=np.float32)
    out[:, :3] = chunk7[:, :3]
    for i in range(chunk7.shape[0]):
        out[i, 3:9] = matrix_to_rotation_6d(axisangle_to_matrix(chunk7[i, 3:6]))
    out[:, 9] = chunk7[:, 6]
    return out
