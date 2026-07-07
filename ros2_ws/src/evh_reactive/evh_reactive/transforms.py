"""Minimal quaternion helpers for the reactive layer (pure numpy, no ROS/robosuite deps).

Convention: quaternions are [x, y, z, w] (robosuite / geometry_msgs order). Rotation deltas in
OSC_POSE actions are axis-angle vectors (direction = axis, norm = angle in radians) applied in
the world frame, matching robosuite's OSC goal update: R_goal = R(delta) @ R_current.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < _EPS:
        return IDENTITY_QUAT.copy()
    return q / n


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2 (apply q2's rotation, then q1's)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def axisangle_to_quat(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa)
    if angle < _EPS:
        return IDENTITY_QUAT.copy()
    axis = aa / angle
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)])


def quat_to_axisangle(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    if q[3] < 0.0:   # canonical hemisphere -> shortest-path rotation
        q = -q
    s = np.linalg.norm(q[:3])
    if s < _EPS:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(s, q[3])
    return (q[:3] / s) * angle
