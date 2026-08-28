"""Pure-numpy tests for the reactive layer's quaternion helpers (no ROS needed).

The invariant that matters for the contract: composing an OSC_POSE axis-angle delta onto the
current orientation and then measuring the error back must recover the same delta — that's the
round trip the reactive node performs every waypoint/tick pair.
"""
import numpy as np
import pytest

from evh_reactive.transforms import (
    axisangle_to_quat,
    quat_conj,
    quat_mul,
    quat_normalize,
    quat_to_axisangle,
)


def test_axisangle_quat_roundtrip():
    for aa in ([0.3, 0.0, 0.0], [0.0, -0.5, 0.2], [1.0, 1.0, 1.0]):
        aa = np.asarray(aa, dtype=float)
        assert np.allclose(quat_to_axisangle(axisangle_to_quat(aa)), aa, atol=1e-9)


def test_zero_rotation():
    assert np.allclose(axisangle_to_quat(np.zeros(3)), [0, 0, 0, 1])
    assert np.allclose(quat_to_axisangle(np.array([0, 0, 0, 1.0])), np.zeros(3))


def test_quat_mul_90deg_about_z():
    qz = axisangle_to_quat([0, 0, np.pi / 2])
    q_full = quat_mul(qz, qz)   # two 90° turns = 180°
    assert np.allclose(quat_to_axisangle(q_full), [0, 0, np.pi], atol=1e-9)


def test_shortest_path_sign():
    # -q represents the same rotation; the error extraction must not return a ~2*pi detour
    q = axisangle_to_quat([0.2, 0.1, -0.3])
    assert np.allclose(quat_to_axisangle(-q), quat_to_axisangle(q), atol=1e-9)


def test_conj_inverts_unit_quat():
    q = axisangle_to_quat([0.4, -0.2, 0.7])
    assert np.allclose(quat_mul(q, quat_conj(q)), [0, 0, 0, 1], atol=1e-12)


def test_delta_compose_then_error_recovers_delta():
    """The reactive node's waypoint->target->error loop, end to end in math."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        q_current = quat_normalize(rng.normal(size=4))
        delta = rng.uniform(-0.5, 0.5, size=3)
        q_target = quat_mul(axisangle_to_quat(delta), q_current)         # latch (waypoint)
        err = quat_to_axisangle(quat_mul(q_target, quat_conj(q_current)))  # measure (tick)
        assert np.allclose(err, delta, atol=1e-9)


def test_normalize_degenerate():
    assert np.allclose(quat_normalize(np.zeros(4)), [0, 0, 0, 1])
    with pytest.raises(ValueError):
        quat_mul(np.zeros(3), np.zeros(4))   # malformed input should not pass silently
