"""Pure-Python tests for the ACT policy backends — no ROS2 required.

These guard the controller's contract: whatever the backend, predict() returns a
[chunk_size, action_dim] float32 array. The stub returns zeros today; when the real ACT load
lands, these still hold and catch shape regressions.
"""
import numpy as np
import pytest

from evh_controller.policy import (
    POLICY_SIDECAR,
    ACTBackend,
    ONNXBackend,
    PyTorchBackend,
    TensorRTBackend,
    act_action_dim,
    make_policy,
    resolve_absolute,
    sidecar_absolute,
)


def _dummy_obs():
    return {
        'agentview': np.zeros((2, 84, 84, 3), dtype=np.uint8),   # stacked history of 2
        'wrist': np.zeros((2, 84, 84, 3), dtype=np.uint8),
        'proprio': np.zeros((2, 9), dtype=np.float32),
    }


@pytest.mark.parametrize('backend,cls', [
    ('pytorch', PyTorchBackend),
    ('act', ACTBackend),
    ('onnx', ONNXBackend),
    ('tensorrt', TensorRTBackend),
])
def test_make_policy_returns_backend(backend, cls):
    policy = make_policy(backend, weights_path='')
    assert isinstance(policy, cls)


def test_make_policy_aliases():
    assert isinstance(make_policy('torch', ''), PyTorchBackend)
    assert isinstance(make_policy('act_lerobot', ''), ACTBackend)
    assert isinstance(make_policy('act_onnx', ''), ONNXBackend)
    assert isinstance(make_policy('trt', ''), TensorRTBackend)


def test_make_policy_rejects_unknown():
    with pytest.raises(ValueError):
        make_policy('jax', '')


@pytest.mark.parametrize('backend', ['pytorch', 'act', 'onnx', 'tensorrt'])
def test_predict_chunk_shape_and_dtype(backend):
    policy = make_policy(backend, '')
    chunk = policy.predict(_dummy_obs())
    assert chunk.shape == (policy.chunk_size, policy.action_dim)
    assert chunk.dtype == np.float32


def test_newest_unwraps_history():
    from evh_controller.policy import newest
    imgs = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    assert np.array_equal(newest(imgs), imgs[1])
    vecs = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(newest(vecs), [3.0, 4.0])
    single = np.zeros((4, 4, 3))
    assert newest(single).shape == (4, 4, 3)   # no history axis -> unchanged


# --- action convention (invariant 1) -------------------------------------------------------
# An ACT checkpoint is 7-dim whether it was trained on delta or absolute actions, so unlike the
# DP backend it cannot derive the mode from its own shapes. It reads a stamp written beside the
# weights; a wrong answer here is silent garbage motion, caught only by the plant's cross-check.

def _stamp(tmp_path, payload):
    (tmp_path / POLICY_SIDECAR).write_text(payload)
    return str(tmp_path)


def test_sidecar_absolute_reads_the_stamp(tmp_path):
    assert sidecar_absolute(_stamp(tmp_path, '{"absolute_actions": true}')) is True


def test_sidecar_absolute_reads_a_delta_stamp(tmp_path):
    assert sidecar_absolute(_stamp(tmp_path, '{"absolute_actions": false}')) is False


def test_sidecar_absolute_is_none_when_unstamped(tmp_path):
    assert sidecar_absolute(str(tmp_path)) is None
    assert sidecar_absolute(str(tmp_path / 'nope')) is None
    assert sidecar_absolute('') is None


def test_sidecar_absolute_is_none_when_the_key_is_missing(tmp_path):
    assert sidecar_absolute(_stamp(tmp_path, '{"chunk_size": 16}')) is None


def test_sidecar_absolute_raises_rather_than_defaulting_on_a_broken_stamp(tmp_path):
    """A truncated stamp must not read as 'delta' — that is the failure it exists to prevent."""
    with pytest.raises(ValueError):
        sidecar_absolute(_stamp(tmp_path, '{"absolute_actions": tru'))


@pytest.mark.parametrize('backend', ['act', 'onnx'])
def test_make_policy_forwards_the_absolute_override(backend):
    assert make_policy(backend, '', absolute=True).absolute_actions is True
    assert make_policy(backend, '', absolute=False).absolute_actions is False


@pytest.mark.parametrize('backend', ['act', 'onnx'])
def test_backends_default_to_delta_without_a_stamp(backend):
    assert make_policy(backend, '').absolute_actions is False


def test_resolve_absolute_prefers_the_override_over_the_stamp():
    assert resolve_absolute(False, True, 'ckpt') is False
    assert resolve_absolute(True, False, 'ckpt') is True


def test_resolve_absolute_falls_back_to_the_stamp_then_to_delta():
    assert resolve_absolute(None, True, 'ckpt') is True
    assert resolve_absolute(None, False, 'ckpt') is False
    assert resolve_absolute(None, None, 'ckpt') is False


def test_act_action_dim_reads_the_head_width_from_the_config(tmp_path):
    (tmp_path / 'config.json').write_text(
        '{"output_features": {"action": {"type": "ACTION", "shape": [10]}}}')
    assert act_action_dim(str(tmp_path)) == 10


def test_act_action_dim_is_none_when_it_cannot_tell(tmp_path):
    assert act_action_dim(str(tmp_path)) is None
    (tmp_path / 'config.json').write_text('{"output_features": {}}')
    assert act_action_dim(str(tmp_path)) is None


def test_a_ten_dim_act_head_announces_absolute_without_a_stamp(tmp_path):
    """The orchestrator must reach the same verdict the backend will, before any node starts."""
    from evh_controller.policy import stamped_absolute

    (tmp_path / 'config.json').write_text(
        '{"output_features": {"action": {"shape": [10]}}}')
    assert stamped_absolute('act', str(tmp_path)) is True


def test_a_seven_dim_act_head_still_needs_its_stamp(tmp_path):
    from evh_controller.policy import stamped_absolute

    (tmp_path / 'config.json').write_text('{"output_features": {"action": {"shape": [7]}}}')
    assert stamped_absolute('act', str(tmp_path)) is None
    (tmp_path / POLICY_SIDECAR).write_text('{"absolute_actions": false}')
    assert stamped_absolute('act', str(tmp_path)) is False
