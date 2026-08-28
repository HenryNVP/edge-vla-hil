"""Pure-Python tests for the ACT policy backends — no ROS2 required.

These guard the controller's contract: whatever the backend, predict() returns a
[chunk_size, action_dim] float32 array. The stub returns zeros today; when the real ACT load
lands, these still hold and catch shape regressions.
"""
import numpy as np
import pytest

from evh_controller.policy import PyTorchBackend, TensorRTBackend, make_policy


def _dummy_obs():
    return {
        'agentview': np.zeros((2, 84, 84, 3), dtype=np.uint8),   # stacked history of 2
        'wrist': np.zeros((2, 84, 84, 3), dtype=np.uint8),
        'proprio': np.zeros((2, 9), dtype=np.float32),
    }


@pytest.mark.parametrize('backend,cls', [
    ('pytorch', PyTorchBackend),
    ('tensorrt', TensorRTBackend),
])
def test_make_policy_returns_backend(backend, cls):
    policy = make_policy(backend, weights_path='')
    assert isinstance(policy, cls)


def test_make_policy_aliases():
    assert isinstance(make_policy('torch', ''), PyTorchBackend)
    assert isinstance(make_policy('trt', ''), TensorRTBackend)


def test_make_policy_rejects_unknown():
    with pytest.raises(ValueError):
        make_policy('jax', '')


@pytest.mark.parametrize('backend', ['pytorch', 'tensorrt'])
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
