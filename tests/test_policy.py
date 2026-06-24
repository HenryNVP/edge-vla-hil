"""Pure-Python tests for the ACT policy backends — no ROS2 required.

These guard the controller's contract: whatever the backend, predict() returns a
[chunk_size, action_dim] float32 array. The stub returns zeros today; when the real ACT load
lands, these still hold and catch shape regressions.
"""
import numpy as np
import pytest

from evh_controller.act_policy import make_policy, PyTorchBackend, TensorRTBackend


def _dummy_obs():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    joint = np.zeros(7, dtype=np.float32)
    return image, joint


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
    image, joint = _dummy_obs()
    chunk = policy.predict(image, joint)
    assert chunk.shape == (policy.chunk_size, policy.action_dim)
    assert chunk.dtype == np.float32
