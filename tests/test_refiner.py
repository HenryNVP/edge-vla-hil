"""Tests for refiner.py's robot-side pieces (features, numpy MLP, refine_fn) — no torch, no
robosuite.

Pinned: the feature layout and padding of a short remainder, positions relative to the EE, the
object entering only as motion since the request, the numpy forward pass against a hand
computation, and the refined action (EE + predicted offset, rotation from the chunk, gripper
from the logit sign).
"""
import numpy as np

import refiner as rf


def _chunk(n):
    c = np.zeros((n, 7), np.float32)
    c[:, 0] = 0.5 + 0.01 * np.arange(n)
    c[:, 3:6] = [0.1, 0.2, 0.3]
    c[:, 6] = -1.0
    return c


def test_features_layout_and_padding():
    p9 = np.array([0.5, 0, 1.0, 0, 0, 0, 1, 0.04, -0.04], np.float32)
    x = rf.features(p9, _chunk(3), [0.1, 0.1, 0.8], [0.1, 0.12, 0.8], 15)
    assert x.shape == (rf.N_FEATURES,)
    pos = x[:rf.H * 3].reshape(rf.H, 3)
    assert np.allclose(pos[:3, 0], [0.0, 0.01, 0.02]) and np.allclose(pos[3:, 0], 0.02)
    assert np.allclose(pos[:, 2], -1.0)
    assert np.allclose(x[rf.H * 4:rf.H * 4 + 3], [0, 0.02, 0])     # motion, not pose
    assert np.allclose(x[-3:], [0.04, -0.04, 0.5])


def _model(tmp_path, bias_out):
    n_out = rf.K * 4
    W0 = np.zeros((rf.N_FEATURES, 4), np.float32)
    W1 = np.zeros((4, n_out), np.float32)
    np.savez(tmp_path / 'r.npz', n_layers=2, W0=W0, b0=np.ones(4, np.float32), W1=W1,
             b1=np.asarray(bias_out, np.float32), x_mean=np.zeros(rf.N_FEATURES),
             x_std=np.ones(rf.N_FEATURES), y_mean=np.full(rf.K * 3, 0.01),
             y_std=np.full(rf.K * 3, 2.0))
    return rf.Refiner(str(tmp_path / 'r.npz'))


def test_refine_action(tmp_path):
    bias = np.zeros(rf.K * 4)
    bias[:3] = [1.0, -1.0, 0.0]                  # normalised -> 2*b + 0.01
    bias[rf.K * 3] = 0.5                         # close
    model = _model(tmp_path, bias)
    p9 = np.array([0.5, 0, 1.0, 0, 0, 0, 1, 0.04, -0.04], np.float32)
    a = rf.refine(model, p9, _chunk(5), np.zeros(3), np.zeros(3), 0)
    assert np.allclose(a[:3], [0.5 + 2.01, -1.99, 1.01])
    assert np.allclose(a[3:6], [0.1, 0.2, 0.3]) and a[6] == 1.0


def test_refine_fn_without_anchor_sees_no_motion(tmp_path):
    seen = []

    class Spy:
        def __call__(self, x):
            seen.append(x)
            return np.zeros(rf.K * 4)

    fn = rf.make_refine_fn(Spy(), lambda o: np.array([0.0, 0.3, 0.8]))
    obs = {'proprio': np.zeros((1, 9), np.float32)}
    a = fn(obs, _chunk(4), None, 7)
    assert np.allclose(seen[-1][rf.H * 4:rf.H * 4 + 3], 0) and seen[-1][-1] == 0
    assert a[6] == -1.0
    fn(obs, _chunk(4), (2, np.array([0.0, 0.25, 0.8])), 7)
    assert np.allclose(seen[-1][rf.H * 4:rf.H * 4 + 3], [0, 0.05, 0])
    assert np.isclose(seen[-1][-1], 5 / 30)
