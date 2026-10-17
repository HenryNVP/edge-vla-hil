import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import interp_corrector as ic  # noqa: E402


class Fake:
    n_obs_steps = 2

    def __init__(self, grip):
        self.grip = grip

    def predict(self, obs):
        c = np.zeros((4, 7))
        c[:, 6] = self.grip
        return c


def test_corrected_shifts_open_chunks_only(monkeypatch):
    monkeypatch.setattr(ic, 'encode', lambda policy, obs: np.zeros(3, np.float32))
    fix = lambda g: np.tile([10.0, -20.0], (len(g), 1))          # mm
    opened = ic.Corrected(Fake(-1.0), fix)
    c = opened.predict({})
    assert np.allclose(c[:, :2], [0.01, -0.02]) and np.allclose(c[:, 2:6], 0)
    assert opened.applied == 1 and opened.n_obs_steps == 2
    closed = ic.Corrected(Fake(1.0), fix)
    assert np.allclose(closed.predict({})[:, :6], 0) and closed.applied == 0
