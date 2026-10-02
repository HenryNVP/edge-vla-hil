"""An alpha-beta filter for a tracked object's position: the reactive layer's estimate of the world.

Object anchoring (evh_controller/chunk_executor/anchored.py) shifts a stale chunk by how far the
object moved since the chunk was planned. Its input is a position measurement made on the robot.
A real measurement is noisy, late (a detector or flow step takes frames), and sometimes missing
(occlusion by the gripper exactly when it matters). Feeding it raw turns the noise straight into
commanded motion, the delay into a lag, and a dropout into a frozen target.

The classical fix is a constant-velocity alpha-beta filter, a fixed-gain Kalman filter:

    predict   x <- x + v * dt                          (every tick, measurement or not)
    correct   r = z - x;  x <- x + alpha r;  v <- v + (beta / dt) r      (when z arrives)

and to report the estimate extrapolated over the measurement's known delay, x + v * latency, so
a late tracker gives a current position instead of an old one. It is deterministic, has no
learned part, and costs a few multiplies per tick. That is the kind of component paper 2's local
layer is meant to be made of.

`update(z)` is called once per control tick with the newest measurement (or None), and returns
the delay-compensated estimate, or None before the first measurement. Measurement time is the
tick it arrives minus `latency` ticks, so the filter state is kept at measurement time and
extrapolated forward on output.
"""
from __future__ import annotations

import numpy as np


class AlphaBetaFilter:
    def __init__(self, alpha: float = 0.5, beta: float = 0.1, latency: int = 0,
                 dt: float = 1.0, max_coast: int = 40) -> None:
        if not (0.0 < alpha <= 1.0 and 0.0 <= beta <= 2.0):
            raise ValueError('alpha in (0, 1], beta in [0, 2]')
        self.alpha, self.beta, self.latency, self.dt = alpha, beta, latency, dt
        self.max_coast = max_coast      # ticks without a measurement before velocity is dropped
        self.reset()

    def reset(self) -> None:
        self.x: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self._coast = 0

    def update(self, z) -> np.ndarray | None:
        if self.x is not None:
            self.x = self.x + self.v * self.dt
        if z is not None:
            z = np.asarray(z, dtype=float)
            if self.x is None:
                self.x, self.v = z.copy(), np.zeros_like(z)
            else:
                r = z - self.x
                self.x = self.x + self.alpha * r
                self.v = self.v + (self.beta / self.dt) * r
            self._coast = 0
        elif self.x is not None:
            self._coast += 1
            if self._coast > self.max_coast:     # a long occlusion: stop extrapolating
                self.v = np.zeros_like(self.v)
        if self.x is None:
            return None
        return self.x + self.v * (self.latency * self.dt)
