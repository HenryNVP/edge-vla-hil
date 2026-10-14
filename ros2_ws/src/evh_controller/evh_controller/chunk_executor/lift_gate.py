"""Lift gating: after a close, hold the chunk's lift until the grasp has actually formed.

The close gate (anchored.py) decides WHEN the gripper closes; nothing decides when the arm may
leave with the object. A chunk times its lift from its own rollout, a few entries after its
close. On a stale chunk those entries arrive while the fingers are still closing, or after a
re-grasp the chunk never planned, and the arm pulls away from a half-formed grasp. Measured on
Square with ACT at d = 20 (docs/paper2-progress.md, 2026-10-15 d): slips and drops (held, then
lost on the lift) rose from 10 to 56 of 180 episodes, while peg misses stayed at 4-9.

`LiftHold` is the local fix, shared by the anchored and continuation executors. After a close is
played, a closed action whose target rises more than `rise_m` above the close pose is held, and
the close pose is commanded instead, until `lift_fn(obs, ticks_since_close)` passes. After
`timeout` held ticks it gives way, like the close gate, so a wrong check cannot deadlock the
episode. It never changes the gripper command and never moves the arm anywhere new: the worst
it can do is wait.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class LiftHold:
    def __init__(self, lift_fn: Callable[[dict, int], bool], rise_m: float = 0.005,
                 timeout: int = 20) -> None:
        self.lift_fn, self.rise_m, self.timeout = lift_fn, rise_m, timeout
        self.holds = 0                       # ticks a lift was held, cumulative, for analysis
        self.reset()

    def reset(self) -> None:
        self._ref: np.ndarray | None = None  # the close pose, as played
        self._ok = False
        self._held = 0
        self._since = 0

    def on_close(self, played) -> None:
        self._ref = np.array(played, dtype=np.float32, copy=True)
        self._ok, self._held, self._since = False, 0, 0

    def on_open(self) -> None:
        self._ref = None

    def check(self, obs, out) -> np.ndarray | None:
        """`out` is the closed action about to be played. Returns the pose to command instead
        (the close pose) while the lift is held, else None."""
        if self._ref is None or self._ok:
            return None
        self._since += 1
        if out[2] <= self._ref[2] + self.rise_m:
            return None                       # not lifting yet
        if self._held >= self.timeout or self.lift_fn(obs, self._since):
            self._ok = True
            return None
        self._held += 1
        self.holds += 1
        return self._ref.copy()
