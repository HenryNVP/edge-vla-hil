"""Object anchoring: keep a stale chunk aimed at where the object is NOW, not where it was.

A chunk is planned from one observation and played up to two round trips later. In a static
world that is harmless (paper 1). In a moving world the chunk's approach aims at the object's old
position, and no execution strategy fixes that: the plan itself is wrong. But the robot can see
the object move. If it can measure the object's position locally (a tracker; privileged state in
the mechanism testbed), the part of the chunk that is *about* the object can be shifted by how
far the object moved since the chunk was planned:

    a[:3]  +=  p_object(now) - p_object(at the request that produced the chunk)

applied only while the chunk's gripper command is OPEN. That is the pre-grasp part of the plan.
Once the gripper closes, the object moves with the hand and the shift would be applied twice.
This is the world-motion half of the staleness decomposition (paper 2, docs/paper2-edge-plan.md).
The robot-motion half needs no measurement, since absolute targets already ignore where the arm
happens to be.

`AnchoredExecutor` is the coverage-aware family with this one change, so it inherits the
never-skip join: `rule='aligned'` is the anchored no-skip queue, `rule='waiting'` is anchored
waiting. `anchor_fn(obs) -> xyz | None` is the tracker: None means "no fix this tick", and the
chunk then plays unshifted. Absolute actions only. A delta action has no position to shift, and
for deltas the correction would have to be re-derived from the target.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from evh_controller.chunk_executor.adaptive import CoverageAwareExecutor

GRIP_OPEN_BELOW = 0.0        # robosuite gripper command: -1 open, +1 closed


class AnchoredExecutor(CoverageAwareExecutor):
    """Coverage-aware execution whose pre-grasp actions follow the tracked object."""
    name = 'anchored'
    needs_guided_resampling = False

    def __init__(self, worker, policy, *, anchor_fn: Callable[[dict], np.ndarray | None],
                 **kw) -> None:
        if not getattr(policy, 'absolute_actions', False):
            raise ValueError('object anchoring shifts absolute targets; this policy emits deltas')
        kw.setdefault('rule', 'aligned')
        kw.setdefault('guided', False)
        self.anchor_fn = anchor_fn
        super().__init__(worker, policy, **kw)

    def _on_reset(self) -> None:
        super()._on_reset()
        self._anchors: dict[int, np.ndarray | None] = {}   # request tick -> object at request
        self._arrived_anchor: np.ndarray | None = None
        self._chunk_ref = None
        self._chunk_anchor: np.ndarray | None = None
        self.shifts: list[float] = []                       # |shift| per shifted action, for analysis

    def _issue(self, obs, t, prefix=None, weights=None) -> bool:
        ok = super()._issue(obs, t, prefix, weights)
        if ok:
            p = self.anchor_fn(obs)
            self._anchors[t] = None if p is None else np.asarray(p, dtype=float).copy()
        return ok

    def _poll(self, t):
        arrival = super()._poll(t)
        if arrival is not None:
            self._arrived_anchor = self._anchors.pop(arrival.t_issue, None)
        return arrival

    def step(self, obs, t):
        a = super().step(obs, t)
        if self._chunk is not self._chunk_ref:          # a newly arrived chunk was adopted
            self._chunk_ref = self._chunk
            self._chunk_anchor = self._arrived_anchor
        if a is None or self._chunk_anchor is None or a[6] >= GRIP_OPEN_BELOW:
            return a
        now = self.anchor_fn(obs)
        if now is None:
            return a
        shift = np.asarray(now, dtype=float) - self._chunk_anchor
        out = np.array(a, dtype=np.float32, copy=True)
        out[:3] += shift.astype(np.float32)
        self.shifts.append(float(np.linalg.norm(shift)))
        return out
