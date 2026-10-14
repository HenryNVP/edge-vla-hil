"""Object anchoring: keep a stale chunk aimed at where the object is NOW, not where it was.

A chunk is planned from one observation and played up to two round trips later. In a static
world that is harmless (paper 1). In a moving world the chunk's approach aims at the object's old
position, and no execution strategy fixes that: the plan itself is wrong. But the robot can see
the object move. If it can measure the object's position locally (a tracker; privileged state in
the mechanism testbed), the part of the chunk that is *about* the object can be shifted by how
far the object moved since the chunk was planned:

    a[:3]  +=  p_object(now) - p_object(at the request that produced the chunk)

applied in full while the chunk's gripper command is OPEN (the pre-grasp part of the plan). When
the gripper closes, the object starts moving with the hand, so tracking it further would apply
the hand's own motion twice. Instead, the shift is FROZEN at the object's position at the moment
of the close: the closing and lifting part of a chunk planned before the grasp keeps the
correction it had, instead of snapping back to the stale grasp pose. A chunk requested after the
close was planned with the object already in hand and is not shifted.
This is the world-motion half of the staleness decomposition (paper 2, docs/paper2-edge-plan.md).
The robot-motion half needs no measurement, since absolute targets already ignore where the arm
happens to be.

Anchoring corrects WHERE the chunk goes, not WHEN its discrete events happen. A chunk decides
to close the gripper at the tick its own rollout says the arm reaches the grasp pose. The real
arm lags its target, and more so behind a moving target, so a stale chunk closes on nothing
(measured on Square at 2 cm/s: closes issued 1-3 cm off the handle, finger width 0.004). The
optional `gate_fn(obs, action) -> bool` is the local fix: an open-to-close transition is
held until the gate passes. Meanwhile the chunk does not advance and the arm keeps servoing to
the anchored pre-grasp target. After `gate_timeout` ticks the gate gives way, so a wrong gate
cannot deadlock the episode. The optional `lift_fn(obs, ticks_since_close) -> bool` does the
same for the lift that follows (lift_gate.py).

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
from evh_controller.chunk_executor.lift_gate import LiftHold

GRIP_OPEN_BELOW = 0.0        # robosuite gripper command: -1 open, +1 closed


class AnchoredExecutor(CoverageAwareExecutor):
    """Coverage-aware execution whose pre-grasp actions follow the tracked object."""
    name = 'anchored'
    needs_guided_resampling = False

    def __init__(self, worker, policy, *, anchor_fn: Callable[[dict], np.ndarray | None],
                 gate_fn: Callable[[dict, np.ndarray], bool] | None = None,
                 gate_timeout: int = 60,
                 lift_fn: Callable[[dict, int], bool] | None = None, **kw) -> None:
        if not getattr(policy, 'absolute_actions', False):
            raise ValueError('object anchoring shifts absolute targets; this policy emits deltas')
        self.lift = None if lift_fn is None else LiftHold(lift_fn)
        kw.setdefault('rule', 'aligned')
        kw.setdefault('guided', False)
        self.anchor_fn = anchor_fn
        self.gate_fn = gate_fn
        self.gate_timeout = gate_timeout
        super().__init__(worker, policy, **kw)

    def _on_reset(self) -> None:
        super()._on_reset()
        self._anchors: dict[int, np.ndarray | None] = {}   # request tick -> object at request
        self._arrived_anchor: tuple[int, np.ndarray] | None = None
        self._chunk_ref = None
        self._chunk_anchor: tuple[int, np.ndarray] | None = None
        self._close: tuple[int, np.ndarray] | None = None  # (tick, object) at the last close
        self.shifts: list[float] = []                       # |shift| per shifted action, for analysis
        self._last_grip = -1.0
        self._gated = 0
        self.gate_holds = 0                                 # ticks a close was held, for analysis
        if self.lift is not None:
            self.lift.reset()

    def _issue(self, obs, t, prefix=None, weights=None) -> bool:
        ok = super()._issue(obs, t, prefix, weights)
        if ok:
            p = self.anchor_fn(obs)
            self._anchors[t] = None if p is None else np.asarray(p, dtype=float).copy()
        return ok

    def _poll(self, t):
        arrival = super()._poll(t)
        if arrival is not None:
            p = self._anchors.pop(arrival.t_issue, None)
            self._arrived_anchor = None if p is None else (arrival.t_issue, p)
        return arrival

    def step(self, obs, t):
        a = super().step(obs, t)
        if self._chunk is not self._chunk_ref:          # a newly arrived chunk was adopted
            self._chunk_ref = self._chunk
            self._chunk_anchor = self._arrived_anchor
        if a is None:
            return a
        closing = a[6] >= GRIP_OPEN_BELOW and self._last_grip < GRIP_OPEN_BELOW
        if closing and self.gate_fn is not None and self._gated < self.gate_timeout:
            pre = self._anchor(obs, a, t, as_open=True)
            if not self.gate_fn(obs, pre):
                # hold the close: un-play this entry, keep commanding the open pre-grasp pose
                self._i -= 1
                self._played -= 1
                self._n_from_chunk -= 1
                self._gated += 1
                self.gate_holds += 1
                pre[6] = self._last_grip
                return pre
        if closing:
            now = self.anchor_fn(obs)
            self._close = None if now is None else (t, np.asarray(now, dtype=float).copy())
        elif a[6] < GRIP_OPEN_BELOW:
            self._close = None
        self._gated = 0
        self._last_grip = float(a[6])
        out = self._anchor(obs, a, t)
        if self.lift is not None:
            if closing:
                self.lift.on_close(out)
            elif a[6] < GRIP_OPEN_BELOW:
                self.lift.on_open()
            else:
                hold = self.lift.check(obs, out)
                if hold is not None:            # un-play this entry, keep the close pose
                    self._i -= 1
                    self._played -= 1
                    self._n_from_chunk -= 1
                    return hold
        return out

    def _anchor(self, obs, a, t, as_open: bool = False):
        out = anchor_shift(a, self._chunk_anchor, self.anchor_fn(obs), self._close, as_open)
        if out is not a:
            self.shifts.append(float(np.linalg.norm(out[:3] - a[:3])))
        return out


def anchor_shift(a, chunk_anchor, now, close, as_open: bool = False):
    """The anchoring rule, shared with continuation.py. `chunk_anchor` is (request tick, object
    at request) or None, `now` the object now or None, `close` (tick, object) at the last close
    or None. Returns `a` itself when nothing applies."""
    if chunk_anchor is None:
        return a
    t_req, at_req = chunk_anchor
    if a[6] < GRIP_OPEN_BELOW or as_open:
        if now is None:
            return a
        shift = np.asarray(now, dtype=float) - at_req
    else:
        if close is None or t_req > close[0]:
            return a                       # planned with the object already in hand
        shift = close[1] - at_req          # frozen at the moment of the close
    out = np.array(a, dtype=np.float32, copy=True)
    out[:3] += shift.astype(np.float32)
    return out
