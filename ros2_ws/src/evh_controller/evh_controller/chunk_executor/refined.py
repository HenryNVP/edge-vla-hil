"""Learned local refinement: a small model, not a hand rule, corrects each stale action.

Anchoring (anchored.py) and close gating are two hand-written corrections for a chunk that was
planned from an old view of a moving world: shift the pre-grasp targets by the object's motion,
hold the close until the arm is there. This executor replaces both with one learned function,
called on the robot every tick:

    action = refine_fn(obs, remaining, chunk_anchor, t)

`remaining` is the rest of the current chunk as the cloud planned it, starting at the entry the
queue would play now; `chunk_anchor` is (request tick, tracked object at that request) or None.
The function sees what anchoring and the gate see (the stale plan, the robot's own state, the
tracked object then and now) and returns the action to play. It decides WHERE (the target) and
WHEN (the gripper) together. The cloud still supplies the intent: the refiner gets the object's
motion since the request, not its pose, so it cannot plan the task by itself.

Queue mechanics are AnchoredExecutor's (the coverage-aware no-skip join), so `refined` is a
drop-in for `anchored` / `anchored_gate` in the comparison. Paper 2, Phase D.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from evh_controller.chunk_executor.anchored import AnchoredExecutor


class RefinedExecutor(AnchoredExecutor):
    """No-skip queue whose every played action comes from a learned local refiner."""
    name = 'refined'

    def __init__(self, worker, policy, *, anchor_fn: Callable[[dict], np.ndarray | None],
                 refine_fn: Callable, **kw) -> None:
        kw.pop('gate_fn', None)                # the refiner times the close itself
        self.refine_fn = refine_fn
        super().__init__(worker, policy, anchor_fn=anchor_fn, gate_fn=None, **kw)

    def _anchor(self, obs, a, t, as_open: bool = False):
        remaining = self._chunk[self._i - 1:]  # `a` was the entry just taken
        out = np.asarray(self.refine_fn(obs, remaining, self._chunk_anchor, t), np.float32)
        self.shifts.append(float(np.linalg.norm(out[:3] - a[:3])))
        return out
