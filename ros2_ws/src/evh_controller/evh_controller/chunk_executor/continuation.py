"""Continuation planning: request the next chunk from where the current plan ENDS, then append it.

With one request in flight, how fast can the robot advance through its plan? Each request
delivers H actions and requests are at least d apart, so the ceiling is min(1, H/d) actions per
tick. Paper 1's strategies sit below it:

    waiting (s = H)   plays H, then idles d                          ->  H / (H + d)
    no-skip queue     the new chunk is planned from the state at REQUEST time, so its first
                      entries re-plan the motion the robot makes during the delay; the join
                      drops them. In steady state the robot plays x = H/2 entries per cycle
                      (it plays the old chunk's x leftovers during the delay, then H - x new
                      ones)                                           ->  H / (2d)   for d >= H/2

So past H/2 the no-skip queue throws away half of every chunk, and past d = H it advances more
slowly than waiting. The waste is not the network's. It comes from planning from the wrong
state.

This executor plans from the RIGHT state. The request carries the current observation with the
robot's part rolled forward to the end of the actions already queued (`roll_fn`), and the arriving
chunk is APPENDED to the queue and played from entry 0, as a continuation. Nothing is re-planned
and nothing is dropped. Requests go out just in time: when the queue is down to the forecast
delay (plus `lead` ticks), so the plan is as fresh as the delay allows and the queue does not
grow without bound when d < H. This is VLASH's roll-forward idea (Tang et al. 2025), made to
work past H/2, where VLASH and the RTC family are not evaluated.

What rolling forward cannot fix is the WORLD: the observation's scene is still d + queue ticks
old. In a moving world combine it with object anchoring (`anchor_fn`, same semantics as
anchored.py). The robot's own state is exact for absolute actions in a static world. With a
learned policy and stale images the rolled-forward proprio contradicts the images, and whether
the policy tolerates that is an empirical question (paper 2, M3).

`roll_fn(obs, queued_actions) -> obs'`: queued_actions is the [n, A] array still to be played
(n may be 0). None means plan from the observation as is, the no-roll ablation, which re-plays
the delay motion.

Rolling forward commits the robot to a predicted future, including predicted outcomes (a grasp
that is assumed to succeed). When reality leaves the prediction, every queued action, and the
chunk in flight, was planned for a state that will not happen. `diverged_fn(obs, recent) -> bool`
is the task's check that this has happened. `recent` holds the last `recent_n` played actions,
oldest first. When it fires, the executor drops the queue and the request in flight (epoch bump,
as at an episode reset) and replans from the measurement. That costs one round trip, and buys
back the reactivity that continuation gave away. The check runs at most once per `cooldown` ticks
so a slow-to-clear condition cannot flush every tick.
"""
from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

from evh_controller.chunk_executor.base import ChunkExecutor

GRIP_OPEN_BELOW = 0.0


class ContinuationExecutor(ChunkExecutor):
    name = 'continuation'

    def __init__(self, worker, policy, *, roll_fn: Callable[[dict, np.ndarray], dict] | None,
                 anchor_fn: Callable[[dict], np.ndarray | None] | None = None,
                 diverged_fn: Callable[[dict, np.ndarray], bool] | None = None,
                 recent_n: int = 20, cooldown: int = 10,
                 lead: int = 1, quantile: float = 0.9, delay_buffer: int = 50) -> None:
        if not getattr(policy, 'absolute_actions', False):
            raise ValueError('continuation appends absolute targets; this policy emits deltas')
        self.roll_fn = roll_fn
        self.anchor_fn = anchor_fn
        self.diverged_fn = diverged_fn
        self.recent_n = recent_n
        self.cooldown = cooldown
        self.flushes = 0                       # divergence flushes, cumulative across episodes
        self.lead = lead
        self.quantile = quantile
        self.delay_buffer = delay_buffer
        self._delays: list[int] = []           # a property of the link, kept across episodes
        super().__init__(worker, policy)

    def forecast_delay(self) -> int:
        if not self._delays:
            return 1
        return int(np.ceil(np.quantile(self._delays[-self.delay_buffer:], self.quantile)))

    def _on_reset(self) -> None:
        self._queue: deque = deque()           # (action, anchor at its request or None)
        self._anchors: dict[int, np.ndarray | None] = {}
        self.shifts: list[float] = []
        self._recent: deque = deque(maxlen=self.recent_n)
        self._since_flush = self.cooldown

    def _check_divergence(self, obs) -> None:
        self._since_flush += 1
        if (self.diverged_fn is None or self._since_flush < self.cooldown
                or not self._recent):
            return
        if self.diverged_fn(obs, np.stack(self._recent)):
            self.flushes += 1
            self.reset()                       # queue and in-flight chunk planned for a lost future
            self._since_flush = 0

    def _queued(self) -> np.ndarray:
        if not self._queue:
            return np.zeros((0, int(self.policy.action_dim)), dtype=np.float32)
        return np.stack([a for a, _ in self._queue])

    def _request(self, obs, t) -> None:
        planned_from = obs if self.roll_fn is None else self.roll_fn(obs, self._queued())
        if self._issue(planned_from, t):
            if self.anchor_fn is not None:
                p = self.anchor_fn(obs)
                self._anchors[t] = None if p is None else np.asarray(p, float).copy()

    def step(self, obs, t):
        self._check_divergence(obs)
        arrival = self._poll(t)
        if arrival is not None:
            self._delays.append(max(0, t - arrival.t_issue))
            anchor = self._anchors.pop(arrival.t_issue, None)
            for a in np.asarray(arrival.chunk):
                self._queue.append((np.asarray(a, dtype=np.float32), anchor))

        if self._pending_t is None and len(self._queue) <= self.forecast_delay() + self.lead:
            self._request(obs, t)

        if not self._queue:
            return None
        a, anchor = self._queue.popleft()
        self._recent.append(a)
        if anchor is None or self.anchor_fn is None or a[6] >= GRIP_OPEN_BELOW:
            return a
        now = self.anchor_fn(obs)
        if now is None:
            return a
        shift = np.asarray(now, float) - anchor
        self.shifts.append(float(np.linalg.norm(shift)))
        out = a.copy()
        out[:3] += shift.astype(np.float32)
        return out
