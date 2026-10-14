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
anchored.py). The two corrections must agree with each other. The roll sees the queued actions
WITH their anchoring shift applied, so the rolled robot state sits in the same object frame as
the observation it is sent with. Rolling raw targets puts the arm where the object USED to be
relative to it. On Square with a moving nut that offset (d ticks of nut travel) exceeded the
descent tolerance, and every appended chunk re-approached instead of grasping. The robot's own state is exact for absolute actions in a static world. With a
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

`inpaint=True` is the way to continue without a state model. The request carries the current
observation unchanged and the queued actions as a hard inpainting prefix (the policy's
`predict_inpaint`, RTC's guided sampling), and on arrival the first len(prefix) entries are
dropped: what remains is the policy's own continuation of the queued plan, conditioned on a
consistent observation. It needs no roll and no rolled proprio that contradicts the images, but
it is capped by the horizon. A queue of n >= H entries leaves nothing to append, so past d = H
it degrades toward waiting.

`gate_fn(obs, action) -> bool` holds an open-to-close gripper transition until the arm is
really at the grasp pose. It has the same semantics as anchored.py, which explains why a stale
chunk needs it. `lift_fn(obs, ticks_since_close) -> bool` holds the lift after it (lift_gate.py).
"""
from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

from evh_controller.chunk_executor.anchored import anchor_shift
from evh_controller.chunk_executor.base import ChunkExecutor
from evh_controller.chunk_executor.lift_gate import LiftHold

GRIP_OPEN_BELOW = 0.0


class ContinuationExecutor(ChunkExecutor):
    name = 'continuation'

    def __init__(self, worker, policy, *, roll_fn: Callable[[dict, np.ndarray], dict] | None,
                 anchor_fn: Callable[[dict], np.ndarray | None] | None = None,
                 diverged_fn: Callable[[dict, np.ndarray], bool] | None = None,
                 recent_n: int = 20, cooldown: int = 10,
                 gate_fn: Callable[[dict, np.ndarray], bool] | None = None,
                 gate_timeout: int = 60,
                 lead: int = 1, quantile: float = 0.9, delay_buffer: int = 50,
                 inpaint: bool = False,
                 lift_fn: Callable[[dict, int], bool] | None = None) -> None:
        if not getattr(policy, 'absolute_actions', False):
            raise ValueError('continuation appends absolute targets; this policy emits deltas')
        if inpaint and roll_fn is not None:
            raise ValueError('inpainting continuation plans from the observation as is; no roll')
        self.roll_fn = roll_fn
        self.inpaint = inpaint
        self.anchor_fn = anchor_fn
        self.diverged_fn = diverged_fn
        self.recent_n = recent_n
        self.cooldown = cooldown
        self.flushes = 0                       # divergence flushes, cumulative across episodes
        self.gate_fn = gate_fn
        self.gate_timeout = gate_timeout
        self.gate_holds = 0
        self.lift = None if lift_fn is None else LiftHold(lift_fn)
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
        self._prefix_n: dict[int, int] = {}    # request tick -> entries given as prefix
        self.shifts: list[float] = []
        self._recent: deque = deque(maxlen=self.recent_n)
        self._since_flush = self.cooldown
        self._last_grip = -1.0
        self._gated = 0
        self._close: tuple[int, np.ndarray] | None = None
        if self.lift is not None:
            self.lift.reset()

    def _check_divergence(self, obs) -> None:
        self._since_flush += 1
        if (self.diverged_fn is None or self._since_flush < self.cooldown
                or not self._recent):
            return
        if self.diverged_fn(obs, np.stack(self._recent)):
            self.flushes += 1
            self.reset()                       # queue and in-flight chunk planned for a lost future
            self._since_flush = 0

    def _queued(self, obs=None) -> np.ndarray:
        """The actions still to be played, as they WILL be played: anchoring shift applied."""
        if not self._queue:
            return np.zeros((0, int(self.policy.action_dim)), dtype=np.float32)
        if obs is None or self.anchor_fn is None:
            return np.stack([a for a, _ in self._queue])
        return np.stack([self._shifted(obs, a, anchor, record=False) for a, anchor in self._queue])

    def _request(self, obs, t) -> None:
        if self.inpaint:
            q = self._queued(obs)
            ok = (self._issue(obs, t, prefix=q, weights=np.ones(len(q))) if len(q)
                  else self._issue(obs, t))
            if ok:
                self._prefix_n[t] = len(q)
        else:
            planned_from = obs if self.roll_fn is None else self.roll_fn(obs, self._queued(obs))
            ok = self._issue(planned_from, t)
        if ok:
            if self.anchor_fn is not None:
                p = self.anchor_fn(obs)
                self._anchors[t] = None if p is None else np.asarray(p, float).copy()

    def step(self, obs, t):
        self._check_divergence(obs)
        arrival = self._poll(t)
        if arrival is not None:
            self._delays.append(max(0, t - arrival.t_issue))
            p = self._anchors.pop(arrival.t_issue, None)
            anchor = None if p is None else (arrival.t_issue, p)
            skip = self._prefix_n.pop(arrival.t_issue, 0)
            for a in np.asarray(arrival.chunk)[skip:]:
                self._queue.append((np.asarray(a, dtype=np.float32), anchor))

        if self._pending_t is None and len(self._queue) <= self.forecast_delay() + self.lead:
            self._request(obs, t)

        if not self._queue:
            return None
        entry = self._queue.popleft()
        a, anchor = entry
        closing = a[6] >= GRIP_OPEN_BELOW and self._last_grip < GRIP_OPEN_BELOW
        if closing and self.gate_fn is not None and self._gated < self.gate_timeout:
            pre = self._shifted(obs, a, anchor, as_open=True)
            if not self.gate_fn(obs, pre):
                self._queue.appendleft(entry)      # not played yet
                self._gated += 1
                self.gate_holds += 1
                pre = pre.copy()
                pre[6] = self._last_grip
                return pre
        if closing:
            now = None if self.anchor_fn is None else self.anchor_fn(obs)
            self._close = None if now is None else (t, np.asarray(now, float).copy())
        elif a[6] < GRIP_OPEN_BELOW:
            self._close = None
        out = self._shifted(obs, a, anchor)
        if self.lift is not None:
            if closing:
                self.lift.on_close(out)
            elif a[6] < GRIP_OPEN_BELOW:
                self.lift.on_open()
            else:
                hold = self.lift.check(obs, out)
                if hold is not None:
                    self._queue.appendleft(entry)  # not played yet
                    return hold
        self._gated = 0
        self._last_grip = float(a[6])
        self._recent.append(a)
        return out

    def _shifted(self, obs, a, anchor, as_open: bool = False, record: bool = True):
        if self.anchor_fn is None:
            return a
        out = anchor_shift(a, anchor, self.anchor_fn(obs), self._close, as_open)
        if record and out is not a:
            self.shifts.append(float(np.linalg.norm(out[:3] - a[:3])))
        return out
