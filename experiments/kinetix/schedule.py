"""Tick schedules of any `chunk_executor` strategy, for replay on Kinetix (`replay.py`).

At a fixed delay d, what a strategy does (when it requests, which chunk entry it plays on each
tick, when it idles) depends only on d, its execution horizon s and the chunk length H, never on
the observations or the actions. So the strategy's own Python code runs once, here, against a
policy whose actions are labels (chunk number, entry) and a `StepWorker` with that delay. The
result is a static per-tick schedule, which the JAX loop then replays with the real policy and
environment. The strategy under test is the one the robosuite harness and LIBERO adapter run, not
a re-implementation of it.

Unguided strategies only: a guided request also carries a prefix of actions, which this replay
does not pass to the policy (RTC is the official loop in `executor.py`).

Conventions match `executor.run`:
  * free first chunk: the episode clock starts when the bootstrap chunk lands; in JAX that chunk
    is computed from the initial observation before the first tick, as the official eval does;
  * a request issued at tick t uses the observation at the start of tick t and is usable from
    tick t + d (`StepWorker`: issued in step t, delivered by the poll of step t + d).

Schedule arrays, one entry per episode tick (length = the deadline):
  issue[t]   ring slot the chunk requested at tick t is written to, or -1
  slot[t]    ring slot of the chunk played at tick t, or -1 (idle)
  entry[t]   entry of that chunk
The bootstrap chunk occupies slot 0. With one request in flight, at most the chunk being played,
its successor in flight and one more are live, so a ring of RING slots never overwrites a chunk
before its last use (checked while simulating).
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
for _p in (os.path.join(_REPO, 'ros2_ws', 'src', 'evh_controller'), os.path.join(_REPO, 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RING = 3
REPLAY_METHODS = ('noskip_async',)


@dataclasses.dataclass(frozen=True)
class Schedule:
    issue: np.ndarray
    slot: np.ndarray
    entry: np.ndarray

    @property
    def idle_frac(self) -> float:
        return float(np.mean(self.slot < 0))

    def joins(self) -> list[int]:
        """The entry at which each newly started chunk was first played."""
        out, prev = [], None
        for s, e in zip(self.slot, self.entry):
            if s >= 0 and s != prev:
                out.append(int(e))
            if s >= 0:
                prev = s
        return out


class _LabelPolicy:
    """Chunk k's entry j is the action (k, j)."""
    action_dim = 2
    guided_resampling = False

    def __init__(self, H: int) -> None:
        self.chunk_size = H
        self.calls = 0

    def predict(self, obs):
        c = np.stack([np.full(self.chunk_size, self.calls), np.arange(self.chunk_size)], axis=1)
        self.calls += 1
        return c.astype(np.float64)

    def predict_inpaint(self, obs, prefix, weights):
        raise ValueError('schedule replay supports unguided strategies only')


def simulate(method: str, d: int, s: int, H: int, deadline: int) -> Schedule:
    """Run strategy `method` for `deadline` episode ticks at fixed delay d; return its schedule."""
    from evh_controller.chunk_executor import _REGISTRY
    from sil_step import StepWorker, make_strategy

    if d < 1:
        raise ValueError(f'delay must be >= 1 tick, got {d}')
    policy = _LabelPolicy(H)
    worker = StepWorker(policy, lambda: d)
    cls = _REGISTRY[method]
    if getattr(cls, 'needs_guided_resampling', False):
        raise ValueError(f'{method} is guided; replay supports unguided strategies only')
    ex = make_strategy(method, worker, policy, s_min=s)

    # bootstrap: the first request, from the initial observation, lands at tick d = episode 0
    for t in range(d):
        a = ex.step({}, t)
        assert a is None and worker.requests == 1, 'nothing but the bootstrap before it lands'
    issue = np.full(deadline, -1, np.int32)
    slot = np.full(deadline, -1, np.int32)
    entry = np.zeros(deadline, np.int32)
    for tau in range(deadline):
        before = worker.requests
        a = ex.step({}, d + tau)
        if worker.requests > before:
            issue[tau] = (worker.requests - 1) % RING
        if a is not None:
            k, j = int(a[0]), int(a[1])
            assert k > worker.requests - 1 - RING, f'chunk {k} overwritten before its last use'
            slot[tau], entry[tau] = k % RING, j
    return Schedule(issue, slot, entry)


def from_cycle_plan(method: str, d: int, s: int, H: int, deadline: int) -> Schedule:
    """`executor.run`'s own schedule (sync / naive), unrolled; validates the replay against it."""
    from executor import IDLE, NEW, OLD, cycle_plan

    P, src, idx = cycle_plan(method, d, s, H)
    issue, slot, entry = [], [], []
    k = 0                                            # chunk in the current cycle's NEW role
    if method == 'sync':
        for j in range(min(s, H)):                   # the free first chunk, from its start
            issue.append(-1), slot.append(0), entry.append(j)
    first_cycle = True
    while len(slot) < deadline + P:
        k += 1
        for i, (sr, j) in enumerate(zip(src, idx)):
            issue.append(k % RING if i == 0 else -1)
            if sr == NEW:
                slot.append(k % RING), entry.append(int(j))
            elif sr == OLD:
                # the previous chunk, shifted by one period; the first cycle's OLD is the free
                # first chunk itself, unshifted
                jj = int(j) if first_cycle else P + int(j)
                ok = (method != 'sync') and jj < H
                slot.append((k - 1) % RING if ok else -1), entry.append(jj if ok else 0)
            else:
                assert sr == IDLE
                slot.append(-1), entry.append(0)
        first_cycle = False
    cut = lambda x: np.asarray(x[:deadline], np.int32)  # noqa: E731
    return Schedule(cut(issue), cut(slot), cut(entry))
