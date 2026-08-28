"""The three prior-work baselines, kept in one file so their splices stay comparable by eye.

They differ only in what happens when a late chunk lands, and reading them side by side is the
point — that difference is what the benchmark measures:

  SynchronousExecutor      execute the chunk fully, then HOLD while the next computes. Success
                           survives (every chunk starts from a fresh observation) but throughput
                           collapses: the pauses grow linearly with inference+network latency.
  NaiveAsyncExecutor       replan on a cadence and jump to new[0] on arrival. No time alignment,
                           so the new chunk replays a past the robot already lived — the
                           chunk-boundary discontinuity RTC exists to fix.
  TemporalEnsembleExecutor ACT-style weighted average over time-aligned overlapping chunks.
                           Smoothness only, no latency model; the weak baseline RTC beats.

References: ACT/temporal ensembling (RSS 2023).
"""
from __future__ import annotations

import numpy as np

from evh_controller.chunk_executor.base import ChunkExecutor


class SynchronousExecutor(ChunkExecutor):
    """Execute the chunk fully, then hold position while the next one computes.

    The prior-work default: success survives (each chunk starts from a fresh observation) but
    throughput collapses — the pauses grow linearly with inference+network latency.
    """
    name = 'synchronous'

    def _on_reset(self) -> None:
        self._chunk: np.ndarray | None = None
        self._i = 0

    def step(self, obs, t):
        arrival = self._poll(t)
        if arrival is not None:
            self._chunk, self._i = arrival.chunk, 0
        if self._chunk is not None:
            a = self._chunk[self._i]
            self._i += 1
            if self._i >= len(self._chunk):
                self._chunk = None
                self._issue(obs, t)   # start the next chunk as the last action goes out
            return a
        self._issue(obs, t)           # bootstrap / retry while worker still busy
        return None                   # paused: emit nothing, the robot holds


class NaiveAsyncExecutor(ChunkExecutor):
    """Replan on a fixed cadence; on arrival jump to new[0] immediately.

    No time alignment: the new chunk was planned from an observation `delay` steps old, so its
    first actions replay a past the robot already lived — the discontinuity RTC fixes. If the old
    chunk runs out before the new one lands, the last action is repeated open-loop.
    """
    name = 'naive_async'

    def __init__(self, worker, policy, replan_every: int = 8) -> None:
        self.replan_every = replan_every
        super().__init__(worker, policy)

    def _on_reset(self) -> None:
        self._chunk: np.ndarray | None = None
        self._i = 0
        self._last_issue: int | None = None

    def step(self, obs, t):
        arrival = self._poll(t)
        if arrival is not None:
            self._chunk, self._i = arrival.chunk, 0
        due = self._last_issue is None or (t - self._last_issue) >= self.replan_every
        if due and self._issue(obs, t):
            self._last_issue = t
        if self._chunk is None:
            return None
        i = min(self._i, len(self._chunk) - 1)
        self._i += 1
        return self._chunk[i]


class TemporalEnsembleExecutor(ChunkExecutor):
    """ACT temporal ensembling: exponentially-weighted average over overlapping chunks.

    Chunks are time-aligned by ISSUE tick (a chunk requested at t_issue predicts actions for
    t_issue, t_issue+1, ...). Following the ACT reference implementation, the OLDEST prediction
    covering the current tick gets the highest weight: w_i = exp(-m * i) with i ranked oldest
    first (m=0.01 there, near-uniform). Smoothness only — no latency model; this is the weak
    baseline RTC beats.
    """
    name = 'temporal_ensemble'

    def __init__(self, worker, policy, m: float = 0.01, replan_every: int = 1) -> None:
        self.m = m
        self.replan_every = replan_every
        super().__init__(worker, policy)

    def _on_reset(self) -> None:
        self._buffer: list[tuple[int, np.ndarray]] = []   # (t_issue, chunk)
        self._last_issue: int | None = None

    def step(self, obs, t):
        arrival = self._poll(t)
        if arrival is not None:
            self._buffer.append((arrival.t_issue, arrival.chunk))
        self._buffer = [(t0, c) for t0, c in self._buffer if t - t0 < len(c)]

        due = self._last_issue is None or (t - self._last_issue) >= self.replan_every
        if due and self._issue(obs, t):
            self._last_issue = t

        covering = sorted(
            ((t0, c) for t0, c in self._buffer if 0 <= t - t0 < len(c)),
            key=lambda item: item[0])                      # oldest first
        if not covering:
            return None
        votes = np.stack([c[t - t0] for t0, c in covering])
        w = np.exp(-self.m * np.arange(len(covering)))
        return np.average(votes, axis=0, weights=w / w.sum())
