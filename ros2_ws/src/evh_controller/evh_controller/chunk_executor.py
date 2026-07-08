"""Chunk-execution strategies — the heart of the experiment and the Wedge-A/B seam.

A chunk policy emits H actions at once; *how* you execute that chunk while the next one is being
computed (under latency) is what the latency-robust-chunking literature is about. This module
makes the strategy pluggable so Wedge A reproduces the baselines and Wedge B drops in a new one
without touching the ROS2 node.

Execution is ASYNCHRONOUS and arrival-based (the model validated in
experiments/pusht_sanity/latency_chunking.py): inference runs on a background worker, the
strategy keeps streaming from the chunk in hand, and a finished chunk arrives some measured
number of control steps after it was requested. The strategy decides when to request and how to
splice the late chunk onto the in-flight actions — that splice is the entire difference between
the baselines.

Strategy contract (called once per control timestep by the controller):

    reset()                      # episode boundary; in-flight results are discarded (epoch tag)
    step(obs, t) -> action | None    # the action to emit THIS tick; None = hold (no waypoint)

Implemented:
  SynchronousExecutor      -- execute chunk fully, hold while the next one computes (prior-work
                              default: full success, throughput collapses with latency)
  NaiveAsyncExecutor       -- replan on a cadence, jump to new[0] on arrival (the chunk-boundary
                              discontinuity RTC exists to fix)
  TemporalEnsembleExecutor -- ACT: weighted average over time-aligned overlapping chunks
  BIDExecutor              -- closed-loop resampling w/ backward coherence      [stub: Step 4]
  RTCExecutor              -- freeze the executing overlap, inpaint the rest, continue
                              time-aligned; delay forecast = max of measured delays
  NetworkAwareExecutor     -- Wedge B: RTC with a quantile (jitter-aware) delay forecast

References: RTC (arXiv:2506.07339), BID (arXiv:2408.17355), ACT/TE (RSS 2023).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from evh_controller.inference_worker import InferenceWorker
from evh_controller.policy import ChunkPolicy


class ChunkExecutor(ABC):
    """Base: owns the request/poll bookkeeping; subclasses own the splice policy."""

    name: str = 'base'

    def __init__(self, worker: InferenceWorker, policy: ChunkPolicy) -> None:
        self.worker = worker
        self.policy = policy          # metadata only (chunk_size, action_dim); never called here
        self._epoch = 0
        self._pending_t: int | None = None
        self._arrival_metrics: tuple[float, int] | None = None
        self.reset()

    def reset(self) -> None:
        """Episode boundary: drop chunk state; tag so in-flight results get discarded."""
        self._epoch += 1
        self._pending_t = None
        self._on_reset()

    @abstractmethod
    def _on_reset(self) -> None:
        ...

    @abstractmethod
    def step(self, obs: dict, t: int) -> np.ndarray | None:
        """Action to emit at control tick `t`, or None to hold. obs: see policy.py contract."""
        ...

    def take_arrival_metrics(self) -> tuple[float, int] | None:
        """(compute_ms, delay_steps) of a chunk that arrived since last call, else None."""
        m, self._arrival_metrics = self._arrival_metrics, None
        return m

    # ----------------------------------------------------------- worker plumbing
    def _issue(self, obs: dict, t: int,
               prefix: np.ndarray | None = None, weights: np.ndarray | None = None) -> bool:
        if self._pending_t is not None:
            return False
        if self.worker.try_request(obs, t, self._epoch, prefix, weights):
            self._pending_t = t
            return True
        return False

    def _poll(self, t: int):
        """Deliver a finished chunk if one is ready; discards stale (pre-reset) results."""
        arrival = self.worker.poll()
        if arrival is None:
            return None
        if arrival.epoch != self._epoch:
            return None          # computed against pre-reset observations
        self._pending_t = None
        delay = max(0, t - arrival.t_issue)
        self._arrival_metrics = (arrival.compute_s * 1e3, delay)
        return arrival


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


class BIDExecutor(NaiveAsyncExecutor):
    """Bidirectional Decoding (stub, Step 4): sample N chunks, pick by backward coherence +
    forward contrast (needs a weak-policy reference). Until then behaves as naive-async with a
    per-chunk cadence; the ctor reserves the BID knobs so launch files don't change later.
    """
    name = 'bid'

    def __init__(self, worker, policy, num_samples: int = 32, keep: int = 3,
                 replan_every: int = 8) -> None:
        self.num_samples = num_samples
        self.keep = keep
        super().__init__(worker, policy, replan_every=replan_every)


class RTCExecutor(ChunkExecutor):
    """Real-Time Chunking: freeze the executing overlap, inpaint the rest, splice time-aligned.

    Continuous replanning: a new request is issued as soon as the worker is free, carrying
    `prefix` — the actions that will execute while inference runs (the next `d_hat` entries of
    the current chunk, `d_hat` = forecast delay) — and the paper's soft-mask weights W_i. On
    arrival, execution continues at the time-aligned index (the measured delay), so the frozen
    overlap is never replayed. Measured delays feed the forecast: RTC uses a conservative max;
    NetworkAwareExecutor (Wedge B) overrides just the forecast.

    Soft-mask weights (RTC paper):
        c_i = (H - s - i) / (H - s - d + 1)
        W_i = 1                          if i < d
              c_i (e^{c_i} - 1)/(e - 1)  if d <= i < H - s
              0                          if i >= H - s
    The guided sampling itself lives in policy.predict_inpaint (soft blending today; true guided
    denoising is Step 4).
    """
    name = 'rtc'

    def __init__(self, worker, policy, exec_horizon_min: int = 1,
                 delay_buffer: int = 20) -> None:
        self.s_min = exec_horizon_min
        self.delay_buffer = delay_buffer
        # delay history is a property of the SYSTEM (network + GPU), not of an episode:
        # it survives reset() on purpose, so the forecast stays warm across episodes.
        self._delays: list[int] = []
        super().__init__(worker, policy)

    def _on_reset(self) -> None:
        self._chunk: np.ndarray | None = None
        self._i = 0

    def forecast_delay(self) -> int:
        """RTC: conservative max over recent delays. Wedge B overrides this."""
        if not self._delays:
            return 1
        return max(self._delays[-self.delay_buffer:])

    @staticmethod
    def freeze_weights(H: int, s: int, d: int) -> np.ndarray:
        w = np.zeros(H, dtype=np.float64)
        denom = max(H - s - d + 1, 1)
        for i in range(H):
            if i < d:
                w[i] = 1.0
            elif i < H - s:
                c = (H - s - i) / denom
                w[i] = c * (np.exp(c) - 1.0) / (np.e - 1.0)
            else:
                w[i] = 0.0
        return w

    def step(self, obs, t):
        arrival = self._poll(t)
        if arrival is not None:
            d_actual = max(0, t - arrival.t_issue)
            self._delays.append(d_actual)
            if d_actual < len(arrival.chunk):
                # index d_actual is the action for THIS tick; [0, d_actual) already executed
                # as the frozen prefix of the outgoing chunk
                self._chunk, self._i = arrival.chunk, d_actual
            elif self._chunk is None or self._i >= len(self._chunk):
                # degenerate regime d >= H (inference slower than a whole chunk): the arrival
                # covers only the past, but nothing is executing — falling back to executing
                # the stale chunk from 0 (synchronous semantics) beats freezing the robot.
                # This is exactly the regime the benchmark should show RTC degrading in.
                self._chunk, self._i = arrival.chunk, 0
            # else: keep executing the current chunk; the fresher request already carries it

        # issue BEFORE emitting: self._i is the action about to execute at tick t, so the
        # overlap that will run during inference is exactly [self._i, self._i + d_hat)
        if self._pending_t is None:
            if self._chunk is None:
                self._issue(obs, t)                    # bootstrap: nothing to freeze
            else:
                H = self.policy.chunk_size
                d_hat = max(self.forecast_delay(), 1)
                # freeze only what can actually execute during inference, and never the whole
                # chunk — at least s_min actions must stay free for the policy to regenerate
                # (matters when d_hat >= H, i.e. inference slower than a full chunk)
                d_frz = min(d_hat, max(H - self.s_min, 0), max(len(self._chunk) - self._i, 0))
                s = max(self.s_min, min(d_hat, H - d_frz))
                prefix = self._chunk[self._i:self._i + d_frz]
                self._issue(obs, t, prefix=prefix, weights=self.freeze_weights(H, s, d_frz))

        if self._chunk is not None and self._i < len(self._chunk):
            a = self._chunk[self._i]
            self._i += 1
            return a
        return None


class NetworkAwareExecutor(RTCExecutor):
    """Wedge B: RTC whose delay forecast uses the measured delay distribution.

    RTC's max-over-buffer assumes a reliable channel; under heavy-tailed jitter/loss a quantile
    (later: loss-aware) estimate should dominate. Only the forecast differs — everything else is
    inherited, which is exactly the point of the seam.
    """
    name = 'network_aware'

    def __init__(self, worker, policy, quantile: float = 0.95,
                 exec_horizon_min: int = 1, delay_buffer: int = 50) -> None:
        super().__init__(worker, policy, exec_horizon_min=exec_horizon_min,
                         delay_buffer=delay_buffer)
        self.quantile = quantile

    def forecast_delay(self) -> int:
        if not self._delays:
            return 1
        recent = self._delays[-self.delay_buffer:]
        return int(np.ceil(np.quantile(recent, self.quantile)))   # TODO: loss-aware term


_REGISTRY = {
    cls.name: cls for cls in (
        SynchronousExecutor, NaiveAsyncExecutor, TemporalEnsembleExecutor,
        BIDExecutor, RTCExecutor, NetworkAwareExecutor,
    )
}


def make_executor(strategy: str, worker: InferenceWorker, policy: ChunkPolicy) -> ChunkExecutor:
    key = strategy.lower()
    if key not in _REGISTRY:
        raise ValueError(f'unknown strategy {strategy!r}; options: {sorted(_REGISTRY)}')
    return _REGISTRY[key](worker, policy)
