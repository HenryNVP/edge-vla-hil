"""Chunk-execution strategies — the heart of the experiment and the Wedge-A/B seam.

A chunk policy emits H actions at once; *how* you execute that chunk while the next one is being
computed (under latency) is what the latency-robust-chunking literature is about. This module makes
the strategy pluggable so Wedge A reproduces the baselines and Wedge B drops in a new one without
touching the ROS2 node.

Strategy contract (called once per control timestep by the controller):

    reset()
    select_action(obs, policy, t) -> action_vector   # the action to emit THIS timestep

The strategy owns its chunk buffer and decides when to (re)invoke the policy. In the real system
policy inference is slow and runs concurrently with execution; the stubs here implement the
synchronous semantics and mark the async-generation hook with TODOs. Implemented:

  SynchronousExecutor   -- predict a chunk, emit it, pause to predict the next (prior-work default)
  NaiveAsyncExecutor    -- replan every `replan_every` steps, switch chunks immediately
  TemporalEnsembleExecutor -- weighted average over overlapping chunks (ACT)        [partial]
  BIDExecutor           -- closed-loop resampling w/ backward coherence              [stub]
  RTCExecutor           -- freeze first d, soft-masked guided inpaint of the rest    [stub+masking]
  NetworkAwareExecutor  -- Wedge B: RTC with distribution-aware delay forecast       [reserved]

References: RTC (arXiv:2506.07339), BID (arXiv:2408.17355), ACT/TE (RSS 2023).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from evh_controller.policy import ChunkPolicy


class ChunkExecutor(ABC):
    name: str = 'base'

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def select_action(self, obs: tuple, policy: ChunkPolicy, t: int) -> np.ndarray:
        """Return the action to emit at control timestep `t`. obs = (image, state)."""
        ...


class SynchronousExecutor(ChunkExecutor):
    """Predict a full chunk, execute it, then pause to predict the next. Prior-work default."""
    name = 'synchronous'

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._chunk: np.ndarray | None = None
        self._i = 0

    def select_action(self, obs, policy, t) -> np.ndarray:
        if self._chunk is None or self._i >= self._chunk.shape[0]:
            self._chunk = policy.predict(*obs)   # TODO: model the inference stall (d timesteps)
            self._i = 0
        a = self._chunk[self._i]
        self._i += 1
        return a


class NaiveAsyncExecutor(ChunkExecutor):
    """Replan every `replan_every` steps; switch to the new chunk immediately (jerky at seams)."""
    name = 'naive_async'

    def __init__(self, replan_every: int = 8) -> None:
        self.replan_every = replan_every
        self.reset()

    def reset(self) -> None:
        self._chunk: np.ndarray | None = None
        self._i = 0
        self._since = 1 << 30

    def select_action(self, obs, policy, t) -> np.ndarray:
        if self._chunk is None or self._since >= self.replan_every:
            self._chunk = policy.predict(*obs)
            self._i = 0
            self._since = 0
        i = min(self._i, self._chunk.shape[0] - 1)
        a = self._chunk[i]
        self._i += 1
        self._since += 1
        return a


class TemporalEnsembleExecutor(ChunkExecutor):
    """ACT temporal ensembling: exponentially-weighted average over overlapping chunk predictions.

    Partial: maintains the overlap buffer and the weighting; wire `m` (decay) and the per-timestep
    average to match ACT. Smoothness only — no latency model (the weak baseline RTC beats).
    """
    name = 'temporal_ensemble'

    def __init__(self, m: float = 0.01, replan_every: int = 1) -> None:
        self.m = m
        self.replan_every = replan_every
        self.reset()

    def reset(self) -> None:
        self._buffer: list[tuple[int, np.ndarray]] = []   # (start_t, chunk)
        self._since = 1 << 30

    def select_action(self, obs, policy, t) -> np.ndarray:
        if self._since >= self.replan_every:
            self._buffer.append((t, policy.predict(*obs)))
            self._since = 0
        self._since += 1

        votes, weights = [], []
        for start, chunk in self._buffer:
            k = t - start
            if 0 <= k < chunk.shape[0]:
                votes.append(chunk[k])
                weights.append(np.exp(-self.m * k))   # TODO: confirm sign/convention vs ACT
        if not votes:
            return self._buffer[-1][1][0]
        w = np.asarray(weights)
        return np.average(np.stack(votes), axis=0, weights=w / w.sum())


class BIDExecutor(ChunkExecutor):
    """Bidirectional Decoding (stub): sample N chunks, pick by backward coherence + forward contrast.

    Needs a strong and a weak policy for the forward-contrast term; reserve those refs here.
    """
    name = 'bid'

    def __init__(self, num_samples: int = 32, keep: int = 3) -> None:
        self.num_samples = num_samples
        self.keep = keep
        self.reset()

    def reset(self) -> None:
        self._prev: np.ndarray | None = None
        self._chunk: np.ndarray | None = None
        self._i = 0

    def select_action(self, obs, policy, t) -> np.ndarray:
        # TODO: draw N candidate chunks; score backward coherence vs self._prev and forward
        # contrast vs a weak policy; choose best. Stub: single sample, no resampling.
        if self._chunk is None or self._i >= self._chunk.shape[0]:
            self._prev = self._chunk
            self._chunk = policy.predict(*obs)
            self._i = 0
        a = self._chunk[self._i]
        self._i += 1
        return a


class RTCExecutor(ChunkExecutor):
    """Real-Time Chunking (stub + real masking weights).

    RTC freezes the first `d` actions (those guaranteed to execute during the inference stall) and
    inpaints the rest via soft-masked guided denoising, keeping cross-chunk consistency. `d` is
    forecast as the (conservative) max of recent observed delays.

    The soft-mask weights follow the paper:
        c_i = (H - s - i) / (H - s - d + 1)
        W_i = 1                          if i < d
              c_i (e^{c_i} - 1)/(e - 1)  if d <= i < H - s
              0                          if i >= H - s
    `freeze_weights` returns W; the guided sampling itself lives in policy.predict_inpaint (TODO).
    """
    name = 'rtc'

    def __init__(self, exec_horizon_min: int = 1) -> None:
        self.s_min = exec_horizon_min
        self.reset()

    def reset(self) -> None:
        self._chunk: np.ndarray | None = None
        self._i = 0
        self._delays: list[int] = []      # observed inference delays (timesteps)

    def forecast_delay(self) -> int:
        """RTC: conservative max over recent delays. Wedge B overrides this."""
        return max(self._delays) if self._delays else 0

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

    def select_action(self, obs, policy, t) -> np.ndarray:
        H = policy.chunk_size
        d = self.forecast_delay()
        s = max(d, self.s_min)
        if self._chunk is None or self._i >= H - s:
            prefix = (self._chunk[self._i:self._i + d]
                      if self._chunk is not None and d > 0
                      else np.zeros((0, policy.action_dim), np.float32))
            weights = self.freeze_weights(H, s, d)[:len(prefix)]
            # TODO: record the real measured delay into self._delays (async generation hook)
            self._chunk = policy.predict_inpaint(obs[0], obs[1], prefix, weights)
            self._i = 0
        a = self._chunk[self._i]
        self._i += 1
        return a


class NetworkAwareExecutor(RTCExecutor):
    """Wedge B (reserved): RTC whose delay forecast uses a measured RTT/jitter distribution.

    RTC's max-over-buffer assumes a reliable channel; under heavy-tailed jitter/loss a quantile or
    loss-aware estimate should dominate. Override forecast_delay with the network model and feed it
    RTT/jitter samples measured at the DDS boundary.
    """
    name = 'network_aware'

    def __init__(self, quantile: float = 0.95, exec_horizon_min: int = 1) -> None:
        super().__init__(exec_horizon_min=exec_horizon_min)
        self.quantile = quantile

    def forecast_delay(self) -> int:
        if not self._delays:
            return 0
        return int(np.ceil(np.quantile(self._delays, self.quantile)))   # TODO: loss-aware term


_REGISTRY = {
    cls.name: cls for cls in (
        SynchronousExecutor, NaiveAsyncExecutor, TemporalEnsembleExecutor,
        BIDExecutor, RTCExecutor, NetworkAwareExecutor,
    )
}


def make_executor(strategy: str) -> ChunkExecutor:
    key = strategy.lower()
    if key not in _REGISTRY:
        raise ValueError(f'unknown strategy {strategy!r}; options: {sorted(_REGISTRY)}')
    return _REGISTRY[key]()
