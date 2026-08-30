"""Real-Time Chunking and the Wedge-B extension built on it.

RTC replans continuously: a request goes out as soon as the worker is free, carrying `prefix` —
the actions that will actually execute while inference runs — plus the paper's soft-mask weights.
On arrival execution continues at the TIME-ALIGNED index (the measured delay), so the frozen
overlap is never replayed. That is the whole difference from naive-async.

NetworkAwareExecutor (Wedge B) lives here rather than in its own file because it overrides exactly
one method — the delay forecast — and the value of the comparison is seeing that next to the RTC
forecast it replaces.

Reference: RTC (arXiv:2506.07339).
"""
from __future__ import annotations

import numpy as np

from evh_controller.chunk_executor.base import ChunkExecutor

# Shared by RTC and its Wedge-B subclass, and shared ON PURPOSE. These two differ only in the
# STATISTIC they take over the delay history (max vs quantile); if they also differed in how much
# history they look at, an observed effect could be the longer memory rather than the estimator,
# and the comparison would prove nothing. They were 20 and 50 — a heavy-tail spike 30 samples old
# sat inside one window and outside the other, which is how the quantile came out ABOVE the max.
#
# 50 and not 20 because delays are small INTEGERS (control steps), so the top few samples of a
# short window are usually the same integer and ceil(p95) lands exactly on the max. Measured over
# tight delay distributions, ceil(p95) == max for 88% of 20-sample windows, 27% at 50, 13% at 100.
# A shorter window is therefore mostly RTC wearing a quantile; a longer one adapts more slowly to
# a changing link. 50 is the compromise — raise it if the two strategies keep tying.
DELAY_BUFFER = 50


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
                 delay_buffer: int = DELAY_BUFFER) -> None:
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
                # guidance is the whole REMAINING PLAN, not just the frozen part: freeze_weights
                # assigns a decaying weight across [d_frz, H - s) too, and those weights need
                # something to pull toward. Sending only the frozen slice left the soft region
                # with no target, so the new chunk was continuous with the executed prefix and
                # then free to jump — the discontinuity RTC exists to remove.
                guide = self._chunk[self._i:self._i + H]
                self._issue(obs, t, prefix=guide, weights=self.freeze_weights(H, s, d_frz))

        if self._chunk is not None and self._i < len(self._chunk):
            a = self._chunk[self._i]
            self._i += 1
            return a
        return None


class NetworkAwareExecutor(RTCExecutor):
    """Wedge B: RTC whose delay forecast uses the measured delay distribution.

    RTC's max-over-buffer assumes a reliable channel; under heavy-tailed jitter/loss a quantile
    (later: loss-aware) estimate should dominate. Only the forecast differs — everything else,
    including the buffer length (see DELAY_BUFFER), is inherited, so any measured difference is
    attributable to the estimator and nothing else.

    Note what this needs from the EXPERIMENT to be testable at all: with a light-tailed delay
    distribution `ceil(p95) == max` for the integer step counts these forecasts produce, and the
    two strategies compute byte-identical freeze horizons. Sweep `jitter_model:=lognormal` or a
    non-zero `drop_prob`, or this reduces to running RTC twice.
    """
    name = 'network_aware'

    def __init__(self, worker, policy, quantile: float = 0.95,
                 exec_horizon_min: int = 1, delay_buffer: int = DELAY_BUFFER) -> None:
        super().__init__(worker, policy, exec_horizon_min=exec_horizon_min,
                         delay_buffer=delay_buffer)
        self.quantile = quantile

    def forecast_delay(self) -> int:
        if not self._delays:
            return 1
        recent = self._delays[-self.delay_buffer:]
        return int(np.ceil(np.quantile(recent, self.quantile)))   # TODO: loss-aware term
