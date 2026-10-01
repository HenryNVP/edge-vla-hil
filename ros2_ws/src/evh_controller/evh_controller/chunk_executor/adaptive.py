"""Coverage-aware execution: overlap while the delay allows it, wait when it does not, never skip.

The two executor families in this package fail in opposite places (paper 1, sections 3 and 5):

  time-aligned (RTC, naive, TE)  a chunk requested at t0 covers ticks t0 .. t0+H-1 and is joined at
                                 entry d. With one request in flight it covers only (H - d)/d of
                                 the ticks until the next arrival: full up to d = H/2, none at d = H.
                                 Past H/2 the robot starves, and the entries meant for the starved
                                 ticks are skipped when the next chunk lands.
  waiting (synchronous)          never skips, but pays d idle ticks on every request: duty cycle
                                 s / (s + d), whatever the delay's distribution.

Real wireless delay is mostly short with rare stalls longer than a chunk, so each fixed strategy
fails on part of it. This executor does three things:

  1. ESTIMATE, per request, the action rate each mode would sustain, from the recent measured
     delays D (the request->arrival ticks every executor already measures):

         time-aligned   E_D[ min(1, max(0, H - d) / d) ]        (coverage; no skipping, see 3)
         waiting        E_D[ s_w / (s_w + d) ]                  (duty cycle)

     and run the mode with the higher rate, with a small hysteresis margin so a noisy estimate
     does not flip modes every request. With s_w = H the two rates cross at d = H(sqrt5-1)/2
     = 0.618 H; a shorter waiting chunk (a policy that tolerates only s_w open-loop steps) moves
     the crossover. The whole distribution enters, so a heavy tail lowers the time-aligned rate
     before the median moves. `rule='half'` replaces this with the fixed d_hat <= H/2 test, for
     the ablation.
  2. TIME-ALIGNED mode is RTC: requests every max(d_hat, s_min) ticks carrying the remaining plan
     as a frozen, soft-masked prefix, so the new chunk continues the old one.
  3. STARVATION-AWARE ALIGNMENT: when the chunk in hand runs out before the next arrives, hold;
     on arrival, join the new chunk at the number of actions actually PLAYED since the request,
     not the ticks elapsed. A stall then costs time instead of skipped actions. With
     `reindex=False` this reduces to RTC's tick-based join (the switching-only ablation).

WAITING mode is synchronous: at the end of a chunk (or after s_w actions), request with the current
observation, hold until it arrives, play it from entry 0.

It needs only the measured delay: no model change, no retraining, and it runs on any backend that
provides `predict_inpaint` (and degrades to unguided splicing on one that does not, like RTC).
"""
from __future__ import annotations

import numpy as np

from evh_controller.chunk_executor.base import ChunkExecutor
from evh_controller.chunk_executor.rtc import DELAY_BUFFER, RTCExecutor

ALIGNED, WAITING = 'aligned', 'waiting'


def expected_rates(delays, H: int, s_wait: int) -> tuple[float, float]:
    """(time-aligned coverage, waiting duty cycle) expected under the delay samples."""
    d = np.maximum(np.asarray(delays, dtype=float), 1.0)
    aligned = np.clip(np.maximum(H - d, 0.0) / d, 0.0, 1.0).mean()
    waiting = (s_wait / (s_wait + d)).mean()
    return float(aligned), float(waiting)


class CoverageAwareExecutor(ChunkExecutor):
    """Time-aligned while its expected coverage beats waiting's duty cycle; never skips."""
    name = 'coverage_aware'
    needs_guided_resampling = True

    def __init__(self, worker, policy, *, rule: str = 'rate', reindex: bool = True,
                 guided: bool = True, s_wait: int | None = None, exec_horizon_min: int = 1,
                 quantile: float = 0.9, margin: float = 0.05,
                 delay_buffer: int = DELAY_BUFFER) -> None:
        if rule not in ('rate', 'half', 'aligned', 'waiting'):
            raise ValueError(f'unknown rule {rule!r}')
        self.rule = rule
        self.reindex = reindex
        self.guided = guided
        self.s_wait = s_wait
        self.s_min = exec_horizon_min
        self.quantile = quantile
        self.margin = margin
        self.delay_buffer = delay_buffer
        # like RTC's forecast, the delay history is a property of the link, not of an episode
        self._delays: list[int] = []
        self.mode = ALIGNED
        self.mode_log: list[str] = []          # one entry per request, for analysis and tests
        super().__init__(worker, policy)

    # ------------------------------------------------------------------ estimation
    def forecast_delay(self) -> int:
        if not self._delays:
            return 1
        return int(np.ceil(np.quantile(self._delays[-self.delay_buffer:], self.quantile)))

    def _horizon(self) -> int:
        return int(self.policy.chunk_size)

    def _s_wait(self) -> int:
        return min(self.s_wait or self._horizon(), self._horizon())

    def choose_mode(self) -> str:
        if self.rule in (ALIGNED, WAITING):
            return self.rule
        H = self._horizon()
        if self.rule == 'half':
            return ALIGNED if self.forecast_delay() <= H / 2 else WAITING
        if not self._delays:
            return self.mode                   # nothing measured yet: keep the current mode
        aligned, waiting = expected_rates(self._delays[-self.delay_buffer:], H, self._s_wait())
        if self.mode == ALIGNED:
            return WAITING if waiting > aligned + self.margin else ALIGNED
        return ALIGNED if aligned > waiting + self.margin else WAITING

    # ------------------------------------------------------------------ execution
    def _on_reset(self) -> None:
        self._chunk: np.ndarray | None = None
        self._i = 0
        self._played = 0                       # actions emitted this episode
        self._played_at_issue = 0
        self._last_issue: int | None = None
        self._n_from_chunk = 0                 # actions played from the chunk in hand

    def _request(self, obs, t) -> None:
        self.mode = self.choose_mode()
        H = self._horizon()
        if self.mode == WAITING:
            # waiting means the robot holds while the plan is computed, so the observation this
            # request carries is still true when the chunk lands; switching into waiting in the
            # middle of a chunk therefore stops the robot here, not at the chunk's end
            self._chunk = None
        if (self.guided and self.mode == ALIGNED and self._chunk is not None
                and self._i < len(self._chunk)):
            d_hat = max(self.forecast_delay(), 1)
            d_frz = min(d_hat, max(H - self.s_min, 0), len(self._chunk) - self._i)
            s = max(self.s_min, min(d_hat, H - d_frz))
            ok = self._issue(obs, t, prefix=self._chunk[self._i:self._i + H],
                             weights=RTCExecutor.freeze_weights(H, s, d_frz))
        else:
            ok = self._issue(obs, t)
        if ok:
            self.mode_log.append(self.mode)
            self._last_issue = t
            self._played_at_issue = self._played

    def step(self, obs, t):
        arrival = self._poll(t)
        if arrival is not None:
            elapsed = max(0, t - arrival.t_issue)
            self._delays.append(elapsed)
            chunk = arrival.chunk
            if self.mode == WAITING:
                start = 0                      # the robot held: the plan's first entry is now
            elif self.reindex:
                start = self._played - self._played_at_issue   # actions actually played
            else:
                start = elapsed                # RTC's join: ticks elapsed, starved or not
            if start < len(chunk):
                self._chunk, self._i = chunk, start
            elif self._chunk is None or self._i >= len(self._chunk):
                self._chunk, self._i = chunk, 0   # wholly in the past: replay rather than freeze
            self._n_from_chunk = 0

        exhausted = self._chunk is None or self._i >= len(self._chunk)
        if self._pending_t is None:
            if self._chunk is None:
                self._request(obs, t)          # bootstrap
            elif self.mode == WAITING:
                if exhausted or self._n_from_chunk >= self._s_wait():
                    self._chunk = None         # stop and wait for a fresh plan
                    self._request(obs, t)
            else:
                d_hat = max(self.forecast_delay(), 1)
                due = t - self._last_issue >= max(d_hat, self.s_min)
                if due or exhausted:
                    self._request(obs, t)

        if self._chunk is not None and self._i < len(self._chunk):
            a = self._chunk[self._i]
            self._i += 1
            self._played += 1
            self._n_from_chunk += 1
            return a
        return None


class NoSkipRTCExecutor(CoverageAwareExecutor):
    """Ablation: always time-aligned, but never skips (starvation-aware alignment alone)."""
    name = 'rtc_noskip'

    def __init__(self, worker, policy, **kw) -> None:
        kw.setdefault('rule', 'aligned')
        super().__init__(worker, policy, **kw)


class SwitchOnlyExecutor(CoverageAwareExecutor):
    """Ablation: the rate-based switch, with RTC's tick-based join (skips when starved)."""
    name = 'switch_only'

    def __init__(self, worker, policy, **kw) -> None:
        kw.setdefault('reindex', False)
        super().__init__(worker, policy, **kw)


class NoSkipAsyncExecutor(CoverageAwareExecutor):
    """Baseline: the deployed default (LeRobot's async client, Armory's robot model).

    Always overlapping, unguided, and joins each chunk at the actions actually played, pausing
    when starved: it never skips, and it never waits on purpose. Prior art, so the comparison any
    claim about skipping or switching has to beat.
    """
    name = 'noskip_async'
    needs_guided_resampling = False

    def __init__(self, worker, policy, **kw) -> None:
        kw.setdefault('rule', 'aligned')
        kw.setdefault('guided', False)
        super().__init__(worker, policy, **kw)
