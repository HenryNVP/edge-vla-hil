"""The ChunkExecutor contract: request/poll bookkeeping every strategy shares.

Execution is ASYNCHRONOUS and arrival-based (the model validated in
experiments/pusht_sanity/latency_chunking.py): inference runs on a background worker, the strategy
keeps streaming from the chunk in hand, and a finished chunk arrives some measured number of
control steps after it was requested. What a strategy overrides is the *splice* — when to request,
and how to graft the late chunk onto the actions already in flight. That splice is the entire
difference between the baselines.

Called once per control timestep by the controller:

    reset()                        # episode boundary; in-flight results discarded (epoch tag)
    step(obs, t) -> action | None  # the action to emit THIS tick; None = hold (no waypoint)

Two pieces of bookkeeping live here so no strategy has to re-derive them: the single-slot request
guard (`_issue`), and the epoch check in `_poll` that throws away a chunk computed against
observations from before a reset. The measured request->arrival delay recorded there is the honest
number RTC's forecast is built on — see rtc.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from evh_controller.inference_worker import InferenceWorker
from evh_controller.policy import ChunkPolicy


class ChunkExecutor(ABC):
    """Base: owns the request/poll bookkeeping; subclasses own the splice policy."""

    name: str = 'base'
    # True for strategies whose contribution IS the guided resample (RTC, BID). Paired with a
    # policy whose `guided_resampling` is False they still run, but on the soft-blend fallback —
    # which is the thing they are supposed to beat. make_executor warns; nothing else would.
    needs_guided_resampling: bool = False

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
