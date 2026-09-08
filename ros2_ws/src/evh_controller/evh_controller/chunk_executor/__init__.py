"""Chunk-execution strategies — the heart of the experiment and the Wedge-A/B seam.

A chunk policy emits H actions at once; *how* you execute that chunk while the next one is being
computed (under latency) is what the latency-robust-chunking literature is about. This package
makes the strategy pluggable so Wedge A reproduces the baselines and Wedge B drops in a new one
without touching the ROS2 node.

    base.py       the ChunkExecutor contract: request/poll, epoch guard, delay measurement
    baselines.py  synchronous | naive_async | temporal_ensemble  (prior work, kept comparable)
    rtc.py        rtc | network_aware                            (RTC + the Wedge-B forecast)
    bid.py        bid                                            [stub: Step 4]

To add a strategy: subclass ChunkExecutor, set a `name`, and add it to _REGISTRY below. The ROS
node never changes — it only ever calls make_executor().

Import path is unchanged from when this was a single module:

    from evh_controller.chunk_executor import make_executor, RTCExecutor
"""
from __future__ import annotations

import logging

from evh_controller.chunk_executor.base import ChunkExecutor
from evh_controller.chunk_executor.baselines import (
    NaiveAsyncExecutor,
    SynchronousExecutor,
    TemporalEnsembleExecutor,
)
from evh_controller.chunk_executor.bid import BIDExecutor
from evh_controller.chunk_executor.rtc import NetworkAwareExecutor, RTCExecutor
from evh_controller.inference_worker import InferenceWorker
from evh_controller.policy import ChunkPolicy

logger = logging.getLogger(__name__)

_REGISTRY = {
    cls.name: cls for cls in (
        SynchronousExecutor, NaiveAsyncExecutor, TemporalEnsembleExecutor,
        BIDExecutor, RTCExecutor, NetworkAwareExecutor,
    )
}


def guidance_warning(strategy_cls, policy: ChunkPolicy) -> str | None:
    """Warn when a strategy's whole mechanism is unavailable on this backend, else None.

    RTC and BID are defined by the guided resample. On a deterministic backend (ACT, ONNX-ACT,
    the PyTorch fallback) `predict_inpaint` degrades to a post-hoc soft blend, and the run then
    produces a full, plausible-looking CSV row under a label claiming RTC. Nothing else in the
    graph notices; this is the one place both capabilities meet.

    Split out of make_executor so the wording is testable without constructing a worker.
    """
    if not getattr(strategy_cls, 'needs_guided_resampling', False):
        return None
    if getattr(policy, 'guided_resampling', False):
        return None
    return (f'strategy {strategy_cls.name!r} needs guided resampling, but the '
            f'{type(policy).__name__} backend does not implement it — predict_inpaint will fall '
            f'back to a post-hoc SOFT BLEND. The run will complete and the numbers will look '
            f'reasonable, but they are not {strategy_cls.name}. Use a diffusion/flow backend '
            f'(dp, dp_onnx) for this strategy, or report the row as soft-blend.')


def make_executor(strategy: str, worker: InferenceWorker, policy: ChunkPolicy) -> ChunkExecutor:
    key = strategy.lower()
    if key not in _REGISTRY:
        raise ValueError(f'unknown strategy {strategy!r}; options: {sorted(_REGISTRY)}')
    cls = _REGISTRY[key]
    problem = guidance_warning(cls, policy)
    if problem is not None:
        logger.warning(problem)
    return cls(worker, policy)


__all__ = [
    'BIDExecutor',
    'ChunkExecutor',
    'NaiveAsyncExecutor',
    'NetworkAwareExecutor',
    'RTCExecutor',
    'SynchronousExecutor',
    'TemporalEnsembleExecutor',
    'guidance_warning',
    'make_executor',
]
