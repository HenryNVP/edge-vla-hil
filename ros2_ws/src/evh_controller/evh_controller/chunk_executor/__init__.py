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


__all__ = [
    'BIDExecutor',
    'ChunkExecutor',
    'NaiveAsyncExecutor',
    'NetworkAwareExecutor',
    'RTCExecutor',
    'SynchronousExecutor',
    'TemporalEnsembleExecutor',
    'make_executor',
]
