"""Cloud context, local policy: the cloud sends features, not actions; the robot plans every tick.

Every other executor receives a chunk of ACTIONS from the remote model and decides how to splice
it. Here the remote model (the cloud half of a split policy, scripts/two_stream.py) returns a
MEMORY: a token sequence encoding the scene as it was at the request. The robot holds the latest
memory and runs the local half on its own fresh observation:

    chunk = policy.local(obs, memory, age)        # age = ticks since that memory's request

and plays `replan` entries of that chunk before planning again. A late memory still drives
fresh actions: the local half sees the current images and proprio and knows how old its context
is, so staleness becomes an input instead of a splice (HiRT, CloudEdgeVLA). Requests go out
whenever the single slot is free, so the memory is as fresh as the link allows.

Before the first memory arrives the robot holds (returns None), like every other executor at
episode start. Query planner, paper 2 (docs/query-planner-plan.md).
"""
from __future__ import annotations

import numpy as np

from evh_controller.chunk_executor.base import ChunkExecutor


class CloudLocalExecutor(ChunkExecutor):
    name = 'cloud_local'

    def __init__(self, worker, policy, *, replan: int = 4, max_age: int | None = None) -> None:
        self.replan = replan
        self.max_age = max_age                # ages beyond the trained range are clamped
        self.ages: list[int] = []             # age of the memory behind each local plan
        super().__init__(worker, policy)

    def _on_reset(self) -> None:
        self._memory = None
        self._t_memory = 0
        self._chunk = None
        self._i = 0

    def step(self, obs, t):
        arrival = self._poll(t)
        if arrival is not None:
            self._memory, self._t_memory = arrival.chunk, arrival.t_issue
        if self._pending_t is None:
            self._issue(obs, t)
        if self._memory is None:
            return None
        if self._chunk is None or self._i >= min(self.replan, len(self._chunk)):
            age = t - self._t_memory
            if self.max_age is not None:
                age = min(age, self.max_age)
            self._chunk = np.asarray(self.policy.local(obs, self._memory, age))
            self._i = 0
            self.ages.append(age)
        a = self._chunk[self._i]
        self._i += 1
        return a
