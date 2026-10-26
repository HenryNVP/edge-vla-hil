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

A policy with a `reset()` (a local half that keeps state between plans, such as the BEV planner's
previous map) is reset with each episode. `refresh=False` (strategy `cloud_local_once`) keeps the
episode's first memory and never asks again: a link outage right after the first map, the stress
test of how long the robot can run on one stale map. `drop_after=N` plans with no memory
(`policy.local(obs, None, age)`) once the memory is more than N ticks old: a map past the
trained age range misleads (outputs/stress/: one map then outage did worse than no map), so a
policy trained with cloud-map dropout falls back to its robot-only behaviour instead. Before the first memory arrives the robot holds (returns None), like every other executor at
episode start. Query planner, paper 2 (docs/query-planner-plan.md).
"""
from __future__ import annotations

import numpy as np

from evh_controller.chunk_executor.base import ChunkExecutor


class CloudLocalExecutor(ChunkExecutor):
    name = 'cloud_local'

    def __init__(self, worker, policy, *, replan: int = 4, max_age: int | None = None,
                 refresh: bool = True, drop_after: int | None = None) -> None:
        self.replan, self.refresh, self.drop_after = replan, refresh, drop_after
        self.max_age = max_age                # ages beyond the trained range are clamped
        self.ages: list[int] = []             # age of the memory behind each local plan
        super().__init__(worker, policy)

    def _on_reset(self) -> None:
        self._memory = None
        self._t_memory = 0
        self._chunk = None
        self._i = 0
        if hasattr(self.policy, 'reset'):      # a local half with its own state (bev_planner)
            self.policy.reset()

    def step(self, obs, t):
        arrival = self._poll(t)
        if arrival is not None:
            self._memory, self._t_memory = arrival.chunk, arrival.t_issue
        if self._pending_t is None and (self.refresh or self._memory is None):
            self._issue(obs, t)
        if self._memory is None:
            return None
        if self._chunk is None or self._i >= min(self.replan, len(self._chunk)):
            age = t - self._t_memory
            memory = self._memory
            if self.drop_after is not None and age > self.drop_after:
                memory = None
            if self.max_age is not None:
                age = min(age, self.max_age)
            self._chunk = np.asarray(self.policy.local(obs, memory, age))
            self._i = 0
            self.ages.append(age)
        a = self._chunk[self._i]
        self._i += 1
        return a
