"""Bidirectional Decoding — its own file because Step 4 replaces the whole body.

Today this is a stub: it inherits naive-async behaviour with a per-chunk cadence, and the ctor
already reserves the BID knobs so launch files and sweeps do not have to change when the real
implementation lands. Real BID samples N chunks per step and picks by backward coherence against
the previous plan plus forward contrast against a weak-policy reference — neither of which the
current single-slot worker can express, so it is a genuine rewrite rather than a fill-in.

Reference: BID (arXiv:2408.17355).
"""
from __future__ import annotations

from evh_controller.chunk_executor.baselines import NaiveAsyncExecutor


class BIDExecutor(NaiveAsyncExecutor):
    """Bidirectional Decoding (stub, Step 4): sample N chunks, pick by backward coherence +
    forward contrast (needs a weak-policy reference). Until then behaves as naive-async with a
    per-chunk cadence; the ctor reserves the BID knobs so launch files don't change later.
    """
    name = 'bid'
    needs_guided_resampling = True

    def __init__(self, worker, policy, num_samples: int = 32, keep: int = 3,
                 replan_every: int = 8) -> None:
        self.num_samples = num_samples
        self.keep = keep
        super().__init__(worker, policy, replan_every=replan_every)
