"""Robot-side execution: an InferenceWorker whose GPU is on the other side of the network.

Executor placement is one of the paper's factors. With the executor on the POLICY side (the
original design, `controller_node` + `InferenceWorker`) the chunk is consumed where it is computed
and streamed to the robot one waypoint per control tick, so every action crosses the link and
every lost packet costs a step. With the executor on the ROBOT side, which is how real offloading
stacks work, the whole chunk crosses once and is executed from a local buffer: a lost packet costs
nothing until that buffer runs out.

`RemoteWorker` is what makes the second placement a drop-in. It has InferenceWorker's interface
(`busy`, `try_request`, `poll`, `shutdown`), so the ChunkExecutor strategies run unchanged on the
robot side; only where the inference happens moves. A request goes out through a `send` callback
(the node publishes it on /policy/request) and the policy server's reply comes back through
`deliver` (the node's /cmd/chunk subscription). Two consequences follow from the strategies being
untouched:

  * The delay a strategy measures (request -> arrival, in robot ticks) is now the whole round
    trip: uplink + inference + downlink. So RTC's forecast sees the network without any extra
    feedback channel; nothing had to be added to the strategies for it.
  * RTC's frozen prefix is computed where the actions execute, from the robot's own index, which
    is exactly what the prefix has to describe. No executed-index report is needed either.

What the local worker never had to handle is LOSS: a request or a reply can vanish, and a
single-slot worker waiting for a reply that will never come would stall the executor for the rest
of the episode. So a pending request times out, and `poll` then returns an Arrival flagged
`lost`, which the executor base treats as "nothing in flight any more". The timeout adapts to the
link: `timeout_factor` times the slowest recent round trip, floored at `min_timeout_s`, with a
generous `first_timeout_s` before any reply has been seen (the first chunk includes model warm-up).
A reply that turns up after its request timed out is recognised by `req_id` and dropped, never
spliced in late.

The timeout is itself a design choice with an effect under bursty loss (how long a client waits
before re-asking), so its parameters are reported with the results. Pure Python; `clock` is
injectable so the tests can drive time.
"""
from __future__ import annotations

import collections
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from evh_controller.chunk_codec import Chunk, Request
from evh_controller.inference_worker import Arrival

logger = logging.getLogger(__name__)


@dataclass
class PolicyInfo:
    """The policy metadata the executors read, announced by the policy server.

    The robot side never loads the model, but the strategies need its shape (RTC sizes its freeze
    from `chunk_size`) and make_executor checks `guided_resampling`.
    """
    chunk_size: int
    action_dim: int
    guided_resampling: bool
    absolute_actions: bool


class RemoteWorker:
    """Single-slot inference over a link that can delay, drop and reorder."""

    def __init__(self, send: Callable[[Request], None], action_dim: int,
                 timeout_factor: float = 2.0, min_timeout_s: float = 0.25,
                 first_timeout_s: float = 10.0, history: int = 20,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._send = send
        self.action_dim = action_dim
        self.timeout_factor = timeout_factor
        self.min_timeout_s = min_timeout_s
        self.first_timeout_s = first_timeout_s
        self._clock = clock
        self._rtts: collections.deque[float] = collections.deque(maxlen=history)
        self._next_id = 0
        self._pending: tuple[int, int, int, float] | None = None   # (req_id, t_issue, epoch, sent)
        self._inbox: Chunk | None = None
        self.lost = 0
        self.stale = 0

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def timeout_s(self) -> float:
        if not self._rtts:
            return self.first_timeout_s
        return max(self.min_timeout_s, self.timeout_factor * max(self._rtts))

    def try_request(self, obs: dict, t_issue: int, epoch: int,
                    prefix: np.ndarray | None = None,
                    weights: np.ndarray | None = None) -> bool:
        """Send a request; False if one is already pending. `obs` is unused: the policy server
        infers from its own (relayed) observation stream, as a server in the field would from
        what the robot last uploaded."""
        if self._pending is not None:
            return False
        req_id = self._next_id
        self._next_id += 1
        self._pending = (req_id, t_issue, epoch, self._clock())
        self._inbox = None
        self._send(Request(req_id, t_issue, epoch, prefix, weights))
        return True

    def deliver(self, chunk: Chunk) -> None:
        """A reply arrived from the policy server. Kept only if it answers the pending request."""
        if self._pending is None or chunk.req_id != self._pending[0]:
            self.stale += 1
            return
        self._inbox = chunk

    def poll(self) -> Arrival | None:
        """The pending request's chunk, a `lost` Arrival once it has timed out, or None."""
        if self._pending is None:
            return None
        req_id, t_issue, epoch, sent = self._pending
        now = self._clock()
        if self._inbox is not None:
            chunk, self._inbox, self._pending = self._inbox, None, None
            self._rtts.append(now - sent)
            return Arrival(chunk.actions, t_issue, epoch, chunk.compute_s)
        if now - sent > self.timeout_s():
            self._pending = None
            self.lost += 1
            logger.debug('request %d timed out after %.3fs', req_id, now - sent)
            return Arrival(np.zeros((0, self.action_dim), np.float32), t_issue, epoch,
                           0.0, lost=True)
        return None

    def shutdown(self) -> None:
        self._pending = None
