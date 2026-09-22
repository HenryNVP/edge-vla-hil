"""The policy side of robot-side execution: answer chunk requests from the latest observation.

With the executor on the robot side (remote_worker.py), `controller_node` stops executing chunks
and becomes a server: requests arrive over the uplink, each is answered with one chunk computed
from the newest observation history the controller holds, and the reply goes back over the
downlink. This module is that loop without ROS, so it is testable in the fast suite.

Two rules shape it:

  * The newest request wins. A robot side that timed out asks again, and the retry supersedes
    whatever was waiting; answering a request its sender has already abandoned would only put a
    stale reply on the wire for RemoteWorker to throw away.
  * Inference stays single-slot on the same InferenceWorker the policy-side placement uses, so
    both placements pay the identical compute cost; only where the chunk is consumed differs.

The request's `t_issue` / `epoch` belong to the robot side and are echoed back untouched, and its
RTC guidance (prefix, weights) is passed to the policy exactly as a local executor would pass it.
"""
from __future__ import annotations

from evh_controller.chunk_codec import Chunk, Request
from evh_controller.inference_worker import InferenceWorker


class ChunkServer:
    """Single-slot request handling in front of an InferenceWorker."""

    def __init__(self, worker: InferenceWorker) -> None:
        self.worker = worker
        self._waiting: Request | None = None     # newest request not yet started
        self._serving: Request | None = None     # request whose chunk is being computed
        self.superseded = 0

    def on_request(self, req: Request) -> None:
        if self._waiting is not None:
            self.superseded += 1
        self._waiting = req

    def step(self, obs: dict | None) -> Chunk | None:
        """Start the waiting request if the worker is free; return a finished reply, if any.

        `obs` is the controller's latest stacked observation (None until it has one). Call often:
        the reply waits here until the next call, so the call rate quantises the inference time.
        """
        reply = None
        arrival = self.worker.poll()
        if arrival is not None and self._serving is not None:
            req, self._serving = self._serving, None
            reply = Chunk(req.req_id, req.t_issue, req.epoch, arrival.compute_s, arrival.chunk)
        if self._waiting is not None and obs is not None and not self.worker.busy:
            req = self._waiting
            if self.worker.try_request(obs, req.t_issue, req.epoch, req.prefix, req.weights):
                self._serving, self._waiting = req, None
        return reply

    def reset(self) -> None:
        """Episode boundary: a request from the old episode is not worth starting."""
        self._waiting = None
