"""Background inference worker: the control tick must never block on the GPU.

This is what makes the latency phenomenon *real* in the ROS graph: policy inference runs
concurrently with action streaming, exactly like the deployed system (a Jetson computing one
chunk while the robot executes the previous one). Strategies request a chunk, keep streaming
from the one in hand, and poll for the result; the delay they observe — in control steps — is
the honest inference (+scheduling) latency, and is what feeds RTC's delay forecast.

At most one request is in flight at a time (`try_request` returns False while busy), mirroring
a single-GPU controller. Results carry the issue tick and an epoch tag so an executor reset
(episode boundary) can discard a chunk that was computed against pre-reset observations.

Note on concurrency: PyTorch releases the GIL inside the heavy ops, so the rclpy executor keeps
ticking while the denoiser runs. Residual Python-level GIL contention in the model wrapper shows
up honestly as tick jitter — which is part of what the testbed measures.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from evh_controller.policy import ChunkPolicy

logger = logging.getLogger(__name__)


@dataclass
class Arrival:
    """A finished chunk, delivered back to the executor."""
    chunk: np.ndarray      # [H, A]
    t_issue: int           # control tick the request was issued at
    epoch: int             # executor epoch at issue time (stale-after-reset guard)
    compute_s: float       # wall-clock inference time


@dataclass
class _Job:
    image: np.ndarray
    state: np.ndarray
    t_issue: int
    epoch: int
    prefix: np.ndarray | None    # RTC: actions that will execute during inference
    weights: np.ndarray | None   # RTC: soft-mask over the chunk (len H)


class InferenceWorker:
    """Single-slot threaded inference: request, keep executing, poll."""

    def __init__(self, policy: ChunkPolicy) -> None:
        self.policy = policy
        self._req: queue.Queue[_Job | None] = queue.Queue(maxsize=1)
        self._res: queue.Queue[Arrival] = queue.Queue()
        self._busy = False   # touched only from the control thread
        self._thread = threading.Thread(target=self._run, daemon=True, name='evh_inference')
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy

    def try_request(self, image: np.ndarray, state: np.ndarray, t_issue: int, epoch: int,
                    prefix: np.ndarray | None = None,
                    weights: np.ndarray | None = None) -> bool:
        """Start computing a chunk from `obs`; False if a request is already in flight."""
        if self._busy:
            return False
        self._busy = True
        # snapshot the observations: the caller's buffers are overwritten by newer messages
        self._req.put(_Job(np.array(image, copy=True), np.array(state, copy=True),
                           t_issue, epoch, prefix, weights))
        return True

    def poll(self) -> Arrival | None:
        """Non-blocking: the finished chunk, or None if still computing / nothing requested."""
        try:
            arrival = self._res.get_nowait()
        except queue.Empty:
            return None
        self._busy = False
        return arrival

    def shutdown(self) -> None:
        try:
            self._req.put_nowait(None)
        except queue.Full:
            pass   # worker mid-compute; it is a daemon thread and dies with the process

    # ------------------------------------------------------------ worker side
    def _run(self) -> None:
        while True:
            job = self._req.get()
            if job is None:
                return
            t0 = time.perf_counter()
            try:
                if job.prefix is not None and len(job.prefix) > 0:
                    chunk = self.policy.predict_inpaint(
                        job.image, job.state, job.prefix, job.weights)
                else:
                    chunk = self.policy.predict(job.image, job.state)
            except Exception:
                logger.exception('policy inference failed; delivering a zero chunk')
                chunk = np.zeros(
                    (self.policy.chunk_size, self.policy.action_dim), dtype=np.float32)
            self._res.put(Arrival(np.asarray(chunk), job.t_issue, job.epoch,
                                  time.perf_counter() - t0))
