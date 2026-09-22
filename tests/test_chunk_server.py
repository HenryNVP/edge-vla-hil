"""Tests for the policy side of robot-side execution — pure Python, no ROS.

The server answers chunk requests from the controller's newest observation history. What it must
get right: newest request wins, the robot's bookkeeping is echoed untouched, RTC guidance reaches
the policy, and nothing is started without an observation.
"""
import numpy as np

from evh_controller.chunk_codec import Request
from evh_controller.chunk_server import ChunkServer
from evh_controller.inference_worker import Arrival


class FakeWorker:
    """Finishes a started request on the next poll."""

    def __init__(self):
        self.busy, self.started, self._done = False, [], None

    def try_request(self, obs, t_issue, epoch, prefix=None, weights=None):
        if self.busy:
            return False
        self.busy = True
        self.started.append((t_issue, epoch, prefix, weights))
        self._done = Arrival(np.full((4, 7), float(t_issue), np.float32), t_issue, epoch, 0.05)
        return True

    def poll(self):
        if not self.busy or self._done is None:
            return None
        arrival, self._done, self.busy = self._done, None, False
        return arrival


OBS = {'agentview': np.zeros((1, 4, 4, 3), np.uint8)}


def test_a_request_is_answered_with_its_own_ids_echoed():
    worker = FakeWorker()
    server = ChunkServer(worker)
    server.on_request(Request(req_id=7, t_issue=30, epoch=2, prefix=None, weights=None))

    assert server.step(OBS) is None                  # started
    reply = server.step(OBS)                         # finished
    assert (reply.req_id, reply.t_issue, reply.epoch) == (7, 30, 2)
    assert reply.compute_s == 0.05 and reply.actions.shape == (4, 7)


def test_nothing_starts_before_the_first_observation():
    worker = FakeWorker()
    server = ChunkServer(worker)
    server.on_request(Request(0, 0, 1, None, None))
    assert server.step(None) is None and worker.started == []
    server.step(OBS)
    assert len(worker.started) == 1, 'the waiting request was lost instead of kept'


def test_the_newest_request_supersedes_one_still_waiting():
    """A robot side that timed out re-asks; answering the abandoned request would only put a
    reply on the wire that RemoteWorker throws away."""
    worker = FakeWorker()
    server = ChunkServer(worker)
    server.on_request(Request(0, 10, 1, None, None))
    server.on_request(Request(1, 20, 1, None, None))
    server.step(OBS)
    assert [s[0] for s in worker.started] == [20] and server.superseded == 1


def test_rtc_guidance_reaches_the_policy_untouched():
    worker = FakeWorker()
    server = ChunkServer(worker)
    prefix, weights = np.ones((3, 7)), np.linspace(1, 0, 4)
    server.on_request(Request(0, 0, 1, prefix, weights))
    server.step(OBS)
    _, _, got_prefix, got_weights = worker.started[0]
    assert np.array_equal(got_prefix, prefix) and np.array_equal(got_weights, weights)


def test_a_reset_drops_a_request_from_the_old_episode():
    worker = FakeWorker()
    server = ChunkServer(worker)
    server.on_request(Request(0, 99, 1, None, None))
    server.reset()
    server.step(OBS)
    assert worker.started == []
