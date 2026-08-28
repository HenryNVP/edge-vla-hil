"""Tests for the real InferenceWorker — no ROS2 required.

test_chunk_executor.py drives the strategies through a FakeWorker, so the actual threaded worker
— the thing that keeps the 20 Hz control tick off the GPU — is otherwise untested. What matters
here is the contract the strategies lean on: exactly one request in flight, observations
snapshotted at request time, the issue tick and epoch surviving the round trip, and (the one that
would deadlock the control loop) a policy that RAISES still delivering an arrival.
"""
import threading
import time

import numpy as np
import pytest

from evh_controller.inference_worker import InferenceWorker

H, A = 8, 7


def _obs():
    return {'agentview': np.zeros((2, 84, 84, 3), np.uint8),
            'proprio': np.zeros((2, 9), np.float32)}


class SlowPolicy:
    """Blocks inside predict() until released, so 'busy' is observable deterministically."""
    chunk_size = H
    action_dim = A

    def __init__(self, blocking=False):
        self.gate = threading.Event()
        if not blocking:
            self.gate.set()
        self.seen: list[dict] = []
        self.inpaint_calls: list[tuple] = []

    def predict(self, obs):
        self.seen.append(obs)
        self.gate.wait(timeout=5.0)
        return np.ones((H, A), np.float32)

    def predict_inpaint(self, obs, prefix, weights):
        self.inpaint_calls.append((obs, prefix, weights))
        return np.full((H, A), 2.0, np.float32)


class ExplodingPolicy:
    chunk_size = H
    action_dim = A

    def predict(self, obs):
        raise RuntimeError('CUDA out of memory')


def _await_arrival(worker, timeout=5.0):
    """Poll like the control tick does; return the Arrival or fail."""
    end = time.time() + timeout
    while time.time() < end:
        arrival = worker.poll()
        if arrival is not None:
            return arrival
        time.sleep(0.005)
    pytest.fail('worker never delivered a chunk')


def test_only_one_request_is_in_flight_at_a_time():
    """Single-slot: mirrors a single-GPU controller. A second request must be refused, not queued
    — a queue here would let requests pile up and report a delay the hardware never had."""
    policy = SlowPolicy(blocking=True)
    worker = InferenceWorker(policy)
    try:
        assert worker.try_request(_obs(), t_issue=0, epoch=0) is True
        assert worker.busy is True
        assert worker.try_request(_obs(), t_issue=1, epoch=0) is False
        assert worker.poll() is None, 'nothing should arrive while the policy is still running'
    finally:
        policy.gate.set()
        worker.shutdown()


def test_slot_is_freed_once_the_chunk_is_collected():
    policy = SlowPolicy()
    worker = InferenceWorker(policy)
    try:
        worker.try_request(_obs(), t_issue=3, epoch=0)
        _await_arrival(worker)
        assert worker.busy is False
        assert worker.try_request(_obs(), t_issue=4, epoch=0) is True
    finally:
        worker.shutdown()


def test_arrival_carries_the_issue_tick_epoch_and_a_real_compute_time():
    """t_issue and epoch are how executors time-align a chunk and discard pre-reset ones."""
    policy = SlowPolicy()
    worker = InferenceWorker(policy)
    try:
        worker.try_request(_obs(), t_issue=17, epoch=4)
        arrival = _await_arrival(worker)
        assert arrival.t_issue == 17
        assert arrival.epoch == 4
        assert arrival.compute_s > 0.0
        assert arrival.chunk.shape == (H, A)
    finally:
        worker.shutdown()


def test_observations_are_snapshotted_at_request_time():
    """The controller overwrites its obs buffers from ROS callbacks while inference runs; the
    worker must copy, or a chunk gets computed against a half-updated observation."""
    policy = SlowPolicy(blocking=True)
    worker = InferenceWorker(policy)
    try:
        obs = _obs()
        obs['proprio'][:] = 1.0
        worker.try_request(obs, t_issue=0, epoch=0)
        obs['proprio'][:] = 99.0          # simulate a newer message landing mid-inference
        policy.gate.set()
        _await_arrival(worker)
        assert policy.seen, 'predict() was never called'
        assert np.all(policy.seen[0]['proprio'] == 1.0), 'worker saw the caller-mutated buffer'
    finally:
        policy.gate.set()
        worker.shutdown()


def test_a_prefix_routes_to_the_inpainting_path():
    """RTC hands the worker the actions that will execute during inference."""
    policy = SlowPolicy()
    worker = InferenceWorker(policy)
    try:
        prefix = np.zeros((3, A), np.float32)
        weights = np.linspace(1.0, 0.0, H, dtype=np.float32)
        worker.try_request(_obs(), t_issue=0, epoch=0, prefix=prefix, weights=weights)
        arrival = _await_arrival(worker)
        assert len(policy.inpaint_calls) == 1
        assert policy.seen == [], 'plain predict() must not also run'
        assert np.all(arrival.chunk == 2.0)
    finally:
        worker.shutdown()


@pytest.mark.parametrize('prefix', [None, np.zeros((0, A), np.float32)])
def test_no_prefix_uses_plain_predict(prefix):
    """An empty prefix is not an inpainting request — len 0 must take the plain path."""
    policy = SlowPolicy()
    worker = InferenceWorker(policy)
    try:
        worker.try_request(_obs(), t_issue=0, epoch=0, prefix=prefix)
        arrival = _await_arrival(worker)
        assert policy.inpaint_calls == []
        assert np.all(arrival.chunk == 1.0)
    finally:
        worker.shutdown()


def test_a_failing_policy_still_delivers_a_chunk():
    """The control tick polls; it never joins the thread. If an exception swallowed the arrival
    the slot would stay busy forever and the robot would silently stop getting new chunks."""
    worker = InferenceWorker(ExplodingPolicy())
    try:
        worker.try_request(_obs(), t_issue=9, epoch=1)
        arrival = _await_arrival(worker)
        assert arrival.chunk.shape == (H, A)
        assert np.all(arrival.chunk == 0.0)
        assert arrival.t_issue == 9 and arrival.epoch == 1
        assert worker.busy is False
        assert worker.try_request(_obs(), t_issue=10, epoch=1) is True, 'slot never recovered'
    finally:
        worker.shutdown()


def test_shutdown_stops_the_worker_thread():
    worker = InferenceWorker(SlowPolicy())
    worker.shutdown()
    worker._thread.join(timeout=5.0)
    assert not worker._thread.is_alive()


def test_shutdown_is_safe_while_a_request_is_in_flight():
    """destroy_node() can land mid-inference; a full request queue must not raise."""
    policy = SlowPolicy(blocking=True)
    worker = InferenceWorker(policy)
    worker.try_request(_obs(), t_issue=0, epoch=0)
    worker.shutdown()          # queue is full / worker mid-compute: must not raise
    policy.gate.set()
