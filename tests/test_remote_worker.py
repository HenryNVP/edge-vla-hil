"""Tests for robot-side execution's inference proxy — pure Python, no ROS.

RemoteWorker has to look exactly like the local InferenceWorker to the strategies (so they run
unchanged on the robot side) while surviving what the local worker never meets: replies that
arrive late, arrive for a request already given up on, or never arrive.
"""
import numpy as np
import pytest

from evh_controller.chunk_codec import Chunk
from evh_controller.chunk_executor import make_executor
from evh_controller.remote_worker import PolicyInfo, RemoteWorker

H, A = 8, 7


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _worker(**kw):
    sent, clock = [], Clock()
    worker = RemoteWorker(sent.append, action_dim=A, clock=clock, **kw)
    return worker, sent, clock


def _reply(req, value=1.0, compute_s=0.05):
    return Chunk(req.req_id, req.t_issue, req.epoch, compute_s, np.full((H, A), value, np.float32))


def test_a_request_goes_out_and_its_reply_comes_back_as_an_arrival():
    worker, sent, clock = _worker()
    assert worker.try_request({}, t_issue=5, epoch=2)
    assert worker.busy and len(sent) == 1

    clock.t = 0.3
    worker.deliver(_reply(sent[0], value=4.0))
    arrival = worker.poll()

    assert arrival.t_issue == 5 and arrival.epoch == 2 and not arrival.lost
    assert np.all(arrival.chunk == 4.0) and arrival.compute_s == pytest.approx(0.05)
    assert not worker.busy


def test_only_one_request_is_in_flight():
    worker, sent, _ = _worker()
    assert worker.try_request({}, 0, 1)
    assert not worker.try_request({}, 1, 1)
    assert len(sent) == 1


def test_nothing_arrives_while_waiting():
    worker, _, clock = _worker()
    worker.try_request({}, 0, 1)
    clock.t = 1.0
    assert worker.poll() is None and worker.busy


def test_a_request_with_no_reply_times_out_as_lost_and_frees_the_slot():
    """Without this a single dropped packet would stall the executor for the whole episode."""
    worker, sent, clock = _worker(first_timeout_s=2.0)
    worker.try_request({}, 7, 3)
    clock.t = 2.5
    arrival = worker.poll()

    assert arrival.lost and arrival.t_issue == 7 and arrival.epoch == 3
    assert not worker.busy and worker.lost == 1
    assert worker.try_request({}, 8, 3), 'slot not freed after the timeout'


def test_a_reply_for_a_request_already_given_up_on_is_dropped():
    """Splicing it in late would execute a plan made for an older observation than the one
    the retry is about to bring."""
    worker, sent, clock = _worker(first_timeout_s=1.0)
    worker.try_request({}, 0, 1)
    clock.t = 1.5
    assert worker.poll().lost
    worker.try_request({}, 30, 1)

    worker.deliver(_reply(sent[0], value=9.0))     # the first request's reply, finally
    assert worker.poll() is None and worker.stale == 1
    worker.deliver(_reply(sent[1], value=2.0))
    assert np.all(worker.poll().chunk == 2.0)


def test_the_timeout_adapts_to_the_measured_round_trip():
    worker, sent, clock = _worker(timeout_factor=2.0, min_timeout_s=0.25, first_timeout_s=10.0)
    assert worker.timeout_s() == 10.0, 'before any reply the first-chunk allowance applies'

    for rtt in (0.4, 0.6, 0.5):
        worker.try_request({}, 0, 1)
        clock.t += rtt
        worker.deliver(_reply(sent[-1]))
        worker.poll()
    assert worker.timeout_s() == pytest.approx(1.2), 'factor x slowest recent round trip'


def test_the_timeout_never_drops_below_its_floor():
    worker, sent, clock = _worker(min_timeout_s=0.25)
    worker.try_request({}, 0, 1)
    clock.t += 0.01
    worker.deliver(_reply(sent[-1]))
    worker.poll()
    assert worker.timeout_s() == 0.25


# ------------------------------------------------ strategies over a lossy fake link
class Link:
    """A policy server behind a link: replies after `delay` ticks, drops the listed requests."""

    def __init__(self, delay=3, drop=()):
        self.delay, self.drop, self.queue, self.served = delay, set(drop), [], 0

    def tick(self, worker, sent, now):
        for req in sent[self.served:]:
            self.served += 1
            if req.req_id not in self.drop:
                self.queue.append((now + self.delay, req))
        for due, req in [q for q in self.queue if q[0] <= now]:
            self.queue.remove((due, req))
            worker.deliver(_reply(req, value=float(req.req_id + 1)))


def _run(strategy, link, ticks, dt=0.05, **kw):
    worker, sent, clock = _worker(first_timeout_s=kw.pop('first_timeout_s', 10.0), **kw)
    info = PolicyInfo(chunk_size=H, action_dim=A, guided_resampling=True, absolute_actions=True)
    executor = make_executor(strategy, worker, info)
    actions = []
    for t in range(ticks):
        clock.t = t * dt
        link.tick(worker, sent, t)
        actions.append(executor.step({}, t))
    return executor, worker, sent, actions


@pytest.mark.parametrize('strategy', ['synchronous', 'naive_async', 'temporal_ensemble', 'rtc',
                                      'network_aware'])
def test_every_strategy_runs_unchanged_on_the_robot_side(strategy):
    _, _, sent, actions = _run(strategy, Link(delay=3), ticks=60)
    assert len(sent) > 1, 'never asked again after the first chunk'
    assert any(a is not None for a in actions)


def test_the_measured_delay_is_the_whole_round_trip():
    """What RTC's forecast was missing: with the executor on the robot side the request->arrival
    delay it records includes both network legs, with no extra feedback channel."""
    executor, _, _, _ = _run('rtc', Link(delay=5), ticks=80)
    assert executor._delays and min(executor._delays) >= 5


def test_a_dropped_request_costs_one_timeout_not_the_episode():
    executor, worker, sent, actions = _run(
        'synchronous', Link(delay=2, drop={1}), ticks=80, first_timeout_s=1.0)
    assert worker.lost == 1
    assert executor.take_lost() == 1
    assert len(sent) >= 4, 'the executor stopped asking after the loss'
    assert actions[-1] is not None or any(a is not None for a in actions[-20:])
