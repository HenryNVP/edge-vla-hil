"""Tests for the shared-server fleet harness — pure Python, no ROS, no robosuite, no GPU.

What is pinned: the queue (first come first served, one inference at a time), that a fleet of
one is exactly the single-robot WiFi replay, and that robots that finish early keep loading the
server until every robot has its episodes.
"""
import numpy as np

from sil_fleet import FleetLink, FleetServer, run_fleet
from sil_step import StepWorker, TraceDelay
from test_sil_step import FakeEnv, FakePolicy, _frames
from evh_controller.chunk_executor import make_executor


def _flat_trace(net_ms):
    times = np.arange(0, 600, 0.05)
    return TraceDelay(times, np.full_like(times, 80.0 + net_ms))   # trace rtt includes 80 ms


def test_requests_at_the_same_tick_queue_behind_each_other():
    server = FleetServer(service_ms=80.0)
    a, b = FleetLink(server, _flat_trace(0.0)), FleetLink(server, _flat_trace(0.0))
    assert a(0) == 2                       # 80 ms -> 2 ticks
    assert b(0) == 4                       # waits 80 ms for a, then 80 ms of its own
    assert b.delays_ms[-1] == 160.0


def test_a_fleet_of_one_is_the_single_robot_replay():
    server = FleetServer(service_ms=80.0)
    net = _flat_trace(230.0)
    link = FleetLink(server, net)
    for t in range(0, 200, 7):                 # never overlapping: one request in flight
        assert link(t) == net(t)


def test_a_free_server_charges_only_network_and_inference():
    server = FleetServer(service_ms=80.0)
    a = FleetLink(server, _flat_trace(100.0))
    a(0)
    b = FleetLink(server, _flat_trace(100.0))
    assert b(20) == 4                          # 1 s later the server is idle: 180 ms -> 4 ticks


def test_episode_ticks_are_counted_from_the_robots_episode_start():
    server = FleetServer(service_ms=80.0)
    a, b = FleetLink(server, _flat_trace(0.0)), FleetLink(server, _flat_trace(0.0))
    a.start_episode(g0=100, scene=0)
    a(0)                                       # global tick 100
    b.start_episode(g0=0, scene=0)
    assert b(100) == 4                         # also global tick 100: queues behind a


def test_every_robot_gets_its_episodes_and_early_finishers_keep_running():
    server = FleetServer(service_ms=80.0)
    robots, rows = [], []
    for goal in (0.05, 0.3):                   # robot 0 finishes far sooner than robot 1
        policy, env = FakePolicy(), FakeEnv(goal)
        link = FleetLink(server, _flat_trace(0.0))
        worker = StepWorker(policy, link)
        robots.append(dict(env=env, obs=env.obs(), policy=policy, link=link, worker=worker,
                           executor=make_executor('synchronous', worker, policy),
                           reset=lambda e=env: (setattr(e, 'x', 0.0), e.obs())[1]))
    ran = run_fleet(robots, episodes=3, deadline_ticks=200,
                    on_episode=lambda i, k, ep, link: rows.append((i, k, ep.success)),
                    episode_kw=dict(is_success=lambda e: e.x >= e.goal, frame_fn=_frames))
    assert sorted((i, k) for i, k, _ in rows) == [(i, k) for i in (0, 1) for k in range(3)]
    assert all(ok for *_, ok in rows)
    assert ran[0] > 3 and ran[1] == 3          # robot 0 kept loading the server, unrecorded
