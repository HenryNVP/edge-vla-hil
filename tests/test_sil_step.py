"""Tests for the step-based SiL harness — pure Python, no ROS, no robosuite, no GPU.

The harness's claim is that it is the testbed's execution with the network replaced by a tick
count. So what is pinned here is the timing (a chunk is usable exactly d ticks after its request,
one request in flight), what an idle tick does to the world and to the deadline, and that the
hold before the first action is the plant's hold, not a zero (invariant 2).
"""
import numpy as np
import pytest

from evh_controller.chunk_executor import make_executor
from evh_plant.env_factory import hold_action
from sil_step import StepWorker, fixed_delay, run_episode, sampled_delay

H = 6


class FakePolicy:
    """Chunks whose x-position counts up from the observed x: a perfect planner on a line."""
    n_obs_steps = 2
    chunk_size = H
    action_dim = 7
    needs_wrist = False
    absolute_actions = True
    guided_resampling = True

    def __init__(self):
        self.inpaint_calls = 0

    def predict(self, obs):
        x = float(obs['proprio'][-1][0])
        chunk = np.zeros((H, 7), np.float32)
        chunk[:, 0] = x + np.arange(1, H + 1) * 0.01
        chunk[:, 3:6] = [np.pi, 0, 0]
        return chunk

    def predict_inpaint(self, obs, prefix, weights):
        self.inpaint_calls += 1
        return self.predict(obs)


class FakeEnv:
    """Moves the end effector to the commanded x; succeeds once x passes `goal`."""

    def __init__(self, goal=10.0):
        self.x = 0.0
        self.goal = goal
        self.actions = []

    def obs(self):
        return {'robot0_eef_pos': np.array([self.x, 0.0, 0.0]),
                'robot0_eef_quat': np.array([1.0, 0.0, 0.0, 0.0])}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.x = float(action[0])
        return self.obs(), 0.0, False, {}


def _frames(obs, _needs_wrist):
    proprio = np.zeros(9, np.float32)
    proprio[0] = obs['robot0_eef_pos'][0]
    return 0, None, proprio


def _run(strategy, d, world_speed=1.0, deadline=60, goal=10.0, absolute=True, hold_fn=None):
    env = FakeEnv(goal)
    policy = FakePolicy()
    worker = StepWorker(policy, fixed_delay(d))
    ex = make_executor(strategy, worker, policy)
    ep = run_episode(env, env.obs(), policy, ex, worker, deadline_ticks=deadline,
                     world_speed=world_speed, absolute=absolute,
                     is_success=lambda e: e.x >= e.goal, frame_fn=_frames, hold_fn=hold_fn)
    return ep, env, policy


# ------------------------------------------------------------------ worker
def test_a_chunk_is_usable_exactly_d_ticks_after_its_request():
    worker = StepWorker(FakePolicy(), fixed_delay(3))
    assert worker.try_request({'proprio': np.zeros((2, 9))}, 7, 0)
    assert not worker.try_request({'proprio': np.zeros((2, 9))}, 7, 0), 'one in flight'
    arrivals = [worker.poll() for _ in range(3)]
    assert arrivals[:2] == [None, None]
    assert arrivals[2] is not None and arrivals[2].t_issue == 7
    assert worker.poll() is None


def test_a_prefix_goes_to_the_guided_sampler():
    policy = FakePolicy()
    worker = StepWorker(policy, fixed_delay(1))
    worker.try_request({'proprio': np.zeros((2, 9))}, 0, 0,
                       prefix=np.zeros((2, 7)), weights=np.ones(H))
    assert policy.inpaint_calls == 1


def test_a_zero_tick_delay_is_refused():
    with pytest.raises(ValueError):
        fixed_delay(0)


def test_sampled_delays_leave_the_scene_rng_alone():
    """Scene k depends on numpy's global RNG; drawing a delay must not shift it."""
    np.random.seed(0)
    expected = np.random.rand()
    np.random.seed(0)
    draw = sampled_delay([2, 5, 9], seed=1)
    for _ in range(10):
        assert draw() in (2, 5, 9)
    assert np.random.rand() == expected


# ------------------------------------------------------------------ episode
def test_synchronous_waits_d_ticks_then_plays_the_whole_chunk():
    ep, env, _ = _run('synchronous', d=4, goal=1e9, deadline=30)
    # ticks 0-3 idle (first chunk in flight), then 6 actions, then idle while the next computes
    assert ep.idle >= 4
    assert ep.ticks == 30
    xs = [a[0] for a in env.actions if a[0] != 0.0]
    # idle ticks repeat the last target (difference 0); nothing ever moves backwards
    assert np.all(np.diff(xs) >= 0), 'synchronous never goes backwards'
    assert xs[-1] > xs[0]


def test_a_frozen_world_does_not_step_on_idle_ticks_but_still_spends_the_deadline():
    ep, env, _ = _run('synchronous', d=4, world_speed=0.0, goal=1e9, deadline=30)
    assert ep.env_steps == ep.ticks - ep.idle == len(env.actions)
    assert ep.ticks == 30


def test_a_moving_world_holds_the_last_target_on_idle_ticks():
    ep, env, _ = _run('synchronous', d=4, world_speed=1.0, goal=1e9, deadline=30)
    assert ep.env_steps == ep.ticks
    # after the first chunk ends, idle ticks repeat its last action exactly
    first_chunk_end = 4 + H
    assert np.array_equal(env.actions[first_chunk_end], env.actions[first_chunk_end - 1])


def test_before_the_first_action_the_hold_is_the_current_pose_not_the_origin():
    ep, env, _ = _run('synchronous', d=4, world_speed=1.0, goal=1e9, deadline=10)
    expected = hold_action(FakeEnv().obs(), True, 7)
    assert np.array_equal(env.actions[0], expected)
    assert np.allclose(env.actions[0][3:6], expected[3:6]) and np.any(expected[3:6] != 0)


def test_success_ends_the_episode_at_that_tick():
    ep, _, _ = _run('synchronous', d=1, goal=0.03, deadline=50)
    assert ep.success
    assert ep.ticks < 50
    assert ep.sim_s == pytest.approx(ep.ticks / 20.0)


@pytest.mark.parametrize('strategy', ['synchronous', 'naive_async', 'temporal_ensemble', 'rtc'])
def test_every_registered_strategy_runs_unchanged(strategy):
    ep, _, _ = _run(strategy, d=3, goal=0.2, deadline=200)
    assert ep.success
    assert ep.requests >= 1


def test_in_delta_mode_an_idle_tick_holds_instead_of_repeating_the_last_delta():
    """Repeating a delta is a velocity command: the arm would keep moving through every wait."""
    marker = np.full(7, -9.0, np.float32)
    ep, env, _ = _run('synchronous', d=4, world_speed=1.0, goal=1e9, deadline=30,
                      absolute=False, hold_fn=lambda _o: marker)
    first_chunk_end = 4 + H
    assert np.array_equal(env.actions[first_chunk_end], marker)
    assert ep.path_m == 0.0, 'no target path is defined for deltas'


# ------------------------------------------------------------------ WiFi trace replay
def _trace(lost_image=(), lost_chunk=(), offset=5.0):
    """Robot sends an image every 50 ms (20 ms uplink); the server sends a chunk every 200 ms
    (10 ms downlink); the server clock runs `offset` s ahead; probes see a symmetric 5 ms each way."""
    robot, server = [], []
    for k in range(200):
        s = 100.0 + 0.05 * k
        robot.append(dict(stream='sent:image', seq=k, sent=s, recv='', t1='', t2='', t3='', t4=''))
        if k not in lost_image:
            server.append(dict(stream='image', seq=k, sent=s, recv=s + 0.020 + offset))
    for k in range(50):
        c = 100.0 + 0.2 * k
        server.append(dict(stream='sent:chunk', seq=k, sent=c + offset, recv=''))
        if k not in lost_chunk:
            robot.append(dict(stream='chunk', seq=k, sent=c + offset, recv=c + 0.010,
                              t1='', t2='', t3='', t4=''))
    for k in range(20):
        t1 = 100.0 + k
        robot.append(dict(stream='probe', seq='', sent='', recv='', t1=t1, t2=t1 + 0.005 + offset,
                          t3=t1 + 0.006 + offset, t4=t1 + 0.011))
    return robot, server


def test_a_round_trip_is_uplink_plus_inference_plus_the_downlink_delay_of_that_moment():
    from sil_step import wifi_rtt_ms
    robot, server = _trace()
    _t, rtt = wifi_rtt_ms(robot, server, inference_ms=80.0, trim_s=0.5)
    # 20 ms up + 80 ms inference + 10 ms down; the 5 Hz send schedule is not charged
    assert np.allclose(rtt, 110.0, atol=1.0)


def test_a_lost_message_costs_the_wait_for_the_next_one_that_arrived():
    from sil_step import wifi_rtt_ms
    robot, server = _trace(lost_image={40})
    t, rtt = wifi_rtt_ms(robot, server, inference_ms=80.0, trim_s=0.5)
    i = int(np.argmin(np.abs(t - (0.05 * 40 - 0.5))))
    assert rtt[i] == pytest.approx(160.0, abs=1.0)      # waits one image period (50 ms) more


def test_trace_delay_is_paired_by_episode_and_scales_only_the_network():
    from sil_step import TraceDelay
    times = np.arange(0, 600, 0.05)
    rtt = np.full_like(times, 280.0)                    # 80 ms inference + 200 ms network
    a = TraceDelay(times, rtt, seed=3).for_episode(7)
    b = TraceDelay(times, rtt, seed=3).for_episode(7)
    assert a.offset_s == b.offset_s                    # same channel for every strategy
    assert a(0) == 6                                   # 280 ms -> 6 ticks of 50 ms
    assert TraceDelay(times, rtt, scale=2.0)(0) == 10  # 80 + 2 * 200 = 480 ms -> 10 ticks


def test_make_strategy_passes_the_coverage_aware_knobs_and_leaves_others_alone():
    from sil_step import make_strategy
    policy = FakePolicy()
    worker = StepWorker(policy, fixed_delay(2))
    ex = make_strategy('coverage_aware', worker, policy, s_wait=3, s_min=5)
    assert ex.s_wait == 3 and ex.s_min == 5
    assert make_strategy('synchronous', worker, policy, s_wait=3).name == 'synchronous'
    for name in ('noskip_async', 'rtc_noskip', 'switch_only'):
        assert make_strategy(name, worker, policy, s_min=5).s_min == 5, name


def test_make_strategy_builds_the_inpainting_continuation():
    from sil_step import make_strategy
    policy = FakePolicy()
    ex = make_strategy('continuation_inpaint', StepWorker(policy, fixed_delay(2)), policy)
    assert ex.name == 'continuation' and ex.inpaint and ex.roll_fn is None
