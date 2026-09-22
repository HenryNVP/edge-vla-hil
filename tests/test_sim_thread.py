"""Tests for the plant's simulator thread — pure Python, a fake env, no robosuite.

The invariant is one line: every env call happens on ONE thread, the one that built it. robosuite
renders through a thread-bound EGL context; a step from any other thread returned the wrong camera
or a half-drawn frame, and the policy acted on it.
"""
import threading
import time

import pytest

from evh_plant.sim_thread import SimThread


class FakeEnv:
    def __init__(self):
        self.threads = {'build': set(), 'step': set(), 'close': set()}
        self.steps = 0

    def build(self):
        self.threads['build'].add(threading.get_ident())

    def step(self):
        self.threads['step'].add(threading.get_ident())
        self.steps += 1

    def close(self):
        self.threads['close'].add(threading.get_ident())


def _wait(pred, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end and not pred():
        time.sleep(0.001)
    return pred()


def test_build_step_and_close_all_run_on_one_thread_that_is_not_the_callers():
    env = FakeEnv()
    sim = SimThread(env.build, env.step, env.close, period_s=0.002).start()
    assert sim.ready.wait(2.0)
    assert _wait(lambda: env.steps >= 5)
    sim.stop()

    threads = env.threads['build'] | env.threads['step'] | env.threads['close']
    assert len(threads) == 1, f'env touched from {len(threads)} threads'
    assert threading.get_ident() not in threads
    assert threads == {sim.thread_id}


def test_it_steps_at_roughly_the_requested_rate():
    env = FakeEnv()
    sim = SimThread(env.build, env.step, env.close, period_s=0.005).start()
    sim.ready.wait(2.0)
    time.sleep(0.5)
    sim.stop()
    assert 60 <= env.steps <= 110, f'{env.steps} steps in 0.5 s at 200 Hz'


def test_an_overrun_is_counted_and_not_caught_up():
    """A ROS timer does not fire a burst to make up missed ticks, and neither may this."""
    t = [0.0]
    env = FakeEnv()

    def slow_step():
        env.step()
        t[0] += 0.05 if env.steps == 3 else 0.001     # the 3rd step overruns by 10 periods

    sim = SimThread(env.build, slow_step, env.close, period_s=0.005,
                    clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s)).start()
    assert _wait(lambda: env.steps >= 20)
    sim.stop()
    assert sim.overruns == 1


def test_a_build_failure_is_reported_not_swallowed():
    def boom():
        raise RuntimeError('no robosuite')

    env = FakeEnv()
    sim = SimThread(boom, env.step, env.close, period_s=0.01).start()
    assert sim.ready.wait(2.0)
    assert isinstance(sim.error, RuntimeError) and env.steps == 0


def test_the_env_is_closed_even_if_a_step_raises():
    env = FakeEnv()

    def bad_step():
        raise ValueError('physics exploded')

    sim = SimThread(env.build, bad_step, env.close, period_s=0.01).start()
    assert _wait(lambda: sim.error is not None)
    sim.stop()
    assert env.threads['close'], 'env left open after a failing step'


@pytest.mark.parametrize('calls', [1, 2])
def test_stop_is_idempotent(calls):
    env = FakeEnv()
    sim = SimThread(env.build, env.step, env.close, period_s=0.01).start()
    sim.ready.wait(2.0)
    for _ in range(calls):
        sim.stop()
    assert len(env.threads['close']) == 1
