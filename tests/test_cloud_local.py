import numpy as np

from evh_controller.chunk_executor.cloud_local import CloudLocalExecutor
from evh_controller.inference_worker import Arrival


class Policy:
    chunk_size, action_dim, absolute_actions = 4, 7, True

    def __init__(self):
        self.calls = []

    def local(self, obs, memory, age):
        self.calls.append((None if memory is None else int(memory), age))
        c = np.zeros((4, 7), np.float32)
        c[:, 0] = np.arange(4)
        return c


class Worker:
    """Returns the request tick as the 'memory', d polls later."""

    def __init__(self, d):
        self.d, self._p = d, None

    def try_request(self, obs, t_issue, epoch, prefix=None, weights=None):
        if self._p is not None:
            return False
        self._p = [self.d, Arrival(np.array(t_issue), t_issue, epoch, 0.0)]
        return True

    def poll(self):
        if self._p is None:
            return None
        self._p[0] -= 1
        if self._p[0] > 0:
            return None
        a, self._p = self._p[1], None
        return a


def test_holds_until_the_first_memory_then_plans_locally_with_its_age():
    p = Policy()
    ex = CloudLocalExecutor(Worker(3), p, replan=2)
    out = [ex.step({}, t) for t in range(10)]
    assert out[:3] == [None] * 3 and all(a is not None for a in out[3:])
    assert [a[0] for a in out[3:7]] == [0, 1, 0, 1]          # two entries per plan
    # memory from t=0 arrives at t=3 (age 3, then 5); memory from t=3 arrives at t=6 (age 3)
    assert p.calls[:3] == [(0, 3), (0, 5), (3, 4)]


def test_age_is_clamped_to_the_trained_range():
    p = Policy()
    ex = CloudLocalExecutor(Worker(30), p, replan=1, max_age=10)
    for t in range(45):
        ex.step({}, t)
    assert max(a for _, a in p.calls) == 10


def test_once_keeps_the_first_memory_and_stops_asking():
    p = Policy()
    ex = CloudLocalExecutor(Worker(3), p, replan=2, refresh=False)
    for t in range(20):
        ex.step({}, t)
    assert {m for m, _ in p.calls} == {0} and p.calls[-1][1] == 19


def test_drop_after_plans_with_no_memory_once_it_is_too_old():
    p = Policy()
    ex = CloudLocalExecutor(Worker(3), p, replan=1, refresh=False, drop_after=5)
    for t in range(10):
        ex.step({}, t)
    assert [m for m, _ in p.calls] == [0, 0, 0] + [None] * 4      # ages 3, 4, 5, then 6..9
