import numpy as np

from evh_controller.chunk_executor.lift_gate import LiftHold


def _a(z, grip=1.0):
    return np.array([0.1, 0.0, z, 0, 0, 0, grip], np.float32)


def test_holds_a_rising_action_until_the_check_passes():
    ok = {'v': False}
    lh = LiftHold(lambda obs, n: ok['v'], rise_m=0.005, timeout=20)
    lh.on_close(_a(0.90))
    assert lh.check({}, _a(0.902)) is None             # not lifting yet
    held = lh.check({}, _a(0.92))
    assert held is not None and held[2] == np.float32(0.90) and held[6] == 1.0
    ok['v'] = True
    assert lh.check({}, _a(0.92)) is None
    ok['v'] = False
    assert lh.check({}, _a(0.95)) is None               # passed once: the rest of this grasp lifts
    assert lh.holds == 1


def test_times_out_and_resets_on_open():
    lh = LiftHold(lambda obs, n: False, timeout=3)
    lh.on_close(_a(0.90))
    assert [lh.check({}, _a(0.95)) is not None for _ in range(5)] == [True] * 3 + [False] * 2
    lh.on_open()
    assert lh.check({}, _a(0.95)) is None


def test_ticks_since_close_reach_the_check():
    seen = []
    lh = LiftHold(lambda obs, n: seen.append(n) or False)
    lh.on_close(_a(0.90))
    for _ in range(3):
        lh.check({}, _a(0.95))
    assert seen == [1, 2, 3]
