import numpy as np

import nut_goal as ng


class Env:
    """Both nuts on the table; step(action) sets the square nut's z to action[0]."""

    def __init__(self):
        self.objects_on_pegs = np.zeros(2)

    def _obs(self, z=0.83):
        return {'SquareNut_pos': np.array([0.0, 0.15, z]), 'SquareNut_quat': np.zeros(4),
                'RoundNut_pos': np.array([0.0, -0.15, 0.83]), 'RoundNut_quat': np.ones(4)}

    def reset(self):
        return self._obs()

    def step(self, action):
        return self._obs(action[0]), 0.0, False, {}

    def _check_success(self):
        self.objects_on_pegs = np.array([0.0, 1.0])     # only the round nut is on its peg


def test_obs_carries_the_goal_and_its_nut():
    w = ng.GoalWorld(Env())
    o = w.reset(goal=1)
    assert o['goal'] == 1 and o['goal_nut_pos'][1] == -0.15 and o['goal_nut_quat'][0] == 1


def test_switch_flips_once_at_its_tick():
    w = ng.GoalWorld(Env())
    w.reset(goal=0, switch_tick=2)
    assert w.step([0.83])[0]['goal'] == 0
    o = w.step([0.83])[0]
    assert o['goal'] == 1 and o['goal_nut_pos'][1] == -0.15 and w.switched
    assert w.step([0.83])[0]['goal'] == 1


def test_switch_is_cancelled_once_the_goal_nut_is_lifted():
    w = ng.GoalWorld(Env())
    w.reset(goal=0, switch_tick=1)
    assert w.step([0.85])[0]['goal'] == 0 and not w.switched


def test_success_is_the_goal_nut_only():
    w = ng.GoalWorld(Env())
    w.reset(goal=0)
    assert not w.success()
    w.reset(goal=1)
    assert w.success()
