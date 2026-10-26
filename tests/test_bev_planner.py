import numpy as np

import bev_planner as bp


def test_goal_consistent_t0_moves_a_pre_switch_memory_to_the_switch():
    goals = np.array([0, 0, 0, 1, 1, 1])                 # switch at frame 3
    assert bp.goal_consistent_t0(goals, 5, 1) == 3
    assert bp.goal_consistent_t0(goals, 2, 0) == 0       # before the switch: unchanged
    assert bp.goal_consistent_t0(goals, 5, 4) == 4       # after it: unchanged
