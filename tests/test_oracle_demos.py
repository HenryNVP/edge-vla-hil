"""Tests for the demo collector's nut_goal episode draws — numpy only, no robosuite, no lerobot.

Pinned: the switch draw (none at probability 0, a tick in [10, 80] at probability 1) and
episode_params — one seeded RNG stream per episode for scene / conveyor / switch, the conveyor
speed drawn from SPEEDS, and the goal alternating on the kept count so the saved dataset is
exactly balanced however the two goals' success rates differ. The env-facing collect loop is
exercised in Docker instead (the collector's verification runs).
"""
import numpy as np

import oracle_demos as od


def test_switch_is_none_at_zero_probability_and_early_at_one():
    assert od.draw_switch(np.random.RandomState(0), 0.0) is None
    for s in range(20):
        tick = od.draw_switch(np.random.RandomState(s), 1.0)
        assert tick is not None and 10 <= tick <= 80


def test_switch_fires_about_half_the_time_at_the_default_probability():
    draws = [od.draw_switch(np.random.RandomState(s), 0.5) is not None for s in range(400)]
    assert 0.4 < np.mean(draws) < 0.6


def test_episode_params_are_seed_deterministic():
    assert (od.episode_params(100_003, 2, 0.5) == od.episode_params(100_003, 2, 0.5))


def test_episode_params_draw_speed_and_alternate_the_goal():
    for kept in range(6):
        speed, goal, switch = od.episode_params(100_000 + kept, kept, 0.0)
        assert speed in od.SPEEDS
        assert goal == kept % 2                       # balanced by construction, not by luck
        assert switch is None


def test_episode_params_switch_stays_in_range():
    for s in range(50):
        _, _, switch = od.episode_params(200_000 + s, s % 2, 1.0)
        assert 10 <= switch <= 80
