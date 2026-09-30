"""Unit tests for the LeRobot/LIBERO adapter's conversions — numpy only, no LeRobot import.

    python3 -m pytest -q experiments/lerobot/test_adapter.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapter import LiberoStepEnv, batch1, rtc_args  # noqa: E402


def test_only_the_frozen_count_crosses_to_lerobot():
    """RTCExecutor's weights: 1 over entries that play during inference, then a decay, then 0."""
    assert rtc_args([1, 1, 1, 0.7, 0.3, 0, 0]) == (3, 5)
    assert rtc_args([0, 0, 0]) == (0, 0)


class _FakeLibero:
    def __init__(self):
        self.actions = []
        self.episode_index = 0
        self.init_state_id = 7

    def reset(self):
        return {'x': 0}, {}

    def step(self, a):
        self.actions.append(np.asarray(a))
        return {'x': 1}, 0.0, False, False, {'is_success': False}


def test_an_idle_tick_stops_the_arm_but_keeps_the_hand_closed():
    """A zero gripper value would half-open the hand and drop the object mid-wait."""
    raw = _FakeLibero()
    env = LiberoStepEnv(raw, postprocess=lambda a: np.asarray(a))
    env.reset()
    env.step(np.array([0.1, 0, 0, 0, 0, 0, 1.0]))      # close the gripper
    env.step(None)                                      # wait
    assert np.array_equal(raw.actions[-1], [0, 0, 0, 0, 0, 0, 1.0])


def test_before_any_action_the_hold_keeps_the_hand_open():
    raw = _FakeLibero()
    env = LiberoStepEnv(raw, postprocess=lambda a: np.asarray(a))
    env.reset()
    env.step(None)
    assert raw.actions[-1][6] == -1.0


def test_rewinding_pairs_episode_k_to_the_same_initial_state_in_every_cell():
    raw = _FakeLibero()
    LiberoStepEnv(raw, postprocess=None).reset_pairing()
    assert raw.init_state_id == raw.episode_index == 0


def test_batch1_reaches_nested_observations():
    out = batch1({'pixels': {'image': np.zeros((4, 4, 3))}, 'robot_state': {'eef': {'pos': np.zeros(3)}}})
    assert out['pixels']['image'].shape == (1, 4, 4, 3)
    assert out['robot_state']['eef']['pos'].shape == (1, 3)


def test_after_the_env_ends_the_episode_further_ticks_are_no_ops():
    """robosuite raises on an action after its horizon; the wrapper must absorb the rest."""
    class _Ending(_FakeLibero):
        def step(self, a):
            self.actions.append(np.asarray(a))
            return {'x': 1}, 0.0, False, True, {'is_success': False}
    raw = _Ending()
    env = LiberoStepEnv(raw, postprocess=lambda a: np.asarray(a))
    env.reset()
    env.step(None)
    env.step(None)
    assert len(raw.actions) == 1
