"""Plant episode bookkeeping and action shaping — stub env, no robosuite (or GPU) needed.

Covers the plant logic that test_plant_hold.py and test_mode_crosscheck.py leave open:

  * `/eval/success` must be published on BOTH task success and horizon timeout (True/False) —
    the recorder needs both or the success rate is computed over the successes only.
  * a reset must drop the last command, or the first physics step of the new episode replays a
    stale action against a freshly randomised scene.
  * `_current_action` pads/truncates to the env's action dim; nothing upstream guarantees width.
  * `_set_control_delta` is the mechanism behind invariant #1 (absolute mode) and has to work on
    both robosuite config shapes, silently doing nothing on neither.

The import needs rclpy, hence the ros2 mark; the node itself is never constructed.
"""
import types

import numpy as np
import pytest
from conftest import requires_ros2


def _stub_plant(**overrides):
    """A PlantNode-shaped stub carrying only what the method under test touches."""
    published: list[bool] = []
    resets: list[int] = []
    stub = types.SimpleNamespace(
        _action_dim=7,
        _last_action=None,
        _obs={'robot0_eef_pos': np.zeros(3), 'robot0_eef_quat': np.array([0.0, 0.0, 0.0, 1.0])},
        absolute_actions=False,
        pub_success=types.SimpleNamespace(publish=lambda m: published.append(bool(m.data))),
        pub_reset=types.SimpleNamespace(publish=lambda _m: resets.append(1)),
        published=published,
        resets=resets,
    )
    stub.__dict__.update(overrides)
    return stub


class FakeEnv:
    """Minimal robosuite stand-in: records the actions it was stepped with."""

    def __init__(self, success=False, done=False):
        self.success = success
        self.done = done
        self.stepped: list[np.ndarray] = []
        self.resets = 0

    def step(self, action):
        self.stepped.append(np.asarray(action))
        return {'obs': 'after-step'}, 0.0, self.done, {}

    def _check_success(self):
        return self.success

    def reset(self):
        self.resets += 1
        return {'obs': 'after-reset'}


# ------------------------------------------------------------------ episode outcomes
@requires_ros2
@pytest.mark.parametrize('success,done,expected', [
    (True, False, True),     # task solved
    (False, True, False),    # robosuite horizon hit -> timeout, counts as a failure
    (True, True, True),      # both at once: still a success
])
def test_episode_outcome_is_published_for_success_and_timeout(success, done, expected):
    from evh_plant.plant_node import PlantNode

    stub = _stub_plant(_env=FakeEnv(success=success, done=done))
    stub._current_action = lambda: np.zeros(7, np.float32)
    stub._reset_episode = lambda: PlantNode._reset_episode(stub)

    PlantNode._step_physics(stub)

    assert stub.published == [expected], 'both outcomes must reach /eval/success'
    assert stub._env.resets == 1, 'a closed episode must reset the env'


@requires_ros2
def test_a_running_episode_publishes_nothing_and_does_not_reset():
    from evh_plant.plant_node import PlantNode

    stub = _stub_plant(_env=FakeEnv())
    stub._current_action = lambda: np.zeros(7, np.float32)
    stub._reset_episode = lambda: pytest.fail('reset mid-episode')

    PlantNode._step_physics(stub)

    assert stub.published == []
    assert stub._obs == {'obs': 'after-step'}, 'obs must advance with the sim'


@requires_ros2
def test_step_is_a_noop_before_the_env_exists():
    """robosuite-less degraded mode: the node still spins, it just has no physics."""
    from evh_plant.plant_node import PlantNode

    stub = _stub_plant(_env=None)
    stub._current_action = lambda: pytest.fail('no env, nothing to command')
    PlantNode._step_physics(stub)
    assert stub.published == []


@requires_ros2
def test_reset_drops_the_last_command_and_announces_the_boundary():
    """A carried-over action would be applied to a freshly randomised scene, and downstream
    nodes (executor chunk buffers, reactive setpoint) need the /episode/reset to clear theirs."""
    from evh_plant.plant_node import PlantNode

    stub = _stub_plant(_env=FakeEnv(), _last_action=np.ones(7, np.float32))
    PlantNode._reset_episode(stub)

    assert stub._last_action is None
    assert stub._obs == {'obs': 'after-reset'}
    assert stub.resets == [1], 'downstream nodes were never told the episode ended'


# -------------------------------------------------------------------- action shaping
@requires_ros2
def test_current_action_falls_back_to_the_hold_before_the_first_command():
    from evh_plant.plant_node import PlantNode

    stub = _stub_plant(_last_action=None)
    stub._hold_action = lambda: np.full(7, 0.5, np.float32)
    assert np.all(PlantNode._current_action(stub) == 0.5)


@requires_ros2
@pytest.mark.parametrize('width', [3, 6, 7, 9])
def test_current_action_is_reshaped_to_the_env_action_dim(width):
    """Waypoints are a 7-dim JointState by contract, but the env's dim is the env's business —
    pad short, truncate long, never hand robosuite a mis-sized array."""
    from evh_plant.plant_node import PlantNode

    stub = _stub_plant(_last_action=np.arange(width, dtype=np.float64))
    action = PlantNode._current_action(stub)

    assert action.shape == (7,)
    assert action.dtype == np.float32
    assert np.array_equal(action[:min(width, 7)], np.arange(min(width, 7)))
    assert np.all(action[width:] == 0.0)   # padding only, no garbage


# ---------------------------------------------------------------- controller config
@requires_ros2
@pytest.mark.parametrize('value', [True, False])
def test_set_control_delta_on_the_flat_robosuite_14_config(value):
    from evh_plant.plant_node import _set_control_delta

    cfg = {'type': 'OSC_POSE', 'control_delta': not value}
    _set_control_delta(cfg, value)
    assert cfg['control_delta'] is value


@requires_ros2
@pytest.mark.parametrize('value', [True, False])
def test_set_control_delta_reaches_into_a_composite_15_config(value):
    from evh_plant.plant_node import _set_control_delta

    cfg = {'body_parts': {'right': {'type': 'OSC_POSE', 'control_delta': not value},
                          'base': {'type': 'JOINT_VELOCITY'},
                          'gripper': 'not-a-dict'}}
    _set_control_delta(cfg, value)

    assert cfg['body_parts']['right']['control_delta'] is value
    assert 'control_delta' not in cfg['body_parts']['base'], 'only OSC parts take control_delta'


@requires_ros2
def test_set_control_delta_tolerates_a_config_it_does_not_recognise():
    from evh_plant.plant_node import _set_control_delta

    cfg = {'type': 'JOINT_POSITION'}
    _set_control_delta(cfg, False)      # must not raise
    assert cfg == {'type': 'JOINT_POSITION'}
