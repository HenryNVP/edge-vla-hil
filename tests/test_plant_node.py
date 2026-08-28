"""Unit tests for plant_node.py — episodes, action shaping, and the hold action.

Everything PlantNode itself decides, driven through a stub so no robosuite env (and no GPU) is
needed. The env construction, observation packing and video recording that used to live in this
module have their own files now (test_env_factory, test_messages, test_video); the
node's cross-check handshake with the controller is test_mode_crosscheck.

What is guarded here:

  * `/eval/success` must be published on BOTH task success and horizon timeout (True/False) —
    the recorder needs both or the success rate is computed over the successes only.
  * a reset must drop the last command, or the first physics step of the new episode replays a
    stale action against a freshly randomised scene.
  * `_current_action` pads/truncates to the env's action dim; nothing upstream guarantees width.
  * `_hold_action` is mode-aware. THE regression: in absolute mode (OSC control_delta=False) a
    zero action is an ABSOLUTE pose target at the world origin, so defaulting to zeros before the
    first /cmd/action yanked the arm off the table on every reset. Delta mode stays zeros.

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


# ------------------------------------------------------------------- hold action
@requires_ros2
def test_hold_action_absolute_commands_current_pose_not_origin():
    """The regression guard: a zero here is a world-origin target, not 'stay put'."""
    from evh_plant.plant_node import PlantNode, _quat_to_axisangle

    ee_pos = np.array([0.4, -0.1, 1.05])
    ee_quat = np.array([0.0, 0.0, 0.0, 1.0])
    stub = _stub_plant(absolute_actions=True,
                       _obs={'robot0_eef_pos': ee_pos, 'robot0_eef_quat': ee_quat})

    hold = PlantNode._hold_action(stub)

    assert hold.shape == (7,)
    assert np.allclose(hold[:3], ee_pos)                      # NOT the origin
    assert np.allclose(hold[3:6], _quat_to_axisangle(ee_quat))
    assert hold[6] == 0.0                                     # gripper neutral
    assert not np.allclose(hold[:3], 0.0)                     # the actual regression guard


@requires_ros2
def test_hold_action_delta_mode_is_zeros():
    from evh_plant.plant_node import PlantNode

    stub = _stub_plant(absolute_actions=False,
                       _obs={'robot0_eef_pos': np.array([0.4, -0.1, 1.05])})
    assert np.allclose(PlantNode._hold_action(stub), np.zeros(7))


@requires_ros2
def test_hold_action_absolute_refuses_to_default_a_missing_pose():
    """A .get() default here would silently rebuild the origin lurch; raising is the point."""
    from evh_plant.plant_node import PlantNode

    stub = _stub_plant(absolute_actions=True, _obs={'unrelated': 1})
    with pytest.raises(RuntimeError, match='world origin'):
        PlantNode._hold_action(stub)


@requires_ros2
def test_hold_action_absolute_without_obs_falls_back_to_zeros():
    """Before the first obs (self._obs is None) there is nothing to hold to; zeros is the only
    safe default. The origin-lurch window here is unavoidable but momentary (pre-first-step)."""
    from evh_plant.plant_node import PlantNode

    stub = _stub_plant(absolute_actions=True, _obs=None)
    assert np.allclose(PlantNode._hold_action(stub), np.zeros(7))


@requires_ros2
def test_quat_to_axisangle_matches_reactive_transforms():
    """plant_node duplicates this helper rather than depending on evh_reactive (the two packages
    deploy to different machines). Duplicated math has to stay identical math."""
    from evh_plant.plant_node import _quat_to_axisangle
    from evh_reactive.transforms import axisangle_to_quat, quat_to_axisangle

    assert np.allclose(_quat_to_axisangle([0.0, 0.0, 0.0, 1.0]), np.zeros(3))
    for aa in ([0.3, -0.1, 0.7], [np.pi / 2, 0.0, 0.0], [0.0, 2.5, 0.0]):
        q = axisangle_to_quat(np.asarray(aa))
        assert np.allclose(_quat_to_axisangle(q), quat_to_axisangle(q), atol=1e-9)
