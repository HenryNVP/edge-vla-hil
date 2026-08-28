"""Unit tests for controller_node.py — the tick, the waypoint packing, the reset.

The node is thin (the strategies, worker, obs buffer and backends all have their own files and
tests), but what remains is not nothing, and until now it was covered only incidentally: by
test_nodes_boot (it starts) and test_mode_crosscheck (it announces its action mode). What is
guarded here is what the node itself decides:

  * the control-tick counter `self._t`, which is the unit `delay_steps` is measured in and what
    RTC's forecast consumes — it must count ticks the POLICY ran, not wall-clock timer firings,
  * `_to_waypoint`'s coercion to the 7-dim action contract,
  * hold semantics: a strategy returning None means "publish no waypoint", not "publish zeros",
    which in absolute mode would be a world-origin lurch (invariant 2).

Driven through a stub, so no ROS graph is spun up; the import needs message types, hence the mark.
"""
import types

import numpy as np
import pytest
from conftest import requires_ros2


class FakeExecutor:
    """Returns a scripted action per step and optionally reports one chunk arrival."""

    def __init__(self, action=None, metrics=None):
        self.action = action
        self.metrics = metrics
        self.steps: list[int] = []
        self.resets = 0

    def step(self, obs, t):
        self.steps.append(t)
        return self.action

    def take_arrival_metrics(self):
        m, self.metrics = self.metrics, None
        return m

    def reset(self):
        self.resets += 1


def _stub_controller(obs_sample=None, action=None, metrics=None):
    """A ControllerNode-shaped stub carrying only what the tick touches."""
    from builtin_interfaces.msg import Time

    published = {'waypoint': [], 'latency': [], 'delay': []}

    def _pub(key):
        return types.SimpleNamespace(publish=lambda m: published[key].append(m))

    cleared = []
    stub = types.SimpleNamespace(
        obs=types.SimpleNamespace(sample=lambda: obs_sample,
                                  clear=lambda: cleared.append(1)),
        chunk_executor=FakeExecutor(action=action, metrics=metrics),
        _t=0,
        pub_waypoint=_pub('waypoint'),
        pub_latency=_pub('latency'),
        pub_delay=_pub('delay'),
        get_clock=lambda: types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(to_msg=lambda: Time(sec=1, nanosec=0))),
        published=published,
        cleared=cleared,
    )
    # _tick calls it as a method; bind the real implementation to the stub
    from evh_controller.controller_node import ControllerNode
    stub._to_waypoint = lambda a: ControllerNode._to_waypoint(stub, a)
    return stub


def _obs():
    return {'agentview': np.zeros((1, 4, 4, 3), np.uint8), 'proprio': np.zeros((1, 9), np.float32)}


# ------------------------------------------------------------------------ the tick
@requires_ros2
def test_the_tick_does_nothing_before_the_first_observations():
    from evh_controller.controller_node import ControllerNode

    stub = _stub_controller(obs_sample=None, action=np.ones(7))
    ControllerNode._tick(stub)

    assert stub.published['waypoint'] == []
    assert stub.chunk_executor.steps == [], 'policy was stepped without an observation'


@requires_ros2
def test_the_tick_counter_only_advances_when_the_policy_actually_ran():
    """`_t` is the clock delay_steps is denominated in. If it ticked while the controller was
    still waiting for its first observations, every measured delay would be inflated by the
    startup gap and RTC would forecast against a number that never happened."""
    from evh_controller.controller_node import ControllerNode

    stub = _stub_controller(obs_sample=None)
    for _ in range(5):
        ControllerNode._tick(stub)
    assert stub._t == 0

    stub.obs.sample = _obs
    for _ in range(3):
        ControllerNode._tick(stub)
    assert stub._t == 3
    assert stub.chunk_executor.steps == [0, 1, 2], 'policy saw the wrong tick indices'


@requires_ros2
def test_an_action_is_published_as_a_waypoint():
    from evh_controller.controller_node import ControllerNode

    stub = _stub_controller(obs_sample=_obs(), action=np.arange(7, dtype=float))
    ControllerNode._tick(stub)

    assert len(stub.published['waypoint']) == 1
    assert list(stub.published['waypoint'][0].position) == [0, 1, 2, 3, 4, 5, 6]


@requires_ros2
def test_a_hold_publishes_no_waypoint_at_all():
    """None means hold: the reactive layer keeps tracking its latched target. Publishing zeros
    instead would command the world origin in absolute mode (invariant 2)."""
    from evh_controller.controller_node import ControllerNode

    stub = _stub_controller(obs_sample=_obs(), action=None)
    ControllerNode._tick(stub)

    assert stub.published['waypoint'] == []
    assert stub._t == 1, 'a hold is still a control step'


@requires_ros2
def test_arrival_metrics_are_published_on_the_tick_a_chunk_lands():
    from evh_controller.controller_node import ControllerNode

    stub = _stub_controller(obs_sample=_obs(), action=np.zeros(7), metrics=(12.5, 3))
    ControllerNode._tick(stub)

    assert [m.data for m in stub.published['latency']] == pytest.approx([12.5])
    assert [m.data for m in stub.published['delay']] == pytest.approx([3.0])

    ControllerNode._tick(stub)   # nothing arrived this tick
    assert len(stub.published['latency']) == 1, 'metrics republished without an arrival'


# ------------------------------------------------------------- the waypoint contract
@requires_ros2
@pytest.mark.parametrize('width', [3, 6, 7, 10])
def test_waypoints_are_coerced_to_the_seven_dim_contract(width):
    """/cmd/waypoint is [pos/dpos(3), axis-angle/drot(3), gripper]. The reactive layer pads too,
    but a wrong width leaving here would silently shift the gripper into a rotation slot."""
    from evh_controller.controller_node import ControllerNode

    stub = _stub_controller()
    msg = ControllerNode._to_waypoint(stub, np.arange(width, dtype=float))

    assert len(msg.position) == 7
    assert list(msg.position[:min(width, 7)]) == list(range(min(width, 7)))


@requires_ros2
def test_a_non_flat_action_is_flattened_not_rejected():
    """A [1, A] chunk row is a shape strategies can plausibly hand back."""
    from evh_controller.controller_node import ControllerNode

    stub = _stub_controller()
    msg = ControllerNode._to_waypoint(stub, np.arange(7, dtype=float).reshape(1, 7))
    assert list(msg.position) == [0, 1, 2, 3, 4, 5, 6]


@requires_ros2
def test_the_waypoint_carries_a_timestamp():
    """The recorder measures waypoint_hz off these; an unstamped message breaks the metric."""
    from evh_controller.controller_node import ControllerNode

    stub = _stub_controller()
    msg = ControllerNode._to_waypoint(stub, np.zeros(7))
    assert msg.header.stamp.sec == 1


# -------------------------------------------------------------------------- resets
@requires_ros2
def test_an_episode_reset_clears_every_piece_of_episode_state():
    """Chunks, the observation history and the tick counter must all go — a surviving chunk would
    splice pre-reset actions onto a freshly randomised scene."""
    from evh_controller.controller_node import ControllerNode

    stub = _stub_controller(obs_sample=_obs(), action=np.zeros(7))
    ControllerNode._tick(stub)
    ControllerNode._tick(stub)
    assert stub._t == 2

    ControllerNode._on_episode_reset(stub, None)

    assert stub._t == 0
    assert stub.chunk_executor.resets == 1
    assert stub.cleared == [1], 'observation history survived the episode boundary'


# ------------------------------------------------------------ observation callbacks
@requires_ros2
def test_image_callbacks_reshape_the_flat_ros_buffer():
    from sensor_msgs.msg import Image

    from evh_controller.controller_node import ControllerNode

    stub = _stub_controller()
    stub.obs = types.SimpleNamespace(image=None, wrist=None, proprio=None)
    msg = Image(height=4, width=5, encoding='rgb8')
    msg.data = bytes(range(4 * 5 * 3))

    ControllerNode._on_image(stub, msg)
    ControllerNode._on_wrist(stub, msg)

    assert stub.obs.image.shape == (4, 5, 3)
    assert stub.obs.wrist.shape == (4, 5, 3)
    assert stub.obs.image.dtype == np.uint8


@requires_ros2
def test_proprio_callback_keeps_the_nine_dim_layout():
    from sensor_msgs.msg import JointState

    from evh_controller.controller_node import ControllerNode

    stub = _stub_controller()
    stub.obs = types.SimpleNamespace(image=None, wrist=None, proprio=None)
    ControllerNode._on_proprio(stub, JointState(position=[float(i) for i in range(9)]))

    assert stub.obs.proprio.shape == (9,)
    assert stub.obs.proprio.dtype == np.float32
