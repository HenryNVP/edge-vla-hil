"""Tests for the absolute/delta action-mode cross-check between plant and policy.

The invariant: the plant's `absolute_actions` (a launch arg) and the policy's `absolute_actions`
(derived from the checkpoint) must agree. Nothing used to compare them, and a mismatch raises no
error anywhere — it just produces garbage motion and plausible-looking metrics. The controller now
announces its mode on a latched /policy/absolute and the plant checks it; the message wording is
what a confused user actually reads, so it is pinned here.
"""
import pytest
from conftest import integration, requires_ros2


@requires_ros2
@pytest.mark.parametrize('mode', [True, False])
def test_agreeing_modes_report_no_problem(mode):
    from evh_plant.plant_node import mode_mismatch_message

    assert mode_mismatch_message(mode, mode) is None


@requires_ros2
def test_delta_plant_with_absolute_policy_says_relaunch_absolute_true():
    from evh_plant.plant_node import mode_mismatch_message

    msg = mode_mismatch_message(plant_absolute=False, policy_absolute=True)
    assert msg is not None
    assert 'absolute:=true' in msg
    assert 'a world-frame pose read as a delta' in msg


@requires_ros2
def test_absolute_plant_with_delta_policy_says_relaunch_absolute_false():
    from evh_plant.plant_node import mode_mismatch_message

    msg = mode_mismatch_message(plant_absolute=True, policy_absolute=False)
    assert msg is not None
    assert 'absolute:=false' in msg
    assert 'a delta read as a world-frame pose' in msg


def _run_mode_handshake(ros, plant_absolute, announced):
    """Boot a plant, publish `announced` on the latched /policy/absolute, return the plant.

    strict_mode_check is off so a detected mismatch sets the flag instead of shutting rclpy down
    underneath the test fixture.
    """
    import time

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.parameter import Parameter
    from std_msgs.msg import Bool

    from evh_plant.plant_node import MODE_QOS, PlantNode

    plant = PlantNode(parameter_overrides=[
        Parameter('action_hz', value=50.0),
        Parameter('absolute_actions', value=plant_absolute),
        Parameter('strict_mode_check', value=False)])
    talker = rclpy.create_node('fake_controller')
    talker.create_publisher(Bool, '/policy/absolute', MODE_QOS).publish(Bool(data=announced))

    ex = SingleThreadedExecutor()
    ex.add_node(plant)
    ex.add_node(talker)
    end = time.time() + 3.0
    while time.time() < end and not plant.mode_mismatch:
        ex.spin_once(timeout_sec=0.05)

    talker.destroy_node()
    return plant


@integration
def test_plant_flags_a_mismatched_policy_mode_over_ros(ros):
    """End-to-end: the check is worthless if the topic name or QoS drifts, so exercise the wire."""
    plant = _run_mode_handshake(ros, plant_absolute=False, announced=True)
    assert plant.mode_mismatch is True
    plant.destroy_node()


@integration
def test_plant_stays_quiet_when_the_policy_mode_agrees(ros):
    plant = _run_mode_handshake(ros, plant_absolute=True, announced=True)
    assert plant.mode_mismatch is False
    plant.destroy_node()


@requires_ros2
def test_plant_and_controller_agree_on_the_mode_topic_qos():
    """A latched publisher only reaches a subscriber whose QoS is compatible; if these drift the
    check silently never fires, which is the exact failure it exists to prevent."""
    from evh_controller.controller_node import MODE_QOS as controller_qos
    from evh_plant.plant_node import MODE_QOS as plant_qos

    assert plant_qos.durability == controller_qos.durability
    assert plant_qos.reliability == controller_qos.reliability
    assert plant_qos.depth == controller_qos.depth
