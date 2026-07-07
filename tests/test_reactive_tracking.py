"""Contract tests for the reactive layer: waypoint in -> tracking /cmd/action out.

Exercises the two modes over real ROS topics:
  * tracking:    latches an absolute target from the local EE pose + delta, emits a clipped
                 7-dim OSC action toward it (gripper preserved), holds after target reached,
                 and clears its target on /episode/reset.
  * passthrough: forwards the raw delta scaled by passthrough_scale, gripper unscaled.
"""
import time

import numpy as np
import pytest

from conftest import requires_ros2


def _make_probe(ros, actions):
    import rclpy
    from sensor_msgs.msg import JointState

    probe = rclpy.create_node('reactive_probe')
    probe.create_subscription(
        JointState, '/cmd/action',
        lambda m: actions.append(np.asarray(m.position, dtype=float)), 10)
    return probe


def _spin_all(nodes, seconds):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    ex = SingleThreadedExecutor()
    for n in nodes:
        ex.add_node(n)
    end = time.time() + seconds
    while time.time() < end and rclpy.ok():
        ex.spin_once(timeout_sec=0.02)


def _publish_ee_pose(node, pos=(0.0, 0.0, 0.0)):
    from geometry_msgs.msg import PoseStamped

    pub = node.create_publisher(PoseStamped, '/obs/ee_pose', 10)
    msg = PoseStamped()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos
    msg.pose.orientation.w = 1.0
    return pub, msg


def _publish_waypoint(node, action):
    from sensor_msgs.msg import JointState

    pub = node.create_publisher(JointState, '/cmd/waypoint', 10)
    msg = JointState()
    msg.position = [float(v) for v in action]
    return pub, msg


@requires_ros2
def test_tracking_emits_action_toward_target(ros):
    from evh_reactive.reactive_node import ReactiveNode

    node = ReactiveNode()
    actions = []
    probe = _make_probe(ros, actions)
    ee_pub, ee_msg = _publish_ee_pose(probe)
    wp_pub, wp_msg = _publish_waypoint(probe, [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

    ee_pub.publish(ee_msg)
    _spin_all([node, probe], 0.2)   # EE pose must land before the waypoint can anchor
    wp_pub.publish(wp_msg)
    _spin_all([node, probe], 0.4)

    assert actions, 'reactive layer emitted no /cmd/action'
    a = actions[-1]
    assert a.shape == (7,)
    # target is +x of current pose -> positive x action; kp=1, err=0.5*pos_scale -> a_x ~ 0.5
    assert a[0] == pytest.approx(0.5, abs=0.05)
    assert abs(a[1]) < 1e-6 and abs(a[2]) < 1e-6
    assert a[6] == -1.0, 'gripper command must be preserved'
    node.destroy_node()
    probe.destroy_node()


@requires_ros2
def test_tracking_holds_at_target(ros):
    """Once the local EE pose reaches the latched target, the emitted action goes to ~zero."""
    from evh_reactive.reactive_node import ReactiveNode

    node = ReactiveNode()
    actions = []
    probe = _make_probe(ros, actions)
    ee_pub, ee_msg = _publish_ee_pose(probe)
    wp_pub, wp_msg = _publish_waypoint(probe, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ee_pub.publish(ee_msg)
    _spin_all([node, probe], 0.2)
    wp_pub.publish(wp_msg)
    _spin_all([node, probe], 0.2)

    # simulate the plant having reached the target (1.0 action * 0.05 pos_scale)
    ee_msg.pose.position.x = 0.05
    ee_pub.publish(ee_msg)
    actions.clear()
    _spin_all([node, probe], 0.3)

    assert actions
    assert np.allclose(actions[-1][:6], 0.0, atol=1e-6), 'should hold at the latched target'
    node.destroy_node()
    probe.destroy_node()


@requires_ros2
def test_episode_reset_clears_target(ros):
    from std_msgs.msg import Empty
    from evh_reactive.reactive_node import ReactiveNode

    node = ReactiveNode()
    actions = []
    probe = _make_probe(ros, actions)
    ee_pub, ee_msg = _publish_ee_pose(probe)
    wp_pub, wp_msg = _publish_waypoint(probe, [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    reset_pub = probe.create_publisher(Empty, '/episode/reset', 10)

    ee_pub.publish(ee_msg)
    _spin_all([node, probe], 0.2)
    wp_pub.publish(wp_msg)
    _spin_all([node, probe], 0.2)
    assert actions, 'sanity: tracking was active before the reset'

    reset_pub.publish(Empty())
    _spin_all([node, probe], 0.2)
    actions.clear()
    _spin_all([node, probe], 0.3)
    assert not actions, 'no stale target may survive an episode reset'
    node.destroy_node()
    probe.destroy_node()


@requires_ros2
def test_passthrough_scales_delta_not_gripper(ros):
    from rclpy.parameter import Parameter
    from evh_reactive.reactive_node import ReactiveNode

    node = ReactiveNode(parameter_overrides=[
        Parameter('passthrough', value=True),
        Parameter('passthrough_scale', value=0.1),
    ])
    actions = []
    probe = _make_probe(ros, actions)
    wp_pub, wp_msg = _publish_waypoint(probe, [1.0, -1.0, 0.5, 0.2, 0.0, 0.0, 1.0])

    wp_pub.publish(wp_msg)
    _spin_all([node, probe], 0.4)

    assert actions, 'passthrough emitted no /cmd/action'
    a = actions[-1]
    assert np.allclose(a[:6], np.array([1.0, -1.0, 0.5, 0.2, 0.0, 0.0]) * 0.1, atol=1e-9)
    assert a[6] == 1.0, 'gripper must pass through unscaled'
    node.destroy_node()
    probe.destroy_node()
