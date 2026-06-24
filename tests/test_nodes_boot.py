"""Boot smoke test: every node constructs, spins briefly, and advertises its contract topics.

Marked ros2. This is the cheap CI guard that catches import errors, bad QoS, typos in topic
names, and broken parameter declarations across the whole graph before any real logic is wired in.
"""
import pytest

from conftest import requires_ros2, spin_until


def _topic_names(node):
    return {name for name, _types in node.get_topic_names_and_types()}


@requires_ros2
def test_plant_boots_and_publishes(ros):
    import rclpy
    from sensor_msgs.msg import Image, JointState
    from evh_plant.plant_node import PlantNode

    node = PlantNode()
    got = {'image': False, 'joint': False}
    sub_node = rclpy.create_node('plant_probe')
    sub_node.create_subscription(Image, '/obs/image', lambda _m: got.update(image=True), 10)
    sub_node.create_subscription(
        JointState, '/obs/joint_state', lambda _m: got.update(joint=True), 10)

    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(sub_node)

    import time
    end = time.time() + 5.0
    while time.time() < end and not all(got.values()):
        ex.spin_once(timeout_sec=0.05)

    assert got['image'], 'plant did not publish /obs/image'
    assert got['joint'], 'plant did not publish /obs/joint_state'
    node.destroy_node()
    sub_node.destroy_node()


@requires_ros2
def test_controller_boots(ros):
    from evh_controller.controller_node import ControllerNode
    node = ControllerNode()  # pytorch stub backend by default
    assert '/cmd/waypoint' in _topic_names(node)
    assert spin_until(node, lambda: True, timeout=0.5)  # spins without raising
    node.destroy_node()


@requires_ros2
def test_reactive_boots(ros):
    from evh_reactive.reactive_node import ReactiveNode
    node = ReactiveNode()
    assert spin_until(node, lambda: True, timeout=0.5)
    node.destroy_node()


@requires_ros2
def test_reactive_passthrough_param(ros):
    import rclpy
    from rclpy.parameter import Parameter
    from evh_reactive.reactive_node import ReactiveNode
    node = ReactiveNode(parameter_overrides=[Parameter('passthrough', value=True)])
    assert node.passthrough is True
    node.destroy_node()


@requires_ros2
def test_latency_boots(ros):
    from evh_latency.latency_node import LatencyNode
    node = LatencyNode()
    assert spin_until(node, lambda: True, timeout=0.5)
    node.destroy_node()
