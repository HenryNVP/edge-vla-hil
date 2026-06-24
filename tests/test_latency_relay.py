"""Smoke + behavior tests for the latency relay (the experimental instrument).

Marked ros2: needs rclpy + std_msgs. Verifies the relay forwards messages end-to-end, that delay
sampling is seeded/reproducible, and that drop_prob=1.0 drops everything.
"""
import random

import pytest

from conftest import requires_ros2


def _relay(rclpy, **overrides):
    """Construct a LatencyNode with parameter overrides (read in its __init__)."""
    from rclpy.parameter import Parameter
    from evh_latency.latency_node import LatencyNode
    params = [Parameter(k, value=v) for k, v in overrides.items()]
    return LatencyNode(parameter_overrides=params)


def _drain_executor(ex, predicate, timeout=5.0):
    import time
    end = time.time() + timeout
    while time.time() < end:
        ex.spin_once(timeout_sec=0.02)
        if predicate():
            return True
    return predicate()


@requires_ros2
def test_relay_forwards_messages(ros):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import String

    relay = _relay(rclpy, input_topic='/in', output_topic='/out',
                   msg_type='std_msgs/msg/String', latency_ms=0.0)
    helper = rclpy.create_node('test_helper')
    pub = helper.create_publisher(String, '/in', 10)
    received: list[str] = []
    helper.create_subscription(String, '/out', lambda m: received.append(m.data), 10)

    ex = SingleThreadedExecutor()
    ex.add_node(relay)
    ex.add_node(helper)

    for i in range(5):
        pub.publish(String(data=f'msg{i}'))

    assert _drain_executor(ex, lambda: len(received) >= 5)
    assert received == [f'msg{i}' for i in range(5)]

    relay.destroy_node()
    helper.destroy_node()


@requires_ros2
def test_drop_prob_one_drops_all(ros):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import String

    relay = _relay(rclpy, input_topic='/in2', output_topic='/out2',
                   msg_type='std_msgs/msg/String', latency_ms=0.0, drop_prob=1.0)
    helper = rclpy.create_node('test_helper2')
    pub = helper.create_publisher(String, '/in2', 10)
    received: list[str] = []
    helper.create_subscription(String, '/out2', lambda m: received.append(m.data), 10)

    ex = SingleThreadedExecutor()
    ex.add_node(relay)
    ex.add_node(helper)
    for i in range(5):
        pub.publish(String(data=str(i)))

    # spin a bounded number of times; nothing should ever arrive
    for _ in range(50):
        ex.spin_once(timeout_sec=0.02)
    assert received == []

    relay.destroy_node()
    helper.destroy_node()


@requires_ros2
def test_delay_sampling_is_seeded(ros):
    import rclpy
    relay = _relay(rclpy, input_topic='/a', output_topic='/b',
                   msg_type='std_msgs/msg/String', latency_ms=50.0, jitter_ms=10.0, seed=42)
    relay._rng = random.Random(42)
    first = [relay._sample_delay_ms() for _ in range(10)]
    relay._rng = random.Random(42)
    second = [relay._sample_delay_ms() for _ in range(10)]

    assert first == second                       # reproducible
    assert all(d >= 0.0 for d in first)          # never negative
    relay.destroy_node()
