"""Smoke + behavior tests for the latency relay (the experimental instrument).

Marked integration: constructs a real relay node and spins it. Verifies the relay forwards messages end-to-end, that delay
sampling is seeded/reproducible, and that drop_prob=1.0 drops everything.
"""
import random

import pytest
from conftest import integration


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


@integration
def test_relay_forwards_messages(ros):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import String

    relay = _relay(rclpy, input_topic='/in', output_topic='/out',
                   msg_type='std_msgs/msg/String', latency_ms=0.0)
    helper = rclpy.create_node('test_helper')
    pub = helper.create_publisher(String, '/in', 10)
    received: list[str] = []
    # the relay publishes best-effort (sensor-data QoS); the probe must match or DDS won't pair
    helper.create_subscription(
        String, '/out', lambda m: received.append(m.data), qos_profile_sensor_data)

    ex = SingleThreadedExecutor()
    ex.add_node(relay)
    ex.add_node(helper)

    for i in range(5):
        pub.publish(String(data=f'msg{i}'))

    assert _drain_executor(ex, lambda: len(received) >= 5)
    assert received == [f'msg{i}' for i in range(5)]

    relay.destroy_node()
    helper.destroy_node()


@integration
def test_drop_prob_one_drops_all(ros):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import String

    relay = _relay(rclpy, input_topic='/in2', output_topic='/out2',
                   msg_type='std_msgs/msg/String', latency_ms=0.0, drop_prob=1.0)
    helper = rclpy.create_node('test_helper2')
    pub = helper.create_publisher(String, '/in2', 10)
    received: list[str] = []
    helper.create_subscription(
        String, '/out2', lambda m: received.append(m.data), qos_profile_sensor_data)

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


@integration
def test_timer_is_idle_until_a_message_arrives(ros):
    """The drain timer is event-driven: cancelled while the queue is empty, so an idle relay
    costs no CPU. A polled tick burned ~17% of a core per relay spinning on an empty heap."""
    import rclpy
    from std_msgs.msg import String

    relay = _relay(rclpy, input_topic='/i3', output_topic='/o3',
                   msg_type='std_msgs/msg/String', latency_ms=50.0)
    assert relay._timer.is_canceled()

    relay._on_msg(String(data='x'))
    assert not relay._timer.is_canceled(), 'a pending message must arm the timer'
    # armed for roughly the configured delay, not a fixed tick
    assert relay._timer.timer_period_ns == pytest.approx(50e6, rel=0.2)

    relay.destroy_node()


@integration
def test_timer_is_cancelled_again_once_the_queue_drains(ros):
    import time

    import rclpy
    from std_msgs.msg import String

    relay = _relay(rclpy, input_topic='/i4', output_topic='/o4',
                   msg_type='std_msgs/msg/String', latency_ms=20.0)
    relay._on_msg(String(data='x'))
    assert not relay._timer.is_canceled()

    end = time.time() + 2.0
    while time.time() < end and relay._heap:
        rclpy.spin_once(relay, timeout_sec=0.01)

    assert relay._heap == [], 'message was never released'
    assert relay._timer.is_canceled(), 'timer must go idle again after the queue drains'
    relay.destroy_node()


@integration
def test_zero_latency_publishes_without_waiting_for_a_tick(ros):
    """latency_ms=0 must add no scheduling delay: the message goes out from the subscription
    callback itself, not on a later timer tick."""
    import rclpy
    from std_msgs.msg import String

    relay = _relay(rclpy, input_topic='/i5', output_topic='/o5',
                   msg_type='std_msgs/msg/String', latency_ms=0.0)
    sent: list[object] = []
    relay.pub.publish = lambda m: sent.append(m)   # observe the release, skip DDS

    relay._on_msg(String(data='now'))
    assert len(sent) == 1, 'zero-delay message should be released inline'
    assert relay._heap == []
    assert relay._timer.is_canceled()
    relay.destroy_node()


@integration
def test_release_order_is_preserved_under_jitter(ros):
    """reorder=False must not let a low-jitter sample overtake an earlier message."""
    import rclpy
    from std_msgs.msg import String

    relay = _relay(rclpy, input_topic='/i6', output_topic='/o6',
                   msg_type='std_msgs/msg/String', latency_ms=50.0, jitter_ms=20.0,
                   reorder=False, seed=7)
    for i in range(20):
        relay._on_msg(String(data=str(i)))

    releases = [entry[0] for entry in sorted(relay._heap, key=lambda e: e[1])]
    assert releases == sorted(releases), 'messages must not overtake each other'
    relay.destroy_node()


@integration
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
