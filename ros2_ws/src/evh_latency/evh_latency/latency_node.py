"""Latency / jitter / packet-drop relay — the core experimental instrument.

A type-generic relay: subscribe to `input_topic`, hold each message for a configurable delay,
then republish on `output_topic`. Models edge network conditions reproducibly and in software, so
the same sweep runs whether the controller is on the same host or across the Ethernet link.

Effects (composable):
  * latency_ms  : constant base one-way delay.
  * jitter_ms   : added zero-mean noise; distribution selected by `jitter_model`.
  * drop_prob   : probability a message is dropped entirely (packet loss).
  * reorder     : if False (default), enforce monotonic release ordering even when jitter would
                  otherwise reorder messages (TCP-like); if True, allow reordering (UDP-like).

Determinism: every run is seeded so a benchmark sweep is exactly reproducible.

Usage (per topic): launch one relay per link you want to degrade, e.g. the observation path
  /obs/image -> /obs/image/delayed and /obs/joint_state -> /obs/joint_state/delayed.
The controller subscribes to the /delayed topics via remap; it is unaware of the relay.
"""
from __future__ import annotations

import heapq
import itertools
import random

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message


class LatencyNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__('evh_latency', **kwargs)

        self.declare_parameter('input_topic', '/obs/image')
        self.declare_parameter('output_topic', '/obs/image/delayed')
        self.declare_parameter('msg_type', 'sensor_msgs/msg/Image')
        self.declare_parameter('latency_ms', 0.0)
        self.declare_parameter('jitter_ms', 0.0)
        self.declare_parameter('jitter_model', 'gaussian')   # gaussian | uniform
        self.declare_parameter('drop_prob', 0.0)
        self.declare_parameter('reorder', False)
        self.declare_parameter('tick_hz', 1000.0)
        self.declare_parameter('seed', 0)

        in_topic = self.get_parameter('input_topic').value
        out_topic = self.get_parameter('output_topic').value
        type_str = self.get_parameter('msg_type').value

        self.latency_ms = float(self.get_parameter('latency_ms').value)
        self.jitter_ms = float(self.get_parameter('jitter_ms').value)
        self.jitter_model = self.get_parameter('jitter_model').value
        self.drop_prob = float(self.get_parameter('drop_prob').value)
        self.reorder = bool(self.get_parameter('reorder').value)
        self.tick_hz = float(self.get_parameter('tick_hz').value)

        self._rng = random.Random(int(self.get_parameter('seed').value))
        self._heap: list[tuple[float, int, object]] = []     # (release_t, seq, msg)
        self._seq = itertools.count()
        self._last_release = 0.0

        msg_cls = get_message(type_str)
        self.pub = self.create_publisher(msg_cls, out_topic, qos_profile_sensor_data)
        self.sub = self.create_subscription(
            msg_cls, in_topic, self._on_msg, qos_profile_sensor_data)
        self.create_timer(1.0 / self.tick_hz, self._drain)

        self.get_logger().info(
            f'evh_latency: {in_topic} -> {out_topic} [{type_str}] '
            f'lat={self.latency_ms}ms jitter={self.jitter_ms}ms drop={self.drop_prob}')

    # --------------------------------------------------------------- ingest
    def _on_msg(self, msg) -> None:
        if self.drop_prob > 0.0 and self._rng.random() < self.drop_prob:
            return  # dropped

        delay_s = self._sample_delay_ms() / 1e3
        now = self._now_s()
        release = now + delay_s
        if not self.reorder:
            release = max(release, self._last_release)   # preserve order (no overtaking)
            self._last_release = release
        heapq.heappush(self._heap, (release, next(self._seq), msg))

    # ---------------------------------------------------------------- drain
    def _drain(self) -> None:
        now = self._now_s()
        while self._heap and self._heap[0][0] <= now:
            _, _, msg = heapq.heappop(self._heap)
            self.pub.publish(msg)

    # -------------------------------------------------------------- helpers
    def _sample_delay_ms(self) -> float:
        d = self.latency_ms
        if self.jitter_ms > 0.0:
            if self.jitter_model == 'uniform':
                d += self._rng.uniform(-self.jitter_ms, self.jitter_ms)
            else:
                d += self._rng.gauss(0.0, self.jitter_ms)
        return max(0.0, d)

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LatencyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
