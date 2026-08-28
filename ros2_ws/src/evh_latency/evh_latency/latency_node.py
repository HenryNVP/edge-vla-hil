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

Release timing is EVENT-DRIVEN: a single timer is armed at the head of the pending queue and
cancelled whenever the queue drains, rather than polling at a fixed tick. The polled version cost
~17% of a core per relay spinning on an empty heap (three relays on the observation path, on a
host also running MuJoCo at 200 Hz with camera renders), and quantized every release onto the tick
grid — ~0.5 ms of extra mean delay at 1 kHz, on top of the latency actually being requested.
Messages whose delay has already elapsed (latency_ms=0) are published straight from the
subscription callback, so the zero-latency baseline adds no scheduling delay at all.

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

# rcl will not take a zero-length timer period; the head of the queue is always strictly in the
# future by the time we arm (_drain pops everything already due), so this only floors the rounding.
_MIN_ARM_NS = 1_000


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
        self.declare_parameter('seed', 0)

        in_topic = self.get_parameter('input_topic').value
        out_topic = self.get_parameter('output_topic').value
        type_str = self.get_parameter('msg_type').value

        self.latency_ms = float(self.get_parameter('latency_ms').value)
        self.jitter_ms = float(self.get_parameter('jitter_ms').value)
        self.jitter_model = self.get_parameter('jitter_model').value
        self.drop_prob = float(self.get_parameter('drop_prob').value)
        self.reorder = bool(self.get_parameter('reorder').value)

        self._rng = random.Random(int(self.get_parameter('seed').value))
        self._heap: list[tuple[float, int, object]] = []     # (release_t, seq, msg)
        self._seq = itertools.count()
        self._last_release = 0.0

        msg_cls = get_message(type_str)
        self.pub = self.create_publisher(msg_cls, out_topic, qos_profile_sensor_data)
        self.sub = self.create_subscription(
            msg_cls, in_topic, self._on_msg, qos_profile_sensor_data)
        # period is rewritten per-message by _arm(); starts cancelled so an idle relay costs nothing
        self._timer = self.create_timer(1.0, self._drain)
        self._timer.cancel()

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
        # publishes immediately when the delay has already elapsed (the latency_ms=0 baseline),
        # otherwise just arms the timer for this message's release
        self._drain()

    # ---------------------------------------------------------------- drain
    def _drain(self) -> None:
        """Publish everything now due, then re-arm for the next release."""
        now = self._now_s()
        while self._heap and self._heap[0][0] <= now:
            _, _, msg = heapq.heappop(self._heap)
            self.pub.publish(msg)
        self._arm()

    def _arm(self) -> None:
        """Point the timer at the head of the queue; cancel it while the queue is empty.

        Assumes the single-threaded executor main() spins: _on_msg and _drain both mutate the heap
        and must not run concurrently. Give this node its own callback group if that ever changes.
        """
        if not self._heap:
            self._timer.cancel()
            return
        wait_ns = int((self._heap[0][0] - self._now_s()) * 1e9)
        self._timer.timer_period_ns = max(wait_ns, _MIN_ARM_NS)
        self._timer.reset()   # also clears the cancelled state

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
