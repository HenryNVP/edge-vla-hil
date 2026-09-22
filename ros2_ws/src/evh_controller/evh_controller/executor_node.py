"""Robot-side chunk executor node: the strategy runs next to the robot, inference across the link.

Wiring only; the behaviour lives in `remote_worker.py` (the inference proxy with its timeout),
`chunk_codec.py` (the wire format) and `chunk_executor/` (the strategies, unchanged). Launched
with `executor:=robot`, paired with a `controller_node` in the same mode serving chunks.

    /policy/info    (latched)  -> the policy's shape; the strategy is built when it arrives
    tick (control_hz)          -> executor.step() -> /cmd/waypoint (local, to the reactive layer)
                                  RemoteWorker.try_request -> /policy/request   (uplink)
    /cmd/chunk                 -> RemoteWorker.deliver                          (downlink)
    /episode/reset             -> executor.reset()

Metrics, named as in the policy-side placement so the recorder reads both the same way:
  /metrics/inference_ms   the server's compute time, carried in the chunk
  /metrics/delay_steps    request -> arrival in robot ticks: here the WHOLE round trip
  /metrics/chunk_age_ms   downlink delay of each chunk (send stamp to arrival): d_act
  /metrics/request_lost   1.0 per request given up on after the timeout
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float32

from evh_controller.chunk_codec import CodecError, Request, decode_chunk, encode_request
from evh_controller.chunk_executor import make_executor
from evh_controller.remote_worker import PolicyInfo, RemoteWorker

# Must match controller_node's profiles: DDS pairs nothing across mismatched QoS, silently.
MODE_QOS = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
WAYPOINT_QOS = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                          history=QoSHistoryPolicy.KEEP_LAST)


def policy_info_from(values) -> PolicyInfo:
    """Decode /policy/info: [chunk_size, action_dim, guided_resampling, absolute_actions]."""
    v = list(values)
    if len(v) != 4:
        raise ValueError(f'/policy/info carries {len(v)} values, expected 4')
    return PolicyInfo(chunk_size=int(v[0]), action_dim=int(v[1]),
                      guided_resampling=bool(v[2]), absolute_actions=bool(v[3]))


class ExecutorNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__('evh_executor', **kwargs)

        self.declare_parameter('strategy', 'synchronous')
        self.declare_parameter('control_hz', 20.0)
        self.declare_parameter('timeout_factor', 2.0)    # x slowest recent round trip
        self.declare_parameter('min_timeout_s', 0.25)
        self.declare_parameter('first_timeout_s', 10.0)  # before any reply (model warm-up)

        self.strategy = self.get_parameter('strategy').value
        self.control_hz = float(self.get_parameter('control_hz').value)
        self.info: PolicyInfo | None = None
        self.worker: RemoteWorker | None = None
        self.chunk_executor = None
        self._t = 0

        self.create_subscription(JointState, '/policy/info', self._on_info, MODE_QOS)
        self.create_subscription(JointState, '/cmd/chunk', self._on_chunk, WAYPOINT_QOS)
        self.create_subscription(Empty, '/episode/reset', self._on_episode_reset, 10)

        self.pub_request = self.create_publisher(JointState, '/policy/request', WAYPOINT_QOS)
        self.pub_waypoint = self.create_publisher(JointState, '/cmd/waypoint', WAYPOINT_QOS)
        self.pub_latency = self.create_publisher(Float32, '/metrics/inference_ms', 10)
        self.pub_delay = self.create_publisher(Float32, '/metrics/delay_steps', 10)
        self.pub_chunk_age = self.create_publisher(Float32, '/metrics/chunk_age_ms', 10)
        self.pub_lost = self.create_publisher(Float32, '/metrics/request_lost', 10)

        self.create_timer(1.0 / self.control_hz, self._tick)
        self.get_logger().info(
            f'evh_executor up (robot side): strategy={self.strategy} ctrl={self.control_hz}Hz; '
            'waiting for /policy/info')

    # ------------------------------------------------------------- callbacks
    def _on_info(self, msg: JointState) -> None:
        if self.chunk_executor is not None:
            return
        self.info = policy_info_from(msg.position)
        self.worker = RemoteWorker(
            self._send_request, action_dim=self.info.action_dim,
            timeout_factor=float(self.get_parameter('timeout_factor').value),
            min_timeout_s=float(self.get_parameter('min_timeout_s').value),
            first_timeout_s=float(self.get_parameter('first_timeout_s').value))
        self.chunk_executor = make_executor(self.strategy, self.worker, self.info)
        self.get_logger().info(f'evh_executor: policy {self.info}; strategy ready')

    def _on_chunk(self, msg: JointState) -> None:
        if self.worker is None:
            return
        try:
            chunk = decode_chunk(msg.position)
        except CodecError as exc:
            self.get_logger().warn(f'dropping a malformed chunk: {exc}')
            return
        sent_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if sent_ns > 0:
            age_ms = (self.get_clock().now().nanoseconds - sent_ns) / 1e6
            self.pub_chunk_age.publish(Float32(data=float(max(0.0, age_ms))))
        self.worker.deliver(chunk)

    def _on_episode_reset(self, _msg: Empty) -> None:
        if self.chunk_executor is not None:
            self.chunk_executor.reset()
        self._t = 0

    # --------------------------------------------------------------- control
    def _tick(self) -> None:
        if self.chunk_executor is None:
            return
        action = self.chunk_executor.step({}, self._t)
        self._t += 1

        metrics = self.chunk_executor.take_arrival_metrics()
        if metrics is not None:
            compute_ms, delay_steps = metrics
            self.pub_latency.publish(Float32(data=float(compute_ms)))
            self.pub_delay.publish(Float32(data=float(delay_steps)))
        for _ in range(self.chunk_executor.take_lost()):
            self.pub_lost.publish(Float32(data=1.0))

        if action is not None:   # None = hold: the reactive layer keeps tracking its target
            self.pub_waypoint.publish(self._to_waypoint(action))

    # --------------------------------------------------------------- helpers
    def _send_request(self, req: Request) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = encode_request(req, self.info.action_dim)
        self.pub_request.publish(msg)

    def _to_waypoint(self, action: np.ndarray) -> JointState:
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size < 7:
            a = np.pad(a, (0, 7 - a.size))
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = [float(v) for v in a[:7]]
        return msg

    def destroy_node(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
