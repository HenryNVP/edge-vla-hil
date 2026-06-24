"""Controller node: runs the ACT policy and emits task-space waypoints.

HiL "Controller" side (Jetson Orin Nano). Subscribes to the (latency-injected) observation
topics, runs ACT inference at its natural rate (~10 Hz honest on the Orin Nano), and publishes a
single target waypoint per cognitive tick on /cmd/waypoint. The high-rate reactive layer
downstream is responsible for actually tracking it.

Also publishes per-inference latency on /metrics/inference_ms for the benchmark recorder.

ACT outputs an action *chunk*; we expose the chunk head as the current waypoint. Temporal
ensembling across overlapping chunks is a TODO knob (`temporal_ensemble`).
"""
from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32

from evh_controller.act_policy import make_policy


class ControllerNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__('evh_controller', **kwargs)

        self.declare_parameter('backend', 'pytorch')        # pytorch | tensorrt
        self.declare_parameter('weights_path', '')           # ckpt dir or .engine
        self.declare_parameter('infer_hz', 10.0)             # cognitive-layer rate
        self.declare_parameter('temporal_ensemble', False)
        self.declare_parameter('prompt', 'pick up the block')

        backend = self.get_parameter('backend').value
        weights = self.get_parameter('weights_path').value
        self.infer_hz = self.get_parameter('infer_hz').value

        self.policy = make_policy(backend, weights)
        self.get_logger().info(f'evh_controller: backend={backend} infer={self.infer_hz}Hz')

        # latest observations (overwritten by callbacks; inference samples the freshest)
        self._image: np.ndarray | None = None
        self._joint: np.ndarray | None = None

        self.create_subscription(Image, '/obs/image', self._on_image, qos_profile_sensor_data)
        self.create_subscription(
            JointState, '/obs/joint_state', self._on_joint, qos_profile_sensor_data)

        self.pub_waypoint = self.create_publisher(PoseStamped, '/cmd/waypoint', 10)
        self.pub_latency = self.create_publisher(Float32, '/metrics/inference_ms', 10)

        self.create_timer(1.0 / self.infer_hz, self._infer)

    # ------------------------------------------------------------- callbacks
    def _on_image(self, msg: Image) -> None:
        self._image = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)

    def _on_joint(self, msg: JointState) -> None:
        self._joint = np.asarray(msg.position, dtype=np.float32)

    # ------------------------------------------------------------- inference
    def _infer(self) -> None:
        if self._image is None or self._joint is None:
            return  # wait for first observations

        t0 = time.perf_counter()
        chunk = self.policy.predict(self._image, self._joint)   # [horizon, action_dim]
        infer_ms = (time.perf_counter() - t0) * 1e3
        self.pub_latency.publish(Float32(data=float(infer_ms)))

        waypoint = chunk[0]  # TODO: temporal ensemble across overlapping chunks
        self.pub_waypoint.publish(self._to_pose(waypoint))

    # --------------------------------------------------------------- helpers
    def _to_pose(self, action: np.ndarray) -> PoseStamped:
        """Map a 6/7-DoF action vector to a task-space pose.

        TODO: define the action convention. ACT typically emits joint deltas or an EE delta;
        decide whether the waypoint is absolute EE pose or a delta and document it here so the
        reactive layer agrees on the contract.
        """
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base'
        if action.shape[0] >= 3:
            msg.pose.position.x = float(action[0])
            msg.pose.position.y = float(action[1])
            msg.pose.position.z = float(action[2])
        msg.pose.orientation.w = 1.0
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
