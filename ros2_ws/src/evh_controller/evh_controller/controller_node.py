"""Controller node: runs the diffusion/flow policy under a pluggable chunk-execution strategy.

HiL "Controller" side (Jetson Orin Nano). Subscribes to the (latency-injected) observation topics
and streams task-space EE-pose targets on /cmd/waypoint at the control rate. *How* the action
chunk is executed under inference latency is delegated to a ChunkExecutor strategy
(synchronous | naive_async | temporal_ensemble | bid | rtc | network_aware), which is the seam for
the Wedge-A baseline comparison and the Wedge-B extension. The high-rate reactive layer downstream
tracks the streamed targets.

Publishes per-tick compute time on /metrics/inference_ms for the benchmark recorder. (When the
real async-generation hook lands, this becomes the true inference delay `d`.)
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

from evh_controller.policy import make_policy
from evh_controller.chunk_executor import make_executor


class ControllerNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__('evh_controller', **kwargs)

        self.declare_parameter('backend', 'pytorch')        # pytorch | tensorrt
        self.declare_parameter('weights_path', '')           # ckpt dir or .engine
        self.declare_parameter('strategy', 'synchronous')    # chunk-execution strategy
        self.declare_parameter('control_hz', 50.0)           # action stream rate
        self.declare_parameter('prompt', 'pick up the block')

        backend = self.get_parameter('backend').value
        weights = self.get_parameter('weights_path').value
        strategy = self.get_parameter('strategy').value
        self.control_hz = self.get_parameter('control_hz').value

        self.policy = make_policy(backend, weights)
        self.executor = make_executor(strategy)
        self.get_logger().info(
            f'evh_controller: backend={backend} strategy={strategy} ctrl={self.control_hz}Hz')

        # latest observations (overwritten by callbacks; the strategy samples the freshest)
        self._image: np.ndarray | None = None
        self._joint: np.ndarray | None = None
        self._t = 0   # control timestep counter

        self.create_subscription(Image, '/obs/image', self._on_image, qos_profile_sensor_data)
        self.create_subscription(
            JointState, '/obs/joint_state', self._on_joint, qos_profile_sensor_data)

        self.pub_waypoint = self.create_publisher(PoseStamped, '/cmd/waypoint', 10)
        self.pub_latency = self.create_publisher(Float32, '/metrics/inference_ms', 10)

        self.create_timer(1.0 / self.control_hz, self._tick)

    # ------------------------------------------------------------- callbacks
    def _on_image(self, msg: Image) -> None:
        self._image = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)

    def _on_joint(self, msg: JointState) -> None:
        self._joint = np.asarray(msg.position, dtype=np.float32)

    # --------------------------------------------------------------- control
    def _tick(self) -> None:
        if self._image is None or self._joint is None:
            return  # wait for first observations

        t0 = time.perf_counter()
        action = self.executor.select_action((self._image, self._joint), self.policy, self._t)
        compute_ms = (time.perf_counter() - t0) * 1e3
        # nonzero only on ticks where the strategy actually invoked the policy
        self.pub_latency.publish(Float32(data=float(compute_ms)))

        self.pub_waypoint.publish(self._to_pose(action))
        self._t += 1

    # --------------------------------------------------------------- helpers
    def _to_pose(self, action: np.ndarray) -> PoseStamped:
        """Map an EE action vector (OSC_POSE: 6-DoF delta + gripper) to a task-space pose target.

        TODO: confirm absolute vs delta convention and orientation encoding so the reactive layer
        agrees on the contract. Position-only is filled here for the skeleton.
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
