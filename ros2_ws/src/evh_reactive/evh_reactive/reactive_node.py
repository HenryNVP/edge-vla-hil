"""Reactive layer: high-rate operational-space impedance controller.

This is the "fix" half of the experiment. It runs co-located with the plant (zero-delay local
state) at a high rate (~200-500 Hz) and tracks the *delayed, low-rate* waypoints arriving from the
ACT controller. Between cognitive updates it interpolates toward the latest waypoint, so the plant
keeps moving smoothly through the network latency instead of acting open-loop on stale commands.

Control law (fixed-gain operational-space impedance):

    F = Kp (x_des - x) - Kd * x_dot          (task space)
    tau = J(q)^T F                            (mapped to joint commands)

Gains Kp/Kd are FIXED in this project. Making them a function of the language prompt is the
documented future-work extension (proposal Section 6); the parameter interface below is shaped to
accept that later without restructuring.

Ablation switch: `passthrough=true` disables interpolation/impedance and forwards the raw waypoint
at the cognitive rate, reproducing the monolithic (no-reactive-layer) baseline for the benchmark.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped


class ReactiveNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__('evh_reactive', **kwargs)

        self.declare_parameter('rate_hz', 250.0)
        self.declare_parameter('kp', [600.0, 600.0, 600.0, 30.0, 30.0, 30.0])
        self.declare_parameter('kd', [40.0, 40.0, 40.0, 5.0, 5.0, 5.0])
        self.declare_parameter('passthrough', False)   # True -> reproduce monolithic baseline

        self.rate_hz = self.get_parameter('rate_hz').value
        self.kp = np.asarray(self.get_parameter('kp').value, dtype=np.float64)
        self.kd = np.asarray(self.get_parameter('kd').value, dtype=np.float64)
        self.passthrough = bool(self.get_parameter('passthrough').value)

        self._target: np.ndarray | None = None     # latest waypoint [x,y,z,(rpy)]
        self._joint: np.ndarray | None = None       # local joint state (zero-delay)

        self.create_subscription(
            PoseStamped, '/cmd/waypoint', self._on_waypoint, 10)
        self.create_subscription(
            JointState, '/obs/joint_state', self._on_joint, qos_profile_sensor_data)

        self.pub_action = self.create_publisher(JointState, '/cmd/action', 10)
        self.create_timer(1.0 / self.rate_hz, self._tick)

        mode = 'PASSTHROUGH (baseline)' if self.passthrough else 'impedance'
        self.get_logger().info(f'evh_reactive up: {self.rate_hz}Hz mode={mode}')

    # ------------------------------------------------------------- callbacks
    def _on_waypoint(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self._target = np.array([p.x, p.y, p.z], dtype=np.float64)

    def _on_joint(self, msg: JointState) -> None:
        self._joint = np.asarray(msg.position, dtype=np.float64)

    # ------------------------------------------------------------- high-rate
    def _tick(self) -> None:
        if self._target is None or self._joint is None:
            return

        if self.passthrough:
            action = self._target          # forward raw waypoint, no local stabilization
        else:
            action = self._impedance_step()

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.position = list(np.asarray(action, dtype=float))
        self.pub_action.publish(out)

    def _impedance_step(self) -> np.ndarray:
        """TODO: real operational-space impedance control.

        Needs forward kinematics + Jacobian for the robot. Two implementation options:
          (a) pull FK/J from the robosuite/MuJoCo model (mjData.site_xpos, mj_jacSite), or
          (b) a pinocchio model loaded from the robot URDF.
        For Phase 1 this returns the current target as a placeholder so the graph runs.
        """
        x = self._target            # placeholder: pretend we already track perfectly
        x_dot = np.zeros_like(x)    # TODO: finite-difference / measured EE velocity
        _F = self.kp[:3] * (self._target - x) - self.kd[:3] * x_dot   # noqa: F841 (wiring TODO)
        return self._target


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ReactiveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
