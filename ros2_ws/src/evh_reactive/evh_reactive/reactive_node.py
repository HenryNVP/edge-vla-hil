"""Reactive layer: high-rate local tracking of the (delayed) cognitive waypoints.

This is the "fix" half of the experiment. It runs co-located with the plant (zero-delay local
state) and tracks the *delayed, low-rate* OSC_POSE deltas arriving from the controller on
/cmd/waypoint. On each new waypoint it latches an ABSOLUTE task-space target

    x_target = x_local + dpos * pos_scale
    q_target = R(drot * rot_scale) ⊗ q_local          (world-frame delta, robosuite convention)

anchored at the *local* EE pose (from /obs/ee_pose, no network in between), then re-emits, every
tick, the OSC_POSE action that drives the current local pose toward that target:

    a_pos = clip(kp_pos * (x_target - x) / pos_scale, -1, 1)
    a_rot = clip(kp_rot * axisangle(q_target ⊗ q⁻¹) / rot_scale, -1, 1)

robosuite's OSC controller underneath supplies the impedance (F = Kp Δx - Kd ẋ); this layer
supplies the zero-delay error recomputation. Because the target is latched absolute, a lost or
late waypoint means "hold the last target" instead of "keep re-applying a stale delta" — that
difference is the reactive layer's contribution under network degradation.

pos_scale / rot_scale mirror the plant OSC's output_max (robosuite defaults 0.05 m / 0.5 rad):
how far a unit action moves the OSC goal in one control step. Keep them in sync with the plant.

Gain scheduling (kp as a function of the language prompt) is the documented future-work
extension; the fixed-gain interface below is shaped to accept that later.

Ablation switch: `passthrough=true` forwards the raw delta instead, scaled by `passthrough_scale`
(the plant re-applies the cached action at action_hz, so a delta meant for one policy step at
control_hz must be shrunk by ~control_hz/action_hz to keep the commanded EE speed). This
reproduces the monolithic (no-reactive-layer) baseline for the benchmark.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty

from evh_reactive.transforms import (
    axisangle_to_quat, quat_conj, quat_mul, quat_normalize, quat_to_axisangle)

ACTION_DIM = 7   # OSC_POSE: dpos(3) + axis-angle drot(3) + gripper


class ReactiveNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__('evh_reactive', **kwargs)

        self.declare_parameter('rate_hz', 250.0)
        self.declare_parameter('kp_pos', 1.0)
        self.declare_parameter('kp_rot', 1.0)
        self.declare_parameter('pos_scale', 0.05)        # m per unit action (OSC output_max)
        self.declare_parameter('rot_scale', 0.5)         # rad per unit action
        self.declare_parameter('passthrough', False)     # True -> monolithic baseline
        self.declare_parameter('passthrough_scale', 0.1)  # ~ controller_hz / plant action_hz

        self.rate_hz = self.get_parameter('rate_hz').value
        self.kp_pos = float(self.get_parameter('kp_pos').value)
        self.kp_rot = float(self.get_parameter('kp_rot').value)
        self.pos_scale = float(self.get_parameter('pos_scale').value)
        self.rot_scale = float(self.get_parameter('rot_scale').value)
        self.passthrough = bool(self.get_parameter('passthrough').value)
        self.passthrough_scale = float(self.get_parameter('passthrough_scale').value)

        # local (zero-delay) EE state
        self._ee_pos: np.ndarray | None = None
        self._ee_quat: np.ndarray | None = None
        # latched absolute target (tracking mode) / raw delta (passthrough mode)
        self._target_pos: np.ndarray | None = None
        self._target_quat: np.ndarray | None = None
        self._gripper = 0.0
        self._raw: np.ndarray | None = None

        self.create_subscription(JointState, '/cmd/waypoint', self._on_waypoint, 10)
        self.create_subscription(
            PoseStamped, '/obs/ee_pose', self._on_ee_pose, qos_profile_sensor_data)
        self.create_subscription(Empty, '/episode/reset', self._on_episode_reset, 10)

        self.pub_action = self.create_publisher(JointState, '/cmd/action', 10)
        self.create_timer(1.0 / self.rate_hz, self._tick)

        mode = 'PASSTHROUGH (baseline)' if self.passthrough else 'tracking'
        self.get_logger().info(f'evh_reactive up: {self.rate_hz}Hz mode={mode}')

    # ------------------------------------------------------------- callbacks
    def _on_ee_pose(self, msg: PoseStamped) -> None:
        p, o = msg.pose.position, msg.pose.orientation
        self._ee_pos = np.array([p.x, p.y, p.z])
        self._ee_quat = quat_normalize(np.array([o.x, o.y, o.z, o.w]))

    def _on_waypoint(self, msg: JointState) -> None:
        a = np.asarray(msg.position, dtype=np.float64).reshape(-1)
        if a.size < ACTION_DIM:
            a = np.pad(a, (0, ACTION_DIM - a.size))
        self._raw = a[:ACTION_DIM]
        self._gripper = float(self._raw[6])
        if self.passthrough:
            return
        if self._ee_pos is None:
            return  # cannot anchor a target before the first local EE pose
        self._target_pos = self._ee_pos + self._raw[:3] * self.pos_scale
        self._target_quat = quat_mul(
            axisangle_to_quat(self._raw[3:6] * self.rot_scale), self._ee_quat)

    def _on_episode_reset(self, _msg: Empty) -> None:
        self._target_pos = None
        self._target_quat = None
        self._raw = None
        self._gripper = 0.0

    # ------------------------------------------------------------- high-rate
    def _tick(self) -> None:
        if self.passthrough:
            action = self._passthrough_action()
        else:
            action = self._tracking_action()
        if action is None:
            return
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.position = [float(v) for v in action]
        self.pub_action.publish(out)

    def _passthrough_action(self) -> np.ndarray | None:
        if self._raw is None:
            return None
        action = self._raw.copy()
        action[:6] *= self.passthrough_scale   # gripper is a command, not a delta: unscaled
        return action

    def _tracking_action(self) -> np.ndarray | None:
        if self._target_pos is None or self._ee_pos is None:
            return None
        err_pos = self._target_pos - self._ee_pos
        err_rot = quat_to_axisangle(quat_mul(self._target_quat, quat_conj(self._ee_quat)))
        action = np.empty(ACTION_DIM)
        action[:3] = np.clip(self.kp_pos * err_pos / self.pos_scale, -1.0, 1.0)
        action[3:6] = np.clip(self.kp_rot * err_rot / self.rot_scale, -1.0, 1.0)
        action[6] = self._gripper
        return action


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
