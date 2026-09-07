"""Reactive layer node: high-rate local tracking of the (delayed) cognitive waypoints.

Wiring only. It runs co-located with the plant, so `/obs/ee_pose` reaches it with zero delay
(deliberately NOT routed through the latency relay — invariant 6), and it re-emits a `/cmd/action`
every tick at `rate_hz` while `/cmd/waypoint` arrives late and sparse from the controller.

The tracking itself — what a waypoint means, how the target is latched, how the setpoint marches —
lives in `tracking.py`, which has no ROS in it and is unit-tested directly. This module chooses a
tracker from the parameters and moves messages in and out of it:

    /cmd/waypoint  --> tracker.set_waypoint()
    /obs/ee_pose   --> the local anchor passed to tracker.step()
    /episode/reset --> tracker.reset()
    tick (rate_hz) --> tracker.step() --> /cmd/action

`/cmd/action` carries a DEADLINE (see ACTION_QOS): the plant needs to tell "this layer is
commanding a hold" from "this layer is gone", because it re-applies whatever it last received at
action_hz until something replaces it.

`absolute_waypoints` must match the plant's `absolute_actions`; every launch file feeds both the
same argument, and the plant aborts if the loaded policy disagrees (invariant 1).
"""
from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty

from evh_reactive.tracking import (
    AbsoluteTracker,
    DeltaTracker,
    PassthroughTracker,
    Pose,
    normalize_waypoint,
)
from evh_reactive.transforms import quat_normalize

# The command path, and the one topic that crosses the network in the split deployment. It is
# BEST_EFFORT with a depth of 1 ON PURPOSE, and the three packages that touch it must agree or DDS
# silently refuses to pair them.
#
# Reliable delivery is the wrong contract here. A waypoint is an ABSOLUTE target and the reactive
# layer latches it, so a lost one costs nothing — it simply keeps tracking the previous target.
# A LATE one costs plenty: reliable QoS retransmits and delivers in order, so a stale waypoint
# arrives after a fresher one was already available and the arm is commanded backwards. That is
# precisely the "re-apply an old command" behaviour the latched-absolute-target design exists to
# prevent. Newest-wins, no retransmit, no head-of-line blocking.
WAYPOINT_QOS = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                          history=QoSHistoryPolicy.KEEP_LAST)

# /cmd/action is a CONTINUOUS stream, not an event: the plant re-applies whichever action it holds
# at action_hz until another arrives, so a reactive layer that dies or stalls is invisible to it.
# In DELTA mode that is not a freeze but a runaway — robosuite's OSC re-derives
# goal = eef + delta * output_max every step, so a cached non-zero delta is a velocity command and
# the arm drifts until it hits a limit, while the episode closes as an ordinary timeout.
#
# The DEADLINE is what makes the silence observable: DDS tells the plant when no message arrived
# within it and the plant drops the stale command (see PlantNode._on_action_deadline). It is part
# of the QoS CONTRACT, not a local setting — a subscriber requesting a deadline the publisher does
# not offer is never paired at all, and /cmd/action would go quiet altogether — so it is duplicated
# in evh_plant (the two deploy to different machines and neither may depend on the other) and
# test_mode_crosscheck.py pins the agreement.
#
# 50 ms = one control period, ~12 reactive ticks at the default 250 Hz: far enough above ordinary
# scheduling jitter never to fire spuriously, and short enough to cap a delta runaway at ten
# physics steps instead of a whole episode.
ACTION_DEADLINE_S = 0.05
ACTION_QOS = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE,
                        history=QoSHistoryPolicy.KEEP_LAST,
                        deadline=Duration(seconds=ACTION_DEADLINE_S))


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
        self.declare_parameter('absolute_waypoints', True)  # abs-action policy (see tracking.py)
        self.declare_parameter('max_step_pos', 0.004)    # m per tick toward the target (abs)
        self.declare_parameter('max_step_rot', 0.02)     # rad per tick toward the target (abs)

        self.rate_hz = self.get_parameter('rate_hz').value
        self.passthrough = bool(self.get_parameter('passthrough').value)
        self.absolute = bool(self.get_parameter('absolute_waypoints').value)
        self.tracker = self._make_tracker()

        self._ee: Pose | None = None   # local, zero-delay EE state

        self.create_subscription(JointState, '/cmd/waypoint', self._on_waypoint, WAYPOINT_QOS)
        self.create_subscription(
            PoseStamped, '/obs/ee_pose', self._on_ee_pose, qos_profile_sensor_data)
        self.create_subscription(Empty, '/episode/reset', self._on_episode_reset, 10)

        self.pub_action = self.create_publisher(JointState, '/cmd/action', ACTION_QOS)
        self.create_timer(1.0 / self.rate_hz, self._tick)

        if 1.0 / self.rate_hz > ACTION_DEADLINE_S:
            self.get_logger().warn(
                f'rate_hz={self.rate_hz} is slower than the /cmd/action deadline '
                f'({ACTION_DEADLINE_S * 1e3:.0f}ms): the plant will read the gaps between ticks '
                'as a dead reactive layer and hold')

        mode = 'PASSTHROUGH (baseline)' if self.passthrough else 'tracking'
        self.get_logger().info(
            f'evh_reactive up: {self.rate_hz}Hz mode={mode} absolute={self.absolute}')

    def _make_tracker(self):
        """Pick the tracking strategy the parameters describe."""
        if self.passthrough:
            return PassthroughTracker(
                absolute=self.absolute,
                scale=float(self.get_parameter('passthrough_scale').value))
        if self.absolute:
            return AbsoluteTracker(
                max_step_pos=float(self.get_parameter('max_step_pos').value),
                max_step_rot=float(self.get_parameter('max_step_rot').value))
        return DeltaTracker(
            kp_pos=float(self.get_parameter('kp_pos').value),
            kp_rot=float(self.get_parameter('kp_rot').value),
            pos_scale=float(self.get_parameter('pos_scale').value),
            rot_scale=float(self.get_parameter('rot_scale').value))

    # ------------------------------------------------------------- callbacks
    def _on_ee_pose(self, msg: PoseStamped) -> None:
        p, o = msg.pose.position, msg.pose.orientation
        self._ee = Pose(pos=np.array([p.x, p.y, p.z]),
                        quat=quat_normalize(np.array([o.x, o.y, o.z, o.w])))

    def _on_waypoint(self, msg: JointState) -> None:
        self.tracker.set_waypoint(normalize_waypoint(msg.position), self._ee)

    def _on_episode_reset(self, _msg: Empty) -> None:
        self.tracker.reset()

    # ------------------------------------------------------------- high-rate
    def _tick(self) -> None:
        if self.tracker.needs_ee and self._ee is None:
            return   # nothing to anchor against yet
        action = self.tracker.step(self._ee)
        if action is None:
            return
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.position = [float(v) for v in action]
        self.pub_action.publish(out)


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
