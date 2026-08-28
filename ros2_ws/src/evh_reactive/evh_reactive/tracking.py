"""The reactive layer's tracking math — the "fix" half of the experiment, with no ROS in it.

Split out of `reactive_node.py` so the node is wiring (params, subscriptions, a timer) and the
behaviour under test lives here, callable from a plain unit test at whatever tick rate the test
wants. Both trackers are stateful objects rather than functions: absolute tracking carries a
setpoint trajectory across ticks, and that state is exactly what makes a late waypoint degrade
gracefully instead of stalling.

Two modes, matching the plant's OSC configuration (invariant 1 — they must agree):

  DeltaTracker (`absolute_waypoints=false`, OSC control_delta=True)
    A waypoint is an OSC_POSE delta. On arrival it is anchored against the LOCAL, zero-delay EE
    pose into an absolute target; every tick then re-emits a clipped error-delta toward it:

        x_target = x_local + dpos * pos_scale
        q_target = R(drot * rot_scale) (x) q_local
        a_pos    = clip(kp_pos * (x_target - x) / pos_scale, -1, 1)
        a_rot    = clip(kp_rot * axisangle(q_target (x) q^-1) / rot_scale, -1, 1)

    Because the target is latched absolute, a lost or late waypoint means "hold the last target"
    rather than "keep re-applying a stale delta". That difference is the reactive layer's whole
    contribution under network degradation.

  AbsoluteTracker (`absolute_waypoints=true`, OSC control_delta=False, the abs-action DP ckpts)
    The waypoint already IS the target. Each tick the setpoint marches toward it, capped at
    max_step_pos / max_step_rot, and the SETPOINT is what the plant receives — giving smooth
    interpolation between sparse delayed targets. The setpoint marches from its own previous
    value, NOT from the measured EE pose: re-anchoring at the measurement would keep the OSC goal
    one step ahead of the arm, so the proportional force never grows and motion crawls.

pos_scale / rot_scale mirror the plant OSC's output_max (robosuite defaults 0.05 m / 0.5 rad):
how far a unit action moves the OSC goal in one control step. They are duplicated parameters, not
shared ones — keep them in sync with the plant (invariant 3).

Gain scheduling (kp as a function of the language prompt) is the documented future-work
extension; the fixed-gain interface below is shaped to accept that later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evh_reactive.transforms import (
    axisangle_to_quat,
    quat_conj,
    quat_mul,
    quat_to_axisangle,
)

ACTION_DIM = 7   # OSC_POSE: dpos(3) + axis-angle drot(3) + gripper


@dataclass
class Pose:
    """A task-space pose. Quaternion is [x, y, z, w] (robosuite / geometry_msgs order)."""
    pos: np.ndarray
    quat: np.ndarray


class Tracker:
    """Common state: the latched target and the gripper command riding along with it."""

    needs_ee = True   # False for trackers that emit without any local pose (passthrough)

    def __init__(self) -> None:
        self.target: Pose | None = None
        self.gripper = 0.0

    def reset(self) -> None:
        """Drop episode-scoped state at an /episode/reset boundary."""
        self.target = None
        self.gripper = 0.0

    def set_waypoint(self, action: np.ndarray, ee: Pose | None) -> None:
        """Latch a new target from a 7-dim waypoint. `ee` is the local, zero-delay EE pose."""
        raise NotImplementedError

    def step(self, ee: Pose) -> np.ndarray | None:
        """The action to send this tick, or None when there is nothing to track yet."""
        raise NotImplementedError


class DeltaTracker(Tracker):
    """Delta-OSC plant: anchor the delta locally, then emit a clipped error-delta each tick."""

    def __init__(self, kp_pos: float = 1.0, kp_rot: float = 1.0,
                 pos_scale: float = 0.05, rot_scale: float = 0.5) -> None:
        super().__init__()
        self.kp_pos = kp_pos
        self.kp_rot = kp_rot
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale

    def set_waypoint(self, action: np.ndarray, ee: Pose | None) -> None:
        self.gripper = float(action[6])
        if ee is None:
            return   # cannot anchor a delta target before the first local EE pose
        self.target = Pose(
            pos=ee.pos + action[:3] * self.pos_scale,
            quat=quat_mul(axisangle_to_quat(action[3:6] * self.rot_scale), ee.quat))

    def step(self, ee: Pose) -> np.ndarray | None:
        if self.target is None:
            return None
        err_pos = self.target.pos - ee.pos
        err_rot = quat_to_axisangle(quat_mul(self.target.quat, quat_conj(ee.quat)))

        action = np.empty(ACTION_DIM)
        action[:3] = np.clip(self.kp_pos * err_pos / self.pos_scale, -1.0, 1.0)
        action[3:6] = np.clip(self.kp_rot * err_rot / self.rot_scale, -1.0, 1.0)
        action[6] = self.gripper
        return action


class AbsoluteTracker(Tracker):
    """Absolute-OSC plant: march a setpoint trajectory toward the target and emit the setpoint."""

    def __init__(self, max_step_pos: float = 0.004, max_step_rot: float = 0.02) -> None:
        super().__init__()
        self.max_step_pos = max_step_pos
        self.max_step_rot = max_step_rot
        self.setpoint: Pose | None = None

    def reset(self) -> None:
        super().reset()
        self.setpoint = None

    def set_waypoint(self, action: np.ndarray, ee: Pose | None) -> None:
        # the waypoint IS the target — no anchoring, so this works before any EE pose arrives
        self.gripper = float(action[6])
        self.target = Pose(pos=action[:3].copy(), quat=axisangle_to_quat(action[3:6]))

    def step(self, ee: Pose) -> np.ndarray | None:
        if self.target is None:
            return None
        if self.setpoint is None:      # first tick of an episode: start from the arm's pose
            self.setpoint = Pose(pos=ee.pos.copy(), quat=ee.quat.copy())

        err_pos = self.target.pos - self.setpoint.pos
        dist = float(np.linalg.norm(err_pos))
        if dist > self.max_step_pos:
            err_pos = err_pos * (self.max_step_pos / dist)
        self.setpoint.pos = self.setpoint.pos + err_pos

        err_rot = quat_to_axisangle(quat_mul(self.target.quat, quat_conj(self.setpoint.quat)))
        angle = float(np.linalg.norm(err_rot))
        if angle > self.max_step_rot:
            err_rot = err_rot * (self.max_step_rot / angle)
        self.setpoint.quat = quat_mul(axisangle_to_quat(err_rot), self.setpoint.quat)

        action = np.empty(ACTION_DIM)
        action[:3] = self.setpoint.pos
        action[3:6] = quat_to_axisangle(self.setpoint.quat)
        action[6] = self.gripper
        return action


class PassthroughTracker(Tracker):
    """Ablation baseline: forward the raw waypoint, reproducing the monolithic (no-reactive) loop.

    In delta mode the plant re-applies the cached action at action_hz, so a delta meant for one
    policy step at control_hz must be shrunk by ~control_hz/action_hz to keep the commanded EE
    speed honest. Absolute commands are idempotent, so they are forwarded unscaled.
    """

    needs_ee = False

    def __init__(self, absolute: bool, scale: float = 0.1) -> None:
        super().__init__()
        self.absolute = absolute
        self.scale = scale
        self.raw: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self.raw = None

    def set_waypoint(self, action: np.ndarray, ee: Pose | None) -> None:
        self.raw = action.copy()
        self.gripper = float(action[6])

    def step(self, ee: Pose | None = None) -> np.ndarray | None:
        if self.raw is None:
            return None
        action = self.raw.copy()
        if not self.absolute:
            action[:6] *= self.scale   # gripper is a command, not a delta: unscaled
        return action


def normalize_waypoint(position) -> np.ndarray:
    """Coerce a JointState.position of any width to the 7-dim action contract."""
    a = np.asarray(position, dtype=np.float64).reshape(-1)
    if a.size < ACTION_DIM:
        a = np.pad(a, (0, ACTION_DIM - a.size))
    return a[:ACTION_DIM]
