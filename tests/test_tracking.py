"""Unit tests for the reactive layer's tracking math — pure Python, no ROS, no spinning nodes.

test_reactive_tracking.py still covers the same behaviour end-to-end over real topics; these run
in milliseconds against the trackers directly, so the tick-by-tick trajectory (which a ROS test
can only sample) is actually assertable. That is what moving the math out of the node bought.
"""
import numpy as np
import pytest

from evh_reactive.tracking import (
    ACTION_DIM,
    AbsoluteTracker,
    DeltaTracker,
    PassthroughTracker,
    Pose,
    normalize_waypoint,
)
from evh_reactive.transforms import axisangle_to_quat, quat_to_axisangle

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def _pose(pos=(0.0, 0.0, 0.0), quat=IDENTITY):
    return Pose(pos=np.asarray(pos, dtype=float), quat=np.asarray(quat, dtype=float))


def _waypoint(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), gripper=0.0):
    return np.array([*pos, *rot, gripper], dtype=float)


# --------------------------------------------------------------------- delta mode
def test_delta_tracker_emits_nothing_before_a_waypoint():
    assert DeltaTracker().step(_pose()) is None


def test_delta_tracker_cannot_latch_a_target_before_the_first_ee_pose():
    """A delta is meaningless without something to anchor it to."""
    tracker = DeltaTracker()
    tracker.set_waypoint(_waypoint(pos=(1.0, 0.0, 0.0)), ee=None)

    assert tracker.target is None
    assert tracker.step(_pose()) is None


def test_delta_target_is_anchored_at_the_local_pose_and_scaled():
    """x_target = x_local + dpos * pos_scale — pos_scale mirrors the plant OSC's output_max."""
    tracker = DeltaTracker(pos_scale=0.05)
    tracker.set_waypoint(_waypoint(pos=(1.0, 0.0, 0.0)), ee=_pose(pos=(0.3, 0.0, 0.0)))

    assert tracker.target.pos == pytest.approx([0.35, 0.0, 0.0])


def test_delta_action_is_clipped_to_the_unit_action_range():
    """OSC takes actions in [-1, 1]; a far target must saturate, not overflow."""
    tracker = DeltaTracker(pos_scale=0.05)
    tracker.set_waypoint(_waypoint(pos=(1.0, 1.0, 1.0)), ee=_pose())
    action = tracker.step(_pose(pos=(-5.0, -5.0, -5.0)))

    assert action.shape == (ACTION_DIM,)
    assert np.all(action[:6] >= -1.0) and np.all(action[:6] <= 1.0)
    assert action[:3] == pytest.approx([1.0, 1.0, 1.0])


def test_delta_error_shrinks_as_the_arm_approaches_the_latched_target():
    """The target is absolute, so the commanded delta must decay as the arm closes in — this is
    what makes a late waypoint 'hold' rather than re-apply a stale delta forever."""
    tracker = DeltaTracker(pos_scale=0.05)
    tracker.set_waypoint(_waypoint(pos=(1.0, 0.0, 0.0)), ee=_pose())   # target x = 0.05

    far = tracker.step(_pose(pos=(0.0, 0.0, 0.0)))[0]
    near = tracker.step(_pose(pos=(0.04, 0.0, 0.0)))[0]
    at = tracker.step(_pose(pos=(0.05, 0.0, 0.0)))[0]

    assert far > near > at
    assert at == pytest.approx(0.0, abs=1e-9), 'must command nothing once the target is reached'


def test_delta_gripper_rides_through_unscaled():
    """The gripper is a command, not a delta — scaling it would half-close the hand."""
    tracker = DeltaTracker()
    tracker.set_waypoint(_waypoint(gripper=1.0), ee=_pose())
    assert tracker.step(_pose())[6] == 1.0


def test_delta_reset_drops_the_target():
    tracker = DeltaTracker()
    tracker.set_waypoint(_waypoint(pos=(1.0, 0.0, 0.0), gripper=1.0), ee=_pose())
    tracker.reset()

    assert tracker.step(_pose()) is None
    assert tracker.gripper == 0.0


# ------------------------------------------------------------------ absolute mode
def test_absolute_target_is_the_waypoint_itself():
    """No anchoring: the abs-action policy already emits a world-frame target."""
    tracker = AbsoluteTracker()
    tracker.set_waypoint(_waypoint(pos=(0.4, 0.1, 0.9)), ee=None)
    assert tracker.target.pos == pytest.approx([0.4, 0.1, 0.9])


def test_absolute_setpoint_is_rate_limited_per_tick():
    tracker = AbsoluteTracker(max_step_pos=0.004)
    tracker.set_waypoint(_waypoint(pos=(1.0, 0.0, 0.0)), ee=None)
    action = tracker.step(_pose(pos=(0.0, 0.0, 0.0)))

    assert action[0] == pytest.approx(0.004), 'setpoint jumped further than max_step_pos'


def test_absolute_setpoint_marches_from_itself_not_from_the_measured_pose():
    """The documented subtlety: re-anchoring the setpoint at the measured EE pose each tick would
    keep the OSC goal exactly one step ahead of the arm, so the proportional force never grows and
    the motion crawls. With the arm held still, the setpoint must keep advancing anyway."""
    tracker = AbsoluteTracker(max_step_pos=0.004)
    tracker.set_waypoint(_waypoint(pos=(1.0, 0.0, 0.0)), ee=None)

    stuck = _pose(pos=(0.0, 0.0, 0.0))     # arm does not move at all
    commanded = [tracker.step(stuck)[0] for _ in range(5)]

    assert commanded == pytest.approx([0.004, 0.008, 0.012, 0.016, 0.020])


def test_absolute_setpoint_converges_and_then_holds_the_target():
    tracker = AbsoluteTracker(max_step_pos=0.004)
    tracker.set_waypoint(_waypoint(pos=(0.02, 0.0, 0.0)), ee=None)

    stuck = _pose()
    for _ in range(20):                    # 0.02 / 0.004 = 5 ticks, plus slack
        action = tracker.step(stuck)

    assert action[0] == pytest.approx(0.02), 'never reached the target'
    assert tracker.step(stuck)[0] == pytest.approx(0.02), 'must hold, not overshoot'


def test_absolute_rotation_is_rate_limited_too():
    tracker = AbsoluteTracker(max_step_rot=0.02)
    tracker.set_waypoint(_waypoint(rot=(0.0, 0.0, 1.0)), ee=None)   # 1 rad about z
    action = tracker.step(_pose())

    assert np.linalg.norm(action[3:6]) == pytest.approx(0.02, rel=1e-6)


def test_absolute_setpoint_starts_at_the_arm_pose_each_episode():
    """First tick after a reset anchors at wherever the arm actually is, so the new episode does
    not begin by marching from the previous episode's setpoint."""
    tracker = AbsoluteTracker(max_step_pos=0.004)
    tracker.set_waypoint(_waypoint(pos=(1.0, 0.0, 0.0)), ee=None)
    tracker.step(_pose(pos=(0.0, 0.0, 0.0)))

    tracker.reset()
    assert tracker.setpoint is None
    tracker.set_waypoint(_waypoint(pos=(1.0, 0.0, 0.0)), ee=None)
    action = tracker.step(_pose(pos=(0.5, 0.0, 0.0)))

    assert action[0] == pytest.approx(0.504), 'did not restart from the arm pose'


def test_absolute_holds_the_last_target_when_waypoints_stop():
    """The whole point under packet loss: no new waypoint means keep going to the last target."""
    tracker = AbsoluteTracker(max_step_pos=0.004)
    tracker.set_waypoint(_waypoint(pos=(0.1, 0.0, 0.0)), ee=None)

    stuck = _pose()
    first = tracker.step(stuck)[0]
    later = [tracker.step(stuck)[0] for _ in range(3)][-1]

    assert later > first, 'stopped advancing once the waypoints dried up'


# --------------------------------------------------------------------- passthrough
@pytest.mark.parametrize('absolute,expected_pos,expected_rot', [
    (False, 0.1, 0.05),    # delta: shrunk by ~control_hz/action_hz so EE speed stays honest
    (True, 1.0, 0.5),      # absolute commands are idempotent — forwarded as-is
])
def test_passthrough_scaling_follows_the_mode(absolute, expected_pos, expected_rot):
    tracker = PassthroughTracker(absolute=absolute, scale=0.1)
    tracker.set_waypoint(_waypoint(pos=(1.0, 0.0, 0.0), rot=(0.5, 0.0, 0.0), gripper=1.0), ee=None)
    action = tracker.step()

    assert action[0] == pytest.approx(expected_pos)
    assert action[3] == pytest.approx(expected_rot)
    assert action[6] == 1.0, 'the gripper is never scaled'


def test_passthrough_needs_no_ee_pose():
    """It forwards rather than anchors, so it must emit even with no /obs/ee_pose at all."""
    assert PassthroughTracker(absolute=True).needs_ee is False


def test_passthrough_emits_nothing_before_a_waypoint():
    assert PassthroughTracker(absolute=True).step() is None


# ------------------------------------------------------------------- the contract
@pytest.mark.parametrize('width', [3, 6, 7, 9])
def test_waypoints_are_normalized_to_the_seven_dim_contract(width):
    a = normalize_waypoint(list(range(width)))
    assert a.shape == (ACTION_DIM,)
    assert np.array_equal(a[:min(width, ACTION_DIM)], np.arange(min(width, ACTION_DIM)))


def test_absolute_rotation_round_trips_through_the_action():
    """action[3:6] is an axis-angle the plant feeds straight to OSC — it must come back out the
    way it went in when the setpoint has reached the target."""
    tracker = AbsoluteTracker(max_step_rot=10.0)   # no rate limit, converge in one tick
    rot = np.array([0.0, 0.0, 0.3])
    tracker.set_waypoint(_waypoint(rot=rot), ee=None)
    action = tracker.step(_pose())

    assert action[3:6] == pytest.approx(quat_to_axisangle(axisangle_to_quat(rot)), abs=1e-9)
