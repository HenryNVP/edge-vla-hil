"""Regression tests for the plant's 'no command yet' hold action (episode start / post-reset).

The bug this guards: in absolute mode (OSC control_delta=False) a zero action is an ABSOLUTE pose
target at the world origin, so defaulting to zeros before the first /cmd/action yanked the arm off
the table every reset. The hold must instead command the CURRENT EE pose. Delta mode stays zeros.

Pure logic — the hold builder and quaternion helper are exercised via a stub, so no robosuite env
(and no GPU) is needed; the import still needs rclpy, hence the ros2 mark.
"""
import types

import numpy as np
import pytest
from conftest import requires_ros2


@requires_ros2
def test_quat_to_axisangle_matches_reactive_transforms():
    from evh_plant.plant_node import _quat_to_axisangle
    from evh_reactive.transforms import axisangle_to_quat, quat_to_axisangle

    # identity -> zero rotation
    assert np.allclose(_quat_to_axisangle([0.0, 0.0, 0.0, 1.0]), np.zeros(3))
    # round-trips a handful of rotations and agrees with the reactive-layer implementation
    for aa in ([0.3, -0.1, 0.7], [np.pi / 2, 0.0, 0.0], [0.0, 2.5, 0.0]):
        q = axisangle_to_quat(np.asarray(aa))
        assert np.allclose(_quat_to_axisangle(q), quat_to_axisangle(q), atol=1e-9)


@requires_ros2
def test_hold_action_absolute_commands_current_pose_not_origin():
    from evh_plant.plant_node import PlantNode, _quat_to_axisangle

    ee_pos = np.array([0.4, -0.1, 1.05])
    ee_quat = np.array([0.0, 0.0, 0.0, 1.0])
    stub = types.SimpleNamespace(
        absolute_actions=True, _action_dim=7,
        _obs={'robot0_eef_pos': ee_pos, 'robot0_eef_quat': ee_quat})

    hold = PlantNode._hold_action(stub)
    assert hold.shape == (7,)
    assert np.allclose(hold[:3], ee_pos)                      # NOT the origin
    assert np.allclose(hold[3:6], _quat_to_axisangle(ee_quat))
    assert hold[6] == 0.0                                     # gripper neutral
    assert not np.allclose(hold[:3], 0.0)                     # the actual regression guard


@requires_ros2
def test_hold_action_delta_mode_is_zeros():
    from evh_plant.plant_node import PlantNode

    stub = types.SimpleNamespace(
        absolute_actions=False, _action_dim=7,
        _obs={'robot0_eef_pos': np.array([0.4, -0.1, 1.05])})
    assert np.allclose(PlantNode._hold_action(stub), np.zeros(7))


@requires_ros2
def test_hold_action_absolute_refuses_to_default_a_missing_pose():
    """A .get() default here would silently rebuild the origin lurch; raising is the point."""
    from evh_plant.plant_node import PlantNode

    stub = types.SimpleNamespace(absolute_actions=True, _action_dim=7, _obs={'unrelated': 1})
    with pytest.raises(RuntimeError, match='world origin'):
        PlantNode._hold_action(stub)


@requires_ros2
def test_hold_action_absolute_without_obs_falls_back_to_zeros():
    """Before the first obs (self._obs is None) there is nothing to hold to; zeros is the only
    safe default. The origin-lurch window here is unavoidable but momentary (pre-first-step)."""
    from evh_plant.plant_node import PlantNode

    stub = types.SimpleNamespace(absolute_actions=True, _action_dim=7, _obs=None)
    assert np.allclose(PlantNode._hold_action(stub), np.zeros(7))
