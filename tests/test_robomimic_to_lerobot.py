"""Pure-logic tests for the robomimic -> LeRobotDataset converter.

Everything here guards a mistake that produces a *plausible* dataset rather than an error:
episodes misordered against the source, the wrist camera landing in the scene camera's slot,
the alternate action convention accidentally becoming a second policy output.
"""
import numpy as np
import pytest

from robomimic_to_lerobot import (
    EEF_TO_CONTROL_QUAT,
    STATE_KEYS,
    action_names,
    build_features,
    camera_keys,
    derive_absolute,
    osc_scales,
    sorted_demos,
    state_names,
)


def test_demos_sort_numerically_not_alphabetically():
    """demo_10 must follow demo_2 — alphabetical order silently misaligns every episode."""
    assert sorted_demos({f'demo_{i}': None for i in (10, 2, 1, 0, 100)}) == [
        'demo_0', 'demo_1', 'demo_2', 'demo_10', 'demo_100']


def test_wrist_camera_is_ordered_last():
    """ACTBackend maps image_keys[0] <- agentview and [1] <- wrist positionally."""
    assert camera_keys({'robot0_eye_in_hand_image': None, 'agentview_image': None}) == [
        'agentview_image', 'robot0_eye_in_hand_image']
    # tool_hang ships sideview instead of agentview; the wrist still goes last
    assert camera_keys({'robot0_eye_in_hand_image': None, 'sideview_image': None}) == [
        'sideview_image', 'robot0_eye_in_hand_image']


def test_camera_keys_ignores_non_image_observations():
    assert camera_keys({'agentview_image': None, 'object': None, 'robot0_eef_pos': None}) == [
        'agentview_image']


def test_state_names_match_the_proprio_contract():
    names = state_names(STATE_KEYS, [3, 4, 2])
    assert len(names) == 9
    assert names[0] == 'robot0_eef_pos_0'
    assert names[3] == 'robot0_eef_quat_0'
    assert names[-1] == 'robot0_gripper_qpos_1'


def test_alt_action_is_invisible_to_the_policy_feature_mapping():
    """lerobot types any key starting with 'action' as a policy output — alt.action must not."""
    features = build_features(['agentview_image'], 9, state_names(STATE_KEYS, [3, 4, 2]), 7,
                              (84, 84, 3), use_videos=False, with_alt=True)
    assert 'alt.action' in features
    assert not any(key.startswith('action') for key in features if key != 'action')
    assert not any(key.startswith('observation') for key in ('alt.action', 'next.done'))


def test_build_features_shapes_and_dtypes():
    features = build_features(['agentview_image', 'robot0_eye_in_hand_image'], 9,
                              state_names(STATE_KEYS, [3, 4, 2]), 7, (84, 84, 3),
                              use_videos=True, with_alt=False)
    assert 'alt.action' not in features
    assert features['observation.state']['shape'] == (9,)
    assert features['action']['shape'] == (7,)
    assert features['observation.images.agentview']['dtype'] == 'video'
    assert features['observation.images.robot0_eye_in_hand']['names'] == [
        'height', 'width', 'channel']


def _identity_quat(n):
    return np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))


def test_derive_absolute_zero_delta_commands_the_current_pose():
    """Invariant 2's other half: in absolute mode a 'hold' is the current pose, not the origin.

    'The current pose' means the controller's tool frame, which is `robot0_eef_quat` turned -90
    degrees about z — commanding the reported frame instead makes the arm fight a 90-degree twist
    for the whole episode (measured: 0/10 on Lift, vs 7/10 for the same demos as deltas).
    """
    from evh_controller.rotation import axisangle_to_matrix
    from evh_reactive.transforms import quat_mul

    pos = np.array([[0.1, -0.2, 1.0]])
    actions = np.zeros((1, 7))
    out = derive_absolute(actions, pos, _identity_quat(1), np.full(3, 0.05), np.full(3, 0.5))
    assert np.allclose(out[:, :3], pos)                      # position frames share an origin

    expected = quat_mul(np.array([0.0, 0.0, 0.0, 1.0]), EEF_TO_CONTROL_QUAT)
    got = axisangle_to_matrix(out[0, 3:6])
    from evh_reactive.transforms import axisangle_to_quat, quat_to_axisangle
    assert np.allclose(got, axisangle_to_matrix(quat_to_axisangle(expected)), atol=1e-9)
    assert np.allclose(axisangle_to_quat(out[0, 3:6])[:3] @ np.array([0.0, 0.0, 1.0]),
                       -np.sin(np.pi / 4), atol=1e-9)        # the -90 deg z twist, explicitly


def test_derive_absolute_scales_by_output_max_and_clips():
    pos = np.zeros((2, 3))
    actions = np.zeros((2, 7))
    actions[0, 0] = 1.0        # full-scale +x
    actions[1, 0] = 4.0        # beyond the OSC input range
    out = derive_absolute(actions, pos, _identity_quat(2), np.full(3, 0.05), np.full(3, 0.5))
    assert np.allclose(out[0, :3], [0.05, 0.0, 0.0])
    assert np.allclose(out[1, :3], [0.05, 0.0, 0.0])   # clipped to the same target


def test_derive_absolute_composes_rotation_in_the_world_frame():
    """R_goal = R(delta) @ R_current_tool — world-frame delta, controller-frame current."""
    quat = np.tile(np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]), (1, 1))  # +90 deg z
    actions = np.zeros((1, 7))
    actions[0, 5] = 1.0                                 # +0.5 rad about z after scaling
    out = derive_absolute(actions, np.zeros((1, 3)), quat, np.full(3, 0.05), np.full(3, 0.5))
    assert np.allclose(out[0, 3:5], 0.0)
    # +90 (pose) - 90 (tool frame) + 0.5 (the commanded delta), all about z
    assert np.isclose(out[0, 5], 0.5)


def test_derive_absolute_passes_the_gripper_through():
    actions = np.zeros((1, 7))
    actions[0, 6] = -1.0
    out = derive_absolute(actions, np.zeros((1, 3)), _identity_quat(1),
                          np.full(3, 0.05), np.full(3, 0.5))
    assert out[0, 6] == -1.0


def test_osc_scales_reads_both_robosuite_config_shapes():
    flat = {'env_kwargs': {'controller_configs': {'type': 'OSC_POSE',
                                                  'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]}}}
    nested = {'env_kwargs': {'controller_configs': {'body_parts': {'right': {
        'type': 'OSC_POSE', 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]}}}}}
    for meta in (flat, nested):
        pos, rot = osc_scales(meta)
        assert np.allclose(pos, 0.05) and np.allclose(rot, 0.5)


def test_osc_scales_refuses_to_guess():
    with pytest.raises(SystemExit):
        osc_scales({'env_kwargs': {'controller_configs': {'type': 'JOINT_VELOCITY'}}})


# --- absolute rotation encoding ------------------------------------------------------------
# Absolute orientation targets for a downward gripper sit on the pi wrap, where axis-angle is
# discontinuous; the 10-dim rot_6d layout is what makes them learnable (and is what ACTBackend
# recognises as absolute without needing a stamp).

def test_action_names_spell_out_the_ten_dim_absolute_layout():
    assert action_names(10) == ['pos_x', 'pos_y', 'pos_z', 'rot6d_0', 'rot6d_1', 'rot6d_2',
                                'rot6d_3', 'rot6d_4', 'rot6d_5', 'gripper']
    assert action_names(7) == ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
    assert action_names(4) == ['action_0', 'action_1', 'action_2', 'action_3']


def test_alt_action_keeps_the_source_width_when_action_is_ten_dim():
    """The 10-dim column is derived; alt.action stays the untouched 7-dim source."""
    features = build_features(['agentview_image'], 9, state_names(STATE_KEYS, [3, 4, 2]), 10,
                              (84, 84, 3), use_videos=False, with_alt=True, alt_dim=7)
    assert features['action']['shape'] == (10,)
    assert features['alt.action']['shape'] == (7,)
