"""Tests for the plant's observation contract — the layout every downstream package assumes.

Needs ROS message classes (hence the ros2 mark) but no node, no robosuite and no simulator: these
build a PlantObservation from a plain dict and check what comes out the other side. The proprio
layout and the [x, y, z, w] quaternion order are the two things that, if reordered, feed the
policy plausible-looking garbage rather than raising anything.
"""
import numpy as np
import pytest
from conftest import requires_ros2


def _obs_dict(**overrides):
    obs = {
        'agentview_image': np.zeros((4, 4, 3), np.uint8),
        'robot0_eye_in_hand_image': np.ones((4, 4, 3), np.uint8),
        'robot0_joint_pos': np.arange(7, dtype=float),
        'robot0_joint_vel': np.arange(7, dtype=float) * 0.1,
        'robot0_eef_pos': np.array([0.1, 0.2, 0.3]),
        'robot0_eef_quat': np.array([0.0, 0.0, 0.7071, 0.7071]),
        'robot0_gripper_qpos': np.array([0.02, -0.02]),
    }
    obs.update(overrides)
    return obs


CAMERAS = ['agentview', 'robot0_eye_in_hand']


def _stamp():
    """ROS header stamps are validated on assignment; any real Time will do here."""
    from builtin_interfaces.msg import Time
    return Time(sec=1, nanosec=0)


@requires_ros2
def test_proprio_is_pos_then_quat_then_gripper():
    """[eef_pos(3), eef_quat(4, xyzw), gripper_qpos(2)] — the controller slices it by index."""
    from evh_plant.messages import PlantObservation

    obs = PlantObservation.from_robosuite(_obs_dict(), CAMERAS, want_wrist=True)
    position = list(obs.proprio_msg(_stamp()).position)   # ROS gives an array('d')

    assert len(position) == 9
    assert position[:3] == [0.1, 0.2, 0.3]
    assert position[3:7] == [0.0, 0.0, 0.7071, 0.7071]
    assert position[7:] == [0.02, -0.02]


@requires_ros2
def test_ee_pose_keeps_the_xyzw_quaternion_order():
    """geometry_msgs stores w last too, but the mapping is written out by hand — a swap here
    rotates every reactive-layer target and nothing errors."""
    from evh_plant.messages import PlantObservation

    obs = PlantObservation.from_robosuite(_obs_dict(), CAMERAS, want_wrist=False)
    pose = obs.ee_pose_msg(_stamp()).pose

    assert (pose.position.x, pose.position.y, pose.position.z) == (0.1, 0.2, 0.3)
    assert pose.orientation.z == pytest.approx(0.7071)
    assert pose.orientation.w == pytest.approx(0.7071)


@requires_ros2
def test_camera_frames_are_flipped_upright():
    """robosuite renders bottom-up; the policy was trained on upright frames."""
    from evh_plant.messages import PlantObservation

    raw = np.zeros((4, 4, 3), np.uint8)
    raw[0, :, :] = 255                      # bottom row as robosuite hands it over
    obs = PlantObservation.from_robosuite(
        _obs_dict(agentview_image=raw), CAMERAS, want_wrist=False)

    assert np.all(obs.frame[-1] == 255), 'frame was not flipped'
    assert np.all(obs.frame[0] == 0)
    assert obs.frame.dtype == np.uint8


@requires_ros2
def test_image_message_describes_the_frame_it_carries():
    from evh_plant.messages import PlantObservation

    obs = PlantObservation.from_robosuite(_obs_dict(), CAMERAS, want_wrist=False)
    msg = obs.image_msg(_stamp())

    assert (msg.height, msg.width) == (4, 4)
    assert msg.encoding == 'rgb8'
    assert msg.step == 12
    assert len(msg.data) == 4 * 4 * 3


@requires_ros2
@pytest.mark.parametrize('want_wrist', [True, False])
def test_the_wrist_frame_is_only_built_when_asked_for(want_wrist):
    """The wrist publisher only exists with a second camera configured; building the frame anyway
    would flip and copy an 84x84 array 20 times a second for nothing."""
    from evh_plant.messages import PlantObservation

    obs = PlantObservation.from_robosuite(_obs_dict(), CAMERAS, want_wrist=want_wrist)
    assert (obs.wrist is not None) is want_wrist
    assert (obs.wrist_msg(_stamp()) is not None) is want_wrist


@requires_ros2
def test_an_obs_without_an_image_yields_nothing_to_publish():
    from evh_plant.messages import PlantObservation

    obs = _obs_dict()
    del obs['agentview_image']
    assert PlantObservation.from_robosuite(obs, CAMERAS, want_wrist=False) is None


@requires_ros2
def test_missing_proprio_fields_fall_back_to_a_neutral_pose():
    """Degraded but publishable: unlike the hold action, an observation must never raise — the
    graph has to keep running so the failure is visible in the metrics, not as a dead node."""
    from evh_plant.messages import PlantObservation

    obs = PlantObservation.from_robosuite(
        {'agentview_image': np.zeros((4, 4, 3), np.uint8)}, CAMERAS, want_wrist=False)

    assert np.all(obs.ee_pos == 0.0)
    assert list(obs.ee_quat) == [0.0, 0.0, 0.0, 1.0], 'identity quaternion, not zeros'


@requires_ros2
def test_the_synthetic_observation_matches_the_configured_image_size():
    """robosuite-less degraded mode still has to produce publishable messages."""
    from evh_plant.messages import PlantObservation

    obs = PlantObservation.synthetic(84, want_wrist=True)

    assert obs.frame.shape == (84, 84, 3) and obs.frame.dtype == np.uint8
    assert obs.wrist is not None
    assert len(obs.proprio_msg(_stamp()).position) == 9
    assert list(obs.ee_quat) == [0.0, 0.0, 0.0, 1.0]
