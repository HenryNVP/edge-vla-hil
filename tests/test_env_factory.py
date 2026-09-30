"""Tests for robosuite env construction — pure Python, no ROS and no robosuite needed.

These used to live in test_plant_node.py and carry a ros2 mark, only because the code was
inside plant_node.py and importing it pulled in rclpy. Moving the simulator-facing logic into its
own module put them in the fast suite, which is the point of the split.

`set_control_delta` is invariant 1's mechanism: it is what actually puts OSC into absolute mode,
and it has to reach the right key on both the flat 1.4 config and the composite 1.5 one.
"""
import pytest

from evh_plant.env_factory import EnvSpec, set_control_delta


@pytest.mark.parametrize('value', [True, False])
def test_set_control_delta_on_the_flat_robosuite_14_config(value):
    cfg = {'type': 'OSC_POSE', 'control_delta': not value}
    set_control_delta(cfg, value)
    assert cfg['control_delta'] is value


@pytest.mark.parametrize('value', [True, False])
def test_set_control_delta_reaches_into_a_composite_15_config(value):
    cfg = {'body_parts': {'right': {'type': 'OSC_POSE', 'control_delta': not value},
                          'base': {'type': 'JOINT_VELOCITY'},
                          'gripper': 'not-a-dict'}}
    set_control_delta(cfg, value)

    assert cfg['body_parts']['right']['control_delta'] is value
    assert 'control_delta' not in cfg['body_parts']['base'], 'only OSC parts take control_delta'


def test_set_control_delta_tolerates_a_config_it_does_not_recognise():
    cfg = {'type': 'JOINT_POSITION'}
    set_control_delta(cfg, False)      # must not raise
    assert cfg == {'type': 'JOINT_POSITION'}


def test_horizon_is_episode_seconds_in_action_steps():
    """robosuite counts the horizon in control steps, and hitting it is a recorded timeout —
    an off-by-a-factor here silently changes every episode length in the sweep."""
    assert EnvSpec(max_episode_s=20.0, action_hz=250.0).horizon == 5000
    assert EnvSpec(max_episode_s=2.5, action_hz=20.0).horizon == 50


def test_spec_defaults_match_the_dp_lift_checkpoint():
    spec = EnvSpec()
    assert spec.image_size == 84, 'DP checkpoints are trained at 84x84'
    assert spec.absolute_actions is True
    assert spec.cameras[0] == 'agentview'


# ------------------------------------------------------------------- timebase
@pytest.mark.parametrize('hz,substeps', [(250.0, 2), (100.0, 5), (20.0, 25), (500.0, 1)])
def test_rates_that_divide_the_timestep_are_accepted(hz, substeps):
    from evh_plant.env_factory import check_timebase
    assert check_timebase(hz, 0.002) == substeps


@pytest.mark.parametrize('hz', [200.0, 300.0, 1000.0])
def test_a_rate_that_would_drop_part_of_a_substep_is_refused(hz):
    """200 Hz asked robosuite for 2.5 substeps and got 2: simulated time ran at 80% of wall
    time and the policy acted every 40 ms of sim time instead of 50."""
    from evh_plant.env_factory import TimebaseError, check_timebase
    with pytest.raises(TimebaseError, match='simulated time would drift'):
        check_timebase(hz, 0.002)


def test_the_default_plant_rate_is_exact():
    from evh_plant.env_factory import EnvSpec, check_timebase
    assert check_timebase(EnvSpec().action_hz, 0.002) == 2


# ------------------------------------------------------------------ hold_action
def test_hold_in_delta_mode_is_zeros():
    import numpy as np

    from evh_plant.env_factory import hold_action
    obs = {'robot0_eef_pos': np.array([0.1, 0.2, 0.9]), 'robot0_eef_quat': np.array([1, 0, 0, 0.])}
    assert not hold_action(obs, False, 7).any()


def test_hold_in_absolute_mode_is_the_current_pose_in_the_controller_frame():
    """Invariants 2 and 7: never the origin, and never the reported frame."""
    import numpy as np

    from evh_plant.env_factory import eef_to_control_quat, hold_action, quat_to_axisangle
    q = np.array([1.0, 0.0, 0.0, 0.0])
    obs = {'robot0_eef_pos': np.array([0.1, 0.2, 0.9]), 'robot0_eef_quat': q}
    hold = hold_action(obs, True, 7)
    assert np.allclose(hold[:3], [0.1, 0.2, 0.9])
    assert np.allclose(hold[3:6], quat_to_axisangle(eef_to_control_quat(q)))
    assert not np.allclose(hold[3:6], quat_to_axisangle(q))
    assert hold[6] == 0.0


def test_hold_in_absolute_mode_refuses_to_guess_a_missing_pose():
    import numpy as np

    from evh_plant.env_factory import hold_action
    with pytest.raises(RuntimeError):
        hold_action({'robot0_eef_pos': np.zeros(3)}, True, 7)
