"""Tests for the launch-argument type coercion — pure Python beyond the launch package itself.

Two separate defects are guarded here, both of the same family: a launch argument is a string, and
what happens when its spelling does not match the type its node declared.

  1. `latency_ms:=40` used to reach a DOUBLE parameter as an INTEGER, killing all three relays at
     startup while the rest of the graph ran on and produced a full, undegraded result under a
     label claiming 40 ms. typed() is what prevents that; test_launch_graph.py checks it is
     applied everywhere it needs to be, and these check that it actually coerces.
  2. `image_size:=84.0` then aborted the whole launch, because launch_ros's int() refuses any
     spelling with a decimal point — even one that is exactly a whole number.
"""
import pytest
from conftest import requires_ros2


def _evaluate(name, raw, value_type):
    from launch import LaunchContext

    from evh_bringup.launch_utils import typed
    ctx = LaunchContext()
    ctx.launch_configurations[name] = raw
    return typed(name, value_type).evaluate(ctx)


# ------------------------------------------------------------------------- floats
@requires_ros2
@pytest.mark.parametrize('raw', ['40', '40.0', '0', '0.0', '1e2', '-5'])
def test_a_float_parameter_accepts_every_reasonable_spelling(raw):
    """`latency_ms:=40` is what a person types; it must not be an integer by the time it lands."""
    value = _evaluate('latency_ms', raw, float)
    assert isinstance(value, float)
    assert value == float(raw)


@requires_ros2
def test_a_non_numeric_float_names_the_argument_it_came_from():
    """The bare launch_ros error is "could not convert string to float: 'abc'" — with a dozen
    typed arguments across three launch files, that is a guessing game."""
    from evh_bringup.launch_utils import LaunchArgumentTypeError

    with pytest.raises(LaunchArgumentTypeError, match='latency_ms'):
        _evaluate('latency_ms', 'abc', float)


# --------------------------------------------------------------------------- ints
@requires_ros2
@pytest.mark.parametrize('raw,expected', [('84', 84), ('84.0', 84), ('16', 16), ('0', 0)])
def test_an_integral_float_is_accepted_for_an_int_parameter(raw, expected):
    """84.0 IS 84. Refusing it aborted the entire launch for a spelling difference."""
    value = _evaluate('image_size', raw, int)
    assert value == expected
    assert isinstance(value, int)


@requires_ros2
@pytest.mark.parametrize('raw', ['84.7', '83.5'])
def test_a_fractional_value_is_refused_rather_than_rounded(raw):
    """Silently truncating to 84 would run the policy at a resolution it was never trained at,
    which is precisely the plausible-but-wrong outcome this coercion exists to prevent."""
    from evh_bringup.launch_utils import LaunchArgumentTypeError

    with pytest.raises(LaunchArgumentTypeError, match='whole number'):
        _evaluate('image_size', raw, int)


@requires_ros2
def test_a_non_numeric_int_names_the_argument_it_came_from():
    from evh_bringup.launch_utils import LaunchArgumentTypeError

    with pytest.raises(LaunchArgumentTypeError, match='image_size'):
        _evaluate('image_size', 'small', int)


# -------------------------------------------------------------------------- bools
@requires_ros2
@pytest.mark.parametrize('raw,expected', [
    ('true', True), ('True', True), ('TRUE', True), ('1', True), ('yes', True),
    ('false', False), ('False', False), ('0', False), ('no', False),
])
def test_bool_spellings_are_left_to_launch_ros_and_all_work(raw, expected):
    """`absolute` and `strict_mode_check` decide whether a run is meaningful at all (invariant 1),
    so every spelling a person might reach for has to land on the right side."""
    assert _evaluate('absolute', raw, expected.__class__) is expected


@requires_ros2
def test_whitespace_around_a_value_is_tolerated():
    assert _evaluate('latency_ms', '  40.0  ', float) == 40.0
    assert _evaluate('image_size', ' 84 ', int) == 84
