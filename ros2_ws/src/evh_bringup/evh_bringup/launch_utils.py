"""Helpers shared by the launch files in `launch/`.

`typed()` exists because of a genuinely nasty failure mode. A launch argument arrives as a string
and launch_ros infers its parameter type by YAML-parsing it, so `latency_ms:=40` becomes the
INTEGER 40 while `evh_latency` declared a DOUBLE. rclpy then raises InvalidParameterTypeException
and the relay dies at startup — but only the relay. The plant, controller and reactive layer come
up fine, the run completes, and you get a full CSV row labelled "40 ms" with no delay injected at
all. Exactly the plausible-but-wrong metrics this repo's invariant list is about, and the sweep
pipes launch output to DEVNULL, so nothing surfaces the three dead nodes.

Declaring the type at the launch boundary fixes it. Use `typed()` for every parameter whose node
declares a non-string type — test_launch_graph.py enforces that.

Two rough edges in launch_ros's own coercion, both handled by `_normalize` below:

  * `int` rejects any spelling with a decimal point, so `image_size:=84.0` aborts the launch even
    though 84.0 IS 84. We accept integral floats and still reject genuinely fractional ones — a
    silently truncated `84.7` would be a resolution the policy was never trained at.
  * every failure raises bare ("invalid literal for int() with base 10: '84.0'") without naming
    the argument, the launch file, or what was expected. With a dozen typed arguments across three
    launch files that is a guessing game; we name the argument and say what would be accepted.

`bool` and `float` are left to launch_ros: it already accepts true/True/TRUE/1/yes (and the
negatives) for bool, and 40 / 40.0 / 1e2 for float.

`degrade_when()` switches a relay on for one executor placement only. Which network link a
chunk's actions cross depends on where the executor runs (streamed waypoints vs whole chunks), so
"delay the action path" means a different relay in each placement. It is a small Substitution
rather than a PythonExpression so test_launch_graph.py can still see which arguments it reads.
"""
from __future__ import annotations

from launch import Substitution
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue


class LaunchArgumentTypeError(ValueError):
    """A launch argument's value cannot be read as the type its node declared."""


def _normalize(name: str, text: str, value_type: type) -> str:
    """Rewrite an argument's raw text into something launch_ros will coerce cleanly."""
    text = text.strip()

    if value_type is int:
        try:
            return str(int(text, 10))
        except ValueError:
            pass
        try:
            number = float(text)
        except ValueError:
            raise LaunchArgumentTypeError(
                f'{name}:={text!r} is not a whole number; {name} is an integer parameter') from None
        if not number.is_integer():
            raise LaunchArgumentTypeError(
                f'{name}:={text!r} is not a whole number; {name} is an integer parameter and '
                f'rounding it would silently change the run') from None
        return str(int(number))   # 84.0 -> 84

    if value_type is float:
        try:
            float(text)
        except ValueError:
            raise LaunchArgumentTypeError(
                f'{name}:={text!r} is not a number; {name} is a float parameter '
                f'(e.g. {name}:=40 or {name}:=40.0)') from None

    return text


class _TypedArgument(Substitution):
    """A LaunchConfiguration whose text is normalized for the parameter type it feeds."""

    def __init__(self, name: str, value_type: type) -> None:
        super().__init__()
        self._name = name
        self._value_type = value_type
        self._source = LaunchConfiguration(name)

    @property
    def source(self) -> LaunchConfiguration:
        """The launch argument this reads. test_launch_graph.py traces the wiring through it."""
        return self._source

    def describe(self) -> str:
        return f"typed('{self._name}', {self._value_type.__name__})"

    def perform(self, context) -> str:
        return _normalize(self._name, self._source.perform(context), self._value_type)


def typed(name: str, value_type: type) -> ParameterValue:
    """A launch argument as a parameter of an explicit type, instead of whatever YAML infers."""
    return ParameterValue(_TypedArgument(name, value_type), value_type=value_type)


_TRUE = ('true', '1', 'yes', 'on')


class _DegradeWhen(Substitution):
    """'true' when the path's delay flag is set AND the executor sits where this relay matters."""

    def __init__(self, flag: str, placement: str) -> None:
        super().__init__()
        self._flag = LaunchConfiguration(flag)
        self._executor = LaunchConfiguration('executor')
        self._placement = placement

    @property
    def sources(self) -> tuple[LaunchConfiguration, LaunchConfiguration]:
        """The two launch arguments this reads, for test_launch_graph.py."""
        return (self._flag, self._executor)

    @property
    def placement(self) -> str:
        return self._placement

    def describe(self) -> str:
        return f"degrade_when('{self._flag.variable_name}', executor={self._placement!r})"

    def perform(self, context) -> str:
        on = self._flag.perform(context).strip().lower() in _TRUE
        here = self._executor.perform(context).strip().lower() == self._placement
        return 'true' if on and here else 'false'


def degrade_when(flag: str, placement: str) -> ParameterValue:
    """A relay's `enabled`: on only if `flag` (delay_obs / delay_act) is set and the launch's
    `executor` argument equals `placement`."""
    return ParameterValue(_DegradeWhen(flag, placement), value_type=bool)


class _ImageMsgType(Substitution):
    """The observation image topics' ROS type, which `image_quality` decides.

    The relay is type-generic but has to be told which type it carries, and DDS pairs nothing
    across a type mismatch — silently. So the relay's type is derived from the same launch argument
    the plant and controller read, rather than written out three times.
    """

    def __init__(self, name: str = 'image_quality') -> None:
        super().__init__()
        self._name = name
        self._source = LaunchConfiguration(name)

    @property
    def source(self) -> LaunchConfiguration:
        """The launch argument this reads, for test_launch_graph.py."""
        return self._source

    def describe(self) -> str:
        return f"image_msg_type('{self._name}')"

    def perform(self, context) -> str:
        quality = int(_normalize(self._name, self._source.perform(context), int))
        return 'sensor_msgs/msg/Image' if quality <= 0 else 'sensor_msgs/msg/CompressedImage'


def image_msg_type(name: str = 'image_quality') -> ParameterValue:
    """A relay's `msg_type` for an observation image link, derived from `image_quality`."""
    return ParameterValue(_ImageMsgType(name), value_type=str)
