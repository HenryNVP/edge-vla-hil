"""Helpers shared by the launch files in `launch/`.

`typed()` exists because of a genuinely nasty failure mode. A launch argument arrives as a string
and launch_ros infers its parameter type by YAML-parsing it, so `latency_ms:=40` becomes the
INTEGER 40 while `evh_latency` declared a DOUBLE. rclpy then raises InvalidParameterTypeException
and the relay dies at startup — but only the relay. The plant, controller and reactive layer come
up fine, the run completes, and you get a full CSV row labelled "40 ms" with no delay injected at
all. Exactly the plausible-but-wrong metrics this repo's invariant list is about, and the sweep
pipes launch output to DEVNULL, so nothing surfaces the three dead nodes.

Declaring the type at the launch boundary fixes it: `40`, `40.0` and `0` all reach the node as a
float. Use it for every parameter whose node declares a non-string type — test_launch_graph.py
enforces that.
"""
from __future__ import annotations

from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue


def typed(name: str, value_type: type) -> ParameterValue:
    """A launch argument as a parameter of an explicit type, instead of whatever YAML infers."""
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)
