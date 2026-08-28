"""Shared test setup.

Makes the package sources importable without a colcon build (handy for the pure-Python logic
tests), and provides an rclpy lifecycle fixture for the node-boot smoke tests. Anything needing
rclpy / ROS2 message packages is skipped automatically when ROS2 is not sourced, so the suite
stays green on a bare Python box and only exercises the full graph on the host/Jetson.
"""
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, 'ros2_ws', 'src')

# Put each package root on sys.path: ros2_ws/src/<pkg> -> import <pkg>.<module>
for _pkg in ('evh_plant', 'evh_controller', 'evh_reactive', 'evh_latency', 'evh_bringup'):
    _path = os.path.join(_SRC, _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _ros2_available() -> bool:
    try:
        import rclpy  # noqa: F401
        import sensor_msgs.msg  # noqa: F401
        return True
    except Exception:
        return False


ROS2 = _ros2_available()
_skip_no_ros2 = pytest.mark.skipif(not ROS2, reason='ROS2 / rclpy not sourced')


def requires_ros2(func):
    """Skip when ROS2 is not sourced, AND tag with the registered `ros2` marker.

    The skip is what keeps the suite green on a bare Python box; the marker is what makes the
    split selectable — `pytest -m "not ros2"` runs only the pure-Python logic tests, so CI can
    assert that set really is import-free rather than inferring it from a pile of skips.

    `ros2` means "needs ROS installed" — most such tests just import a message type and run in
    milliseconds. For "needs a running ROS graph", use `integration` below.
    """
    return pytest.mark.ros2(_skip_no_ros2(func))


def integration(func):
    """Tag a test that spins real nodes and talks over DDS. Implies requires_ros2.

    These are a different animal from the rest of the ros2-marked tests: ~25 of them account for
    almost all of the suite's wall-clock time, and they are the only ones exposed to cross-talk
    from a zombie container or a stale node on the same ROS_DOMAIN_ID — the trap that makes
    metrics look self-contradictory. Having them selectable is what lets you give them their own
    domain instead of hoping the ambient one is clean:

        ROS_DOMAIN_ID=$RANDOM pytest -m integration     # isolated, ~18s
        pytest -m 'not integration'                     # everything else, ~0.2s
    """
    return pytest.mark.integration(requires_ros2(func))


@pytest.fixture()
def ros():
    """Init/shutdown rclpy around a test."""
    import rclpy
    rclpy.init()
    try:
        yield rclpy
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def spin_until(node, predicate, timeout=5.0, period=0.02):
    """Spin a node until predicate() is true or timeout; return predicate()'s final value."""
    import time

    import rclpy
    end = time.time() + timeout
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=period)
        if predicate():
            return True
    return predicate()
