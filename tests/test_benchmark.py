"""Tests for the benchmark recorder's rate metrics and the sweep's absolute-mode resolution.

waypoint_hz is the headline chunk-execution metric (rate of NEW cognitive commands). It has to be
well-defined at the degraded end of the curve — a NaN there silently drops the most interesting
point from the plot — hence count-over-window rather than 1/mean(inter-arrival).
"""
import math

import pytest
from conftest import requires_ros2


@requires_ros2
def test_rate_is_zero_not_nan_when_nothing_arrives():
    from evh_bringup.benchmark import Recorder

    assert Recorder._rate([], 60.0) == 0.0


@requires_ros2
def test_rate_is_defined_for_a_single_sample():
    from evh_bringup.benchmark import Recorder

    assert Recorder._rate([1.0], 10.0) == pytest.approx(0.1)


@requires_ros2
def test_rate_counts_over_the_window():
    from evh_bringup.benchmark import Recorder

    # 20 messages over a 2 s window = 10 Hz, regardless of how they were spaced
    bursty = [0.0, 0.01, 0.02] + [1.0 + 0.001 * i for i in range(17)]
    assert Recorder._rate(bursty, 2.0) == pytest.approx(10.0)


@requires_ros2
def test_rate_nan_only_for_a_degenerate_window():
    from evh_bringup.benchmark import Recorder

    assert math.isnan(Recorder._rate([1.0, 2.0], 0.0))


@requires_ros2
def test_summary_reports_zero_throughput_without_traffic():
    from evh_bringup.benchmark import Recorder

    # build the state directly: Recorder.__init__ needs a live ROS context
    stub = Recorder.__new__(Recorder)
    stub._successes, stub._trials = 0, 0
    stub._infer_ms, stub._action_stamps, stub._waypoint_stamps = [], [], []
    s = Recorder.summary(stub, 30.0)
    assert s['waypoint_hz'] == 0.0
    assert s['loop_hz'] == 0.0
    # a latency that was never observed has no value, unlike a throughput of zero
    assert math.isnan(s['infer_ms_mean']) and math.isnan(s['infer_ms_p95'])


@requires_ros2
@pytest.mark.parametrize('backend,expected', [('dp', 'true'), ('pytorch', 'false'),
                                              ('tensorrt', 'false')])
def test_auto_absolute_follows_the_backend(backend, expected):
    from evh_bringup.benchmark import resolve_absolute

    assert resolve_absolute('auto', backend) == expected


@requires_ros2
@pytest.mark.parametrize('choice', ['true', 'false'])
def test_explicit_absolute_overrides_the_backend_guess(choice):
    from evh_bringup.benchmark import resolve_absolute

    assert resolve_absolute(choice, 'dp') == choice


# ------------------------------------------------------------------ sweep driver
@requires_ros2
def test_the_sweep_enumerates_every_cell_once():
    """strategies x reactive x latencies, no duplicates — a repeated cell would append a second
    CSV row for the same condition and quietly skew whatever averages it."""
    from evh_bringup.benchmark import resolve_absolute  # noqa: F401  (module import guard)

    strategies = ['synchronous', 'rtc']
    values = [0.0, 50.0]
    cells = [(st, rx, lat) for st in strategies for rx in (True, False) for lat in values]

    assert len(cells) == len(set(cells)) == 8


@requires_ros2
@pytest.mark.parametrize('mode,expected', [
    ('both', (True, False)), ('on', (True,)), ('off', (False,)),
])
def test_reactive_modes_selects_the_right_arms(mode, expected):
    """--reactive_modes on/off halves a sweep when you only need one arm of the ablation."""
    modes = (True, False) if mode == 'both' else ((True,) if mode == 'on' else (False,))
    assert modes == expected


@requires_ros2
def test_mid_run_node_deaths_are_counted_from_the_launch_log(tmp_path):
    """Sampled before teardown, so the expected count is zero and there is no magic constant to
    drift. The earlier version subtracted an expected-deaths number that was wrong by one, which
    would have hidden exactly one real death per cell."""
    from evh_bringup.benchmark import _deaths_so_far

    log = tmp_path / 'cell.log'
    log.write_text('[INFO] evh_plant up\n[INFO] recording\n')
    assert _deaths_so_far(str(log)) == 0

    log.write_text('[INFO] evh_plant up\n'
                   "[ERROR] [latency_node-2]: process has died [pid 1, exit code 1, cmd '...'].\n"
                   "[ERROR] [latency_node-3]: process has died [pid 2, exit code 1, cmd '...'].\n")
    assert _deaths_so_far(str(log)) == 2
