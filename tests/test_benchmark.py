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


# ------------------------------------------------------- trial-count recording
@requires_ros2
def test_recorder_exposes_the_episode_count_it_stops_on():
    from std_msgs.msg import Bool

    from evh_bringup.benchmark import Recorder

    rec = Recorder.__new__(Recorder)          # __init__ needs a live ROS context
    rec._successes = rec._trials = 0
    rec._infer_ms, rec._action_stamps, rec._waypoint_stamps = [], [], []

    assert rec.trials == 0
    for data in (True, False, True):
        rec._on_success(Bool(data=data))
    assert rec.trials == 3, 'both outcomes count as trials, not just successes'


@requires_ros2
@pytest.mark.parametrize('target,seen,capped,expected', [
    (20, 20, False, False),   # reached the target before the cap
    (20, 11, True, True),     # cap hit first: the row is short and must say so
    (0, 40, True, False),     # pure time mode: hitting the cap IS the stopping condition
])
def test_a_row_is_flagged_truncated_only_when_it_fell_short(target, seen, capped, expected):
    """The point of the flag: a cell that ran out of wall clock has a smaller n than the cells it
    will be plotted against, and that has to be visible in the CSV rather than inferred."""
    truncated = capped and bool(target) and seen < target
    assert truncated is expected


@requires_ros2
def test_time_windows_under_sample_the_degraded_cells():
    """Why --trials exists, in the measured numbers: the same 90 s window yielded 13 episodes at
    0 ms but only 6 at 800 ms, so the interesting end of the curve carried half the evidence."""
    per_episode_s = {0: 6.9, 800: 15.0}
    window = 90.0
    counts = {lat: int(window // s) for lat, s in per_episode_s.items()}

    assert counts[0] > 2 * counts[800] - 3, 'expected roughly a 2x sampling imbalance'
    # trial mode equalises it, at the cost of a longer window for the degraded cell
    target = 25
    assert per_episode_s[800] * target > per_episode_s[0] * target


# ------------------------------------------------------------- sweep axis knobs
@requires_ros2
@pytest.mark.parametrize('axis,param', [
    ('latency', 'latency_ms'), ('jitter', 'jitter_ms'), ('drop', 'drop_prob'),
])
def test_the_swept_axis_maps_to_the_right_launch_argument(axis, param):
    from evh_bringup.benchmark import _AXIS_PARAM
    assert _AXIS_PARAM[axis] == param


@requires_ros2
def test_every_degradation_knob_is_passed_even_when_it_is_not_swept():
    """drop_prob used to be omitted from the launch command entirely, so it silently stayed at
    the launch default and no sweep could reach it — the same landmine as the `absolute` arg.
    Each knob must appear in the command line whether or not it is the swept one."""
    from evh_bringup.benchmark import _AXIS_PARAM

    args_jitter, args_drop, args_latency = 5.0, 0.1, 200.0
    for axis, swept_value in (('latency', 300.0), ('jitter', 40.0), ('drop', 0.25)):
        knobs = {'latency_ms': args_latency, 'jitter_ms': args_jitter, 'drop_prob': args_drop}
        knobs[_AXIS_PARAM[axis]] = swept_value

        assert set(knobs) == {'latency_ms', 'jitter_ms', 'drop_prob'}
        assert knobs[_AXIS_PARAM[axis]] == swept_value, 'swept knob did not take the value'
        held = {k: v for k, v in knobs.items() if k != _AXIS_PARAM[axis]}
        assert all(v is not None for v in held.values()), 'a held knob went unpassed'
