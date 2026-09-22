"""Tests for the WiFi measurement analysis — pure Python, no ROS, no radio.

The numbers this produces become relay parameters for the paper's "realistic" cells, so the fit
has to recover what a known channel actually did, and the clock-offset correction has to be right
or every one-way delay is off by the offset.
"""
import random

import pytest
from wifi_trace import analyze, estimate_offset, fit_gilbert, loss_runs, quantiles_ms

from evh_latency.channel import GilbertElliott


def test_loss_runs_counts_sent_and_groups_consecutive_gaps():
    sent, runs = loss_runs([0, 1, 4, 5, 9], first=0, last=10)
    assert sent == 11
    assert runs == [2, 3, 1]          # 2-3, 6-8, 10


def test_nothing_lost_means_no_runs_and_no_invented_burst():
    sent, runs = loss_runs(list(range(20)), 0, 19)
    assert runs == []
    assert fit_gilbert(sent, runs, 0.05) == {'loss': 0.0, 'burst_ms': None, 'bursts': 0}


def test_the_fit_recovers_a_known_gilbert_elliott_channel():
    """Round trip through the relay's own model: sample it at 20 Hz, fit, get the knobs back."""
    ge = GilbertElliott(loss=0.1, burst_ms=400.0, seed=4)
    received = [i for i in range(40000) if not ge.bad(1000.0 + i * 0.05)]
    sent, runs = loss_runs(received, 0, 39999)
    fit = fit_gilbert(sent, runs, 0.05)

    assert fit['loss'] == pytest.approx(0.1, abs=0.02)
    # message-sampled runs slightly overstate a continuous outage (partial periods round up)
    assert 350.0 < fit['burst_ms'] < 500.0


def test_offset_is_recovered_from_symmetric_probes_despite_queueing():
    """Server clock 250 ms ahead; one-way 4 ms each way, plus one-sided queueing on most probes.
    The minimum-delay filter must see through the queueing."""
    rng = random.Random(0)
    probes = []
    for k in range(200):
        t1 = 100.0 + k * 0.1
        up, down = 0.004 + rng.expovariate(1 / 0.02), 0.004 + rng.expovariate(1 / 0.02)
        t2 = t1 + up + 0.25
        t3 = t2 + 0.0005
        t4 = t3 - 0.25 + down
        probes.append((t1, t2, t3, t4))
    off = estimate_offset(probes)
    assert off.offset_s == pytest.approx(0.25, abs=0.003)
    assert off.rtt_s < 0.02


def test_no_probes_is_an_error_not_a_zero_offset():
    with pytest.raises(ValueError):
        estimate_offset([])


def test_quantiles_are_in_milliseconds_and_empty_is_none():
    q = quantiles_ms([0.001 * i for i in range(1, 101)])
    assert q['p50'] == pytest.approx(50.0) and q['max'] == pytest.approx(100.0)
    assert quantiles_ms([])['p95'] is None


def _rows(stream, seqs, sent_at, delay, clock_shift=0.0):
    return [{'stream': stream, 'seq': s, 'sent': sent_at(s), 'recv': sent_at(s) + delay
             + clock_shift} for s in seqs]


def test_analyze_corrects_each_direction_by_the_clock_offset():
    """Up-stream receive times are on the server clock, down-stream ones on the robot clock;
    getting the sign wrong would add the offset to one direction and subtract it from the other."""
    offset = 0.5                                  # server clock ahead of the robot's
    period = 0.05

    def at(s):
        return 10.0 + s * period

    robot = ([{'stream': 'sent:image', 'seq': s, 'sent': at(s)} for s in range(100)]
             + _rows('waypoint', range(100), lambda s: at(s) + offset, 0.030, -offset)
             + [{'stream': 'probe', 't1': 1.0, 't2': 1.002 + offset, 't3': 1.0021 + offset,
                 't4': 1.0041}] * 5)
    server = ([{'stream': 'sent:waypoint', 'seq': s, 'sent': at(s) + offset} for s in range(100)]
              + _rows('image', [s for s in range(100) if s not in (10, 11, 12)], at, 0.012,
                      offset))
    result = analyze(robot, server, 'unit', trim_s=0.0)

    assert result['streams']['image']['delay_ms']['p50'] == pytest.approx(12.0, abs=0.5)
    assert result['streams']['waypoint']['delay_ms']['p50'] == pytest.approx(30.0, abs=0.5)
    assert result['streams']['image']['gilbert']['loss'] == pytest.approx(0.03)
    assert result['streams']['image']['gilbert']['burst_ms'] == pytest.approx(150.0)
    assert result['streams']['waypoint']['gilbert']['loss'] == 0.0


def test_start_and_stop_are_not_counted_as_loss():
    """Messages sent before the receiver started listening are missing too, but not lost."""
    def at(s):
        return 10.0 + s * 0.05

    robot = ([{'stream': 'sent:proprio', 'seq': s, 'sent': at(s)} for s in range(200)]
             + [{'stream': 'probe', 't1': 1.0, 't2': 1.001, 't3': 1.0011, 't4': 1.0021}])
    server = _rows('proprio', range(10, 200), at, 0.005)      # receiver joined 0.5 s late
    stats = analyze(robot, server, trim_s=1.0)['streams']['proprio']
    assert stats['gilbert']['loss'] == 0.0
