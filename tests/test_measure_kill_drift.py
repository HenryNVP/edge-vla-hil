import numpy as np
import pytest

import measure_kill_drift as mkd


def _line(speed_before, speed_after, t_kill=1.0, window=4.0, hz=50):
    t = np.arange(0, t_kill + window + 0.5, 1 / hz)
    z = np.where(t <= t_kill, speed_before * t,
                 speed_before * t_kill + speed_after * (t - t_kill))
    return t, np.stack([np.zeros_like(t), np.zeros_like(t), z], axis=1)


def test_a_standing_velocity_drifts_speed_times_window():
    t, xyz = _line(0.05, 0.05)
    r = mkd.drift_after_kill(t, xyz, 1.0, 4.0)
    assert r['drift_mm'] == pytest.approx(200, abs=1.5)
    assert r['speed_at_kill_mm_s'] == pytest.approx(50, abs=0.5)
    assert r['path_mm'] == pytest.approx(r['drift_mm'], rel=1e-6)


def test_a_hold_after_the_kill_drifts_nothing():
    r = mkd.drift_after_kill(*_line(0.05, 0.0), 1.0, 4.0)
    assert r['drift_mm'] == pytest.approx(0, abs=1.1)


def test_path_counts_a_back_and_forth_that_displacement_hides():
    t = np.linspace(0, 2, 201)
    xyz = np.zeros((t.size, 3))
    xyz[:, 0] = np.where(t <= 1.5, 0.1 * np.clip(t - 1, 0, None), 0.1 * (2 - t))
    r = mkd.drift_after_kill(t, xyz, 1.0, 1.0)
    assert r['drift_mm'] < 1 and r['path_mm'] == pytest.approx(100, abs=1)


def test_missing_samples_on_either_side_of_the_kill_raise():
    t, xyz = _line(0.05, 0.05)
    with pytest.raises(ValueError):
        mkd.drift_after_kill(t, xyz, -1.0, 4.0)
    with pytest.raises(ValueError):
        mkd.drift_after_kill(t, xyz, t[-1] + 1, 4.0)


def test_summarize_is_per_arm():
    rows = [{'arm': 'a', 'drift_mm': d, 'speed_at_kill_mm_s': 10} for d in (1, 2, 9)] + \
           [{'arm': 'b', 'drift_mm': 100, 'speed_at_kill_mm_s': 40}]
    s = mkd.summarize(rows)
    assert s['a'] == {'n': 3, 'drift_median': 2, 'drift_min': 1, 'drift_max': 9,
                      'speed_median': 10}
    assert s['b']['n'] == 1
