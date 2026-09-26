"""Pins the arithmetic behind the second term, and the filter that makes cells comparable.

The finding is that blocking's commanded path length is invariant under degradation while
naive_async's is not. Both halves rest on `commanded_path_mm` being a product of three per-cell means
and on `load_cells` comparing only cells that differ in the factor under study -- a cell with jitter,
loss, the reactive layer on, or the executor on the policy side is not comparable, and silently
including one would move the mean without moving anything visible.
"""
import csv

import pytest

import analyze_path


def test_commanded_path_is_rate_times_step_times_time():
    # 20 waypoints/s x 10 mm each for 8 s = 1600 mm of commanded travel.
    assert analyze_path.commanded_path_mm(20.0, 10.0, 8.0) == pytest.approx(1600.0)


def test_halving_the_rate_at_constant_path_doubles_the_time():
    """The blocking signature: duty falls, time grows, distance does not change.

    This is the whole claim in one assertion -- if a strategy commands the same distance at half the
    rate, it takes twice as long, and success rate then depends only on where the deadline sits.
    """
    fast = analyze_path.commanded_path_mm(20.0, 10.0, 8.0)
    slow = analyze_path.commanded_path_mm(10.0, 10.0, 16.0)
    assert fast == pytest.approx(slow)


def test_ols_recovers_a_planted_slope():
    xs = [2, 6, 10, 18, 34]
    ys = [100 + 7 * x for x in xs]
    intercept, slope, t = analyze_path.ols(xs, ys)
    assert slope == pytest.approx(7.0)
    assert intercept == pytest.approx(100.0)
    assert t > 1e6          # an exact fit has no residual, so the t statistic diverges


def _write(tmp_path, name, rows, episodes):
    """A sweep pair on disk: the summary CSV and the per-episode file beside it."""
    cols = ['condition', 'strategy', 'placement', 'executor', 'reactive', 'jitter_ms', 'drop_prob',
            'd_chunk_steps_p50', 'wp_rx_hz', 'wp_step_mm', 'success_rate',
            'max_chunk_actions', 'max_episode_s']
    with open(tmp_path / f'{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})
    with open(tmp_path / f'{name}.episodes.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'scene', 'success', 'wall_s'])
        for cond, scene, ok, wall in episodes:
            w.writerow([cond, scene, ok, wall])


_BASE = dict(strategy='synchronous', placement='act', executor='robot', reactive='False',
             jitter_ms='0', drop_prob='0', d_chunk_steps_p50='10', wp_rx_hz='12',
             wp_step_mm='10', success_rate='0.9', max_chunk_actions='15', max_episode_s='20')


def test_a_comparable_cell_is_loaded_with_its_median_and_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, 'e', [dict(_BASE, condition='c')],
           [('c', 0, 1, 99.0),          # scene 0 is dropped as an unreliable upper bound
            ('c', 1, 1, 10.0), ('c', 2, 1, 12.0), ('c', 3, 0, 20.0)])
    cells = analyze_path.load_cells('*.csv')
    assert len(cells) == 1
    assert cells[0]['median_s'] == pytest.approx(11.0)       # median of 10 and 12, not of 99
    assert cells[0]['path_mm'] == pytest.approx(12 * 10 * 11.0)
    assert cells[0]['duty'] == pytest.approx(15 / 25)


@pytest.mark.parametrize('override, why', [
    ({'jitter_ms': '20'}, 'jitter changes the delay distribution, not just its mean'),
    ({'drop_prob': '0.05'}, 'loss removes commands, which moves path length on its own'),
    ({'reactive': 'True'}, 'the reactive layer interpolates its own waypoints'),
    ({'executor': 'policy'}, 'streamed execution has no buffer, so k is not the horizon'),
    ({'placement': 'obs'}, 'observation-path delay does not enter the round trip the same way'),
    ({'strategy': 'rtc'}, 'overlap methods are not a waiting strategy and have no path law'),
    ({'d_chunk_steps_p50': '0'}, 'a cell with no measured round trip has no duty cycle'),
])
def test_incomparable_cells_are_excluded(tmp_path, monkeypatch, override, why):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, 'e', [dict(_BASE, condition='c', **override)],
           [('c', 1, 1, 10.0), ('c', 2, 1, 12.0)])
    assert analyze_path.load_cells('*.csv') == [], why


def test_a_cell_where_nothing_finished_has_no_path_rather_than_a_nan(tmp_path, monkeypatch):
    """Median completion is undefined with no successes, and a NaN would poison the mean silently."""
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, 'e', [dict(_BASE, condition='c', success_rate='0.0')],
           [('c', 1, 0, 20.0), ('c', 2, 0, 20.0)])
    assert analyze_path.load_cells('*.csv') == []


def test_cells_predating_a_factor_fall_back_to_the_value_that_was_in_force(tmp_path, monkeypatch):
    """E1 and E3 were recorded before either column existed; they ran full chunks and a 20 s deadline.

    Reading a missing column as 0 would put those cells at duty 0 and a deadline of 0, which is how a
    schema change quietly rewrites history.
    """
    monkeypatch.chdir(tmp_path)
    old = {k: v for k, v in _BASE.items() if k not in ('max_chunk_actions', 'max_episode_s')}
    _write(tmp_path, 'e', [dict(old, condition='c')], [('c', 1, 1, 10.0), ('c', 2, 1, 12.0)])
    cell = analyze_path.load_cells('*.csv')[0]
    assert cell['k'] == 15
    assert cell['deadline'] == 20.0
