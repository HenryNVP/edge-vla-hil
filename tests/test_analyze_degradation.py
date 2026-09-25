"""Tests for the degradation analysis — pure Python, no ROS, no experiment data.

The analysis rests on one empirical fact: every failure is a horizon timeout, so success rate is a
censored reading of completion time. `completed_by` is the estimator that makes that explicit, and
it has to treat a censored episode as not-yet-completed rather than as a completion at the horizon.
"""
from analyze_degradation import PAT, completed_by


def _eps(*rows):
    return list(rows)


def test_a_censored_episode_does_not_count_as_a_completion():
    """A timeout is 'had not finished by 20 s', never 'finished at 20 s' — the difference is the
    whole distinction between degrading slowly and not working."""
    eps = _eps((1, 1, 9.0), (2, 0, 21.4), (3, 1, 13.0))
    assert completed_by(eps, 10) == 1 / 3
    assert completed_by(eps, 15) == 2 / 3
    assert completed_by(eps, 25) == 2 / 3      # the failure never becomes a completion


def test_completed_by_is_monotone_in_the_deadline():
    eps = _eps((1, 1, 8.0), (2, 1, 12.0), (3, 1, 18.0), (4, 0, 21.0))
    xs = [completed_by(eps, t) for t in (5, 8, 10, 12, 15, 18, 20)]
    assert xs == sorted(xs)
    assert xs[0] == 0.0 and xs[-1] == 0.75


def test_the_condition_label_parses_both_swept_axes():
    """Cells label the swept value `latency=` or `drop=`; a parser that handles only one silently
    drops half the experiment."""
    for tail, swept in (('latency=800', '800'), ('drop=0.5', '0.5')):
        m = PAT.match('strat=temporal_ensemble_reactive=False_place=both_exec=robot'
                      f'_jit=burst:300_loss=gilbert:0.01:400_lat=0_{tail}')
        assert m is not None, tail
        assert m.group('strat') == 'temporal_ensemble'
        assert m.group('swept') == swept
