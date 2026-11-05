import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from interp_repairable import scores  # noqa: E402
from interp_nuisance import parse, roll_quat, yaw_quat  # noqa: E402


def fake(n_eps, rng, code):
    """Episodes whose features carry y = code(y) plus noise."""
    X, Y, G = [], [], []
    for g in range(n_eps):
        y = rng.normal(0, 50, (30, 2))
        X.append(np.c_[code(y), rng.normal(0, 1, (30, 6))] + rng.normal(0, 0.5, (30, 8)))
        Y.append(y)
        G.append(np.full(30, g))
    return np.vstack(X), np.vstack(Y), np.concatenate(G)


def test_scores_separate_moved_from_lost_information():
    rng = np.random.default_rng(0)
    base = fake(10, rng, lambda y: y / 10)
    moved = fake(10, rng, lambda y: -y / 10)                 # same info, different direction
    lost = fake(10, rng, lambda y: 0 * y)                    # info gone
    s_moved, s_lost = scores(base, moved), scores(base, lost)
    assert s_moved['joint'] < 0.9 or s_moved['transfer'] < 0      # one map can't serve both signs
    assert s_moved['transfer'] < 0 and s_lost['joint'] < 0.1 and s_lost['transfer'] < 0.1
    same = fake(10, rng, lambda y: y / 10)
    assert scores(base, same)['joint'] > 0.9 and scores(base, same)['transfer'] > 0.9


def test_nuisance_parse_and_yaw():
    items = parse('light=0.3; cam=0,0.1,0,90')
    assert items[0][0] == 'light' and np.allclose(items[1][1], [0, 0.1, 0, 90])
    q = yaw_quat(np.array([1.0, 0, 0, 0]), 90)
    assert np.allclose(q, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])


def test_roll_is_right_multiplied():
    q = yaw_quat(np.array([1.0, 0, 0, 0]), 90)              # some orientation
    assert np.allclose(roll_quat(np.array([1.0, 0, 0, 0]), 90), q)   # from identity: same turn
    r = roll_quat(q, 180)
    assert np.isclose(np.linalg.norm(r), 1) and not np.allclose(r, q)
