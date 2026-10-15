import numpy as np

import two_stream as ts


def test_sample_index_stays_in_episode_and_respects_age():
    rng = np.random.default_rng(0)
    lens = np.array([10, 50, 30])
    e, t, t0 = ts.sample_index(lens, 2000, max_age=8, rng=rng)
    assert (t < lens[e]).all() and (t0 >= 0).all() and (t0 <= t).all()
    assert (t - t0).max() == 8 and set(np.unique(e)) == {0, 1, 2}
    _, t, t0 = ts.sample_index(lens, 200, max_age=8, rng=rng, fixed_age=3)
    assert ((t - t0) == np.minimum(t, 3)).all()           # clipped at the episode start


def test_chunk_targets_pad_with_the_last_action():
    a = np.arange(20, dtype=np.float32).reshape(10, 2)
    tgt, mask = ts.chunk_targets(a, 7, h=5)
    assert tgt.shape == (5, 2) and mask.tolist() == [False, False, False, True, True]
    assert (tgt[3:] == a[-1]).all()
    tgt, mask = ts.chunk_targets(a, 0, h=5)
    assert not mask.any() and (tgt == a[:5]).all()
