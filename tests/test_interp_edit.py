import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from interp_act_sites import site_change  # noqa: E402
from interp_edit import frames, low_rank  # noqa: E402


def test_frames_pads_chunks_with_last_action():
    ep = dict(img=[np.zeros((3, 2, 2, 3), np.uint8)] * 2, state=np.zeros((3, 9), np.float32),
              act=np.arange(3, dtype=np.float32)[:, None] * np.ones((1, 10)))
    img, state, chunk, pad, epi = frames([ep], H=4)
    assert chunk.shape == (3, 4, 10) and len(img[0]) == 3
    assert np.allclose(chunk[1, :, 0], [1, 2, 2, 2]) and pad[1].tolist() == [0, 0, 1, 1]


def test_low_rank_starts_at_parent_and_has_rank_r():
    torch = pytest.importorskip('torch')
    lr = low_rank((6, 5), r=2, seed=0)
    W = torch.randn(6, 5)
    assert torch.equal(lr(W), W)
    with torch.no_grad():
        lr.B.normal_()
    assert torch.linalg.matrix_rank(lr(W) - W) == 2


def test_site_change_uses_baseline_row():
    G, P = 3, 1
    d = {'dec': np.zeros((P, G, G, 4, 2)), 'idx': np.full((P, G, G), 2), 'radius': np.zeros(P)}
    d['dec'][0, 0, 0, 2] = [1, 1]                    # only row 2 moves
    d['dec'][0, 0, 0, 1] = [9, 9]
    dF = site_change(d, 'dec', np.ones(P, bool))
    assert dF.shape == (P, G, G, 2) and np.allclose(dF[0, 0, 0], [1, 1]) and np.allclose(dF[0, 1, 1], 0)


def test_transplant_fit_reads_the_ignored_excess():
    from interp_edit import transplant_fit
    rng = np.random.default_rng(0)
    g = np.linspace(-0.08, 0.08, 9)
    O = np.stack(np.meshgrid(g, g, indexing='ij'), -1)
    clamp = np.clip(O, -0.03, 0.03)
    f = np.concatenate([clamp, O - clamp], -1)             # the parent reads only the clamp part
    P = 6
    F = np.broadcast_to(f[None, :, :, None], (P, 9, 9, 2, 4)) + rng.normal(0, 1e-3, (P, 9, 9, 2, 4))
    F = np.concatenate([F, rng.normal(0, 1e-3, (P, 9, 9, 2, 2))], -1)
    W = np.zeros((10, 6))
    W[0, 0] = W[1, 1] = 10.0                                # slope 10 per m, in range only
    b = np.zeros(10)
    d = dict(dec=F, grid=g, nut_xy=np.zeros((P, 2)), scene=np.arange(P))
    dW, db, _ = transplant_fit(d, W, b)
    out = F @ (W[:2] + dW).T + db
    assert np.allclose(out[0, 8, 4, 0], [0.8, 0.0], atol=0.05)       # x = +8 cm now followed
    assert np.allclose(out[0, 4, 4, 0], [0.0, 0.0], atol=0.02)       # centre unchanged
