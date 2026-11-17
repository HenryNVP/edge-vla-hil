import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
interp_collect = pytest.importorskip('interp_collect')


def test_align_to_dp_puts_now_at_row_n_obs_minus_1():
    chunk = np.arange(16)[:, None] * np.ones((1, 10))     # row j = ACT action j steps ahead
    out = interp_collect.align_to_dp(chunk, 2)
    assert out.shape == chunk.shape
    assert out[1, 0] == 0 and out[15, 0] == 14            # DP row 1 = now = ACT row 0
    assert out[0, 0] == 0                                 # the past step repeats "now"
    assert interp_collect.align_to_dp(chunk, 1) is chunk
