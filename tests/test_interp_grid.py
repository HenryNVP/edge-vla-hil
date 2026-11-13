import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from interp_grid import verdict  # noqa: E402
from plot_sweep import wilson  # noqa: E402


def cell(k, n):
    lo, hi = wilson(k, n)
    return dict(k=k, n=n, p=k / n, lo=lo, hi=hi)


def test_verdict_rule():
    # clears base and control, base well below in-distribution
    assert verdict(cell(36, 100), cell(85, 100), cell(50, 100), (100, 100)) == dict(failure=True, head_limited=True)
    # retrained interval overlaps the control's: not head-limited
    assert not verdict(cell(4, 100), cell(63, 100), cell(50, 100), (100, 100))['head_limited']
    # base indistinguishable from in-distribution: not a failure
    assert not verdict(cell(45, 50), cell(50, 50), cell(30, 50), (50, 50))['failure']
