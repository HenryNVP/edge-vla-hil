import numpy as np

import square_failures as sf

PEG = np.array([0.23, 0.10, 0.85])
Z0 = 0.83


def _ep(n=10, held=(), lift=(), near=(), on=(), success=False):
    h = np.zeros(n, bool); h[list(held)] = True
    nut = np.tile([0.0, -0.1, Z0], (n, 1))
    nut[list(lift), 2] = Z0 + 0.05
    nut[list(near), :2] = PEG[:2] + 0.01
    on_peg = np.zeros(n, bool); on_peg[list(on)] = True
    closing = h.copy()
    return sf.classify(h, nut, on_peg, success, closing, Z0, PEG)


def test_classes():
    assert _ep()['cls'] == 'no_grasp'
    assert _ep(held=[3, 4])['cls'] == 'slip'
    assert _ep(held=[3, 4, 5], lift=[4, 5])['cls'] == 'drop'
    assert _ep(held=[3, 4, 5], lift=[4, 5], near=[5])['cls'] == 'peg_miss'
    assert _ep(held=[3, 4], lift=[4], on=[8])['cls'] == 'no_release'
    assert _ep(held=[3], on=[8], success=True)['cls'] == 'success'


def test_milestones():
    r = _ep(held=[3, 4, 5, 7], lift=[4, 5])
    assert r['t_held'] == 3 and r['t_lift'] == 4 and r['closes'] == 2 and r['drops'] == 1
    assert r['t_near_peg'] == -1 and r['max_lift_mm'] == 50.0
