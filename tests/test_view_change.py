import numpy as np

from view_change import changed_patches


def test_only_the_patch_that_changed_is_flagged():
    a = np.zeros((16, 16, 3), np.uint8)
    b = a.copy()
    b[8:16, 0:8] = 50                                 # one 8x8 patch changes a lot
    b[0, 15] = 255                                    # one pixel: below a patch's mean threshold
    m = changed_patches(a, b, patch=8, thresh=4.0)
    assert m.shape == (2, 2) and m.tolist() == [[False, False], [True, False]]
