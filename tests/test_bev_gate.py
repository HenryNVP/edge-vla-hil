import numpy as np

import bev_gate as bg


def test_cell_xy_row_major_over_x_then_y():
    c = bg.cell_xy(4, (-0.3, 0.3), (-0.3, 0.3))
    assert c.shape == (16, 2)
    assert np.allclose(c[0], [-0.3, -0.3]) and np.allclose(c[1], [-0.3, -0.1])
    assert np.allclose(c[4], [-0.1, -0.3])


def test_regions_priority_and_rest():
    cells = np.array([[0.0, 0.0], [0.1, 0.0], [0.227, 0.101], [-0.3, -0.3]])
    reg = bg.regions(cells, nut_now=[0.0, 0.0, 0.8], nut_then=[0.0, 0.0, 0.8],
                     eef=[0.1, 0.0, 1.0], radius=0.03)
    names = [bg.REGIONS[i] for i in reg]
    assert names == ['nut_now', 'gripper', 'pegs', 'rest']   # nut_now wins over nut_then


def test_region_means_nan_for_empty():
    mu = bg.region_means(np.array([1.0, 0.0, 0.5]), np.array([0, 4, 4]))
    assert mu[0] == 1.0 and mu[4] == 0.25 and np.isnan(mu[1])


def test_attn_centroid_is_the_weighted_mean():
    cells = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(bg.attn_centroid(np.array([0.5, 0.5, 0.0]), cells), [0.5, 0.0])
