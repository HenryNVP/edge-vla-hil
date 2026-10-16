import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
import bev_model as bm  # noqa: E402
import top_camera as tc  # noqa: E402


def _P():
    """A pinhole looking straight down from 1 m above the origin, 84 px, f = 60 px."""
    K = np.array([[60, 0, 41.5], [0, 60, 41.5], [0, 0, 1]], float)
    Rt = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1.0]], float)   # z points down
    return K @ Rt


def test_sample_coords_agree_with_top_camera_project():
    P = _P()
    pts = torch.tensor([[[0.0, 0.0, 0.0], [0.1, -0.05, 0.2]]])
    g, valid = bm.sample_coords(torch.tensor(P, dtype=torch.float32)[None], pts)
    px = tc.project(P, pts.numpy().astype(float))
    assert np.allclose((g[0].numpy() + 1) / 2 * 84 - 0.5, px, atol=1e-3)
    assert valid.all()


def test_points_outside_the_image_or_behind_are_invalid():
    P = torch.tensor(_P(), dtype=torch.float32)[None]
    pts = torch.tensor([[[5.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])      # far off-image; above the camera
    _, valid = bm.sample_coords(P, pts)
    assert not valid.any()


def test_bev_sample_reads_the_projected_pixel():
    P = torch.tensor(_P(), dtype=torch.float32)[None]
    fmap = torch.zeros(1, 1, 84, 84)
    fmap[0, 0, 41, 41] = fmap[0, 0, 41, 42] = fmap[0, 0, 42, 41] = fmap[0, 0, 42, 42] = 1.0
    pts = torch.tensor([[[0.0, 0.0, 0.0]], [[0.3, 0.3, 0.0]]])     # centre cell; a corner cell
    g, valid = bm.sample_coords(P, pts)
    out = bm.bev_sample([fmap], g, valid)
    assert out.shape == (1, 2, 1) and out[0, 0, 0] > 0.9 and out[0, 1, 0] == 0


def test_pillar_points_cover_the_grid():
    p = bm.pillar_points(4)
    assert p.shape == (16, len(bm.Z_LEVELS), 3)
    assert p[..., 0].min() == bm.X_RANGE[0] and p[..., 1].max() == bm.Y_RANGE[1]
