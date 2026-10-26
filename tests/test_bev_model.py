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


def test_bev_sample_per_height_keeps_the_heights_apart():
    P = torch.tensor(_P(), dtype=torch.float32)[None]
    fmap = torch.zeros(1, 2, 84, 84)
    fmap[:, :, 40:44, 40:44] = torch.tensor([1.0, 2.0])[:, None, None]
    pts = torch.tensor([[[0.0, 0.0, 0.0], [0.3, 0.3, 0.5]]])        # on the spot; off it
    g, valid = bm.sample_coords(P, pts)
    out = bm.bev_sample([fmap], g, valid, per_height=True)
    assert out.shape == (1, 1, 4)                                   # [B, N, Z*D], height-major
    assert torch.allclose(out[0, 0], torch.tensor([1.0, 2.0, 0.0, 0.0]))
    assert torch.allclose(bm.bev_sample([fmap], g, valid)[0, 0], torch.tensor([0.5, 1.0]))


def test_no_cloud_memory_is_zero_and_needs_some_input():
    P = torch.tensor(np.stack([_P(), _P()]), dtype=torch.float32)
    m = bm.BEVPlanner(P, d=32, heads=4, local_enc='tiny', use_cloud=False)
    img = torch.zeros(2, 84, 84, 3, dtype=torch.uint8)
    assert m.cloud is None and not m.memory(img, img, torch.zeros(2, 9)).any()
    with pytest.raises(ValueError):
        bm.BEVPlanner(P, d=32, heads=4, local_enc='tiny', use_cloud=False, use_local=False)


def test_pillar_points_cover_the_grid():
    p = bm.pillar_points(4)
    assert p.shape == (16, len(bm.Z_LEVELS), 3)
    assert p[..., 0].min() == bm.X_RANGE[0] and p[..., 1].max() == bm.Y_RANGE[1]


@pytest.mark.parametrize('kw', [{'use_ta': False}, {'local_views': 'agent'},
                                {'local_views': 'top'}, {'use_local': False},
                                {'use_cloud': False}, {'height_channels': True},
                                {'height_channels': True, 'corr': True, 'local_views': 'agent'}])
def test_ablations_plan(kw):
    P = torch.tensor(np.stack([_P(), _P()]), dtype=torch.float32)
    m = bm.BEVPlanner(P, d=32, heads=4, local_enc='tiny', **kw).eval()
    img = torch.zeros(1, 84, 84, 3, dtype=torch.uint8)
    st = torch.zeros(1, 9)
    with torch.no_grad():
        out = m(img, img, st, img, img, img, st, torch.tensor([3]))
    assert out.shape == (1, 16, 10)
    if kw.get('use_ta') is False:
        assert not any('.ta.' in n for n, _ in m.named_parameters())


def test_gate_is_per_cell_in_unit_interval_and_absent_by_default():
    P = torch.tensor(np.stack([_P(), _P()]), dtype=torch.float32)
    m = bm.BEVPlanner(P, d=32, heads=4, local_enc='tiny', gate=True, local_views='agent').eval()
    img = torch.zeros(2, 84, 84, 3, dtype=torch.uint8)
    st = torch.zeros(2, 9)
    with torch.no_grad():
        mem = m.memory(img, img, st)
        out, gate = m.plan(mem, img, img, img, st, torch.tensor([0, 30]), return_gate=True)
    assert out.shape == (2, 16, 10) and gate.shape == (2, 256)
    assert ((gate > 0) & (gate < 1)).all()
    plain = bm.BEVPlanner(P, d=32, heads=4, local_enc='tiny')
    assert not hasattr(plain, 'gate') and hasattr(plain, 'fuse')


def test_corr_window_and_identity_match():
    P = torch.tensor(np.stack([_P(), _P()]), dtype=torch.float32)
    m = bm.BEVPlanner(P, d=32, heads=4, local_enc='tiny', corr=True, corr_radius=2,
                      local_views='agent').eval()
    assert (m.corr_idx[0, 3] < 0) and (m.corr_idx[0, 2] >= 0)       # 3 cells away: outside
    torch.manual_seed(0)
    fresh = torch.randn(1, 256, 32)
    fresh[0, 1] *= 5                    # a large-norm neighbour must not outscore an exact match
    cloud = torch.randn(1, 256, 32)
    with torch.no_grad():
        m.corr_qk.weight.copy_(torch.eye(32))
        m.corr_qk.bias.zero_()
        m.corr_logtau.fill_(4.0)                                       # tau ~55: sharp matching
        _, attn = m._corr(fresh, fresh, cloud)                         # nothing moved
    assert attn.shape == (1, 256, 256)
    assert (attn.diagonal(dim1=-2, dim2=-1) > 0.9).all()
    assert torch.allclose(attn.sum(-1), torch.ones(1, 256), atol=1e-5)
    img = torch.zeros(1, 84, 84, 3, dtype=torch.uint8)
    st = torch.zeros(1, 9)
    with torch.no_grad():
        assert m(img, img, st, img, img, img, st, torch.tensor([5])).shape == (1, 16, 10)


def test_corr_recon_loss_zero_for_identity_on_a_still_scene():
    P = torch.tensor(np.stack([_P(), _P()]), dtype=torch.float32)
    m = bm.BEVPlanner(P, d=32, heads=4, local_enc='tiny', corr=True, local_views='agent')
    img = torch.randint(0, 255, (2, 84, 84, 3), dtype=torch.uint8)
    c = m.cell_colors(img, img)
    assert c.shape == (2, 256, len(bm.Z_LEVELS) * 3)
    eye = torch.eye(256).expand(2, -1, -1)
    assert m.corr_recon_loss(eye, img, img, img, img) < 1e-6
    blur = torch.full((2, 256, 256), 1 / 256)
    assert m.corr_recon_loss(blur, img, img, img, img) > 0.05     # many cells fall off this image
