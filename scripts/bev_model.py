"""BEV query planner (step 2): the cloud writes a table-grid map, the robot fuses it with a fresh,
cheap map of its own and plans a chunk from it.

    cloud  (frame t0)   ResNet-18 to layer3 on agentview + top, run at 2x (168 px -> 11x11 maps)
                        spatial cross-attention, simple form: every cell of a 16x16 grid over the
                        table owns a pillar of Z points; each is projected into each view with the
                        exact sim camera matrix and bilinearly sampled; the samples are averaged
                        over the points that land in an image (no learned offsets yet)
                        -> 2 self-attention layers over the cells -> BEV_c [256, D]
    local  (frame t)    MobileNetV3-Small (or TinyCNN) on agentview + top -> the same projection
                        -> fresh BEV_l [256, D] (a grid_sample, no attention); wrist -> 36 tokens
                        per cell: x = Linear([BEV_l ; BEV_c]) + cell position
                        2 layers: self-attention over [x ; wrist ; proprio ; age], then temporal
                        attention: x reads BEV_c (+ age embedding) and, when given, the robot's
                        own previous map (`prev`, BEVFormer's history), so a cell can pull the
                        nut's features from the cell it occupied when the cloud saw it
    head                H learnable action queries cross-attend to the local tokens (ACT)

The projections are fixed (cameras do not move), so they are buffers set from the data's
P matrices and travel in the checkpoint. Plan: docs/query-planner-plan.md, step 2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from two_stream_model import MobileLocal, TinyCNN, _images

X_RANGE = (-0.32, 0.32)       # table frame, metres; pegs at x = 0.23, the nut starts near x = -0.11
Y_RANGE = (-0.32, 0.32)
Z_LEVELS = (0.82, 0.86, 0.90, 0.94)   # table top and three heights above it (the nut is lifted)
GRID = 16
IMAGE = 84


def pillar_points(grid: int = GRID) -> torch.Tensor:
    """[grid*grid, Z, 3] world points, row-major over (x, y)."""
    xs = torch.linspace(*X_RANGE, grid)
    ys = torch.linspace(*Y_RANGE, grid)
    gx, gy = torch.meshgrid(xs, ys, indexing='ij')
    z = torch.tensor(Z_LEVELS)
    pts = torch.stack(torch.broadcast_tensors(gx[..., None], gy[..., None], z), -1)
    return pts.reshape(grid * grid, len(Z_LEVELS), 3)


def sample_coords(P: torch.Tensor, pts: torch.Tensor, size: int = IMAGE):
    """Projection of `pts` [N, Z, 3] by P [V, 3, 4] -> grid_sample coords [V, N, Z, 2] in
    [-1, 1] and a validity mask [V, N, Z] (in front of the camera and inside the image)."""
    h = torch.cat([pts, torch.ones_like(pts[..., :1])], -1)          # [N, Z, 4]
    uvw = torch.einsum('vij,nzj->vnzi', P, h)
    w = uvw[..., 2:3]
    uv = uvw[..., :2] / w.clamp(min=1e-6)
    g = (uv + 0.5) / size * 2 - 1
    valid = (w[..., 0] > 0) & (g.abs() <= 1).all(-1)
    return g, valid


def bev_sample(maps: list[torch.Tensor], coords: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """maps: per view [B, D, h, w]; coords [V, N, Z, 2]; valid [V, N, Z] -> [B, N, D], the
    mean over the valid (view, height) samples of each cell."""
    b = maps[0].shape[0]
    acc, cnt = 0, 0
    for v, m in enumerate(maps):
        g = coords[v][None].expand(b, -1, -1, -1).to(m.dtype)          # [B, N, Z, 2]
        s = F.grid_sample(m, g, align_corners=False)                    # [B, D, N, Z]
        mask = valid[v].to(m.dtype)[None, None]
        acc = acc + (s * mask).sum(-1)
        cnt = cnt + mask.sum(-1)
    return (acc / cnt.clamp(min=1)).transpose(1, 2)


class CloudBEV(nn.Module):
    def __init__(self, d: int, state_dim: int, layers: int = 2, heads: int = 8,
                 upsample: int = 2) -> None:
        super().__init__()
        from torchvision.ops.misc import FrozenBatchNorm2d
        r = torchvision.models.resnet18(weights='IMAGENET1K_V1', norm_layer=FrozenBatchNorm2d)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2,
                                      r.layer3)
        self.proj = nn.Conv2d(256, d, 1)
        self.upsample = upsample
        self.state = nn.Linear(state_dim, d)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, 0.0, batch_first=True, norm_first=True),
            layers, norm=nn.LayerNorm(d))

    def fmap(self, img: torch.Tensor) -> torch.Tensor:
        x = _images(img)
        if self.upsample > 1:
            x = F.interpolate(x, scale_factor=self.upsample, mode='bilinear', align_corners=False)
        return self.proj(self.backbone(x))

    def forward(self, views, state, coords, valid, pos) -> torch.Tensor:
        """views: list of uint8 images (agentview, top) -> memory [B, 256 + 1, D]."""
        bev = bev_sample([self.fmap(v) for v in views], coords, valid) + pos
        x = torch.cat([bev, self.state(state)[:, None]], 1)
        return self.enc(x)


class LocalLayer(nn.Module):
    """Self-attention over the local tokens, temporal attention to the stale maps, MLP."""

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ta = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mem):
        h = self.n1(x)
        x = x + self.sa(h, h, h, need_weights=False)[0]
        x = x + self.ta(self.n2(x), mem, mem, need_weights=False)[0]
        return x + self.mlp(self.n3(x))


class BEVPlanner(nn.Module):
    def __init__(self, P: torch.Tensor | None = None, d: int = 256, state_dim: int = 9,
                 action_dim: int = 10, chunk: int = 16, max_age: int = 48, heads: int = 8,
                 local_enc: str = 'mobile', use_prev: bool = False, use_local: bool = True,
                 grid: int = GRID) -> None:
        super().__init__()
        self.chunk, self.max_age, self.grid = chunk, max_age, grid
        self.use_prev, self.use_local = use_prev, use_local
        n = grid * grid
        self.register_buffer('P', P if P is not None else torch.zeros(2, 3, 4))
        self.register_buffer('pts', pillar_points(grid))
        self.cloud = CloudBEV(d, state_dim)
        self.pos = nn.Parameter(torch.randn(1, n, d) * 0.02)
        self.local_cnn = ({'tiny': TinyCNN, 'mobile': MobileLocal}[local_enc](d)
                          if use_local else None)
        self.fuse = nn.Linear(2 * d, d)
        self.state = nn.Linear(state_dim, d)
        self.age = nn.Embedding(max_age + 1, d)
        self.wrist_pos = nn.Parameter(torch.randn(1, 36, d) * 0.02)
        self.src_emb = nn.Parameter(torch.randn(2, 1, 1, d) * 0.02)   # cloud map / previous map
        self.no_prev = nn.Parameter(torch.zeros(1, 1, d))
        self.layers = nn.ModuleList(LocalLayer(d, heads) for _ in range(2))
        self.norm = nn.LayerNorm(d)
        self.queries = nn.Parameter(torch.randn(1, chunk, d) * 0.02)
        self.head_dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, heads, 4 * d, 0.0, batch_first=True, norm_first=True),
            2, norm=nn.LayerNorm(d))
        self.out = nn.Linear(d, action_dim)

    def _coords(self):
        return sample_coords(self.P, self.pts)

    def memory(self, agent0, top0, state0) -> torch.Tensor:
        """The cloud half: [B, 257, D] (256 cells + a proprio token)."""
        coords, valid = self._coords()
        return self.cloud([agent0, top0], state0, coords, valid, self.pos)

    def plan(self, mem, agent, top, wrist, state, age, prev=None, return_bev=False):
        """The local half. `prev`: the robot's own BEV from its previous plan [B, 256, D] or
        None. Returns actions [B, H, action_dim] (and this plan's BEV when return_bev)."""
        b, n = state.shape[0], self.grid * self.grid
        a = self.age(age.clamp(0, self.max_age))[:, None]
        bev_c = mem[:, :n]
        if self.local_cnn is not None:
            coords, valid = self._coords()
            fresh = bev_sample([self._fmap(agent), self._fmap(top)], coords, valid)
            wrist_tok = self.local_cnn(wrist) + self.wrist_pos
        else:
            fresh = torch.zeros_like(bev_c)
            wrist_tok = self.wrist_pos.expand(b, -1, -1)
        x = self.fuse(torch.cat([fresh, bev_c], -1)) + self.pos
        x = torch.cat([x, wrist_tok, self.state(state)[:, None], a, mem[:, n:]], 1)
        stale = [mem + a + self.src_emb[0]]
        if self.use_prev:
            p = (prev + self.pos) if prev is not None else self.no_prev.expand(b, n, -1)
            stale.append(p + self.src_emb[1])
        stale = torch.cat(stale, 1)
        for layer in self.layers:
            x = layer(x, stale)
        x = self.norm(x)
        out = self.out(self.head_dec(self.queries.expand(b, -1, -1), x))
        return (out, x[:, :n]) if return_bev else out

    def _fmap(self, img):
        """The local encoder's spatial map [B, D, 6, 6] (its tokens, unflattened)."""
        t = self.local_cnn(img)
        s = int(round(t.shape[1] ** 0.5))
        return t.transpose(1, 2).reshape(t.shape[0], -1, s, s)

    def forward(self, agent0, top0, state0, agent, top, wrist, state, age, prev=None):
        return self.plan(self.memory(agent0, top0, state0), agent, top, wrist, state, age, prev)
