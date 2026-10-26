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
                        (BEV_l averages the views per cell with equal weight, so an occluded view
                        dilutes it; `local_views` restricts it to agentview or top alone)
                        with `gate`: x = g * W_l BEV_l + (1 - g) * W_c BEV_c + cell position, where
                        the per-cell staleness gate g in (0, 1) is an MLP of [W_l BEV_l ; W_c BEV_c ;
                        |W_l BEV_l - W_c BEV_c| ; age]: how far the cell's fresh view disagrees
                        with the cloud's, given how old the cloud's is. Nothing supervises g; the
                        nut poses in the demos check afterwards whether it opens where the nut
                        moved (bev_gate.py)
                        with `corr` instead: the robot also maps the frame it sent (t0) with its own
                        encoder, R0, and each fresh cell attends over a window of R0 cells
                        (queries R, keys R0 through one shared projection + a learned per-offset
                        bias; values the cloud's cells): "where at t0 did my own features look like
                        what I see here now", then the cloud's richer feature from there. Same
                        encoder on both sides, so a match means motion, not encoder (the gate's
                        |W_l BEV_l - W_c BEV_c| compared two unaligned spaces; outputs/gate/
                        align.py). x = Linear([BEV_l ; warped BEV_c]); staleness = 1 - the
                        attention a cell keeps on itself. Trained on the action loss alone it
                        became local pooling (a cell kept ~20% on itself even with R0 == R), so
                        `corr_recon_loss` adds a self-supervised target the encoder cannot shape:
                        the same (head-mean) attention must rebuild each cell's image colours at t
                        from those at t0 (colourise by pointing, Vondrick et al. 2018)
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
LOCAL_VIEWS = {'both': [0, 1], 'agent': [0], 'top': [1]}   # the robot map's views (P rows)
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


def bev_sample(maps: list[torch.Tensor], coords: torch.Tensor, valid: torch.Tensor,
               per_height: bool = False) -> torch.Tensor:
    """maps: per view [B, D, h, w]; coords [V, N, Z, 2]; valid [V, N, Z] -> [B, N, D], the
    mean over the valid (view, height) samples of each cell; with `per_height`, [B, N, Z*D],
    the mean over views at each height, heights stacked as channels (keeps what the mean over
    heights throws away: at what height a feature sits)."""
    b = maps[0].shape[0]
    acc, cnt = 0, 0
    for v, m in enumerate(maps):
        g = coords[v][None].expand(b, -1, -1, -1).to(m.dtype)          # [B, N, Z, 2]
        s = F.grid_sample(m, g, align_corners=False)                    # [B, D, N, Z]
        mask = valid[v].to(m.dtype)[None, None]
        acc = acc + (s * mask if per_height else (s * mask).sum(-1))
        cnt = cnt + (mask if per_height else mask.sum(-1))
    out = acc / cnt.clamp(min=1)
    return out.permute(0, 2, 3, 1).flatten(2) if per_height else out.transpose(1, 2)


class CloudBEV(nn.Module):
    def __init__(self, d: int, state_dim: int, layers: int = 2, heads: int = 8,
                 upsample: int = 2, heights: int = 0, goals: int = 0) -> None:
        super().__init__()
        from torchvision.ops.misc import FrozenBatchNorm2d
        r = torchvision.models.resnet18(weights='IMAGENET1K_V1', norm_layer=FrozenBatchNorm2d)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2,
                                      r.layer3)
        self.proj = nn.Conv2d(256, d, 1)
        self.upsample = upsample
        self.zproj = nn.Linear(heights * d, d) if heights else None      # height channels -> D
        self.state = nn.Linear(state_dim, d)
        self.goal = nn.Embedding(goals, d) if goals else None    # the task goal, cloud-only
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, 0.0, batch_first=True, norm_first=True),
            layers, norm=nn.LayerNorm(d))

    def fmap(self, img: torch.Tensor) -> torch.Tensor:
        x = _images(img)
        if self.upsample > 1:
            x = F.interpolate(x, scale_factor=self.upsample, mode='bilinear', align_corners=False)
        return self.proj(self.backbone(x))

    def forward(self, views, state, coords, valid, pos, goal=None) -> torch.Tensor:
        """views: list of uint8 images (agentview, top) -> memory [B, 256 + 1, D]. `goal` [B]
        (long) joins the proprio token; the encoder spreads it over the cells."""
        bev = bev_sample([self.fmap(v) for v in views], coords, valid, self.zproj is not None)
        if self.zproj is not None:
            bev = self.zproj(bev)
        bev = bev + pos
        tok = self.state(state)
        if self.goal is not None:
            if goal is None:
                raise ValueError('this cloud encoder is goal-conditioned: pass goal')
            tok = tok + self.goal(goal)
        x = torch.cat([bev, tok[:, None]], 1)
        return self.enc(x)


class LocalLayer(nn.Module):
    """Self-attention over the local tokens, temporal attention to the stale maps (unless
    `use_ta` is off: the ablation where the cloud map enters only through the per-cell fusion),
    MLP."""

    def __init__(self, d: int, heads: int, use_ta: bool = True) -> None:
        super().__init__()
        self.n1, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        if use_ta:
            self.n2 = nn.LayerNorm(d)
            self.ta = nn.MultiheadAttention(d, heads, batch_first=True)
        self.use_ta = use_ta
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mem):
        h = self.n1(x)
        x = x + self.sa(h, h, h, need_weights=False)[0]
        if self.use_ta:
            x = x + self.ta(self.n2(x), mem, mem, need_weights=False)[0]
        return x + self.mlp(self.n3(x))


class BEVPlanner(nn.Module):
    def __init__(self, P: torch.Tensor | None = None, d: int = 256, state_dim: int = 9,
                 action_dim: int = 10, chunk: int = 16, max_age: int = 48, heads: int = 8,
                 local_enc: str = 'mobile', use_prev: bool = False, use_local: bool = True,
                 use_ta: bool = True, local_views: str = 'both', gate: bool = False,
                 corr: bool = False, corr_radius: int = 4, use_cloud: bool = True,
                 height_channels: bool = False, goals: int = 0, grid: int = GRID) -> None:
        super().__init__()
        if gate and corr:
            raise ValueError('gate and corr are alternatives')
        if corr and not use_local:
            raise ValueError('corr needs the local encoder')
        if not (use_cloud or use_local):
            raise ValueError('needs the cloud map, the robot images or both')
        if local_views not in LOCAL_VIEWS:
            raise ValueError(f'local_views {local_views!r} not in {tuple(LOCAL_VIEWS)}')
        self.chunk, self.max_age, self.grid = chunk, max_age, grid
        self.use_prev, self.use_local = use_prev, use_local
        self.local_views = LOCAL_VIEWS[local_views]
        n = grid * grid
        self.register_buffer('P', P if P is not None else torch.zeros(2, 3, 4))
        self.register_buffer('pts', pillar_points(grid))
        hz = len(Z_LEVELS) if height_channels else 0
        self.d = d
        self.cloud = CloudBEV(d, state_dim, heights=hz, goals=goals) if use_cloud else None
        self.pos = nn.Parameter(torch.randn(1, n, d) * 0.02)
        self.local_cnn = ({'tiny': TinyCNN, 'mobile': MobileLocal}[local_enc](d)
                          if use_local else None)
        self.local_zproj = nn.Linear(hz * d, d) if hz and use_local else None
        self.use_gate = gate
        if gate:
            self.w_l, self.w_c = nn.Linear(d, d), nn.Linear(d, d)
            self.gate = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))
        else:
            self.fuse = nn.Linear(2 * d, d)
        self.use_corr = corr
        if corr:
            self.corr_heads = 4
            self.corr_qk = nn.Linear(d, d)
            self.corr_v = nn.Linear(d, d)
            self.corr_o = nn.Linear(d, d)
            w = 2 * corr_radius + 1
            self.corr_bias = nn.Parameter(torch.zeros(self.corr_heads, w * w))
            self.corr_logtau = nn.Parameter(torch.full((self.corr_heads, 1, 1), 2.3))   # tau 10
            ij = torch.stack(torch.meshgrid(torch.arange(grid), torch.arange(grid),
                                            indexing='ij'), -1).reshape(-1, 2)
            off = ij[None, :] - ij[:, None]                       # [query, key, 2]
            inside = (off.abs() <= corr_radius).all(-1)
            idx = (off[..., 0] + corr_radius) * w + off[..., 1] + corr_radius
            self.register_buffer('corr_idx', torch.where(inside, idx, -1), persistent=False)
        self.state = nn.Linear(state_dim, d)
        self.age = nn.Embedding(max_age + 1, d)
        self.wrist_pos = nn.Parameter(torch.randn(1, 36, d) * 0.02)
        self.src_emb = nn.Parameter(torch.randn(2, 1, 1, d) * 0.02)   # cloud map / previous map
        self.no_prev = nn.Parameter(torch.zeros(1, 1, d))
        self.layers = nn.ModuleList(LocalLayer(d, heads, use_ta) for _ in range(2))
        self.norm = nn.LayerNorm(d)
        self.queries = nn.Parameter(torch.randn(1, chunk, d) * 0.02)
        self.head_dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, heads, 4 * d, 0.0, batch_first=True, norm_first=True),
            2, norm=nn.LayerNorm(d))
        self.out = nn.Linear(d, action_dim)

    def _coords(self):
        return sample_coords(self.P, self.pts)

    def memory(self, agent0, top0, state0, goal0=None) -> torch.Tensor:
        """The cloud half: [B, 257, D] (256 cells + a proprio token); zeros without the cloud
        (`use_cloud` off: the robot-only control)."""
        if self.cloud is None:
            return self.pos.new_zeros(state0.shape[0], self.grid ** 2 + 1, self.d)
        coords, valid = self._coords()
        return self.cloud([agent0, top0], state0, coords, valid, self.pos, goal0)

    def local_map(self, agent, top) -> torch.Tensor:
        """The robot encoder's fresh map [B, 256, D] (with `corr`, also run on the t0 frame)."""
        coords, valid = self._coords()
        v = self.local_views
        imgs = [agent, top]
        z = self.local_zproj
        bev = bev_sample([self._fmap(imgs[i]) for i in v], coords[v], valid[v], z is not None)
        return z(bev) if z is not None else bev

    def cell_colors(self, agent, top) -> torch.Tensor:
        """[B, N, V*Z*3] raw image colours at each cell's pillar points in the robot map's views
        (0 where a point falls outside an image): the fixed target of corr_recon_loss."""
        coords, valid = self._coords()
        imgs = [agent, top]
        out = []
        for v in self.local_views:
            x = _images(imgs[v])
            g = coords[v][None].expand(x.shape[0], -1, -1, -1).to(x.dtype)
            c = F.grid_sample(x, g, align_corners=False) * valid[v].to(x.dtype)[None, None]
            out.append(c.permute(0, 2, 3, 1).flatten(2))                 # [B, N, Z*3]
        return torch.cat(out, -1)

    def corr_recon_loss(self, attn, agent0, top0, agent, top) -> torch.Tensor:
        """L1 between attn @ colours(t0) and colours(t); attn [B, N, N] from plan(return_attn)."""
        c0, c = self.cell_colors(agent0, top0), self.cell_colors(agent, top)
        return (attn.float() @ c0.float() - c.float()).abs().mean()

    def _corr(self, fresh, local0, bev_c):
        """Window attention, queries `fresh`, keys `local0` (same encoder, shared projection),
        values the cloud cells -> (warped cloud map [B, N, D], attention [B, N, N] head mean).
        Cosine logits with a learned temperature per head: a plain dot product lets a neighbour
        with a larger norm outscore an exact match, which is wrong for correspondence."""
        b, n, d = fresh.shape
        h = self.corr_heads
        split = lambda t: t.view(b, n, h, d // h).transpose(1, 2)        # [B, h, N, d/h]
        q, k, v = split(self.corr_qk(fresh)), split(self.corr_qk(local0)), split(self.corr_v(bev_c))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        logits = self.corr_logtau.exp() * (q @ k.transpose(-1, -2))
        idx = self.corr_idx
        bias = self.corr_bias[:, idx.clamp(min=0)]                       # [h, N, N]
        logits = (logits + bias).masked_fill(idx < 0, float('-inf'))
        attn = logits.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.corr_o(out), attn.mean(1)

    def plan(self, mem, agent, top, wrist, state, age, prev=None, return_bev=False,
             return_gate=False, local0=None, return_attn=False):
        """The local half. `prev`: the robot's own BEV from its previous plan [B, 256, D] or
        None. `local0`: with `corr`, the robot's own map of the t0 frame (local_map). Returns
        actions [B, H, action_dim] (and this plan's BEV when return_bev; the gate / staleness
        [B, 256] when return_gate, ones with neither; the corr attention [B, 256, 256] when
        return_attn)."""
        b, n = state.shape[0], self.grid * self.grid
        a = self.age(age.clamp(0, self.max_age))[:, None]
        bev_c = mem[:, :n]
        if self.local_cnn is not None:
            fresh = self.local_map(agent, top)
            wrist_tok = self.local_cnn(wrist) + self.wrist_pos
        else:
            fresh = torch.zeros_like(bev_c)
            wrist_tok = self.wrist_pos.expand(b, -1, -1)
        if self.use_gate:
            f, c = self.w_l(fresh), self.w_c(bev_c)
            g = torch.sigmoid(self.gate(torch.cat([f, c, (f - c).abs(), a.expand_as(f)], -1)))
            x = g * f + (1 - g) * c + self.pos
            gate = g[..., 0]
        elif self.use_corr:
            if local0 is None:
                raise ValueError('corr needs local0, the robot map of the t0 frame')
            warped, attn = self._corr(fresh, local0, bev_c)
            x = self.fuse(torch.cat([fresh, warped], -1)) + self.pos
            gate = 1 - attn.diagonal(dim1=-2, dim2=-1)
        else:
            x = self.fuse(torch.cat([fresh, bev_c], -1)) + self.pos
            gate = torch.ones(b, n, device=x.device, dtype=x.dtype)
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
        extra = ([x[:, :n]] if return_bev else []) + ([gate] if return_gate else [])
        if return_attn:
            extra.append(attn if self.use_corr else None)
        return (out, *extra) if extra else out

    def _fmap(self, img):
        """The local encoder's spatial map [B, D, 6, 6] (its tokens, unflattened)."""
        t = self.local_cnn(img)
        s = int(round(t.shape[1] ** 0.5))
        return t.transpose(1, 2).reshape(t.shape[0], -1, s, s)

    def forward(self, agent0, top0, state0, agent, top, wrist, state, age, prev=None, goal0=None):
        local0 = self.local_map(agent0, top0) if self.use_corr else None
        return self.plan(self.memory(agent0, top0, state0, goal0), agent, top, wrist, state, age,
                         prev, local0=local0)
