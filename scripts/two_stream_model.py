"""Two-stream chunking policy: a cloud encoder writes a token memory, a local net plans from it.

Split so that each half runs where it would in deployment:

    cloud  (once per request, frame t0)   ResNet-18 to layer3 on agentview + wrist (6x6 tokens
                                          each at 84 px), proprio(t0) token, 2 encoder layers
                                          -> memory M [73, D]
    local  (every plan, frame t, age)     tiny CNN on agentview + wrist (6x6 tokens each),
                                          proprio(t) and age tokens; 2 layers that self-attend
                                          and cross-attend to M + age embedding (BEVFormer's
                                          temporal-attention role: fresh queries read stale
                                          context) -> L
    head                                  H learnable action queries (DETR/ACT) cross-attend to
                                          [L; M] -> [H, 10] = [pos, rot_6d, gripper]

Modes, for the ablations that decide whether the cloud is worth anything:
    two    the above
    local  no cloud: M is one learned null token (does the tiny local net solve it alone?)
    cloud  no local images: the local side has only proprio(t) and age (stale ACT + age)
    fresh  cloud on frame t, age 0, no local images (an ACT-like upper bound)

Run 1 options (step 0's local stream overfit: a scratch TinyCNN on 320 demos):
    local_enc='mobile'  an ImageNet MobileNetV3-Small for the local images (MobileLocal)
    local_enc='shared'  the local images go through the cloud's own ResNet + projection, so both
                        halves write tokens in one feature space (a fresh token can stand in for a
                        stale one, the later changed-patch idea)
    freeze=True         the ImageNet backbone is frozen (run under no_grad)
    final_norm=True     a closing LayerNorm on each pre-norm stack. Without it the step-0 models ran
                        2x worse open loop in eval than with dropout on (the head reads an
                        unnormalised residual stream; outputs/qp1/dropgap.py)
    local_drop=p        in training, a sample's local image tokens are replaced by one learned
                        token with probability p, so the plan cannot rest on them alone

torch only; imported lazily by two_stream.py so the bare-box tests never load it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODES = ('two', 'local', 'cloud', 'fresh')


def _images(x: torch.Tensor) -> torch.Tensor:
    """uint8 [B, H, W, 3] -> normalized float [B, 3, H, W]."""
    x = x.permute(0, 3, 1, 2).float() / 255.0
    m = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    s = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - m) / s


class CloudEncoder(nn.Module):
    def __init__(self, d: int, state_dim: int, layers: int = 2, heads: int = 8,
                 freeze: bool = False, final_norm: bool = False, dropout: float = 0.1) -> None:
        super().__init__()
        self.freeze = freeze
        from torchvision.ops.misc import FrozenBatchNorm2d
        r = torchvision.models.resnet18(weights='IMAGENET1K_V1', norm_layer=FrozenBatchNorm2d)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2,
                                      r.layer3)
        if freeze:
            self.backbone.requires_grad_(False)
        self.proj = nn.Conv2d(256, d, 1)
        self.state = nn.Linear(state_dim, d)
        self.pos = nn.Parameter(torch.randn(1, 2 * 36 + 1, d) * 0.02)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True),
            layers, norm=nn.LayerNorm(d) if final_norm else None)

    def tokens(self, img: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                f = self.backbone(_images(img))
        else:
            f = self.backbone(_images(img))
        f = self.proj(f)                                     # [B, D, 6, 6] at 84 px
        return f.flatten(2).transpose(1, 2)

    def forward(self, agent, wrist, state) -> torch.Tensor:
        x = torch.cat([self.tokens(agent), self.tokens(wrist), self.state(state)[:, None]], 1)
        return self.enc(x + self.pos[:, :x.shape[1]])


class TinyCNN(nn.Module):
    """84 px -> 6x6 tokens with four stride-2 convs (~0.1 M parameters)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        chans = (3, 32, 64, 64, 128)
        layers = []
        for a, b in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(a, b, 3, 2, 1), nn.GroupNorm(8, b), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.proj = nn.Conv2d(128, d, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(_images(img))).flatten(2).transpose(1, 2)


class MobileLocal(nn.Module):
    """ImageNet MobileNetV3-Small cut at stride 16 (48 channels): 84 px -> 6x6 tokens at ~8 M
    MACs per image, against ~17 M for TinyCNN and ~0.35 G for the cloud's ResNet to layer3."""

    def __init__(self, d: int) -> None:
        super().__init__()
        m = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.backbone = m.features[:9]        # named so training gives it the backbone lr
        self.proj = nn.Conv2d(48, d, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(_images(img))).flatten(2).transpose(1, 2)


class TwoStream(nn.Module):
    def __init__(self, mode: str = 'two', d: int = 256, state_dim: int = 9, action_dim: int = 10,
                 chunk: int = 16, max_age: int = 48, heads: int = 8, local_enc: str = 'tiny',
                 freeze: bool = False, local_drop: float = 0.0, final_norm: bool = False,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'mode {mode!r} not in {MODES}')
        if local_enc == 'shared' and mode != 'two':
            raise ValueError("local_enc='shared' needs mode 'two'")
        self.mode, self.chunk, self.max_age = mode, chunk, max_age
        self.local_enc, self.local_drop = local_enc, local_drop
        self.cloud = None if mode == 'local' else CloudEncoder(
            d, state_dim, freeze=freeze, final_norm=final_norm, dropout=dropout)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        if local_drop > 0:
            self.dropped = nn.Parameter(torch.zeros(1, 1, d))
        local_cls = {'tiny': TinyCNN, 'mobile': MobileLocal}.get(local_enc)
        self.local_cnn = (local_cls(d) if mode in ('two', 'local') and local_cls else None)
        self.state = nn.Linear(state_dim, d)
        self.age = nn.Embedding(max_age + 1, d)
        self.local_pos = nn.Parameter(torch.randn(1, 2 + 2 * 36, d) * 0.02)
        def dec():
            return nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                           norm_first=True),
                2, norm=nn.LayerNorm(d) if final_norm else None)
        self.local = dec()
        self.queries = nn.Parameter(torch.randn(1, chunk, d) * 0.02)
        self.head_dec = dec()
        self.out = nn.Linear(d, action_dim)

    def memory(self, agent0, wrist0, state0) -> torch.Tensor:
        """The cloud half: [B, 73, D] (or the null token when there is no cloud)."""
        if self.cloud is None:
            return self.null.expand(agent0.shape[0], -1, -1)
        return self.cloud(agent0, wrist0, state0)

    def plan(self, mem, agent, wrist, state, age) -> torch.Tensor:
        """The local half: [B, H, action_dim] in normalized action units."""
        b = state.shape[0]
        a = self.age(age.clamp(0, self.max_age))[:, None]          # [B, 1, D]
        x = [self.state(state)[:, None], a]
        enc = self.cloud.tokens if self.local_enc == 'shared' else self.local_cnn
        if enc is not None and self.mode in ('two', 'local'):
            img = torch.cat([enc(agent), enc(wrist)], 1)
            if self.training and self.local_drop > 0:
                drop = torch.rand(b, 1, 1, device=img.device) < self.local_drop
                img = torch.where(drop, self.dropped.expand_as(img), img)
            x += [img]
        x = torch.cat(x, 1)
        x = x + self.local_pos[:, :x.shape[1]]
        m = mem + a                                                  # the memory knows its age
        x = self.local(x, m)
        q = self.head_dec(self.queries.expand(b, -1, -1), torch.cat([x, m], 1))
        return self.out(q)

    def forward(self, agent0, wrist0, state0, agent, wrist, state, age) -> torch.Tensor:
        if self.mode == 'fresh':
            agent0, wrist0, state0 = agent, wrist, state
            age = torch.zeros_like(age)
        return self.plan(self.memory(agent0, wrist0, state0), agent, wrist, state, age)
