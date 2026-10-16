"""Two-stream query planner, step 0: train on the oracle Square demos, run in the step harness.

The cloud half encodes frame t0; the local half plans the chunk for frame t from its own fresh
images, the cloud's memory, and the age t - t0 (two_stream_model.py). Training draws the age
uniformly from [0, max_age] for every sample. That is HiRT's trick for a stale latent, and here
it is the whole point: one model must work at every delay the link produces.

    train  python3 scripts/two_stream.py train --data data/oracle_square_moving/npz \
               --mode two --out outputs/qp/two.pt
    eval   python3 scripts/dp_mech.py --task square --backend two_stream --weights outputs/qp/two.pt \
               --strategies cloud_local ...

Plan: docs/query-planner-plan.md (branch paper2-query-planner).
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

H = 16
MAX_AGE = 48


def load_npz(dirs: str) -> list[dict]:
    """Oracle demo episodes (oracle_demos.py collect): agentview, wrist, state [T,9], action [T,10]."""
    eps = []
    for d in dirs.split(','):
        for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
            z = np.load(f)
            eps.append({k: z[k] for k in ('agentview', 'wrist', 'state', 'action')})
    return eps


def sample_index(lens: np.ndarray, n: int, max_age: int, rng: np.random.Generator,
                 fixed_age: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(episode, t, t0) for n samples: t uniform within its episode, age uniform in [0, max_age]
    (or `fixed_age`), t0 = t - age clipped at the episode start, so the true age can be smaller
    near the start, which is also what happens at the start of a real episode."""
    p = lens / lens.sum()
    e = rng.choice(len(lens), size=n, p=p)
    t = (rng.random(n) * lens[e]).astype(int)
    age = (np.full(n, fixed_age) if fixed_age is not None
           else rng.integers(0, max_age + 1, size=n))
    t0 = np.maximum(t - age, 0)
    return e, t, t0


def chunk_targets(action: np.ndarray, t: int, h: int = H) -> tuple[np.ndarray, np.ndarray]:
    """actions[t:t+h], padded with the last action, and the pad mask (True = padding)."""
    a = action[t:t + h]
    pad = h - len(a)
    mask = np.zeros(h, bool)
    if pad:
        a = np.concatenate([a, np.repeat(action[-1:], pad, 0)])
        mask[h - pad:] = True
    return a, mask


def random_shift(x, pad: int = 4):
    """DrQ-style augmentation on uint8 [B, H, W, 3]: replicate-pad by `pad`, crop back at a
    random offset per sample."""
    import torch
    import torch.nn.functional as F
    b, h, w, _ = x.shape
    p = F.pad(x.permute(0, 3, 1, 2).float(), (pad,) * 4, mode='replicate')
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    rows = (oy[:, None] + torch.arange(h, device=x.device))[:, None, :, None]
    cols = (ox[:, None] + torch.arange(w, device=x.device))[:, None, None, :]
    out = p[torch.arange(b, device=x.device)[:, None, None, None],
            torch.arange(3, device=x.device)[None, :, None, None], rows, cols]
    return out.permute(0, 2, 3, 1).to(torch.uint8)


def stats(eps: list[dict]) -> dict:
    s = np.concatenate([e['state'] for e in eps])
    a = np.concatenate([e['action'] for e in eps])
    return {'state_mean': s.mean(0), 'state_std': s.std(0) + 1e-6,
            'action_mean': a.mean(0), 'action_std': a.std(0) + 1e-6}


def train(args) -> None:
    import torch
    import torch.nn.functional as F
    from two_stream_model import TwoStream
    eps = load_npz(args.data)
    st = stats(eps)
    lens = np.array([len(e['action']) for e in eps])
    print(f'{len(eps)} episodes, {lens.sum()} frames, mode {args.mode}', flush=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    cfg = {'local_enc': args.local_enc, 'freeze': args.freeze, 'local_drop': args.local_drop,
           'final_norm': args.final_norm, 'dropout': args.dropout}
    model = TwoStream(args.mode, max_age=args.max_age, **cfg).to(dev)
    print(f'config {cfg}, shift {args.shift}', flush=True)
    print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters', flush=True)
    bb = [p for n, p in model.named_parameters() if '.backbone.' in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if '.backbone.' not in n]
    lrs = [args.lr, args.lr * 0.1]                     # pretrained backbone: 10x lower, like ACT
    opt = torch.optim.AdamW([{'params': rest}, {'params': bb}], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lrs if bb else lrs[:1] * 2,
                                                total_steps=args.steps, pct_start=0.02)
    rng = np.random.default_rng(args.seed)
    sm, ss = (torch.tensor(st[k], device=dev) for k in ('state_mean', 'state_std'))
    am, as_ = (torch.tensor(st[k], device=dev) for k in ('action_mean', 'action_std'))
    t_start = time.time()
    for step in range(1, args.steps + 1):
        e, t, t0 = sample_index(lens, args.batch, args.max_age, rng)
        acts, masks = zip(*(chunk_targets(eps[i]['action'], j) for i, j in zip(e, t)))

        def grab(key, idx):
            x = torch.from_numpy(np.stack([eps[i][key][j] for i, j in zip(e, idx)])).to(dev)
            if args.shift and key in ('agentview', 'wrist'):
                x = random_shift(x, args.shift)
            return x
        state0 = (grab('state', t0) - sm) / ss
        state = (grab('state', t) - sm) / ss
        target = (torch.from_numpy(np.stack(acts)).to(dev) - am) / as_
        mask = torch.from_numpy(np.stack(masks)).to(dev)
        age = torch.from_numpy(t - t0).long().to(dev)
        pred = model(grab('agentview', t0), grab('wrist', t0), state0,
                     grab('agentview', t), grab('wrist', t), state, age)
        loss = (F.l1_loss(pred, target, reduction='none') * ~mask[..., None]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            print(f'step {step} loss {loss.item():.4f} ({time.time() - t_start:.0f} s)', flush=True)
        if step % args.save_every == 0 or step == args.steps:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            torch.save({'model': model.state_dict(), 'mode': args.mode, 'max_age': args.max_age,
                        'stats': st, 'step': step, 'cfg': cfg}, args.out)


def load_model(ck: dict):
    """A TwoStream from a checkpoint dict. Run-1 checkpoints carry an unused `dropped` token
    when trained without local dropout; it is ignored."""
    from two_stream_model import TwoStream
    model = TwoStream(ck['mode'], max_age=ck['max_age'], **ck.get('cfg', {}))
    sd = dict(ck['model'])
    if not hasattr(model, 'dropped'):
        sd.pop('dropped', None)
    model.load_state_dict(sd)
    return model


class TwoStreamPolicy:
    """The harness's ChunkPolicy for a trained TwoStream: `predict(obs)` is the cloud half and
    returns the memory (it travels through StepWorker like a chunk); `local(obs, memory, age)`
    is the robot half and returns a 7-dim absolute chunk. Pair with the 'cloud_local' executor."""
    absolute_actions = True
    needs_wrist = True
    n_obs_steps = 1
    action_dim = 7
    chunk_size = H
    guided_resampling = False
    denoise_steps = 1

    def __init__(self, path: str, device: str = 'cuda') -> None:
        import torch
        from two_stream_model import TwoStream
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.dev = device if torch.cuda.is_available() else 'cpu'
        self.mode, self.max_age = ck['mode'], ck['max_age']
        self.model = load_model(ck)
        self.model.to(self.dev).eval()
        self.st = {k: torch.tensor(v, device=self.dev) for k, v in ck['stats'].items()}

    def _obs(self, obs):
        import torch
        g = lambda k: torch.from_numpy(np.ascontiguousarray(np.asarray(obs[k])[-1:])).to(self.dev)
        state = (g('proprio').float() - self.st['state_mean']) / self.st['state_std']
        return g('agentview'), g('wrist'), state

    def predict(self, obs) -> np.ndarray:
        import torch
        with torch.no_grad():
            if self.mode == 'fresh':
                return np.zeros((1, 1), np.float32)      # fresh mode plans from the frame at t
            return self.model.memory(*self._obs(obs))[0].cpu().numpy()

    def predict_inpaint(self, obs, prefix, weights):
        return self.predict(obs)

    def local(self, obs, memory, age: int) -> np.ndarray:
        import torch
        from evh_controller.rotation import abs10_to_abs7
        with torch.no_grad():
            agent, wrist, state = self._obs(obs)
            a = torch.tensor([int(age)], device=self.dev)
            if self.mode == 'fresh':
                out = self.model(agent, wrist, state, agent, wrist, state, a)
            else:
                mem = torch.from_numpy(np.asarray(memory, np.float32))[None].to(self.dev)
                out = self.model.plan(mem, agent, wrist, state, a)
            out = out[0] * self.st['action_std'] + self.st['action_mean']
        return abs10_to_abs7(out.cpu().numpy())


class TwoStreamChunkPolicy(TwoStreamPolicy):
    """The same checkpoint as an ordinary remote chunk policy: `predict(obs)` runs both halves
    on the request frame (age 0) and returns the 7-dim chunk, which then arrives stale and is
    played by any stale-chunk executor (synchronous, anchored_gate, ...). With a 'fresh'
    checkpoint this is the same-model baseline for cloud_local."""

    def predict(self, obs) -> np.ndarray:
        return self.local(obs, super().predict(obs), 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', default='data/oracle_square_moving/npz')
    t.add_argument('--mode', default='two', choices=['two', 'local', 'cloud', 'fresh'])
    t.add_argument('--out', required=True)
    t.add_argument('--steps', type=int, default=60000)
    t.add_argument('--batch', type=int, default=32)
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--max-age', type=int, default=MAX_AGE)
    t.add_argument('--save-every', type=int, default=10000)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--local-enc', default='tiny', choices=['tiny', 'mobile', 'shared'])
    t.add_argument('--freeze', action='store_true', help='freeze the ImageNet backbone')
    t.add_argument('--local-drop', type=float, default=0.0)
    t.add_argument('--final-norm', action='store_true', help='LayerNorm closing each stack')
    t.add_argument('--dropout', type=float, default=0.1)
    t.add_argument('--shift', type=int, default=0, help='random-shift augmentation pad (px)')
    args = ap.parse_args()
    if args.cmd == 'train':
        train(args)


if __name__ == '__main__':
    main()
