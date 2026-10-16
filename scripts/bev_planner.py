"""BEV query planner, step 2: train bev_model.BEVPlanner on the top-camera oracle demos, and run
it in the step harness as the cloud_local executor's policy.

Training is two_stream.py's (random age in [0, max_age], chunk L1, random shift), on demos that
also carry the close top view and the camera matrices (oracle_demos.py collect --top). With
--prev, each sample also runs the local half on the frame `replan` ticks earlier (same memory, no
gradient, as BEVFormer treats its history) and feeds that map as the robot's previous BEV; it is
withheld with probability 0.2 so the planner also works at an episode's first plan.

    python3 scripts/bev_planner.py --data data/oracle_square_top/npz --out outputs/bev/bev.pt
    python3 scripts/dp_mech.py --task square --backend bev --weights outputs/bev/bev.pt \
        --strategies cloud_local ...
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

from two_stream import H, MAX_AGE, chunk_targets, random_shift, sample_index, stats

KEYS = ('agentview', 'top', 'wrist', 'state', 'action')


def load_npz(dirs: str) -> tuple[list[dict], np.ndarray]:
    """Episodes with the top view, and the [2, 3, 4] (agentview, top) projection, which is the
    same in every episode (fixed cameras); checked."""
    eps, P = [], None
    for d in dirs.split(','):
        for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
            z = np.load(f)
            p = np.stack([z['P_agentview'], z['P_birdview']]).astype(np.float32)
            if P is None:
                P = p
            elif not np.allclose(P, p, atol=1e-3):
                raise ValueError(f'{f}: camera matrices differ from the first episode')
            eps.append({k: z[k] for k in KEYS})
    return eps, P


def train(args) -> None:
    import torch
    import torch.nn.functional as F
    from bev_model import BEVPlanner
    eps, P = load_npz(args.data)
    st = stats(eps)
    lens = np.array([len(e['action']) for e in eps])
    print(f'{len(eps)} episodes, {lens.sum()} frames', flush=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    cfg = {'local_enc': args.local_enc, 'use_prev': args.prev, 'use_local': not args.no_local,
           'max_age': args.max_age}
    model = BEVPlanner(torch.from_numpy(P), **cfg).to(dev)
    print(f'config {cfg}; {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters',
          flush=True)
    bb = [p for n, p in model.named_parameters() if '.backbone.' in n]
    rest = [p for n, p in model.named_parameters() if '.backbone.' not in n]
    opt = torch.optim.AdamW([{'params': rest}, {'params': bb}], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [args.lr, args.lr * 0.1],
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
            if key in ('agentview', 'top', 'wrist'):
                return random_shift(x, args.shift) if args.shift else x
            return (x - sm) / ss
        target = (torch.from_numpy(np.stack(acts)).to(dev) - am) / as_
        mask = torch.from_numpy(np.stack(masks)).to(dev)
        age = torch.from_numpy(t - t0).long().to(dev)
        mem = model.memory(grab('agentview', t0), grab('top', t0), grab('state', t0))
        prev = None
        if args.prev and rng.random() > 0.2:
            tp = np.maximum(t - args.replan, t0)        # the previous plan, same memory
            with torch.no_grad():
                _, prev = model.plan(mem, grab('agentview', tp), grab('top', tp),
                                     grab('wrist', tp), grab('state', tp),
                                     torch.from_numpy(tp - t0).long().to(dev), return_bev=True)
        pred = model.plan(mem, grab('agentview', t), grab('top', t), grab('wrist', t),
                          grab('state', t), age, prev)
        loss = (F.l1_loss(pred, target, reduction='none') * ~mask[..., None]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            print(f'step {step} loss {loss.item():.4f} ({time.time() - t_start:.0f} s)', flush=True)
        if step % args.save_every == 0 or step == args.steps:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            torch.save({'model': model.state_dict(), 'cfg': cfg, 'stats': st, 'step': step,
                        'replan': args.replan}, args.out)


class BEVPolicy:
    """The harness's policy for a trained BEVPlanner (cloud_local executor): `predict(obs)` is
    the cloud half (memory), `local(obs, memory, age)` the robot half, which keeps its previous
    map between plans when trained with --prev; `reset()` clears it per episode."""
    absolute_actions = True
    needs_wrist = True
    needs_top = True
    n_obs_steps = 1
    action_dim = 7
    chunk_size = H
    guided_resampling = False
    denoise_steps = 1

    def __init__(self, path: str, device: str = 'cuda') -> None:
        import torch
        from bev_model import BEVPlanner
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.dev = device if torch.cuda.is_available() else 'cpu'
        self.model = BEVPlanner(**ck['cfg'])
        self.model.load_state_dict(ck['model'])
        self.model.to(self.dev).eval()
        self.max_age = ck['cfg']['max_age']
        self.st = {k: torch.tensor(v, device=self.dev) for k, v in ck['stats'].items()}
        self._prev = None

    def reset(self) -> None:
        self._prev = None

    def _obs(self, obs):
        import torch
        g = lambda k: torch.from_numpy(np.ascontiguousarray(np.asarray(obs[k])[-1:])).to(self.dev)
        state = (g('proprio').float() - self.st['state_mean']) / self.st['state_std']
        return g('agentview'), g('top'), g('wrist'), state

    def predict(self, obs) -> np.ndarray:
        import torch
        agent, top, _, state = self._obs(obs)
        with torch.no_grad():
            return self.model.memory(agent, top, state)[0].cpu().numpy()

    def predict_inpaint(self, obs, prefix, weights):
        return self.predict(obs)

    def local(self, obs, memory, age: int) -> np.ndarray:
        import torch
        from evh_controller.rotation import abs10_to_abs7
        with torch.no_grad():
            agent, top, wrist, state = self._obs(obs)
            a = torch.tensor([int(age)], device=self.dev)
            mem = torch.from_numpy(np.asarray(memory, np.float32))[None].to(self.dev)
            out, bev = self.model.plan(mem, agent, top, wrist, state, a, self._prev,
                                       return_bev=True)
            if self.model.use_prev:
                self._prev = bev
            out = out[0] * self.st['action_std'] + self.st['action_mean']
        return abs10_to_abs7(out.cpu().numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default='data/oracle_square_top/npz')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--max-age', type=int, default=MAX_AGE)
    ap.add_argument('--save-every', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--local-enc', default='mobile', choices=['tiny', 'mobile'])
    ap.add_argument('--prev', action='store_true', help="feed the robot's previous map")
    ap.add_argument('--no-local', action='store_true', help='ablation: no fresh local images')
    ap.add_argument('--replan', type=int, default=4, help='ticks between plans (for --prev)')
    ap.add_argument('--shift', type=int, default=4)
    train(ap.parse_args())


if __name__ == '__main__':
    main()
