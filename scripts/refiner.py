"""Phase D: a learned local refiner for stale chunks on Square with a moving nut (paper 2).

The hand-written corrections (anchoring for WHERE, the close gate for WHEN) are replaced by one
small MLP that runs on the robot every tick (chunk_executor/refined.py). It is trained on the
oracle: in a state s_t, the privileged planner's own plan is the right action, so the target is
plan(s_t), and the input is what the robot would actually hold at t:

    the stale chunk  plan(s_{t-k})[i:]       planned k ticks ago, i entries already played
    the robot        proprio(t)              EE position, fingers
    the object       o(t) - o(t-k)           tracked motion since the request, NOT its pose
    the age          k

Positions are relative to the current EE. Without the object's pose the refiner cannot plan the
task alone: where to go comes from the cloud's chunk, the model learns how the object's motion
and the arm's lag should bend it. Rotation passes through from the chunk (the conveyor only
translates the nut).

Three stages:

    collect  oracle_square's sweep (planner, chunk 16, the anchored executors at real delays and
             speeds) with every visited state recorded, so training states are the ones stale
             execution actually reaches, misses and recoveries included.
    train    pairs (t - k, i, t) from the recorded episodes -> MLP (torch), saved as plain numpy
             weights so the robot side needs no torch.
             With --refiner and --strategies refined, a DAgger round: the states the refiner
             itself reaches, labelled by the planner as before.
    (eval)   oracle_square.py / dp_mech.py --refiner <weights.npz> --strategies refined

Importable without torch or robosuite: `features`, `Refiner`, `make_refine_fn`.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

H = 16              # stale entries the refiner sees (ACT's chunk); shorter remainders are padded
K = 4               # entries predicted (only the first is played; the rest regularise)
MAX_AGE = 45        # ticks
SPEEDS = (0.0, 0.005, 0.01, 0.015, 0.02)


def features(proprio9, remaining, o_req, o_now, age) -> np.ndarray:
    """[stale positions - EE (H*3), stale gripper (H), object motion (3), fingers (2), age/30]."""
    p = np.asarray(proprio9, np.float32)
    r = np.asarray(remaining, np.float32)[:H]
    if len(r) < H:
        r = np.concatenate([r, np.repeat(r[-1:], H - len(r), axis=0)])
    return np.concatenate([(r[:, :3] - p[:3]).ravel(), r[:, 6],
                           np.asarray(o_now, np.float32) - np.asarray(o_req, np.float32),
                           p[7:9], [age / 30.0]]).astype(np.float32)


N_FEATURES = H * 3 + H + 3 + 2 + 1


def targets(proprio9, plan_now) -> np.ndarray:
    """[plan positions - EE (K*3), gripper closed (K) in {0, 1}]."""
    p = np.asarray(proprio9, np.float32)
    q = np.asarray(plan_now, np.float32)[:K]
    return np.concatenate([(q[:, :3] - p[:3]).ravel(), (q[:, 6] > 0).astype(np.float32)])


class Refiner:
    """ReLU MLP forward pass in numpy, from the weights `train` writes."""

    def __init__(self, path: str) -> None:
        z = np.load(path)
        self.layers = [(z[f'W{j}'], z[f'b{j}']) for j in range(int(z['n_layers']))]
        self.x_mean, self.x_std = z['x_mean'], z['x_std']
        self.y_mean, self.y_std = z['y_mean'], z['y_std']

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = (x - self.x_mean) / self.x_std
        for j, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if j < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        pos = h[:K * 3] * self.y_std + self.y_mean
        return np.concatenate([pos, h[K * 3:]])          # positions (m), gripper logits


def refine(model, proprio9, remaining, o_req, o_now, age, lead: int = 0) -> np.ndarray:
    """The action: position of predicted entry `lead` (a fresh plan's entry 0 is one step from
    the EE, and the OSC closes ~30% of the error per tick, so commanding it every tick crawls;
    a stale chunk's targets lead the arm by a few steps), gripper of entry 0."""
    out = model(features(proprio9, remaining, o_req, o_now, age))
    a = np.array(remaining[0], dtype=np.float32, copy=True)
    a[:3] = np.asarray(proprio9[:3], np.float32) + out[lead * 3:lead * 3 + 3]
    a[6] = 1.0 if out[K * 3] > 0 else -1.0
    return a


def make_refine_fn(model, anchor_fn, lead: int = 0):
    """The executor's refine_fn(obs, remaining, chunk_anchor, t). No tracker fix, or a chunk
    without an anchor: the object is taken as not having moved."""
    def refine_fn(obs, remaining, chunk_anchor, t):
        p9 = np.asarray(obs['proprio'])[-1][:9]
        now = anchor_fn(obs)
        if chunk_anchor is None or now is None:
            return refine(model, p9, remaining, np.zeros(3), np.zeros(3), 0, lead)
        t_req, o_req = chunk_anchor
        return refine(model, p9, remaining, o_req, now, min(t - t_req, MAX_AGE), lead)
    return refine_fn


def logging_refine_fn(refine_fn, path: str, every: int = 50):
    """Wraps a refine_fn to record its inputs on the robot: features and the oracle layout
    [proprio(9), object(3), object quat(4)] from obs['object'], written to `path` every `every`
    calls. The planner labels them in `train`, so the refiner can learn on chunks from the real
    cloud policy (ACT), not only the planner's."""
    X, S = [], []

    def save():
        np.savez(path, X=np.stack(X), S=np.stack(S))

    def fn(obs, remaining, chunk_anchor, t):
        p9 = np.asarray(obs['proprio'])[-1][:9]
        o = np.asarray(obs['object'])
        if chunk_anchor is None:
            X.append(features(p9, remaining, np.zeros(3), np.zeros(3), 0))
        else:
            X.append(features(p9, remaining, chunk_anchor[1], o[7:10],
                              min(t - chunk_anchor[0], MAX_AGE)))
        S.append(np.concatenate([p9, o[0:7]]).astype(np.float32))
        if len(X) % every == 0:
            save()
        return refine_fn(obs, remaining, chunk_anchor, t)
    fn.save = save
    return fn


# ------------------------------------------------------------------ collect
class Recorder:
    """Wraps the world; records the oracle layout [eef, quat, fingers, nut, nut quat] per tick."""

    def __init__(self, world, episodes: list) -> None:
        self.world, self.episodes, self._ep = world, episodes, None

    def __getattr__(self, name):
        return getattr(self.world, name)

    def _log(self, o) -> None:
        self._ep.append(np.concatenate([o['robot0_eef_pos'], o['robot0_eef_quat'],
                                        o['robot0_gripper_qpos'], o['SquareNut_pos'],
                                        o['SquareNut_quat']]).astype(np.float32))

    def reset(self):
        obs = self.world.reset()
        self._ep = []
        self.episodes.append(self._ep)
        self._log(obs)
        return obs

    def step(self, action):
        out = self.world.step(action)
        self._log(out[0])
        return out


def collect(args) -> None:
    import oracle_lift as ol
    import oracle_square as sq
    cfg = sq.SquareConfig(chunk_size=H)
    tracker = ol.tracker_from(args)
    deadline = int(args.deadline_s * args.hz)
    base = sq.make_env(args.hz, deadline + 10)
    episodes: list = []
    refine_fn = None
    if args.refiner:                       # DAgger: states the refiner itself reaches
        refine_fn = make_refine_fn(Refiner(args.refiner), sq.nut_tracker, args.refiner_lead)

    def held(o):
        return sq.is_held(o['robot0_eef_pos'], o['robot0_gripper_qpos'], o['SquareNut_pos'],
                          o['SquareNut_quat'])

    npz = os.path.splitext(args.out)[0] + '.npz'
    ol.run_sweep(args, policy=sq.SquarePolicy(cfg), tracker=tracker, deadline=deadline,
                 make_world=lambda v: Recorder(ol.MovingCube(
                     base, v, args.hz, joint='SquareNut_joint0', pos_key='SquareNut_pos',
                     held_fn=held, half_range=args.conveyor_range), episodes),
                 frame_fn=sq.make_frames(tracker),
                 executor_kw={'roll_fn': sq.square_roll, 'anchor_fn': sq.nut_tracker,
                              'diverged_fn': sq.square_diverged, 'gate_fn': sq.square_gate,
                              'refine_fn': refine_fn})
    lens = np.array([len(e) for e in episodes])
    np.savez_compressed(npz, states=np.concatenate(episodes), lens=lens)
    print(f'{len(episodes)} episodes, {lens.sum()} states -> {npz}')


# ------------------------------------------------------------------ train
def make_pairs(states: np.ndarray, lens: np.ndarray, per_tick: int, seed: int = 0):
    import oracle_square as sq
    cfg = sq.SquareConfig(chunk_size=H)
    rng = np.random.default_rng(seed)
    X, Y = [], []
    start = 0
    for n in lens:
        ep = states[start:start + n]
        start += n
        plans = [sq.plan_chunk(*sq.unpack(s), cfg) for s in ep]
        for t in range(1, n):
            for _ in range(per_tick):
                k = int(rng.integers(1, min(MAX_AGE, t) + 1))
                i = int(rng.integers(0, min(k, H - 1) + 1))
                X.append(features(ep[t][:9], plans[t - k][i:], ep[t - k][9:12], ep[t][9:12], k))
                Y.append(targets(ep[t][:9], plans[t]))
    return np.stack(X), np.stack(Y)


def train(args) -> None:
    import torch
    import oracle_square as sq
    cfg = sq.SquareConfig(chunk_size=H)
    files = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    Xs, Ys = [], []
    for j, f in enumerate(files):
        z = np.load(f)
        if 'X' in z:                       # logged on the robot (logging_refine_fn)
            X = np.repeat(z['X'], args.log_weight, axis=0)
            Y = np.repeat(np.stack([targets(s[:9], sq.plan_chunk(*sq.unpack(s), cfg))
                                    for s in z['S']]), args.log_weight, axis=0)
        else:
            X, Y = make_pairs(z['states'], z['lens'], args.per_tick, seed=j)
        Xs.append(X)
        Ys.append(Y)
        print(f'{f}: {len(X)} pairs', flush=True)
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    x_mean, x_std = X.mean(0), X.std(0) + 1e-6
    P = K * 3
    y_mean, y_std = Y[:, :P].mean(0), Y[:, :P].std(0) + 1e-6
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xt = torch.tensor((X - x_mean) / x_std, device=dev)
    Pt = torch.tensor((Y[:, :P] - y_mean) / y_std, device=dev)
    Gt = torch.tensor(Y[:, P:], device=dev)
    n_val = len(X) // 20
    perm = torch.randperm(len(X), device=dev)
    val, tr = perm[:n_val], perm[n_val:]
    dims = [N_FEATURES] + [args.width] * args.depth + [K * 4]
    mods = []
    for a, b in zip(dims[:-1], dims[1:]):
        mods += [torch.nn.Linear(a, b), torch.nn.ReLU()]
    net = torch.nn.Sequential(*mods[:-1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    bce = torch.nn.BCEWithLogitsLoss()

    def loss_on(idx):
        out = net(Xt[idx])
        return (torch.nn.functional.mse_loss(out[:, :P], Pt[idx]), bce(out[:, P:], Gt[idx]))

    for ep in range(args.epochs):
        net.train()
        for b in tr[torch.randperm(len(tr), device=dev)].split(args.batch):
            lp, lg = loss_on(b)
            opt.zero_grad()
            (lp + lg).backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            lp, lg = loss_on(val)
            out = net(Xt[val])
            err = ((out[:, :3] * torch.tensor(y_std[:3], device=dev)
                    - Pt[val][:, :3] * torch.tensor(y_std[:3], device=dev)).norm(dim=1))
            acc = ((out[:, P] > 0).float() == Gt[val][:, 0]).float().mean()
        print(f'epoch {ep + 1}: val pos {lp.item():.4f} grip {lg.item():.4f} | entry-0 error '
              f'median {err.median().item() * 1e3:.1f} mm, p90 {err.quantile(0.9).item() * 1e3:.1f}'
              f' mm, gripper acc {acc.item():.3f}', flush=True)
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    w = {f'W{j}': m.weight.detach().cpu().numpy().T for j, m in enumerate(lin)}
    w.update({f'b{j}': m.bias.detach().cpu().numpy() for j, m in enumerate(lin)})
    np.savez(args.out, n_layers=len(lin), x_mean=x_mean, x_std=x_std, y_mean=y_mean,
             y_std=y_std, **w)
    print(f'-> {args.out}')


def main() -> None:
    import oracle_lift as ol
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    ol.add_common_args(c)
    c.set_defaults(chunk=H)                # --out: the sweep's csv; states go next to it (.npz)
    c.add_argument('--strategies', default='anchored,anchored_gate')
    c.add_argument('--deadline-s', type=float, default=40.0)
    c.add_argument('--conveyor-range', type=float, default=0.06)
    c.add_argument('--refiner', default='', help="weights for 'refined' (a DAgger round)")
    c.add_argument('--refiner-lead', type=int, default=1)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True, help='directory of collect npz files')
    t.add_argument('--out', required=True)
    t.add_argument('--per-tick', type=int, default=4)
    t.add_argument('--log-weight', type=int, default=4, help='repeats of each logged pair')
    t.add_argument('--width', type=int, default=256)
    t.add_argument('--depth', type=int, default=3)
    t.add_argument('--epochs', type=int, default=30)
    t.add_argument('--batch', type=int, default=4096)
    t.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    collect(args) if args.cmd == 'collect' else train(args)


if __name__ == '__main__':
    main()
