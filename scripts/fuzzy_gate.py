"""A neuro-fuzzy close gate for Square (paper 2, after Phase D).

The learned refiner lost to the hand rules on ACT because it overrode every action (Phase D,
docs/paper2-progress.md). This learns only the one decision the hard gate makes, whether to close
now, as a zero-order Takagi-Sugeno system with trainable Gaussian memberships (ANFIS-style). Its
output is a convex combination of per-rule logits, so it is bounded and every rule stays readable:
"IF xy is NEAR and dz is LOW and lag is SMALL and yaw is ALIGNED THEN close (+3.1)".

Inputs, all local (the robot's own state, the tracked nut, the command it is about to play):

    dxy   |EE xy - handle xy|              (m)
    dz    EE z - handle z - grasp_dz       (m, signed)
    lag   |command position - EE|          (m: is the arm still catching up with its target?)
    yaw   |grasp turn|                     (rad)

Labels are counterfactual: on every open tick near the handle, `Prober` snapshots the simulator,
closes the gripper at the commanded pose, lifts a little, checks the nut came with it, and
restores the snapshot, so the episode runs on unchanged. The label is "a close issued now would
have held", which is what the gate should predict; the hard gate's thresholds are a guess at it.

    collect  dp_mech.py --probe-log <npz>      (any strategy; the world is wrapped)
    train    fuzzy_gate.py train --data <dir>  -> <weights.npz>  (prints the rules)
    eval     dp_mech.py / oracle_square.py --fuzzy-gate <weights.npz>  (replaces square_gate)
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os

import numpy as np

SCALE = np.array([0.01, 0.01, 0.01, 0.1])          # feature units: cm, cm, cm, 0.1 rad
NAMES = ('dxy', 'dz', 'lag', 'yaw')
NEAR_XY, NEAR_Z = 0.04, 0.05                        # probe only within this box of the handle


def gate_features(eef, eq, nut, nq, command) -> np.ndarray:
    import oracle_square as sq
    eef = np.asarray(eef, float)
    h = sq.handle_pos(nut, nq)
    cfg = sq.SquareConfig()
    return np.array([np.linalg.norm(eef[:2] - h[:2]), eef[2] - h[2] - cfg.grasp_dz,
                     np.linalg.norm(np.asarray(command[:3], float) - eef),
                     abs(sq.grasp_turn(eq, nq))], dtype=np.float32)


class FuzzyGate:
    """Zero-order TS inference in numpy: rules = every combination of each input's memberships."""

    def __init__(self, path: str) -> None:
        z = np.load(path)
        self.c, self.s, self.q = z['centers'], z['widths'], z['consequents']
        self.threshold = float(z['threshold'])
        self.rules = np.array(list(itertools.product(*[range(self.c.shape[1])] * self.c.shape[0])))

    def logit(self, f: np.ndarray) -> float:
        x = np.asarray(f, float) / SCALE
        mu = np.exp(-0.5 * ((x[:, None] - self.c) / self.s) ** 2)     # [inputs, mfs]
        w = np.prod(mu[np.arange(len(x)), self.rules], axis=1)         # [rules]
        return float(w @ self.q / (w.sum() + 1e-12))

    def __call__(self, f) -> bool:
        return self.logit(f) > self.threshold


def make_gate_fn(gate: FuzzyGate, tracker_fn):
    """gate_fn(obs, action) for the executors, reading the oracle layout like square_gate."""
    import dp_mech
    import oracle_square as sq

    def gate_fn(obs, action):
        layout = obs if 'object' not in obs else dp_mech.oracle_layout(obs, 'square')
        eef, eq, _grip, _nut, nq = sq.unpack(np.asarray(layout['proprio'])[-1])
        tracked = tracker_fn(obs)
        if tracked is None:
            return True
        return gate(gate_features(eef, eq, tracked, nq, action))
    return gate_fn


# ------------------------------------------------------------------ collect
def _data_arrays(d) -> dict:
    out = {}
    for k in dir(d):
        if k.startswith('_'):
            continue
        try:
            v = getattr(d, k)
        except Exception:
            continue
        if isinstance(v, np.ndarray) and v.size and v.flags.writeable:
            out[k] = v.copy()
    return out


def _numeric_state(obj) -> dict:
    import copy
    return {k: copy.deepcopy(v) for k, v in vars(obj).items()
            if isinstance(v, (np.ndarray, float, int, bool, np.generic)) or
            type(v).__name__ == 'DeltaBuffer' or type(v).__name__ == 'RingBuffer'}


class Prober:
    """Wraps the (MovingCube) world. Before each step with an open command near the handle, try
    the close counterfactually and record (features, held)."""

    def __init__(self, world, path: str, every: int = 200) -> None:
        self.world, self.path, self.every = world, path, every
        self.F, self.Y, self._obs = [], [], None

    def __getattr__(self, name):
        return getattr(self.world, name)

    def reset(self):
        self._obs = self.world.reset()
        return self._obs

    def step(self, action):
        o = self._obs
        if o is not None and action[6] < 0:
            f = gate_features(o['robot0_eef_pos'], o['robot0_eef_quat'], o['SquareNut_pos'],
                              o['SquareNut_quat'], action)
            if f[0] < NEAR_XY and abs(f[1]) < NEAR_Z:
                self.F.append(f)
                self.Y.append(self._try_close(action))
                if len(self.F) % self.every == 0:
                    self.save()
        out = self.world.step(action)
        self._obs = out[0]
        return out

    def _try_close(self, action) -> float:
        import oracle_square as sq
        env = self.world.env
        sim, grip = env.sim, env.robots[0].gripper
        d = sim.data
        # every mjData array, restored bitwise: get_state() returns live views and leaves out the
        # solver warm start and the derived quantities the next step starts from
        arrays = _data_arrays(d)
        snap = (d.time, env.timestep, env.cur_time, grip.current_action.copy(),
                self.world._dir, self.world._obs, env.done)
        robot = env.robots[0]
        objs = (robot, robot.controller, grip)       # numeric internal state (goals, buffers)
        saved = [_numeric_state(x) for x in objs]
        z0 = float(self._obs['SquareNut_pos'][2])
        a = np.array(action, dtype=np.float32, copy=True)
        a[6] = 1.0
        o = None
        for _ in range(sq.SquareConfig().close_ticks + 2):
            o, *_ = self.world.step(a)
        for _ in range(3):
            a[2] += 0.008
            o, *_ = self.world.step(a)
        held = (sq.is_held(o['robot0_eef_pos'], o['robot0_gripper_qpos'], o['SquareNut_pos'],
                           o['SquareNut_quat']) and float(o['SquareNut_pos'][2]) > z0 + 0.01)
        d.time, env.timestep, env.cur_time, grip.current_action, self.world._dir, \
            self.world._obs, env.done = snap
        for k, v in arrays.items():
            cur = getattr(d, k)
            if cur.shape == v.shape:        # contact-sized (efc_*) arrays are rebuilt each step
                cur[...] = v
        for x, st in zip(objs, saved):
            for k, v in st.items():
                setattr(x, k, v)
        return float(held)

    def save(self) -> None:
        if self.F:
            np.savez(self.path, F=np.stack(self.F), Y=np.array(self.Y, np.float32))


# ------------------------------------------------------------------ train
def train(args) -> None:
    import torch
    F, Y = [], []
    for f in sorted(glob.glob(os.path.join(args.data, '*.npz'))):
        z = np.load(f)
        F.append(z['F'])
        Y.append(z['Y'])
    X = torch.tensor(np.concatenate(F) / SCALE, dtype=torch.float32)
    Yt = torch.tensor(np.concatenate(Y))
    n_in, m = X.shape[1], args.mfs
    print(f'{len(X)} probes, {Yt.mean().item():.3f} held')
    perm = torch.randperm(len(X))
    val, tr = perm[:len(X) // 5], perm[len(X) // 5:]
    qs = torch.quantile(X[tr], torch.linspace(0.1, 0.9, m), dim=0).T          # [inputs, mfs]
    c = torch.nn.Parameter(qs.clone())
    log_s = torch.nn.Parameter(torch.log((X[tr].std(0)[:, None] / m).expand(n_in, m).clone()))
    rules = torch.tensor(list(itertools.product(*[range(m)] * n_in)))
    q = torch.nn.Parameter(torch.zeros(len(rules)))
    opt = torch.optim.Adam([c, log_s, q], lr=args.lr)
    pos_w = (1 - Yt[tr].mean()) / Yt[tr].mean().clamp(min=1e-3)

    def logits(x):
        mu = torch.exp(-0.5 * ((x[:, :, None] - c) / log_s.exp()) ** 2)      # [N, inputs, mfs]
        w = torch.stack([mu[:, i, rules[:, i]] for i in range(n_in)], 0).prod(0)  # [N, rules]
        return (w * q).sum(1) / (w.sum(1) + 1e-12)

    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for ep in range(args.epochs):
        for b in tr[torch.randperm(len(tr))].split(args.batch):
            loss = lossf(logits(X[b]), Yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % max(1, args.epochs // 10) == 0:
            with torch.no_grad():
                lv = logits(X[val])
                pred = lv > 0
                y = Yt[val] > 0.5
                prec = (pred & y).sum() / pred.sum().clamp(min=1)
                rec = (pred & y).sum() / y.sum().clamp(min=1)
                print(f'epoch {ep + 1}: val loss {lossf(lv, Yt[val]).item():.4f} '
                      f'precision {prec.item():.3f} recall {rec.item():.3f}', flush=True)
    with torch.no_grad():
        # hard-gate baseline on the same probes, for reference
        import oracle_square as sq
        cfg = sq.SquareConfig()
        Xv = X[val].numpy() * SCALE
        hard = ((Xv[:, 0] <= cfg.grasp_xy_tol) & (np.abs(Xv[:, 1]) <= cfg.z_tol)
                & (Xv[:, 3] <= 2 * cfg.yaw_tol))
        y = Yt[val].numpy() > 0.5
        print(f'hard gate on val: precision {(hard & y).sum() / max(hard.sum(), 1):.3f} '
              f'recall {(hard & y).sum() / max(y.sum(), 1):.3f}')
    C, S, Q = c.detach().numpy(), log_s.exp().detach().numpy(), q.detach().numpy()
    np.savez(args.out, centers=C, widths=S, consequents=Q, threshold=args.threshold)
    print('memberships (feature units: cm, cm, cm, 0.1 rad):')
    for i, n in enumerate(NAMES):
        print(f'  {n}: ' + ', '.join(f'N({C[i, j]:.2f}, {S[i, j]:.2f})' for j in range(m)))
    print('strongest rules (membership index per input -> logit):')
    for r in np.argsort(-np.abs(Q))[:8]:
        print(f'  {dict(zip(NAMES, rules[r].tolist()))} -> {Q[r]:+.2f}')
    print(f'-> {args.out}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--mfs', type=int, default=2, help='memberships per input')
    t.add_argument('--epochs', type=int, default=200)
    t.add_argument('--batch', type=int, default=1024)
    t.add_argument('--lr', type=float, default=0.02)
    t.add_argument('--threshold', type=float, default=0.0, help='close when logit > this')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
