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


RAW_KEYS = (('robot0_eef_pos', 3), ('robot0_eef_quat', 4), ('robot0_gripper_qpos', 2),
            ('SquareNut_pos', 3), ('SquareNut_quat', 4))
RAW_LAYOUT = 'now [eef3 quat4 grip2 nut3 nutquat4], previous tick [same], command [7]'


def raw_state(obs, prev, command) -> np.ndarray:
    """Everything a richer gate could use, for offline feature search: RAW_LAYOUT."""
    def flat(o):
        return np.concatenate([np.asarray(o[k], np.float32)[:n] for k, n in RAW_KEYS])
    return np.concatenate([flat(obs), flat(prev), np.asarray(command, np.float32)])


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
        self.F, self.Y, self.R, self._obs, self._prev = [], [], [], None, None

    def __getattr__(self, name):
        return getattr(self.world, name)

    def reset(self):
        self._obs = self.world.reset()
        self._prev = self._obs
        return self._obs

    def step(self, action):
        o = self._obs
        if o is not None and action[6] < 0:
            f = gate_features(o['robot0_eef_pos'], o['robot0_eef_quat'], o['SquareNut_pos'],
                              o['SquareNut_quat'], action)
            if f[0] < NEAR_XY and abs(f[1]) < NEAR_Z:
                self.F.append(f)
                self.R.append(raw_state(o, self._prev, action))
                self.Y.append(self._try_close(action))
                if len(self.F) % self.every == 0:
                    self.save()
        out = self.world.step(action)
        self._prev, self._obs = self._obs, out[0]
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
            np.savez(self.path, F=np.stack(self.F), Y=np.array(self.Y, np.float32),
                     R=np.stack(self.R))


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


# ------------------------------------------------------------------ feature search
def derived(R: np.ndarray) -> dict:
    """Candidate gate inputs from raw_state rows, positions in the handle frame (x along the
    handle, y across it: the fingers close along y)."""
    import oracle_square as sq
    cols = {}
    rows = []
    for r in R:
        eef, eq, grip, nut, nq = r[0:3], r[3:7], r[7:9], r[9:12], r[12:16]
        eef0, nut0, cmd = r[16:19], r[25:28], r[32:39]
        rot = sq.rz(-sq.yaw_of(nq))
        e = rot @ (eef - sq.handle_pos(nut, nq))
        v_rel = rot @ ((eef - eef0) - (nut - nut0))
        lag = rot @ (cmd[:3] - eef)
        pred = e + 3 * v_rel
        rows.append([np.hypot(e[0], e[1]), e[2], np.linalg.norm(cmd[:3] - eef),
                     abs(sq.grasp_turn(eq, nq)),
                     e[0], e[1], sq.grasp_turn(eq, nq),
                     v_rel[0], v_rel[1], v_rel[2], np.linalg.norm(v_rel),
                     np.linalg.norm(eef - eef0), np.linalg.norm(nut - nut0),
                     lag[0], lag[1], lag[2],
                     pred[0], pred[1], pred[2], np.hypot(pred[0], pred[1]),
                     float(np.sum(np.abs(grip)))])
    names = ['dxy', 'dz', 'lag', 'yaw', 'e_along', 'e_across', 'yaw_signed',
             'vrel_along', 'vrel_across', 'vrel_z', 'vrel', 'v_eef', 'v_nut',
             'lag_along', 'lag_across', 'lag_z', 'pred_along', 'pred_across', 'pred_z',
             'pred_xy', 'width']
    A = np.array(rows, np.float32)
    for j, n in enumerate(names):
        cols[n] = A[:, j]
    return cols


FEATURE_SETS = {
    'base4 (current gate)': ['dxy', 'dz', 'lag', 'yaw'],
    '+ signed, handle frame': ['e_along', 'e_across', 'dz', 'lag', 'yaw_signed'],
    '+ velocities': ['e_along', 'e_across', 'dz', 'lag', 'yaw_signed', 'vrel_along',
                     'vrel_across', 'vrel_z', 'v_eef', 'v_nut'],
    '+ lag vector': ['e_along', 'e_across', 'dz', 'yaw_signed', 'vrel_along', 'vrel_across',
                     'vrel_z', 'lag_along', 'lag_across', 'lag_z'],
    'prediction (3 ticks)': ['pred_along', 'pred_across', 'pred_z', 'yaw_signed', 'vrel'],
    'all': None,
}


def _auc_ap(score, y):
    order = np.argsort(-score)
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    ap = float((prec * ys).sum() / max(ys.sum(), 1))
    ranks = np.empty(len(score))
    ranks[np.argsort(score)] = np.arange(1, len(score) + 1)
    npos, nneg = ys.sum(), len(ys) - ys.sum()
    auc = float((ranks[y > 0.5].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))
    return auc, ap


def search(args) -> None:
    import torch
    files = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    cols, Y, G = {}, [], []
    for g, f in enumerate(files):
        z = np.load(f)
        for k, v in derived(z['R']).items():
            cols.setdefault(k, []).append(v)
        Y.append(z['Y'])
        G.append(np.full(len(z['Y']), g))
    cols = {k: np.concatenate(v) for k, v in cols.items()}
    Y, G = np.concatenate(Y), np.concatenate(G)
    print(f'{len(Y)} probes in {len(files)} files, {Y.mean():.3f} held; '
          f'base rate = the AP of a random gate')
    print(f'{"feature set":28s} {"logistic AUC/AP":>18s} {"MLP AUC/AP":>14s}')
    for name, keys in FEATURE_SETS.items():
        keys = keys or list(cols)
        X = np.stack([cols[k] for k in keys], 1)
        res = {}
        for model in ('logistic', 'mlp'):
            score = np.zeros(len(Y))
            for g in range(len(files)):                 # leave one file out
                tr, te = G != g, G == g
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                Xt = torch.tensor((X[tr] - mu) / sd)
                Yt = torch.tensor(Y[tr])
                net = (torch.nn.Linear(X.shape[1], 1) if model == 'logistic' else
                       torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(),
                                           torch.nn.Linear(64, 64), torch.nn.ReLU(),
                                           torch.nn.Linear(64, 1)))
                opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
                pw = ((1 - Yt.mean()) / Yt.mean().clamp(min=1e-3)).detach()
                for _ in range(300):
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        net(Xt).squeeze(1), Yt, pos_weight=pw)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    score[te] = net(torch.tensor((X[te] - mu) / sd)).squeeze(1).numpy()
            res[model] = _auc_ap(score, Y)
        print(f'{name:28s} {res["logistic"][0]:8.3f}/{res["logistic"][1]:.3f}'
              f' {res["mlp"][0]:8.3f}/{res["mlp"][1]:.3f}', flush=True)
    print('single features (AUC of the feature or its negative, whichever is higher):')
    single = sorted(((max(_auc_ap(v, Y)[0], _auc_ap(-v, Y)[0]), k) for k, v in cols.items()),
                    reverse=True)
    for auc, k in single:
        print(f'  {k:14s} {auc:.3f}')


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
    f = sub.add_parser('search', help='offline: which inputs predict a held close?')
    f.add_argument('--data', required=True)
    args = ap.parse_args()
    train(args) if args.cmd == 'train' else search(args)


if __name__ == '__main__':
    main()
