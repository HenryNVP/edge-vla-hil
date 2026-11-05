#!/usr/bin/env python3
"""Interp weight edit: a rank-r update to ONE layer of an ACT policy, fitted on a few shifted
demo replays, to close the out-of-range action gap (plan §11, H-E1).

W' = W + B A on the chosen --site (torch parametrize; only A, B train; B starts at zero, so the
edit starts as the parent). Loss on the inference path (eval mode: zero latent, no dropout):
L1 on normalised action chunks of the edit episodes + --beta x L1 to the frozen parent's outputs
on unshifted states (preservation). The merged model is saved as a normal pretrained_model dir,
so interp_gainmap / interp_collect run on it unchanged.

Data are interp_augment's replays, read straight from the HDF5 exactly as robomimic_to_lerobot
builds the training set (abs7 -> abs10 actions, eef_pos + eef_quat + gripper_qpos state, images
as stored). Split by SOURCE demo: the first 160 sources are the edit pool, the rest validation.
--labels shifted uses the first n/2 sources' 2 shifted copies; unshifted (control) uses copy 0
of the first n sources. Preservation states come from the base set's pool sources.

    python3 scripts/interp_edit.py --parent <act_lift_aug_base pretrained_model> --site head \
        --rank 2 --axis x --n 40 --out outputs/interp/edit/x_head_r2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sil_step  # noqa: E402,F401  (puts the ROS-free packages on sys.path)
from robomimic_to_lerobot import STATE_KEYS, sorted_demos  # noqa: E402

AUG = 'outputs/interp/aug'
CAMS = ('agentview_image', 'robot0_eye_in_hand_image')
POOL = 160                                   # sources below this rank: edit pool; rest: validation
SITES = {                                    # name -> (module path under ACT.model, parameter)
    'img_proj': ('encoder_img_feat_input_proj', 'weight'),
    'enc3_ff2': ('encoder.layers.3.linear2', 'weight'),
    'xattn_in': ('decoder.layers.0.multihead_attn', 'in_proj_weight'),
    'xattn_out': ('decoder.layers.0.multihead_attn.out_proj', 'weight'),
    'dec_ff1': ('decoder.layers.0.linear1', 'weight'),
    'dec_ff2': ('decoder.layers.0.linear2', 'weight'),
    'head': ('action_head', 'weight'),
}


def load(path: str, pick) -> list[dict]:
    """Episodes of a replay HDF5 for which pick(source rank, shift) is true."""
    import h5py
    from evh_controller.rotation import abs7_to_abs10
    eps = []
    with h5py.File(path, 'r') as f:
        data = f['data']
        names = sorted_demos(data)
        sources = sorted({str(data[n].attrs['source']) for n in names},
                         key=lambda s: int(s.split('_')[1]))
        rank = {s: i for i, s in enumerate(sources)}
        for n in names:
            ep = data[n]
            shift = np.asarray(ep.attrs['shift'])
            if pick(rank[str(ep.attrs['source'])], shift):
                eps.append(dict(img=[ep['obs'][c][:] for c in CAMS],
                                state=np.concatenate([ep['obs'][k][:] for k in STATE_KEYS], 1)
                                .astype(np.float32),
                                act=abs7_to_abs10(ep['actions_abs'][:].astype(np.float32)),
                                shift=shift, rank=rank[str(ep.attrs['source'])]))
    return eps


def frames(eps: list[dict], H: int):
    """All (images per camera, state, chunk, pad mask, episode) frames; chunks padded with the
    last action."""
    img, state, chunk, pad, epi = [[], []], [], [], [], []
    for e, ep in enumerate(eps):
        T = len(ep['act'])
        for t in range(T):
            epi.append(e)
            j = np.minimum(np.arange(t, t + H), T - 1)
            for c in range(2):
                img[c].append(ep['img'][c][t])
            state.append(ep['state'][t])
            chunk.append(ep['act'][j])
            pad.append(np.arange(t, t + H) >= T)
    return [np.stack(i) for i in img], np.stack(state), np.stack(chunk), np.stack(pad), np.array(epi)


def subset(data, idx):
    return [i[idx] for i in data[0]], *(x[idx] for x in data[1:])


def low_rank(shape, r: int, seed: int):
    import torch
    from torch import nn

    class LowRank(nn.Module):
        def __init__(self):
            super().__init__()
            out, inp = shape[0], int(np.prod(shape[1:]))
            g = torch.Generator().manual_seed(seed)
            self.A = nn.Parameter(torch.randn(r, inp, generator=g) / inp ** 0.5)
            self.B = nn.Parameter(torch.zeros(out, r))

        def forward(self, W):
            return W + (self.B @ self.A).view_as(W)
    return LowRank()


def targets(site: str, model) -> list[tuple]:
    """(module, parameter name) pairs an edit touches; 'all' = every site and every encoder /
    decoder linear and attention projection."""
    if site != 'all':
        path, name = SITES[site]
        return [(model.get_submodule(path), name)]
    out = [(model.get_submodule(p), n) for p, n in SITES.values()]
    for _, mod in list(model.encoder.named_modules()) + list(model.decoder.named_modules()):
        for n in ('weight', 'in_proj_weight'):
            w = getattr(mod, n, None) if n in mod._parameters else None
            if w is not None and w.ndim == 2 and (mod, n) not in out:
                out.append((mod, n))
    return out


def train(args, act, edit, pres, parent_pres, run, norm, dev) -> int:
    """Fit W + B A on the site (or every weight, 'full'); returns the number of trained values."""
    import torch
    import torch.nn.utils.parametrize as P
    act.requires_grad_(args.site == 'full')
    if args.site == 'full':
        params = list(act.parameters())
    else:
        params, tgts = [], targets(args.site, act.model)
        for i, (mod, name) in enumerate(tgts):
            lr_mod = low_rank(getattr(mod, name).shape, args.rank, args.seed + i).to(dev)
            P.register_parametrization(mod, name, lr_mod)
            params += list(lr_mod.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)
    for step in range(args.steps):
        i = rng.integers(len(edit[1]), size=args.batch)
        m = ~torch.from_numpy(edit[3][i]).to(dev)
        loss = (run(edit, i) - norm(edit[2][i])).abs().mean(-1)[m].mean()
        if args.beta:
            j = rng.integers(len(pres[1]), size=args.batch)
            loss = loss + args.beta * (run(pres, j) - parent_pres[j]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == args.steps - 1:
            print(f'step {step}: loss {loss.item():.4f}', flush=True)
    for mod, name in ([] if args.site == 'full' else tgts):
        P.remove_parametrizations(mod, name, leave_parametrized=True)
    return sum(p.numel() for p in params)


def ridge_head(X, Y, w, groups, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    """Weighted ridge with intercept, Y ~ X dW^T + db; lambda (x weighted mean squared row norm /
    D) by 5-fold CV over groups. Returns (dW, db, lambda)."""
    w = w / w.sum()

    def fit(m, lam):
        ww = w[m] / w[m].sum()
        mx, my = ww @ X[m], ww @ Y[m]
        Xc = X[m] - mx
        scale = ww @ (Xc ** 2).sum(1) / X.shape[1]
        dW = np.linalg.solve((Xc * ww[:, None]).T @ Xc + lam * scale * np.eye(X.shape[1]),
                             (Xc * ww[:, None]).T @ (Y[m] - my)).T
        return dW, my - dW @ mx
    folds = np.array_split(np.random.default_rng(0).permutation(np.unique(groups)), 5)
    err = []
    for lam in lams:
        e = 0.0
        for fold in folds:
            te = np.isin(groups, fold)
            dW, db = fit(~te, lam)
            e += w[te] @ ((X[te] @ dW.T + db - Y[te]) ** 2).sum(1)
        err.append(e)
    lam = lams[int(np.argmin(err))]
    return (*fit(np.ones(len(X), bool), lam), lam)


def transplant_rows(d: dict, W: np.ndarray, b: np.ndarray):
    """H-E2, label-free targets from an interp_act_sites sweep (sim placements only, no demos):
    each chunk row's xy head output A(o) (normalised) extended out of the demos' range as the
    parent's own in-range behaviour, label A(0) + R o with R, A(0) the least-squares fit over that
    operating point's in-range grid points; in range the label is A (no change). Returns the head
    inputs X (rows, D), the change Y = label - A (rows, 2), scene groups, and the in-range mask."""
    from interp_gainmap import TASKS
    F = d['dec'].astype(np.float64)                          # (P, G, G, 16, D)
    P_, G, _, R_ = F.shape[:4]
    A = F @ W[:2].T + b[:2]
    O = np.stack(np.meshgrid(d['grid'], d['grid'], indexing='ij'), -1)
    pos = d['nut_xy'][:, None, None] + O
    inr = np.all((pos >= TASKS['lift']['lo']) & (pos <= TASKS['lift']['hi']), -1)
    label = A.copy()
    for p in range(P_):
        if inr[p].sum() < 4:
            continue
        X1 = np.c_[O[inr[p]], np.ones(inr[p].sum())]
        for j in range(R_):
            coef = np.linalg.lstsq(X1, A[p][inr[p]][:, j], rcond=None)[0]   # (3, 2)
            label[p][~inr[p], j] = O[~inr[p]] @ coef[:2] + coef[2]
    return (F.reshape(-1, F.shape[-1]), (label - A).reshape(-1, 2),
            np.repeat(d['scene'], G * G * R_), np.repeat(inr.reshape(-1), R_))


def transplant_fit(d: dict, W: np.ndarray, b: np.ndarray, keep: np.ndarray | None = None,
                   teacher: np.ndarray | None = None):
    """Closed-form head edit from transplant_rows (xy rows), or, given `teacher` (another
    policy's normalised chunks at the sweep's probes, (P, G, G, 16, rows)), towards those on every
    row. In-range and out-of-range sweep rows weigh the same, and `keep` (head inputs of the
    parent's own training states, change 0) weighs as much as the sweep. Returns (dW, db, lambda)."""
    X, Y, groups, inr = transplant_rows(d, W, b)
    if teacher is not None:
        Y = teacher.reshape(-1, teacher.shape[-1]) - (X @ W.T + b)
    w = np.where(inr, 0.5 / max(inr.sum(), 1), 0.5 / max((~inr).sum(), 1))
    if keep is not None:
        X = np.vstack([X, keep])
        Y = np.vstack([Y, np.zeros((len(keep), Y.shape[1]))])
        w = np.r_[w, np.full(len(keep), 1.0 / len(keep))]
        groups = np.r_[groups, groups.max() + 1 + np.arange(len(keep)) % 5]
    return ridge_head(X, Y, w, groups)


def relu_edit(args, act, pres, run, dev) -> int:
    """H-E2 as weights, label-free: 4 of the decoder FFN's least-active hidden units become
    "excess beyond the demo box" units (+x, +y, -x, -y). Input weights: a ridge readout of the
    absolute cube xy (mm) from the FFN input, fitted on an interp_act_sites sweep (sim placements,
    no demos), shifted so each unit switches on --margin past the box edge. Output weights (4
    columns of linear2) by Adam through the rest of the network, which is a per-row function of
    the FFN input: xy rows towards transplant_rows' labels (the parent's own in-range slope
    extended), every other output and the parent's training states (keep) unchanged."""
    import torch
    from interp_gainmap import TASKS
    d = dict(np.load(args.relu))
    L, dn, head = act.model.decoder.layers[0], act.model.decoder.norm, act.model.action_head
    W0, b0 = (t.detach().cpu().double().numpy() for t in (head.weight, head.bias))
    _, Ylab, groups, inr = transplant_rows(d, W0, b0)
    X = d['ffn_in'].astype(np.float32).reshape(-1, d['ffn_in'].shape[-1])
    g = d['grid'] * 1e3
    pos = (d['nut_xy'][:, None, None] * 1e3 + np.stack(np.meshgrid(g, g, indexing='ij'), -1))
    pos = np.repeat(pos.reshape(-1, 2), d['ffn_in'].shape[3], axis=0)
    Pr, p0, lam = ridge_head(X.astype(np.float64), pos, np.ones(len(X)), groups)
    grab = {}
    hook = L.linear1.register_forward_pre_hook(lambda _m, i: grab.__setitem__('x', i[0]))
    keep_idx = np.random.default_rng(args.seed).choice(len(pres[1]), min(2000, len(pres[1])), replace=False)
    with torch.no_grad():
        kp = subset(pres, keep_idx)
        keep = torch.cat([(run(kp, np.arange(i, min(i + 256, len(kp[1])))), grab['x'])[1]
                          .transpose(0, 1).reshape(-1, X.shape[1]) for i in range(0, len(kp[1]), 256)])
    hook.remove()
    Xt = torch.from_numpy(X).to(dev)
    with torch.no_grad():
        act_all = torch.relu(L.linear1(torch.cat([Xt, keep])))
        units = torch.argsort(act_all.mean(0))[:4]
        before_act = act_all[:, units].mean(0).cpu().numpy()
    lo, hi = np.array(TASKS['lift']['lo']) * 1e3, np.array(TASKS['lift']['hi']) * 1e3
    m = args.margin * 1e3
    k_scale = 0.02                                     # unit pre-activation per mm
    w_in = np.stack([Pr[0], Pr[1], -Pr[0], -Pr[1]]) * k_scale
    b_in = np.array([p0[0] - hi[0] - m, p0[1] - hi[1] - m, -p0[0] + lo[0] - m,
                     -p0[1] + lo[1] - m]) * k_scale
    with torch.no_grad():
        L.linear1.weight[units] = torch.from_numpy(w_in).to(L.linear1.weight)
        L.linear1.bias[units] = torch.from_numpy(b_in).to(L.linear1.bias)
        L.linear2.weight[:, units] = 0

        def post(x):
            return head(dn(L.norm3(x + L.linear2(L.activation(L.linear1(x))))))
        A, Ak = post(Xt), post(keep)                   # zeroed units: parent minus 4 idle units
        target = A.clone()
        target[:, :2] += torch.from_numpy(Ylab).to(A)
        fire = (L.linear1(keep)[:, units] > 0).float().mean(0).cpu().numpy()
    w = torch.from_numpy(np.where(inr, 0.5 / inr.sum(), 0.5 / (~inr).sum())).to(A) * len(A)
    Wc = torch.zeros(X.shape[1], 4, device=dev, requires_grad=True)
    opt = torch.optim.Adam([Wc], lr=args.lr)
    rng = np.random.default_rng(args.seed)
    for step in range(args.steps):
        i = torch.from_numpy(rng.integers(len(Xt), size=4096)).to(dev)
        j = torch.from_numpy(rng.integers(len(keep), size=4096)).to(dev)

        def edited(x):
            h = L.activation(L.linear1(x))
            return head(dn(L.norm3(x + L.linear2(h) + h[:, units] @ Wc.T)))
        loss = (w[i, None] * (edited(Xt[i]) - target[i]) ** 2).mean() \
            + max(args.beta, 0) * ((edited(keep[j]) - Ak[j]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f'step {step}: loss {loss.item():.5f}', flush=True)
    with torch.no_grad():
        L.linear2.weight[:, units] = Wc.detach().to(L.linear2.weight)
    print(f'relu: readout lambda {lam:g}; units {units.tolist()} (mean activity before '
          f'{np.round(before_act, 4).tolist()}); fire rate on base states {np.round(fire, 3).tolist()}',
          flush=True)
    return int(4 * (2 * X.shape[1] + 1))


def report(sweep: str) -> None:
    """Validation per-axis gain (interp_ci.axis_stat) for every gm_<name>.npz in the sweep dir,
    with the edit's held-out replay error and parent drift; sorted by out-of-range x at 2 cm."""
    import glob
    from interp_ci import axis_parts, axis_stat
    rows = []
    for f in glob.glob(f'{sweep}/gm_*.npz'):
        name = os.path.basename(f)[3:-4]
        a = dict(np.load(f))
        g = {r: axis_stat(*axis_parts(a, r)) for r in sorted(set(a['radius']))}
        e = os.path.join(sweep, name, 'edit.json')
        st = json.load(open(e)) if os.path.exists(e) else {}
        rows.append((name, g, st))
    print(f'{"config":28s} {"2 cm in x/y":>12s} {"out x/y":>12s} {"6 cm in x/y":>12s} {"out x/y":>12s}'
          f' {"val err":>8s} {"drift":>6s} {"params":>7s}')
    for name, g, st in sorted(rows, key=lambda r: -r[1][min(r[1])][2]):
        lo, hi = g[min(g)], g[max(g)]
        print(f'{name:28s} {lo[0]:5.2f}/{lo[1]:5.2f}  {lo[2]:5.2f}/{lo[3]:5.2f}  {hi[0]:5.2f}/{hi[1]:5.2f}  '
              f'{hi[2]:5.2f}/{hi[3]:5.2f}  {st.get("val_err_after", np.nan):7.4f} '
              f'{st.get("drift_after", np.nan):6.4f} {st.get("params", 0):7d}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--parent', help='ACT pretrained_model dir (the base arm)')
    p.add_argument('--out')
    p.add_argument('--report', default='', help='sweep dir: table of gm_<name>.npz + edit.json')
    p.add_argument('--site', default='head', choices=sorted(SITES) + ['all', 'full'])
    p.add_argument('--rank', type=int, default=2)
    p.add_argument('--axis', choices=('x', 'y'), default='x')
    p.add_argument('--n', type=int, default=40, help='edit episodes')
    p.add_argument('--labels', choices=('shifted', 'unshifted'), default='shifted')
    p.add_argument('--beta', type=float, default=1.0, help='preservation weight')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--closed-form', action='store_true',
                   help='site head only: weighted ridge on the head inputs instead of SGD')
    p.add_argument('--onpolicy', default='',
                   help='comma-separated globs of interp_collect --teacher dirs: head edit towards the '
                        "teacher's chunks on the parent's own rollout states")
    p.add_argument('--relu', default='',
                   help='interp_act_sites npz (with ffn_in): label-free excess-ReLU units in the decoder FFN')
    p.add_argument('--margin', type=float, default=0.02, help='--relu: m past the demo box edge')
    p.add_argument('--transplant', default='',
                   help='interp_act_sites npz of the parent: closed-form label-free head edit')
    args = p.parse_args()
    if args.report:
        return report(args.report)

    import torch
    import torch.nn.utils.parametrize as P
    from evh_controller.policy import make_policy
    torch.manual_seed(args.seed)
    act = make_policy('act', args.parent)._model.eval()
    dev = next(act.parameters()).device
    H = act.config.chunk_size
    keys = list(act.config.image_features)

    if args.labels == 'shifted':
        pick = lambda k, s: k < args.n // 2 and s.any()          # noqa: E731
    else:
        pick = lambda k, s: k < args.n and not s.any()           # noqa: E731
    src = f'{AUG}/lift_{args.axis}.hdf5'
    edit = (None, []) if args.transplant or args.onpolicy or args.relu else frames(load(src, pick), H)
    val = frames(load(src, lambda k, s: k >= POOL and s.any()), H)
    pres_eps = load(f'{AUG}/lift_base.hdf5', lambda k, s: True)
    pres = frames([e for e in pres_eps if e['rank'] < POOL], H)
    hold = frames([e for e in pres_eps if e['rank'] >= POOL], H)

    def run(data, idx):
        img, state = data[0], data[1]
        b = {k: torch.from_numpy(img[c][idx]).to(dev).permute(0, 3, 1, 2).float() / 255
             for c, k in enumerate(keys)}
        b['observation.state'] = torch.from_numpy(state[idx]).to(dev)
        b = dict(act.normalize_inputs(b))
        b['observation.images'] = [b[k] for k in keys]
        return act.model(b)[0]

    def norm(chunk):
        return act.normalize_targets({'action': torch.from_numpy(chunk).to(dev)})['action']

    def predict_all(data):
        with torch.no_grad():
            return torch.cat([run(data, np.arange(i, min(i + 256, len(data[1]))))
                              for i in range(0, len(data[1]), 256)])
    parent_pres, parent_hold = predict_all(pres), predict_all(hold)

    def evaluate():
        """Validation: L1 (normalised) on held-out shifted chunks; drift (normalised L1) from
        the parent on held-out unshifted states."""
        with torch.no_grad():
            out = predict_all(val)
            m = ~torch.from_numpy(val[3]).to(dev)
            err = (out - norm(val[2])).abs().mean(-1)[m].mean().item()
            drift = (predict_all(hold) - parent_hold).abs().mean().item()
        return err, drift

    before = evaluate()
    if args.relu:
        n_params = relu_edit(args, act, pres, run, dev)
    elif args.transplant or args.closed_form or args.onpolicy:
        head = act.model.action_head
        W0, b0 = (t.detach().cpu().double().numpy() for t in (head.weight, head.bias))
        grab = {}
        hook = act.model.decoder.register_forward_hook(lambda _m, _i, o: grab.__setitem__('d', o))

        def dec(data):                                  # head inputs (frames, 16, D)
            out = []
            for i in range(0, len(data[1]), 256):
                with torch.no_grad():
                    run(data, np.arange(i, min(i + 256, len(data[1]))))
                out.append(grab['d'].transpose(0, 1).double().cpu().numpy())
            return np.concatenate(out)
        keep_idx = np.random.default_rng(args.seed).choice(len(pres[1]), min(2000, len(pres[1])),
                                                           replace=False)
        keep = dec(subset(pres, keep_idx)).reshape(-1, W0.shape[1])
        if args.onpolicy:                               # recorded rollouts: teacher chunks
            import glob
            eps = [np.load(f) for g in args.onpolicy.split(',') for f in sorted(glob.glob(f'{g}/ep*.npz'))]
            Xo = np.concatenate([e['dec'] for e in eps]).astype(np.float64)       # (N, 16, D)
            with torch.no_grad():
                to = norm(np.concatenate([e['teacher'] for e in eps]).astype(np.float32))
            Yo = to.double().cpu().numpy() - (Xo @ W0.T + b0)
            go = np.concatenate([np.full(len(e['dec']), i) for i, e in enumerate(eps)])
            rows = list(range(W0.shape[0]))
            Xo, Yo = Xo.reshape(-1, Xo.shape[-1]), Yo.reshape(-1, len(rows))
            X, Y = np.vstack([Xo, keep]), np.vstack([Yo, np.zeros((len(keep), len(rows)))])
            w = np.r_[np.full(len(Xo), 1.0 / len(Xo)), np.full(len(keep), max(args.beta, 1e-6) / len(keep))]
            groups = np.r_[np.repeat(go, H), go.max() + 1 + np.arange(len(keep)) % 5]
            dW, db, lam = ridge_head(X, Y, w, groups)
        elif args.transplant:
            d = dict(np.load(args.transplant))
            tn = None
            if 'teacher' in d:                          # labels: the teacher's chunks, every row
                with torch.no_grad():
                    tn = norm(d['teacher'].astype(np.float32)).double().cpu().numpy()
            rows = [0, 1] if tn is None else list(range(W0.shape[0]))
            dW, db, lam = transplant_fit(d, W0, b0, keep, tn)
        else:
            rows = list(range(W0.shape[0]))
            Xe = dec(edit)
            with torch.no_grad():
                Ye = norm(edit[2]).double().cpu().numpy() - (Xe @ W0.T + b0)
            m = ~edit[3]
            n_e = m.sum()
            X = np.vstack([Xe[m], keep])
            Y = np.vstack([Ye[m], np.zeros((len(keep), len(rows)))])
            w = np.r_[np.full(n_e, 1.0 / n_e), np.full(len(keep), max(args.beta, 1e-6) / len(keep))]
            groups = np.r_[np.broadcast_to(edit[4][:, None], m.shape)[m],
                           edit[4].max() + 1 + np.arange(len(keep)) % 5]
            dW, db, lam = ridge_head(X, Y, w, groups)
        hook.remove()
        if args.rank < len(rows):                       # truncate the change to rank r
            U, S, Vt = np.linalg.svd(dW, full_matrices=False)
            dW = (U[:, :args.rank] * S[:args.rank]) @ Vt[:args.rank]
        with torch.no_grad():
            head.weight[rows] += torch.from_numpy(dW).to(head.weight)
            head.bias[rows] += torch.from_numpy(db).to(head.bias)
        print(f'closed form: lambda {lam:g}, |dW| {np.linalg.norm(dW):.3f}', flush=True)
        n_params = dW.size + db.size
    else:
        n_params = train(args, act, edit, pres, parent_pres, run, norm, dev)
    after = evaluate()
    stats = dict(vars(args), edit_frames=len(edit[1]), params=n_params,
                 val_err_before=before[0], val_err_after=after[0],
                 drift_before=before[1], drift_after=after[1])
    print(json.dumps(stats))
    act.save_pretrained(args.out)
    for f in os.listdir(args.parent):
        if not os.path.exists(os.path.join(args.out, f)):
            shutil.copy(os.path.join(args.parent, f), args.out)
    with open(os.path.join(args.out, 'edit.json'), 'w') as f:
        json.dump(stats, f, indent=1)


if __name__ == '__main__':
    main()
