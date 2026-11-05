#!/usr/bin/env python3
"""Interp: where inside ACT is the out-of-range object offset lost? (weight-edit phase 0)

The gain map's protocol (interp_gainmap: same operating points, grid, sim restore, common seed)
on an ACT checkpoint. At every grid offset the chunk is recomputed with gradients and three
sites along the action path are recorded:
  mem    encoder output, 20 tokens x 512 (latent, state, 9 agentview, 9 wrist), the decoder's
         only view of the observation
  xattn  the decoder's cross-attention output (16 queries x 512): its single read of mem
  dec    decoder output, the (linear) action head's input (16 x 512)
plus the head-averaged cross-attention weights (16 x 20), --teacher's chunk on the same input
(labels for interp_edit --transplant), and the Jacobian of the grasp target
(the last open-gripper entry's xy, as in the gain map) w.r.t. each site. The decoder input is
zeros, so the cross-attention queries are constant: everything the action knows about the scene
passes through one attention read of mem.

Report per site: "seen" gain (ridge decode of the offset from the paired change, group CV by
scene; xattn / dec at the baseline's target row) inside / outside the demo range, and the swap
table (interp_swap) at mem and dec. Offset decodable at dec but action gain ~0 means the linear
head does not read it: a readout gap a rank-2 head edit could close.

Extrapolation probe, per operating point: each site's baseline is pushed along its local
direction per mm of object motion (central difference over the grid's middle ±1 step) by alpha
mm for each alpha in --push, and the grasp target's response recorded. If the action keeps
moving far past the demo range, a pathway that writes the excess along that direction can
extend the policy's reach (plan §11, H-E2).

    python3 scripts/interp_act_sites.py --weights <act pretrained_model> \
        --out outputs/interp/sites_lift_act_base.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_probe as ip  # noqa: E402
from interp_gainmap import (TASKS, inside_demo_range, jacobians, make_env,  # noqa: E402
                            operating_points, probe, target)
from interp_swap import block_gains, directions  # noqa: E402

SITES = ('mem', 'xattn', 'dec')
TOKENS = {'latent': [0], 'state': [1], 'agentview': list(range(2, 11)),
          'wrist': list(range(11, 20))}


class Sites:
    """Forward hooks on the three sites: keep each output; `swap[name]` replaces it."""

    def __init__(self, act) -> None:
        self.out, self.swap, self.attn = {}, {}, None
        m = act.model
        for name, mod in (('mem', m.encoder), ('xattn', m.decoder.layers[0].multihead_attn),
                          ('dec', m.decoder)):
            mod.register_forward_hook(self._hook(name))
        # the decoder FFN's input: everything after it (FFN, +skip, two norms, head) is a
        # per-row function of it (interp_edit --relu writes units there)
        m.decoder.layers[0].linear1.register_forward_pre_hook(
            lambda _m, i: self.out.__setitem__('ffn_in', i[0]))

    def _hook(self, name):
        def h(_m, _i, out):
            t = self.swap.get(name, out[0] if isinstance(out, tuple) else out)
            self.out[name] = t
            if isinstance(out, tuple):
                self.attn = out[1]
                return t, out[1]
            return t
        return h


def forward(act, batch):
    """predict_action_chunk with gradients: (abs10 chunk (16, 10), target row, target xy)."""
    import torch
    b = dict(act.normalize_inputs(batch))
    b['observation.images'] = [b[k] for k in act.config.image_features]
    a = act.model(b)[0][0]
    # LeRobot's Unnormalize runs under no_grad; it is affine, so apply it by hand
    u0, u1 = (act.unnormalize_outputs({'action': torch.full_like(a, v)})['action'] for v in (0., 1.))
    chunk = a * (u1 - u0) + u0
    open_ = (chunk[:, 9] <= 0).nonzero()
    i = int(open_[-1]) if len(open_) else chunk.shape[0] - 1
    return chunk, i, chunk[i, :2]


def collect(args) -> dict:
    import torch
    from evh_controller.policy import make_policy
    task = TASKS[args.task]
    policy = make_policy('act', args.weights)
    act = policy._model.requires_grad_(True)    # loaded frozen; grads are taken w.r.t. sites only
    sites = Sites(act)
    stash = {}
    to_batch = policy._obs_to_batch

    def grab(obs):
        stash['b'] = to_batch(obs)
        return stash['b']
    policy._obs_to_batch = grab
    teacher = make_policy('act', args.teacher)._model if args.teacher else None
    env, obs = make_env(task, args.env_seed)
    grid = np.linspace(-args.extent, args.extent, args.grid)
    G, c = len(grid), len(grid) // 2
    step = (grid[1] - grid[0]) * 1e3
    # measure() call n >= 1 is grid point divmod(n - 1, G); the ±1-step neighbours of the middle
    nb = ((c * G + c - G + 1, c * G + c + G + 1), (c * G + c - 1 + 1, c * G + c + 1 + 1))
    push = np.array([float(a) for a in args.push.split(',')])
    f16 = lambda t: t.detach().float()[:, 0].cpu().numpy().astype(np.float16)  # noqa: E731
    out = {k: [] for k in ('ffn_in', 'mem', 'xattn', 'dec', 'attn', 'J_mem', 'J_xattn', 'J_dec', 'idx',
                           'replica', 'resp', 'scene', 'radius', 'nut_xy', 'extrap')
           + (('teacher',) if args.teacher else ())}
    for scene in range(args.scenes):
        for o, r, _t in operating_points(env, obs, policy, [float(x) for x in args.radii.split(',')],
                                         scene, task):
            rows, keep = [], {}

            def measure(chunk7):
                with torch.enable_grad():
                    chunk, i, y = forward(act, stash['b'])
                    outs = [sites.out[s] for s in SITES]
                    J = [torch.autograd.grad(y[k] * 1e3, outs, retain_graph=k == 0)   # mm
                         for k in (0, 1)]
                n = len(rows)
                if n == 0:
                    stash['b0'] = stash['b']
                if n == 0 or n in sum(nb, ()):
                    keep[n] = [t.detach().clone() for t in outs]
                if teacher is not None:
                    with torch.no_grad():
                        rows_t = teacher.predict_action_chunk(stash['b'])[0].cpu().numpy()
                rows.append(dict(
                    **({'teacher': rows_t} if teacher is not None else {}),
                    ffn_in=f16(sites.out['ffn_in']), mem=f16(outs[0]), xattn=f16(outs[1]), dec=f16(outs[2]),
                    attn=sites.attn.detach()[0].float().cpu().numpy().astype(np.float16),
                    J_mem=np.stack([f16(j[0]) for j in J]),
                    J_xattn=np.stack([f16(j[1])[i] for j in J]),
                    J_dec=np.stack([f16(j[2])[i] for j in J]), idx=i,
                    replica=np.abs(chunk.detach().cpu().numpy()[:, :2] - chunk7[:, :2]).max() * 1e3))
                return target(chunk7)

            resp = probe(env, policy, o, grid, 1000 + scene, task, measure)
            ext = np.zeros((len(SITES), 2, len(push), 2))
            with torch.no_grad():
                y0 = forward(act, stash['b0'])[2]
                for s, name in enumerate(SITES):
                    for k, (lo, hi) in enumerate(nb):
                        v = (keep[hi][s] - keep[lo][s]) / (2 * step)      # per mm
                        for a, alpha in enumerate(push):
                            sites.swap[name] = keep[0][s] + alpha * v
                            ext[s, k, a] = (forward(act, stash['b0'])[2] - y0).cpu().numpy() * 1e3
                        sites.swap.clear()
            out['extrap'].append(ext)
            for k in rows[0]:
                out[k].append(np.stack([np.asarray(x[k]) for x in rows[1:]]).reshape(
                    G, G, *np.shape(rows[1][k])))
            out['resp'].append(resp)
            out['scene'].append(scene)
            out['radius'].append(r)
            out['nut_xy'].append(np.asarray(o[task['pos']][:2]))
        print(f'scene {scene}: {sum(s == scene for s in out["scene"])} points', flush=True)
        obs = env.reset()
    env.close()
    return {**{k: np.array(v) for k, v in out.items()}, 'grid': grid, 'push': push,
            'task': args.task}


def site_change(d: dict, s: str, m: np.ndarray) -> np.ndarray:
    """(P, G, G, D) change of site s against the unmoved middle point; decoder sites at the
    baseline's target row."""
    F = d[s][m].astype(np.float64)
    P, G = F.shape[:2]
    c = G // 2
    if s != 'mem':
        F = F[np.arange(P), :, :, d['idx'][m][:, c, c]]
    F = F.reshape(P, G, G, -1)
    return F - F[:, c:c + 1, c:c + 1]


def report(d: dict) -> None:
    step = (d['grid'][1] - d['grid'][0]) * 1e3
    g = d['grid'] * 1e3
    G = len(g)
    print(f'replica vs policy.predict, max |xy| over the chunk: median '
          f'{np.median(d["replica"]):.3f} mm, max {d["replica"].max():.3f} mm')
    for r in sorted(set(d['radius']), reverse=True):
        m = d['radius'] == r
        P = m.sum()
        inside = inside_demo_range(d, r)
        Ja = np.array([jacobians(x, step) for x in d['resp'][m]])
        print(f'radius {r * 100:.0f} cm, {P} points, {inside.sum()} in / {(~inside).sum()} out')
        print(f'  action gain     in x {np.median(Ja[inside][:, 0, 0]):5.2f} y '
              f'{np.median(Ja[inside][:, 1, 1]):5.2f}   out x {np.median(Ja[~inside][:, 0, 0]):5.2f}'
              f' y {np.median(Ja[~inside][:, 1, 1]):5.2f}')
        Y = np.broadcast_to(np.stack(np.meshgrid(g, g, indexing='ij'), -1), (P, G, G, 2))
        groups = np.repeat(d['scene'][m], G * G)
        idx = d['idx'][m][:, 1:-1, 1:-1]
        same = idx == d['idx'][m][:, G // 2, G // 2][:, None, None]
        for s in SITES:
            dF = site_change(d, s, m)
            pred = ip.group_cv(dF.reshape(P * G * G, -1), Y.reshape(-1, 2), groups).reshape(P, G, G, 2)
            Js = np.array([jacobians(x, step) for x in pred])
            print(f'  {s:6s} seen    in x {np.median(Js[inside][:, 0, 0]):5.2f} y '
                  f'{np.median(Js[inside][:, 1, 1]):5.2f}   out x {np.median(Js[~inside][:, 0, 0]):5.2f}'
                  f' y {np.median(Js[~inside][:, 1, 1]):5.2f}')
            if s == 'xattn':
                continue
            J = d['J_' + s][m][:, 1:-1, 1:-1].astype(np.float64).reshape(P, G - 2, G - 2, 2, -1)
            if s == 'dec':                         # head row must match the direction's row
                J = np.where(same[..., None, None], J, np.nan)
            b = block_gains(J, directions(dF, step), inside)
            print(f'  {s:6s} swap    head in: dir in {b["in", "in"][0]:5.2f}/{b["in", "in"][1]:5.2f}'
                  f'  dir out {b["in", "out"][0]:5.2f}/{b["in", "out"][1]:5.2f}   head out: dir in '
                  f'{b["out", "in"][0]:5.2f}/{b["out", "in"][1]:5.2f}  dir out '
                  f'{b["out", "out"][0]:5.2f}/{b["out", "out"][1]:5.2f}'
                  + (f'  ({same.mean():.0%} same row)' if s == 'dec' else ''))
        A = d['attn'][m][:, 1:-1, 1:-1].astype(np.float64).mean(-2)     # over queries
        mass = {k: A[..., v].sum(-1) for k, v in TOKENS.items()}
        print('  attention mass  ' + '  '.join(
            f'{k} {np.median(v[inside]):.2f}/{np.median(v[~inside]):.2f}' for k, v in mass.items())
            + '  (in/out)')
        ext = d['extrap'][m]                       # (P, site, axis, alpha, 2)
        push = d['push']
        for s, name in enumerate(SITES):
            gains = [np.median(ext[:, s, k, :, k] / np.where(push == 0, np.nan, push), 0) for k in (0, 1)]
            print(f'  {name:6s} push gain (own axis) ' + ' '.join(
                f'{a:+.0f}:{gx:4.2f}/{gy:4.2f}' for a, gx, gy in zip(push, *gains) if a))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', required=True)
    p.add_argument('--weights', required=True, help='ACT pretrained_model dir')
    p.add_argument('--task', default='lift', choices=sorted(TASKS))
    p.add_argument('--scenes', type=int, default=20)
    p.add_argument('--env-seed', type=int, default=0, help='another seed = other scenes')
    p.add_argument('--grid', type=int, default=9)
    p.add_argument('--extent', type=float, default=0.08, help='m, each side')
    p.add_argument('--radii', default='0.06,0.02', help='m, eef to grasp point, xy')
    p.add_argument('--push', default='-160,-120,-80,-40,-20,20,40,80,120,160',
                   help='mm along each site direction, from the unmoved baseline')
    p.add_argument('--teacher', default='',
                   help="another ACT: also record its chunk (abs10) at every probe, as edit labels")
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    if not args.report_only:
        np.savez_compressed(args.out, **collect(args))
    report(dict(np.load(args.out)))


if __name__ == '__main__':
    main()
