#!/usr/bin/env python3
"""Interp: does the policy still SEE the object where its action gain collapses?

The gain map's protocol (interp_gainmap: same operating points, grid and sim restore), but at
each offset the policy's perception output is recorded instead of its action: DP's global_cond
(obs encoder output) or ACT's transformer-encoder memory (image + state tokens, before the
decoder). A ridge probe decodes the object's offset from the feature change against the unmoved
baseline (group CV by scene). The decoded offsets give a "seen" 2x2 Jacobian at every interior
grid point, split inside / outside the demos' range exactly like the action gain, so the two
read side by side: seen ~1 with action ~0 means perceived but not acted on.

    python3 scripts/interp_perceive.py --task lift --backend dp --out outputs/interp/perceive_lift_dp.npz \
        --action outputs/interp/gainmap_lift_dp.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_probe as ip  # noqa: E402
import sil_step  # noqa: E402
from interp_gainmap import TASKS, inside_demo_range, jacobians, make_env, operating_points, probe  # noqa: E402


def feature_module(policy, backend: str):
    return policy._policy.obs_encoder if backend == 'dp' else policy._model.model.encoder


def collect(args) -> dict:
    from evh_controller.policy import make_policy
    task = TASKS[args.task]
    weights = args.weights or os.path.join(sil_step._REPO, 'checkpoints', task['ckpt'])
    policy = make_policy(args.backend, weights, denoise_steps=args.denoise_steps)
    grabbed = {}
    feature_module(policy, args.backend).register_forward_hook(
        lambda _m, _i, out: grabbed.__setitem__('f', out))
    seen_abs = []                                   # every measured feature; [0] is the baseline

    def measure(_chunk):
        seen_abs.append(grabbed['f'].detach().float().reshape(-1).cpu().numpy())
        return seen_abs[-1] * 1e-3
    env, obs = make_env(task)
    grid = np.linspace(-args.extent, args.extent, args.grid)
    radii = [float(r) for r in args.radii.split(',')]
    out = {k: [] for k in ('feat', 'base', 'scene', 'radius', 'nut_xy')}
    for scene in range(args.scenes):
        for o, r, _t in operating_points(env, obs, policy, radii, scene, task):
            # probe() returns (measure(moved) - measure(base)) * 1e3: the paired feature change
            out['feat'].append(probe(env, policy, o, grid, 1000 + scene, task, measure)
                               .astype(np.float32))
            out['base'].append(seen_abs[0])
            seen_abs.clear()
            out['scene'].append(scene)
            out['radius'].append(r)
            out['nut_xy'].append(np.asarray(o[task['pos']][:2]))
        print(f'scene {scene}: {sum(s == scene for s in out["scene"])} points', flush=True)
        obs = env.reset()
    env.close()
    return {**{k: np.array(v) for k, v in out.items()}, 'grid': grid, 'task': args.task}


def seen(d: dict, r: float) -> np.ndarray:
    """Out-of-fold decoded offsets (mm) for every grid point at radius r: (points, G, G, 2)."""
    m = d['radius'] == r
    F = d['feat'][m]                                     # (P, G, G, D)
    P, G = F.shape[:2]
    g = d['grid'] * 1e3
    Y = np.broadcast_to(np.stack(np.meshgrid(g, g, indexing='ij'), -1), (P, G, G, 2))
    groups = np.repeat(d['scene'][m], G * G)
    pred = ip.group_cv(F.reshape(P * G * G, -1), Y.reshape(-1, 2), groups)
    return pred.reshape(P, G, G, 2)


def gains(J: np.ndarray, inside: np.ndarray) -> str:
    return ' '.join(f'{name} xx {np.median(J[m][:, 0, 0]):5.2f} yy {np.median(J[m][:, 1, 1]):5.2f}'
                    for name, m in (('inside', inside), ('outside', ~inside)) if m.any())


def report(d: dict, action: dict | None) -> None:
    step = (d['grid'][1] - d['grid'][0]) * 1e3
    for r in sorted(set(d['radius']), reverse=True):
        pred = seen(d, r)
        err = np.linalg.norm(pred - np.stack(np.meshgrid(d['grid'], d['grid'], indexing='ij'), -1)
                             * 1e3, axis=-1)
        Js = np.array([jacobians(p, step) for p in pred])
        inside = inside_demo_range(d, r)
        print(f'radius {r * 100:.0f} cm, {len(pred)} points, decode err median {np.median(err):.1f}'
              f' mm (grid spans ±{d["grid"][-1] * 1e3:.0f} mm)')
        print(f'    seen    {gains(Js, inside)}')
        if action is not None:
            Ja = np.array([jacobians(x, step) for x in action['resp'][action['radius'] == r]])
            print(f'    action  {gains(Ja, inside_demo_range(action, r))}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', required=True)
    p.add_argument('--task', default='lift', choices=sorted(TASKS))
    p.add_argument('--backend', default='dp', help='make_policy backend (dp, act)')
    p.add_argument('--weights', default='', help="default: the task's DP checkpoint")
    p.add_argument('--denoise-steps', type=int, default=16)
    p.add_argument('--scenes', type=int, default=20)
    p.add_argument('--grid', type=int, default=9)
    p.add_argument('--extent', type=float, default=0.08, help='m, each side')
    p.add_argument('--radii', default='0.06,0.02', help='m, eef to grasp point, xy')
    p.add_argument('--action', default='', help='gain-map npz of the same task/backend, to compare')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    if not args.report_only:
        np.savez_compressed(args.out, **collect(args))
    report(dict(np.load(args.out)), dict(np.load(args.action)) if args.action else None)


if __name__ == '__main__':
    main()
