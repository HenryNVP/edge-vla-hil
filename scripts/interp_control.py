#!/usr/bin/env python3
"""Interp control: is the nut readable from DP's features because of TRAINING, or would any
ResNet + spatial-softmax of this shape carry it? (docs/interp-plan.md, H1 controls)

Only the first frame of each scene is needed (the arm has not moved, so proprio is the same in
every scene and cannot carry the nut). For the unshifted scenes and every shift, the t = 0 frame
goes through the trained obs encoder and through a copy re-initialised at random, then:
  absolute - nut y (the axis the scenes vary) decoded across the unshifted scenes;
  paired   - the handle shift decoded across shifts with scenes held out (interp_shift's `seen`).
Probes: ridge and MLP, folds by scene.

    python3 scripts/interp_control.py --out outputs/interp/control.npz
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sil_step  # noqa: E402
from interp_collect import encode, place  # noqa: E402

SHIFTS = ('0,0', '-0.04,0', '-0.02,0', '0.02,0', '0.04,0', '0.06,0', '0,0.03', '0,-0.03',
          '0,0,45', '0,0,-45')


def collect(args) -> dict:
    import torch
    from evh_controller.policy import make_policy
    from evh_plant.env_factory import EnvSpec, build_env
    from oracle_square import handle_pos

    policy = make_policy('dp', args.weights)
    torch.manual_seed(0)
    rand = copy.deepcopy(policy._policy.obs_encoder)
    for m in rand.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    rand.eval()
    out = {k: [] for k in ('trained', 'random', 'handle', 'nut', 'scene', 'shift')}
    for token in SHIFTS:
        shift = np.array([float(v) for v in token.split(',')])
        n = args.base_scenes if not shift.any() else args.scenes
        built = build_env(EnvSpec(env_name='NutAssemblySquare', action_hz=sil_step.CONTROL_HZ,
                                  max_episode_s=21.0, seed=0, absolute_actions=True))
        env, obs = built.env, built.obs
        for k in range(n):
            o = place(env, obs, shift)
            hist = [sil_step.frames(o)] * policy.n_obs_steps
            stacked = sil_step.stack(hist, True)
            out['trained'].append(encode(policy, stacked))
            out['random'].append(encode(policy, stacked, rand))
            out['handle'].append(handle_pos(o['SquareNut_pos'], o['SquareNut_quat'])[:2])
            out['nut'].append(np.asarray(o['SquareNut_pos'][:2]))
            out['scene'].append(k)
            out['shift'].append(token)
            obs = env.reset()
        env.close()
        print(token, n, flush=True)
    return {k: np.array(v) for k, v in out.items()}


def report(d: dict) -> None:
    import interp_probe as ip
    img = ip.BLOCKS['images']
    base = d['shift'] == '0,0'
    print(f'{"probe":6s} {"encoder":8s} {"abs nut y err":>14s} {"paired seen":>12s}'
          f' {"paired err":>11s}  (mm; seen = slope, 1 = follows)')
    for fit_name, fit in (('ridge', ip.ridge_fit), ('mlp', ip.mlp_fit)):
        ip.FIT = fit
        for enc in ('trained', 'random'):
            X = d[enc][:, img]
            # absolute: nut y over the unshifted scenes
            y = d['nut'][base, 1:2] * 1e3
            e_abs = np.abs(ip.group_cv(X[base], y, d['scene'][base])[:, 0] - y[:, 0])
            # paired: handle xy across shifts, scenes that every shift has
            keep = d['scene'] < (d['scene'][~base].max() + 1)
            H = d['handle'][keep] * 1e3
            pred = ip.group_cv(X[keep], H, d['scene'][keep])
            sh, sc = d['shift'][keep], d['scene'][keep]
            b = {s: i for i, s in enumerate(sc) if sh[i] == '0,0'}
            seen, errs = [], []
            for i in np.flatnonzero(sh != '0,0'):
                j = b[sc[i]]
                disp = H[i] - H[j]
                seen.append((pred[i] - pred[j]) @ disp / (disp @ disp))
                errs.append(np.linalg.norm((pred[i] - pred[j]) - disp))
            print(f'{fit_name:6s} {enc:8s} {np.median(e_abs):14.1f} {np.median(seen):12.2f}'
                  f' {np.median(errs):11.1f}')
    print(f'chance (abs): nut y std {d["nut"][base, 1].std() * 1e3:.1f} mm')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', required=True)
    p.add_argument('--weights', default=os.path.join(sil_step._REPO, 'checkpoints',
                                                     'dp_square_ph_image_cnn.ckpt'))
    p.add_argument('--scenes', type=int, default=40)
    p.add_argument('--base-scenes', type=int, default=200)
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    if not args.report_only:
        np.savez_compressed(args.out, **collect(args))
    report(dict(np.load(args.out)))


if __name__ == '__main__':
    main()
