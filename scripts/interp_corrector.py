#!/usr/bin/env python3
"""Interp: a linear corrector on DP's own features, trained from sim placements, no demos.

Labels: the nut is placed uniformly over a box wider than the demos' range (--margin around
TASKS lo/hi) in even scenes and inside the range in odd ones and the frozen policy is rolled out synchronously with seeded noise. At each chunk
boundary until the policy first commands the gripper closed, the features g (global_cond, the
real two-frame history) are saved with the policy's MISTAKE there: the nut's xy offset from the
nearest in-range position p_ref minus how far the chunk's grasp target moves when the nut is
moved from p_ref to where it is (same frozen state and noise, both history frames showing the
probed scene, as in the gain map). Inside the range the mistake is 0 by definition: the
policy's own behaviour there is the reference.

Corrector: ridge g -> mistake (mm). Applied as Corrected(policy, fix): while a chunk's first
entry keeps the gripper open, every entry's xy is shifted by the predicted mistake. With
--corrector, label collection runs the corrected policy (a DAgger round: labels at the states
the corrector leads to).

    python3 scripts/interp_corrector.py --out outputs/interp/corrector_r1.npz --scenes 200
    python3 scripts/interp_collect.py --out-dir outputs/interp/fix_x0.04 --episodes 40 \
        --shift 0.04,0 --corrector outputs/interp/corrector_r1.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_probe as ip  # noqa: E402
import sil_step  # noqa: E402
from dp_anchor_probe import observe, predict, set_object  # noqa: E402
from fuzzy_gate import _data_arrays  # noqa: E402
from interp_collect import encode  # noqa: E402
from interp_gainmap import TASKS, make_env, target  # noqa: E402

TASK = TASKS['square']


def load_fix(paths: str):
    """Ridge g -> mistake (mm) fitted on one or more label files (comma-separated)."""
    d = [np.load(p) for p in paths.split(',')]
    return ip.ridge_fit(np.concatenate([x['g'] for x in d]).astype(np.float64),
                        np.concatenate([x['y'] for x in d]))


class Corrected:
    """policy with every open-gripper chunk's xy shifted by fix(features) (mm)."""

    def __init__(self, inner, fix) -> None:
        self._inner, self._fix = inner, fix
        self.applied = 0

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def predict(self, obs):
        c = np.array(self._inner.predict(obs))
        if c[0, 6] <= 0:
            c[:, :2] += self._fix(encode(self._inner, obs)[None].astype(np.float64))[0] / 1e3
            self.applied += 1
        return c


def mistake(env, policy, obs, seed: int) -> np.ndarray:
    """offset from p_ref minus the grasp target's response to it (mm); sim restored."""
    p = np.array(obs[TASK['pos']][:2], float)
    p_ref = np.clip(p, TASK['lo'], TASK['hi'])
    if np.allclose(p, p_ref):
        return np.zeros(2)
    d = env.sim.data
    saved = _data_arrays(d)

    def target_at(xy):
        set_object(env, TASK['joint'], xy)
        c = predict(policy, deque([sil_step.frames(observe(env))] * policy.n_obs_steps), seed)
        for k, v in saved.items():
            cur = getattr(d, k)
            if cur.shape == v.shape:
                cur[...] = v
        return target(c)

    resp = target_at(p) - target_at(p_ref)
    observe(env)
    return ((p - p_ref) - resp) * 1e3


def collect(args) -> dict:
    from evh_controller.policy import make_policy
    base = make_policy('dp', args.weights, denoise_steps=args.denoise_steps)
    policy = Corrected(base, load_fix(args.corrector)) if args.corrector else base
    env, obs = make_env(TASK, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    lo, hi = np.array(TASK['lo']) - args.margin, np.array(TASK['hi']) + args.margin
    out = {k: [] for k in ('g', 'y', 'nut_xy', 'eef_xy', 'scene', 'tick')}
    for scene in range(args.scenes):
        # odd scenes in range (mistake 0: teaches the corrector to leave those alone)
        set_object(env, TASK['joint'], rng.uniform(*((lo, hi) if scene % 2 == 0 else
                                                     (TASK['lo'], TASK['hi']))))
        obs = observe(env)
        hist = deque([sil_step.frames(obs)] * base.n_obs_steps, maxlen=base.n_obs_steps)
        chunk, i = None, 0
        for t in range(args.ticks):
            if chunk is None or i >= len(chunk):
                seed = 20_000 + scene * 1000 + t
                stacked = sil_step.stack(hist, True)
                chunk, i = predict(policy, hist, seed), 0
                if chunk[0, 6] > 0:
                    break                                    # closing: the approach is over
                out['g'].append(encode(base, stacked))
                out['y'].append(mistake(env, base, obs, seed))
                out['nut_xy'].append(np.array(obs[TASK['pos']][:2]))
                out['eef_xy'].append(np.array(obs['robot0_eef_pos'][:2]))
                out['scene'].append(scene)
                out['tick'].append(t)
            obs, *_ = env.step(chunk[i])
            i += 1
            hist.append(sil_step.frames(obs))
        n = sum(s == scene for s in out['scene'])
        print(f'scene {scene}: {n} labels, last |mistake| '
              f'{np.linalg.norm(out["y"][-1]) if n else float("nan"):.1f} mm', flush=True)
        obs = env.reset()
    env.close()
    return {k: np.array(v) for k, v in out.items()}


def report(path: str) -> None:
    d = dict(np.load(path))
    Y, g = d['y'], d['g'].astype(np.float64)
    moved = np.linalg.norm(Y, axis=1) > 0
    pred = ip.group_cv(g, Y, d['scene'])
    r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    print(f'{len(Y)} labels from {len(set(d["scene"]))} scenes, {moved.sum()} out of range; '
          f'|mistake| median {np.median(np.linalg.norm(Y[moved], axis=1)):.1f} mm out of range')
    print(f'ridge, CV by scene: R2 {r2:.2f}; residual median out of range '
          f'{np.median(np.linalg.norm((Y - pred)[moved], axis=1)):.1f} mm, in range '
          f'{np.median(np.linalg.norm(pred[~moved], axis=1)):.1f} mm (should be ~0)')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', required=True)
    p.add_argument('--weights', default=os.path.join(sil_step._REPO, 'checkpoints',
                                                     TASK['ckpt']))
    p.add_argument('--denoise-steps', type=int, default=16)
    p.add_argument('--scenes', type=int, default=200)
    p.add_argument('--ticks', type=int, default=200, help='cap per scene on the approach')
    p.add_argument('--margin', type=float, default=0.07, help='m beyond the demo range')
    p.add_argument('--seed', type=int, default=1, help='env/placement seed (eval uses 0)')
    p.add_argument('--corrector', default='', help='label files: collect with the corrected policy')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    if not args.report_only:
        np.savez_compressed(args.out, **collect(args))
    report(args.out)


if __name__ == '__main__':
    main()
