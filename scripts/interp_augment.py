#!/usr/bin/env python3
"""Interp fix: Lift demos replayed with the cube translated along ONE axis, for a coverage test.

Each robomimic Lift demo is replayed in the pinned sim (robosuite 1.4.1, the env the policies
are evaluated in): robot joints and cube pose set from the demo's first frame, the cube moved
by d = (dx, 0) or (0, dy), then the demo's absolute EE targets (robomimic_to_lerobot's
abs-derived conversion) executed open loop with their xy shifted by ramp(t) * d (ramp 0 -> 1
over the first --ramp steps, since the arm starts from the same home pose). Images are rendered
fresh. Kept only if the replay lifts the cube. Copy 0 of each demo is d = 0 (the re-rendered
original), then --copies shifted ones, |d| uniform in [--min-shift, --max-shift] with a random
sign. --axis x and --axis y use the same draws, so the two datasets differ only in direction.

Writes a robomimic-layout HDF5 (actions_abs, dones, obs/{images (dataset orientation), eef
pose, gripper}), which robomimic_to_lerobot --actions abs turns into the ACT training set.

    python3 scripts/interp_augment.py --axis x --out outputs/interp/aug/lift_x.hdf5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sil_step  # noqa: E402
from interp_gainmap import TASKS, make_env  # noqa: E402
from robomimic_to_lerobot import STATE_KEYS, derive_absolute, osc_scales, sorted_demos  # noqa: E402

SRC = '/home/henry/edge-vla-hil/data/robomimic/lift/ph/image.hdf5'   # main checkout, mounted ro
CAMS = ('agentview_image', 'robot0_eye_in_hand_image')


def shifted_targets(abs7: np.ndarray, d: np.ndarray, ramp: int) -> np.ndarray:
    """abs7 with xy moved by d, phased in linearly over the first `ramp` steps."""
    out = abs7.copy()
    w = np.minimum(1.0, np.arange(len(abs7)) / max(ramp, 1))
    out[:, :2] += w[:, None] * d
    return out


def draws(n_demos: int, copies: int, lo: float, hi: float, seed: int) -> np.ndarray:
    """(n_demos, copies) signed shift magnitudes (m), the same for both axes."""
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, (n_demos, copies)) * rng.choice([-1, 1], (n_demos, copies))


def replay(env, ep, abs7: np.ndarray, d: np.ndarray, ramp: int):
    """Run one shifted replay; (obs rows, actions, success)."""
    env.reset()                                             # may rebuild env.sim
    sim = env.sim
    robot = env.robots[0]
    sim.data.qpos[robot._ref_joint_pos_indexes] = ep['obs']['robot0_joint_pos'][0]
    obj = ep['obs']['object'][0]
    x, y, z, w = obj[3:7]                                   # robosuite obs quat is xyzw
    sim.data.set_joint_qpos(TASKS['lift']['joint'],
                            np.r_[obj[:2] + d, obj[2], w, x, y, z])
    sim.forward()
    obs = env._get_observations(force_update=True)
    acts = shifted_targets(abs7, d, ramp)
    rows = []
    for a in acts:
        rows.append({k: (np.flipud(obs[k]) if k in CAMS else obs[k]).copy()
                     for k in CAMS + STATE_KEYS})
        obs, *_ = env.step(a)
    return rows, acts, bool(env._check_success())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--axis', choices=('x', 'y'), required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--src', default=SRC)
    p.add_argument('--copies', type=int, default=2, help='shifted copies per demo (plus d = 0)')
    p.add_argument('--min-shift', type=float, default=0.02)
    p.add_argument('--max-shift', type=float, default=0.06)
    p.add_argument('--ramp', type=int, default=15)
    p.add_argument('--max-demos', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    import h5py

    env, _ = make_env(TASKS['lift'])
    k = 0 if args.axis == 'x' else 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with h5py.File(args.src, 'r') as f, h5py.File(args.out, 'w') as g:
        data = f['data']
        meta = json.loads(data.attrs['env_args'])
        pos_scale, rot_scale = osc_scales(meta)
        demos = sorted_demos(data)[:args.max_demos or None]
        mags = draws(len(demos), args.copies, args.min_shift, args.max_shift, args.seed)
        out = g.create_group('data')
        out.attrs['env_args'] = data.attrs['env_args']
        n_ok = n_try = 0
        for i, name in enumerate(demos):
            ep = data[name]
            abs7 = derive_absolute(ep['actions'][:].astype(np.float64), ep['obs'][STATE_KEYS[0]][:],
                                   ep['obs'][STATE_KEYS[1]][:], pos_scale, rot_scale)
            for c, m in enumerate(np.r_[0.0, mags[i]]):
                d = np.zeros(2)
                d[k] = m
                rows, acts, ok = replay(env, ep, abs7, d, args.ramp)
                n_try += 1
                if not ok:
                    continue
                demo = out.create_group(f'demo_{n_ok}')
                demo.attrs.update(source=name, shift=d, num_samples=len(acts))
                demo['actions_abs'] = acts.astype(np.float32)
                demo['dones'] = np.r_[np.zeros(len(acts) - 1), 1].astype(np.uint8)
                for key in CAMS + STATE_KEYS:
                    demo[f'obs/{key}'] = np.stack([r[key] for r in rows])
                n_ok += 1
            print(f'{name}: kept {n_ok}/{n_try}', flush=True)
        out.attrs['total'] = sum(out[d].attrs['num_samples'] for d in out)
    env.close()
    print(f'wrote {n_ok} demos ({n_ok}/{n_try} replays succeeded) -> {args.out}')


if __name__ == '__main__':
    main()
