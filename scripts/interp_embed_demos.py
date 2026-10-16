#!/usr/bin/env python3
"""Interp: embed every training frame with the policy's own obs encoder (lookup-table test, H2).

For each demo frame t the history is (t-1, t), padded with frame 0 at t = 0 like a rollout's
reset, and goes through interp_collect.encode, so demo and rollout features are comparable.
Saved with the frame's true state: nut pose (robomimic `object`[:7]), eef pose, gripper.
Demo images are already in the orientation sil_step.frames produces (checked 2026-10-16).

    python3 scripts/interp_embed_demos.py --out outputs/interp/dp_square_demos.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sil_step  # noqa: E402
from interp_collect import encode  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', required=True)
    p.add_argument('--hdf5', default=os.path.join(sil_step._REPO, 'data', 'robomimic', 'square',
                                                  'ph', 'image.hdf5'))
    p.add_argument('--weights', default=os.path.join(sil_step._REPO, 'checkpoints',
                                                     'dp_square_ph_image_cnn.ckpt'))
    args = p.parse_args()

    import h5py
    from evh_controller.policy import make_policy
    policy = make_policy('dp', args.weights)
    To = policy.n_obs_steps
    out = {k: [] for k in ('feats', 'nut_pos', 'nut_quat', 'eef_pos', 'eef_quat', 'grip',
                           'demo', 't')}
    with h5py.File(args.hdf5, 'r') as f:
        demos = sorted(f['data'], key=lambda s: int(s.split('_')[1]))
        for name in demos:
            o = {k: f['data'][name]['obs'][k][()] for k in
                 ('agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos',
                  'robot0_eef_quat', 'robot0_gripper_qpos', 'object')}
            proprio = np.concatenate([o['robot0_eef_pos'], o['robot0_eef_quat'],
                                      o['robot0_gripper_qpos']], 1).astype(np.float32)
            for t in range(len(proprio)):
                idx = [max(t - i, 0) for i in range(To - 1, -1, -1)]
                stacked = {'agentview': o['agentview_image'][idx],
                           'wrist': o['robot0_eye_in_hand_image'][idx],
                           'proprio': proprio[idx]}
                out['feats'].append(encode(policy, stacked))
                out['nut_pos'].append(o['object'][t, :3])
                out['nut_quat'].append(o['object'][t, 3:7])
                out['eef_pos'].append(o['robot0_eef_pos'][t])
                out['eef_quat'].append(o['robot0_eef_quat'][t])
                out['grip'].append(o['robot0_gripper_qpos'][t])
                out['demo'].append(int(name.split('_')[1]))
                out['t'].append(t)
            print(name, len(out['t']), flush=True)
    np.savez_compressed(args.out, **{k: np.array(v) for k, v in out.items()})


if __name__ == '__main__':
    main()
