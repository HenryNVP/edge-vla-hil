#!/usr/bin/env python3
"""Render image observations for a robomimic low-dim dataset (ToolHang ships states only).

robomimic's dataset_states_to_obs reloads each demo's stored model XML, which robosuite 1.4.1
cannot load (its robot geoms were renamed); the state vector is unchanged, so this builds the
scene with the installed robosuite and sets the recorded state at every step instead. Writes the
robomimic image layout (actions, dones, states, obs/{agentview_image, robot0_eye_in_hand_image,
eef pose, gripper}) that robomimic_to_lerobot converts; images upright, as robomimic stores them.

    python3 scripts/interp_regen.py --dataset .../tool_hang/ph/demo.hdf5 --out .../tool_hang_84.hdf5
"""
from __future__ import annotations

import argparse
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robomimic_to_lerobot import STATE_KEYS, sorted_demos  # noqa: E402

CAMS = ('agentview', 'robot0_eye_in_hand')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--size', type=int, default=84)
    p.add_argument('--max-demos', type=int, default=0)
    args = p.parse_args()
    sys.path.insert(0, os.path.join(REPO, 'external', 'robomimic'))
    # robomimic picks a render GPU through egl_probe, which the image does not ship
    sys.modules.setdefault('egl_probe', types.SimpleNamespace(get_available_devices=lambda: [0]))
    import h5py
    import robomimic.utils.env_utils as EU
    import robomimic.utils.file_utils as FU
    meta = FU.get_env_metadata_from_dataset(args.dataset)
    env = EU.create_env_for_data_processing(env_meta=meta, camera_names=list(CAMS),
                                            camera_height=args.size, camera_width=args.size,
                                            reward_shaping=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with h5py.File(args.dataset, 'r') as f, h5py.File(args.out, 'w') as g:
        src, out = f['data'], g.create_group('data')
        for k, v in src.attrs.items():
            out.attrs[k] = v
        total = 0
        for name in sorted_demos(src)[:args.max_demos or None]:
            ep = src[name]
            states = ep['states'][:]
            env.reset()
            rows = []
            for s in states:
                env.reset_to({'states': s})
                o = env.get_observation()
                rows.append({k: np.asarray(o[k]) for k in [f'{c}_image' for c in CAMS] + list(STATE_KEYS)})
            d = out.create_group(name)
            d.attrs['num_samples'] = len(states)
            d['actions'] = ep['actions'][:]
            d['states'] = states
            d['dones'] = np.r_[np.zeros(len(states) - 1), 1].astype(np.uint8)
            for k in rows[0]:
                v = np.stack([r[k] for r in rows])
                if v.ndim == 4 and v.shape[1] == 3:                # CHW float -> HWC uint8
                    v = (np.transpose(v, (0, 2, 3, 1)) * (255 if v.max() <= 1 else 1)).astype(np.uint8)
                d[f'obs/{k}'] = v
            total += len(states)
            print(f'{name}: {len(states)} steps', flush=True)
        out.attrs['total'] = total
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
