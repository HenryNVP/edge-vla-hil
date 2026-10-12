"""Does a DP chunk move rigidly with the object? The assumption anchoring makes, measured.

Anchoring shifts a stale chunk's pre-grasp targets by the object's displacement since the
request. That is right only if the policy itself would have planned the shifted chunk. Here the
object is teleported by delta (in x and y) and the chunk re-predicted with the same diffusion
noise. The slope of the targets' shift against delta, over the gripper-open entries, is the
answer: 1 = rigid (anchoring is what the policy would do), 0 = the chunk ignores the object.
Probed at several points along the approach (after `--warm` synchronous ticks), since early in
an episode the policy may still be moving to a generic pre-grasp pose.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

import dp_mech
from sil_step import frames, stack

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_object(env, joint: str, xy) -> None:
    sim = env.sim
    adr = sim.model.get_joint_qpos_addr(joint)[0]
    sim.data.qpos[adr:adr + 2] = xy
    sim.data.qvel[sim.model.get_joint_qvel_addr(joint)[0]:][:6] = 0.0
    sim.forward()


def observe(env):
    return env._get_observations(force_update=True)


def predict(policy, hist, seed: int) -> np.ndarray:
    import torch
    torch.manual_seed(seed)
    return np.asarray(policy.predict(stack(hist, True)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--task', default='lift', choices=sorted(dp_mech.TASKS))
    ap.add_argument('--scenes', type=int, default=10)
    ap.add_argument('--warm', default='0,10,20', help='synchronous ticks before probing')
    ap.add_argument('--delta-m', type=float, default=0.02)
    ap.add_argument('--denoise-steps', type=int, default=4)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    from collections import deque

    from evh_controller.policy import make_policy
    from evh_plant.env_factory import EnvSpec, build_env
    env_name, ckpt, pos_key, _q, joint, _r, _c = dp_mech.TASKS[args.task]
    policy = make_policy('dp', os.path.join(_REPO, 'checkpoints', ckpt),
                         denoise_steps=args.denoise_steps)
    env = build_env(EnvSpec(env_name=env_name, action_hz=20.0, max_episode_s=30.0, seed=0,
                            absolute_actions=True)).env
    rows = []
    for scene in range(args.scenes):
        for warm in [int(w) for w in args.warm.split(',')]:
            np.random.seed(scene)
            obs = env.reset()
            hist = deque([frames(obs)] * policy.n_obs_steps, maxlen=policy.n_obs_steps)
            chunk, i = None, 0
            for _t in range(warm):                       # synchronous: replan when exhausted
                if chunk is None or i >= len(chunk):
                    chunk, i = predict(policy, hist, 10_000 + scene), 0
                obs, *_ = env.step(chunk[i])
                i += 1
                hist.append(frames(obs))
            if obs[pos_key][2] > 0.86 if args.task == 'lift' else False:
                continue                                  # already lifted: nothing to probe
            state = env.sim.get_state()
            xy0 = np.array(obs[pos_key][:2], float)
            seed = 20_000 + scene
            base = predict(policy, hist, seed)
            for axis in (0, 1):
                for sign in (-1.0, 1.0):
                    d = np.zeros(2)
                    d[axis] = sign * args.delta_m
                    set_object(env, joint, xy0 + d)
                    h2 = deque(hist, maxlen=policy.n_obs_steps)
                    h2[-1] = frames(observe(env))         # newest frame shows the moved object
                    moved = predict(policy, h2, seed)
                    env.sim.set_state(state)
                    env.sim.forward()
                    open_ = (base[:, 6] < 0) & (moved[:, 6] < 0)
                    if not open_.any():
                        continue
                    shift = (moved[open_, axis] - base[open_, axis]).mean()
                    rows.append({'scene': scene, 'warm': warm, 'axis': 'xy'[axis],
                                 'delta': d[axis], 'slope': shift / d[axis],
                                 'open_entries': int(open_.sum())})
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for warm in sorted({r['warm'] for r in rows}):
        for ax in 'xy':
            s = [r['slope'] for r in rows if r['warm'] == warm and r['axis'] == ax]
            if s:
                print(f'warm={warm:3d} axis={ax}: slope median {np.median(s):.2f} '
                      f'IQR [{np.percentile(s, 25):.2f}, {np.percentile(s, 75):.2f}] n={len(s)}')


if __name__ == '__main__':
    main()
