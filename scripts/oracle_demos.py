"""Scripted Square demos with a moving nut, rendered, as a LeRobotDataset for ACT (paper 2, 5d).

The DP Square checkpoint never saw a moving nut and fails even at d = 1, which caps every
execution strategy. The privileged-state planner of oracle_square.py does handle motion, so it
is the demonstrator: each episode draws a conveyor speed, the planner replans every `--replan`
ticks (one chunk, 15) from the true state, and the episode is kept only if it succeeds.
Replanning every 5 ticks halved its success (3/6 vs 8/9): the planner's phase logic assumes
chunk-length execution. The policy trained on it sees only images and proprio, never the nut state.

Two stages, so collection parallelises without merging datasets:

    collect   one worker: episodes -> <npz-dir>/ep_<seed>.npz (images, 9-dim state, 10-dim
              abs action). Seeds start at --seed-base (default 100000), far from the eval
              scenes (seed 0 + k), so no evaluation scene is ever a training scene.
    write     every npz -> one LeRobotDataset, with robomimic_to_lerobot.py's feature schema
              (camera order, state layout, 10-dim [pos, rot_6d, gripper] action, the action
              mode file the ACT stamp reads).

The recorded frames are exactly what the harness feeds a policy (sil_step.frames: upright
images, [eef_pos, eef_quat, gripper_qpos]), and the actions are the targets actually sent to the
env, converted with the same abs7_to_abs10 as the robomimic converter.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np

SPEEDS = (0.0, 0.005, 0.01, 0.015, 0.02)
TASK = 'place the square nut on the peg'


def collect(args) -> None:
    import oracle_lift as ol
    import oracle_square as sq
    from evh_controller.rotation import abs7_to_abs10
    from evh_plant.env_factory import EnvSpec, build_env
    from sil_step import frames as image_frames

    os.makedirs(args.npz_dir, exist_ok=True)
    cfg = sq.SquareConfig(chunk_size=15)
    planner = sq.SquarePolicy(cfg)
    oracle_frames = sq.make_frames(None)
    base = build_env(EnvSpec(env_name='NutAssemblySquare', action_hz=20.0,
                             max_episode_s=args.max_s + 1.0, seed=0,
                             absolute_actions=True)).env

    def held(o):
        return sq.is_held(o['robot0_eef_pos'], o['robot0_gripper_qpos'], o['SquareNut_pos'],
                          o['SquareNut_quat'])

    env = ol.MovingCube(base, 0.0, 20.0, joint='SquareNut_joint0', pos_key='SquareNut_pos',
                        held_fn=held, half_range=args.conveyor_range)
    kept = tried = 0
    seed = args.seed_base + args.worker
    while kept < args.episodes:
        seed_k, seed = seed, seed + args.n_workers
        out = os.path.join(args.npz_dir, f'ep_{seed_k}.npz')
        if os.path.exists(out):
            kept += 1
            continue
        tried += 1
        np.random.seed(seed_k)
        env.speed = float(np.random.choice(SPEEDS))
        obs = env.reset()
        agent, wrist, state, actions = [], [], [], []
        chunk, i = None, 0
        ok = False
        for _t in range(int(args.max_s * 20)):
            if chunk is None or i >= min(args.replan, len(chunk)):
                p = oracle_frames(obs)[2]
                chunk, i = planner.predict({'proprio': p[None]}), 0
            a, w, s = image_frames(obs, True)
            agent.append(a)
            wrist.append(w)
            state.append(s)
            actions.append(chunk[i].astype(np.float32))
            obs, *_ = env.step(chunk[i])
            i += 1
            if env._check_success():
                ok = True
                break
        if ok:
            np.savez_compressed(out, agentview=np.stack(agent), wrist=np.stack(wrist),
                                state=np.stack(state),
                                action=abs7_to_abs10(np.stack(actions)).astype(np.float32),
                                speed=env.speed)
            kept += 1
        print(f'worker {args.worker} seed {seed_k} speed {env.speed:.3f}: '
              f'{"ok" if ok else "fail"} ({len(actions)} ticks) kept {kept}/{tried}', flush=True)


def write(args) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from robomimic_to_lerobot import ACTION_MODE_FILE, STATE_KEYS, build_features, state_names
    files = sorted(glob.glob(os.path.join(args.npz_dir, 'ep_*.npz')),
                   key=lambda f: int(Path(f).stem.split('_')[1]))
    if args.max_episodes:
        files = files[:args.max_episodes]
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    probe = np.load(files[0])
    cams = ['agentview_image', 'robot0_eye_in_hand_image']    # scene first, wrist last
    features = build_features(cams, 9, state_names(STATE_KEYS, [3, 4, 2]), 10,
                              tuple(probe['agentview'].shape[1:]), use_videos=False,
                              with_alt=False)
    ds = LeRobotDataset.create(repo_id=f'local/{out.name}', fps=20, root=out,
                               robot_type='robosuite_oracle', features=features,
                               use_videos=False, image_writer_threads=4)
    speeds = []
    for k, f in enumerate(files):
        ep = np.load(f)
        n = len(ep['action'])
        for t in range(n):
            ds.add_frame({'action': ep['action'][t],
                          'next.done': np.array([t == n - 1]),
                          'observation.state': ep['state'][t],
                          'observation.images.agentview': ep['agentview'][t],
                          'observation.images.robot0_eye_in_hand': ep['wrist'][t]}, task=TASK)
        ds.save_episode()
        speeds.append(float(ep['speed']))
        if (k + 1) % 50 == 0:
            print(f'  {k + 1}/{len(files)} episodes', flush=True)
    (out / 'meta' / ACTION_MODE_FILE).write_text(json.dumps(
        {'absolute_actions': True, 'actions': 'oracle-abs', 'rot_repr': 'rot_6d'}, indent=2) + '\n')
    hist = {s: speeds.count(s) for s in sorted(set(speeds))}
    print(f'wrote {ds.meta.total_episodes} episodes / {ds.meta.total_frames} frames -> {out}; '
          f'speeds {hist}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--npz-dir', required=True)
    c.add_argument('--episodes', type=int, default=50, help='successful episodes per worker')
    c.add_argument('--worker', type=int, default=0)
    c.add_argument('--n-workers', type=int, default=1)
    c.add_argument('--seed-base', type=int, default=100_000)
    c.add_argument('--replan', type=int, default=15,
                   help='ticks between planner calls (5 halved the planner\'s success: its phase '
                        'logic assumes chunk-length execution)')
    c.add_argument('--max-s', type=float, default=40.0)
    c.add_argument('--conveyor-range', type=float, default=0.06)
    w = sub.add_parser('write')
    w.add_argument('--npz-dir', required=True)
    w.add_argument('--out', required=True)
    w.add_argument('--max-episodes', type=int, default=0)
    args = ap.parse_args()
    collect(args) if args.cmd == 'collect' else write(args)


if __name__ == '__main__':
    main()
