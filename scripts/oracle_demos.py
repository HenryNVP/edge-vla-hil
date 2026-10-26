"""Scripted Square demos with a moving nut, rendered, as a LeRobotDataset for ACT (paper 2, 5d).

The DP Square checkpoint never saw a moving nut and fails even at d = 1, which caps every
execution strategy. The privileged-state planner of oracle_square.py does handle motion, so it
is the demonstrator: each episode draws a conveyor speed, the planner replans every `--replan`
ticks (one chunk, 15) from the true state, and the episode is kept only if it succeeds.
Replanning every 5 ticks halved its success (3/6 vs 8/9): the planner's phase logic assumes
chunk-length execution. The policy trained on it sees only images and proprio, never the nut state.

`collect --task nut_goal` runs the same collector on nut_goal's NutAssembly (both nuts on the
table, scripts/nut_goal.py's GoalWorld owning which one the episode wants): the conveyor moves
the episode's initial goal nut, the goal alternates on the kept count (exactly balanced however
the success rates differ), and with probability --switch-prob GoalWorld flips it once at a tick
uniform in [10, 80] — cancelled if that nut is already lifted. The planner re-targets at the
next replan (it opens and approaches the new nut), and an episode is kept only if the final
goal nut is on its peg. Beyond the square keys, each episode saves goal [T] int8, switch_tick
(int, -1 if none or cancelled), nut [T, 7] (the goal nut's pose) and nuts [T, 2, 7] (both
nuts). Default npz dir data/oracle_nut_goal/npz; the write stage does not support this task.

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
SWITCH_RANGE = (10, 80)             # goal-switch ticks (before any grasp can have lifted a nut)


def draw_switch(rng, prob: float) -> int | None:
    """One goal switch at a tick uniform in SWITCH_RANGE with probability `prob`, else None."""
    return int(rng.randint(*SWITCH_RANGE)) if rng.random() < prob else None


def episode_params(seed_k: int, kept: int, switch_prob: float) -> tuple[float, int, int | None]:
    """(speed, goal, switch_tick) for one nut_goal episode, seeded like the square collector
    (one RNG stream per episode: scene, conveyor direction, switch).

    The goal alternates on the kept count instead of being drawn: a drawn goal drifts with the
    two goals' differing success rates and collapses for an even --n-workers (every worker then
    owns one parity of seed), while kept % 2 leaves the saved dataset exactly balanced however
    collection goes.
    """
    np.random.seed(seed_k)
    speed = float(np.random.choice(SPEEDS))
    return speed, kept % 2, draw_switch(np.random, switch_prob)


def collect(args) -> None:
    import oracle_lift as ol
    import oracle_square as sq
    import nut_goal as ng
    from evh_controller.rotation import abs7_to_abs10
    from evh_plant.env_factory import EnvSpec, build_env
    from sil_step import frames as image_frames

    nut_task = args.task == 'nut_goal'
    os.makedirs(args.npz_dir, exist_ok=True)
    cfg = sq.SquareConfig(chunk_size=15)
    planner = sq.SquarePolicy(cfg)
    oracle_frames = sq.make_frames(None, spec=None if nut_task else sq.SQUARE)
    cams = ('agentview', 'robot0_eye_in_hand') + (('birdview',) if args.top else ())
    base = build_env(EnvSpec(env_name=ng.ENV_NAME if nut_task else 'NutAssemblySquare',
                             action_hz=20.0, max_episode_s=args.max_s + 1.0, seed=0,
                             absolute_actions=True, cameras=cams)).env
    if args.top:
        import top_camera

    def held(o):
        return sq.is_held(o['robot0_eef_pos'], o['robot0_gripper_qpos'], o['SquareNut_pos'],
                          o['SquareNut_quat'])

    env = None if nut_task else ol.MovingCube(base, 0.0, 20.0, joint='SquareNut_joint0',
                                              pos_key='SquareNut_pos', held_fn=held,
                                              half_range=args.conveyor_range)
    kept = tried = 0
    seed = args.seed_base + args.worker
    while kept < args.episodes:
        seed_k, seed = seed, seed + args.n_workers
        out = os.path.join(args.npz_dir, f'ep_{seed_k}.npz')
        if os.path.exists(out):
            kept += 1
            continue
        tried += 1
        goal0, switch_tick = 0, None
        if nut_task:
            speed, goal0, switch_tick = episode_params(seed_k, kept, args.switch_prob)
            spec0 = sq.spec_for(goal0)           # the conveyor rides the initial goal nut
            nut0 = spec0.name
            env = ng.GoalWorld(
                ol.MovingCube(base, speed, 20.0, joint=f'{nut0}_joint0', pos_key=f'{nut0}_pos',
                              held_fn=lambda o, s=spec0: sq.is_held(
                                  o['robot0_eef_pos'], o['robot0_gripper_qpos'],
                                  o[f'{s.name}_pos'], o[f'{s.name}_quat'], s),
                              half_range=args.conveyor_range))
            obs = env.reset(goal=goal0, switch_tick=switch_tick)   # reset(), not the ctor:
            # GoalWorld.reset overwrites switch_tick with its (default None) argument
        else:
            np.random.seed(seed_k)
            env.speed = float(np.random.choice(SPEEDS))
            speed = env.speed
            obs = env.reset()
        extra = {}
        if args.top:
            obs = {**obs, **top_camera.apply(base)}   # re-rendered images, goal keys kept
            extra = {f'P_{c}': m for c, m in top_camera.matrices(
                base, ('agentview', 'birdview'), base.camera_heights[0]).items()}
        agent, wrist, state, actions = [], [], [], []
        top, nut = [], []
        goal_t, nut_t, nuts_t = [], [], []
        chunk, i = None, 0
        ok = False
        for _t in range(int(args.max_s * 20)):
            if chunk is None or i >= min(args.replan, len(chunk)):
                if nut_task:
                    planner.spec = sq.spec_for(obs['goal'])    # re-target on a goal switch
                p = oracle_frames(obs)[2]
                chunk, i = planner.predict({'proprio': p[None]}), 0
            a, w, s = image_frames(obs, True)
            agent.append(a)
            wrist.append(w)
            state.append(s)
            if args.top:
                top.append(np.flipud(np.asarray(obs['birdview_image'])).astype(np.uint8))
                if not nut_task:
                    nut.append(np.concatenate([obs['SquareNut_pos'], obs['SquareNut_quat']]))
            if nut_task:
                goal_t.append(obs['goal'])
                nut_t.append(np.concatenate([obs['goal_nut_pos'], obs['goal_nut_quat']]))
                nuts_t.append(np.concatenate([obs['SquareNut_pos'], obs['SquareNut_quat'],
                                              obs['RoundNut_pos'], obs['RoundNut_quat']]))
            actions.append(chunk[i].astype(np.float32))
            obs, *_ = env.step(chunk[i])
            i += 1
            if (env.success() if nut_task else env._check_success()):
                ok = True
                break
        if ok:
            if args.top:
                extra.update(top=np.stack(top))
                if not nut_task:
                    extra.update(nut=np.stack(nut).astype(np.float32))
            if nut_task:
                extra.update(goal=np.stack(goal_t).astype(np.int8),
                             switch_tick=np.int64(env.switch_tick if env.switched else -1),
                             nut=np.stack(nut_t).astype(np.float32),
                             nuts=np.stack(nuts_t).astype(np.float32).reshape(-1, 2, 7))
            np.savez_compressed(out, agentview=np.stack(agent), wrist=np.stack(wrist), **extra,
                                state=np.stack(state),
                                action=abs7_to_abs10(np.stack(actions)).astype(np.float32),
                                speed=speed)
            kept += 1
        print(f'worker {args.worker} seed {seed_k} speed {speed:.3f}'
              + (f' goal {goal0} switch {env.switch_tick if env.switched else -1}'
                 if nut_task else '')
              + f': {"ok" if ok else "fail"} ({len(actions)} ticks) kept {kept}/{tried}', flush=True)


def write(args) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from robomimic_to_lerobot import ACTION_MODE_FILE, STATE_KEYS, build_features, state_names
    files = sorted((f for d in args.npz_dir.split(',') for f in glob.glob(os.path.join(d, 'ep_*.npz'))),
                   key=lambda f: int(Path(f).stem.split('_')[1]))
    if args.max_episodes:
        files = files[:args.max_episodes]
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    probe = np.load(files[0])
    if 'goal' in probe:
        raise SystemExit('nut_goal episodes carry goal / switch_tick / nut / nuts keys this '
                         'writer does not handle; write is square-task only')
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
    c.add_argument('--task', default='square', choices=['square', 'nut_goal'],
                   help='square: NutAssemblySquare, the square nut; nut_goal: nut_goal.py\'s '
                        'both-nuts task with a goal (and a possible one-shot switch)')
    c.add_argument('--npz-dir', default=None,
                   help='episode npz dir (default data/oracle_nut_goal/npz for --task '
                        'nut_goal; required for square)')
    c.add_argument('--switch-prob', type=float, default=0.5,
                   help='nut_goal: probability of one goal switch at a tick uniform in [10, 80]')
    c.add_argument('--episodes', type=int, default=50, help='successful episodes per worker')
    c.add_argument('--worker', type=int, default=0)
    c.add_argument('--n-workers', type=int, default=1)
    c.add_argument('--seed-base', type=int, default=100_000)
    c.add_argument('--replan', type=int, default=15,
                   help='ticks between planner calls (5 halved the planner\'s success: its phase '
                        'logic assumes chunk-length execution)')
    c.add_argument('--max-s', type=float, default=40.0)
    c.add_argument('--top', action='store_true',
                   help='also record the close top camera (top_camera.py), the nut pose '
                        '[pos, quat xyzw] and both scene cameras\' projection matrices')
    c.add_argument('--conveyor-range', type=float, default=0.06)
    w = sub.add_parser('write')
    w.add_argument('--npz-dir', required=True, help='one or more, comma-separated')
    w.add_argument('--out', required=True)
    w.add_argument('--max-episodes', type=int, default=0)
    args = ap.parse_args()
    if args.cmd == 'collect':
        if args.npz_dir is None:
            if args.task != 'nut_goal':
                c.error('--npz-dir is required for the square task')
            args.npz_dir = 'data/oracle_nut_goal/npz'
        collect(args)
    else:
        write(args)


if __name__ == '__main__':
    main()
