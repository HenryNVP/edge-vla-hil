#!/usr/bin/env python3
"""Co-located validation of the diffusion_policy Lift checkpoint — the Phase 2 gate.

Runs the pretrained image policy directly against robosuite (no ROS, no network, no executor):
predict a chunk, execute `--act-steps` actions, repeat — the same protocol as the original
repo's robomimic eval. This is the number every HiL result is anchored to: if co-located
success is low, fix THIS (obs alignment, controller config, robosuite-version gap, finetune)
before blaming the network.

Usage (inside the host container):
    bash scripts/setup_dp_deps.sh
    MUJOCO_GL=egl python3 scripts/eval_dp_colocated.py --episodes 10
    # faster sampler (what the HiL loop will use):
    MUJOCO_GL=egl python3 scripts/eval_dp_colocated.py --episodes 10 --denoise-steps 16
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'ros2_ws', 'src', 'evh_controller'))

from evh_controller.dp_repo_policy import DiffusionPolicyRepoBackend  # noqa: E402


def make_controller_config():
    try:  # robosuite <= 1.4
        from robosuite.controllers import load_controller_config
        return load_controller_config(default_controller='OSC_POSE')
    except Exception:
        pass
    try:  # robosuite >= 1.5
        from robosuite.controllers import load_composite_controller_config
        return load_composite_controller_config(controller='BASIC')
    except Exception:
        return None


def set_control_delta(config: dict, value: bool) -> None:
    """Set OSC control_delta across robosuite config shapes (1.4 flat / 1.5 composite)."""
    if config is None:
        return
    if 'control_delta' in config:                    # 1.4 flat OSC config
        config['control_delta'] = value
        return
    for part_cfg in config.get('body_parts', {}).values():   # 1.5 composite
        if isinstance(part_cfg, dict) and part_cfg.get('type', '').startswith('OSC'):
            part_cfg['control_delta'] = value


def build_env(args, absolute_actions=False):
    import robosuite as suite

    kwargs = dict(
        env_name=args.env,
        robots='Panda',
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=['agentview', 'robot0_eye_in_hand'],
        camera_heights=args.image_size,
        camera_widths=args.image_size,
        control_freq=20,             # the rate the checkpoint was trained at
        horizon=args.horizon,
        ignore_done=True,            # robomimic protocol: terminate on success, not on horizon
        reward_shaping=False,
        seed=args.seed,
    )
    controller = make_controller_config()
    if controller is not None:
        if absolute_actions:
            set_control_delta(controller, False)   # abs-action checkpoints drive OSC absolute
        kwargs['controller_configs'] = controller
    elif absolute_actions:
        raise RuntimeError('abs-action policy needs an OSC controller config to flip control_delta')
    try:
        return suite.make(**kwargs)
    except TypeError:                # robosuite 1.4 has no seed kwarg
        kwargs.pop('seed', None)
        return suite.make(**kwargs)


def obs_to_frames(obs, size):
    """robosuite obs dict -> (agentview, wrist, proprio) in the backend's convention."""
    agent = np.flipud(obs['agentview_image']).astype(np.uint8)
    wrist = np.flipud(obs['robot0_eye_in_hand_image']).astype(np.uint8)
    proprio = np.concatenate([
        np.asarray(obs['robot0_eef_pos'], dtype=np.float32),
        np.asarray(obs['robot0_eef_quat'], dtype=np.float32),
        np.asarray(obs['robot0_gripper_qpos'], dtype=np.float32),
    ])
    return agent, wrist, proprio


def run_episode(env, policy, args, writer=None):
    obs = env.reset()
    To = policy.n_obs_steps
    hist = collections.deque(maxlen=To)
    hist.append(obs_to_frames(obs, args.image_size))

    infer_times = []
    t = 0
    while t < args.horizon:
        stacked = {
            'agentview': np.stack([h[0] for h in hist]),
            'wrist': np.stack([h[1] for h in hist]),
            'proprio': np.stack([h[2] for h in hist]),
        }
        t0 = time.perf_counter()
        chunk = policy.predict(stacked)
        infer_times.append(time.perf_counter() - t0)

        for action in chunk[:args.act_steps]:
            obs, _reward, _done, _info = env.step(action)
            hist.append(obs_to_frames(obs, args.image_size))
            if writer is not None:
                writer.append_data(np.flipud(obs['agentview_image']).astype(np.uint8))
            t += 1
            if env._check_success():
                return True, t, infer_times
            if t >= args.horizon:
                break
    return False, t, infer_times


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--ckpt', default=os.path.join(_REPO, 'checkpoints', 'dp_lift_ph_image_cnn.ckpt'))
    p.add_argument('--env', default='Lift')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--horizon', type=int, default=400)
    p.add_argument('--act-steps', type=int, default=8, help='actions executed per prediction')
    p.add_argument('--image-size', type=int, default=84)
    p.add_argument('--denoise-steps', type=int, default=0,
                   help='0 = checkpoint default (DDPM 100); >0 swaps in DDIM with this many steps')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--video', default='', help='record the first episode (agentview mp4)')
    args = p.parse_args()

    policy = DiffusionPolicyRepoBackend(
        args.ckpt, device=args.device,
        denoise_steps=args.denoise_steps if args.denoise_steps > 0 else None)
    print(f'[eval] policy: chunk_size={policy.chunk_size} action_dim={policy.action_dim} '
          f'n_obs_steps={policy.n_obs_steps} denoise={policy._policy.num_inference_steps} '
          f'absolute_actions={policy.absolute_actions}')

    env = build_env(args, absolute_actions=policy.absolute_actions)
    np.random.seed(args.seed)

    writer = None
    if args.video:
        import imageio
        os.makedirs(os.path.dirname(args.video) or '.', exist_ok=True)
        writer = imageio.get_writer(args.video, fps=20)

    successes, all_infer = 0, []
    for ep in range(args.episodes):
        ok, steps, infer = run_episode(env, policy, args, writer if ep == 0 else None)
        successes += ok
        all_infer += infer
        print(f'[eval] episode {ep}: success={ok} steps={steps} '
              f'infer={np.mean(infer):.3f}s/chunk')
        if writer is not None and ep == 0:
            writer.close()
            print(f'[eval] wrote {args.video}')

    print(f'\n[eval] SUCCESS RATE: {successes}/{args.episodes} '
          f'({100.0 * successes / args.episodes:.0f}%)  '
          f'mean inference {np.mean(all_infer):.3f}s '
          f'(= {np.mean(all_infer) * 20:.1f} control steps @ 20 Hz)')
    env.close()


if __name__ == '__main__':
    main()
