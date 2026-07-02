#!/usr/bin/env python3
"""Run the robosuite/MuJoCo plant in isolation — no ROS — with visualization + observability.

Use this to verify the task, action space, camera rendering, and observation keys BEFORE wiring
the same `suite.make(...)` into evh_plant._build_env. It does not need the dataset or a policy;
actions are scripted (zero / random / sine) so you can watch the arm move and inspect what the
env emits.

Examples
--------
On-screen viewer (needs a display):
    python scripts/run_sim_standalone.py --env Lift --actions random --viewer

Headless -> save a video (set the GL backend for offscreen rendering):
    MUJOCO_GL=egl python scripts/run_sim_standalone.py --env Can --actions sine \
        --video out.mp4 --steps 200

Multi-camera grid video (comma-separated names):
    MUJOCO_GL=egl python scripts/run_sim_standalone.py --env PickPlaceCan --actions sine \
        --camera agentview,frontview,robot0_eye_in_hand --video grid.mp4 --steps 200

Just print the observation/action spec and exit:
    python scripts/run_sim_standalone.py --env Square --inspect-only

Notes
-----
* OSC_POSE controller -> 7-D action (6-DoF EE delta + gripper), matching the project's contract.
* robosuite's controller-config API moved between versions; we try the known variants.
* On a headless box, on-screen `--viewer` will fail — use `--video` with MUJOCO_GL=egl (or osmesa).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def parse_cameras(camera_arg: str) -> list[str]:
    cameras = [c.strip() for c in camera_arg.split(',') if c.strip()]
    if not cameras:
        raise ValueError('--camera must name at least one view')
    return cameras


def make_controller_config():
    """Return (kwarg_name, value) for an OSC_POSE (Cartesian) controller, across robosuite versions."""
    try:  # robosuite <= 1.4
        from robosuite.controllers import load_controller_config
        return 'controller_configs', load_controller_config(default_controller='OSC_POSE')
    except Exception:
        pass
    try:  # robosuite >= 1.5 (composite controllers)
        from robosuite.controllers import load_composite_controller_config
        return 'controller_configs', load_composite_controller_config(controller='BASIC')
    except Exception:
        return None, None  # fall back to env default controller


def build_env(args):
    import robosuite as suite

    cameras = args.cameras
    kwargs = dict(
        env_name=args.env,
        robots=args.robot,
        has_renderer=args.viewer,
        has_offscreen_renderer=bool(args.video) or args.use_camera,
        use_camera_obs=args.use_camera,
        control_freq=args.control_hz,
        horizon=args.steps,
        render_camera=cameras[0],  # mjviewer supports one fixed cam at a time
    )
    if args.use_camera or args.video:
        n = len(cameras)
        kwargs.update(
            camera_names=cameras,
            camera_heights=[args.image_size] * n,
            camera_widths=[args.image_size] * n,
        )

    ckey, cval = make_controller_config()
    if cval is not None:
        kwargs[ckey] = cval
        print(f'[sim] using OSC_POSE-style controller ({ckey})')
    else:
        print('[sim] WARNING: could not load OSC_POSE config; using env default controller')

    cam_label = cameras[0] if len(cameras) == 1 else f'[{", ".join(cameras)}]'
    print(f'[sim] suite.make({args.env}, robot={args.robot}, control_hz={args.control_hz}, camera={cam_label})')
    return suite.make(**kwargs)


def render_camera_frame(env, obs, camera: str, size: int) -> np.ndarray:
    img = obs.get(f'{camera}_image')
    if img is None:
        img = env.sim.render(camera_name=camera, height=size, width=size)
    return np.flipud(np.asarray(img)).astype(np.uint8)


def stitch_grid(frames: list[np.ndarray]) -> np.ndarray:
    """Tile frames into a roughly square grid (row-major)."""
    n = len(frames)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    h, w = frames[0].shape[:2]
    grid = np.zeros((nrow * h, ncol * w, frames[0].shape[2]), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, ncol)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame
    return grid


def describe(env, obs) -> None:
    """Observability: print observation keys/shapes/ranges, action spec, control freq."""
    print('\n=== observation spec ===')
    for k, v in sorted(obs.items()):
        v = np.asarray(v)
        rng = f'[{v.min():.3g}, {v.max():.3g}]' if v.size else '[]'
        print(f'  {k:32s} shape={str(v.shape):16s} dtype={str(v.dtype):8s} range={rng}')
    low, high = env.action_spec
    print('\n=== action spec ===')
    print(f'  action_dim = {len(low)}   low={np.round(low, 2)}   high={np.round(high, 2)}')
    print(f'  control_freq = {getattr(env, "control_freq", "?")} Hz')
    print('========================\n')


def sample_action(env, mode: str, t: int) -> np.ndarray:
    low, high = env.action_spec
    dim = len(low)
    if mode == 'zero':
        return np.zeros(dim)
    if mode == 'random':
        return np.random.uniform(low, high) * 0.3      # gentle, stays in-bounds
    if mode == 'sine':                                  # smooth, watchable motion
        a = np.zeros(dim)
        a[:min(3, dim)] = 0.3 * np.sin(2 * np.pi * 0.5 * t / env.control_freq + np.arange(min(3, dim)))
        return a
    raise ValueError(mode)


def main() -> None:
    p = argparse.ArgumentParser(description='Standalone robosuite plant (no ROS)')
    p.add_argument('--env', default='Lift', help='Lift | PickPlaceCan | NutAssemblySquare | ...')
    p.add_argument('--robot', default='Panda')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--control-hz', type=float, default=20.0)
    p.add_argument('--actions', choices=['zero', 'random', 'sine'], default='sine')
    p.add_argument('--camera', default='agentview',
                   help='camera name, or comma-separated list for a grid video '
                        '(e.g. agentview,frontview,robot0_eye_in_hand)')
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--viewer', action='store_true', help='on-screen MuJoCo viewer (needs display)')
    p.add_argument('--use-camera', action='store_true', help='include camera obs in observation dict')
    p.add_argument('--video', default='', help='save an offscreen-rendered mp4 to this path')
    p.add_argument('--inspect-only', action='store_true', help='print specs after reset and exit')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    args.cameras = parse_cameras(args.camera)
    if args.viewer and len(args.cameras) > 1:
        print(f'[sim] --viewer uses first camera only: {args.cameras[0]}')

    np.random.seed(args.seed)
    env = build_env(args)
    obs = env.reset()
    describe(env, obs)
    if args.inspect_only:
        env.close()
        return

    frames, writer = [], None
    if args.video:
        import imageio
        writer = imageio.get_writer(args.video, fps=int(args.control_hz))

    succeeded = False
    for t in range(args.steps):
        action = sample_action(env, args.actions, t)
        obs, reward, done, info = env.step(action)

        if args.viewer:
            env.render()
        if writer is not None:
            tiles = [render_camera_frame(env, obs, cam, args.image_size) for cam in args.cameras]
            writer.append_data(tiles[0] if len(tiles) == 1 else stitch_grid(tiles))

        if t % 20 == 0 or done:
            print(f'  t={t:4d}  reward={reward:6.3f}  done={done}')
        if hasattr(env, '_check_success') and env._check_success():
            succeeded = True
        if done:
            break

    if writer is not None:
        writer.close()
        print(f'[sim] wrote {args.video}')
    print(f'[sim] finished {t + 1} steps; task success seen: {succeeded}')
    env.close()


if __name__ == '__main__':
    try:
        main()
    except ImportError as e:
        print(f'Missing dependency: {e}\nInstall the host deps: pip install -r requirements-host.txt',
              file=sys.stderr)
        sys.exit(1)
