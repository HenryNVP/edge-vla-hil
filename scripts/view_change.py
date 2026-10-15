"""Which camera view changes more per control tick: agentview, a top-down view, or the wrist?

Motivation (paper 2, query planner): a local model whose queries attend only to what changed
since the cloud's (stale) feature map is cheap only if few patches change. This runs the scripted
Square planner (synchronous, d = 0) on a static or moving nut, renders each camera every tick,
and reports the fraction of patches whose mean absolute pixel change exceeds a threshold.

    python3 scripts/view_change.py --episodes 3 --speeds 0,0.02
"""
from __future__ import annotations

import argparse

import numpy as np

CAMERAS = ('agentview', 'birdview', 'robot0_eye_in_hand')


def changed_patches(prev: np.ndarray, cur: np.ndarray, patch: int, thresh: float) -> np.ndarray:
    """Boolean [H/patch, W/patch]: patches whose mean |pixel change| (0-255, over channels)
    exceeds `thresh`."""
    d = np.abs(cur.astype(np.float32) - prev.astype(np.float32)).mean(axis=-1)
    h, w = d.shape[0] // patch, d.shape[1] // patch
    return d[:h * patch, :w * patch].reshape(h, patch, w, patch).mean(axis=(1, 3)) > thresh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--speeds', default='0,0.02')
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--patch', type=int, default=8)
    ap.add_argument('--thresh', type=float, default=4.0, help='mean |delta| per patch, 0-255')
    ap.add_argument('--replan', type=int, default=4, help='chunk entries played per plan')
    ap.add_argument('--ticks', type=int, default=400)
    ap.add_argument('--ages', default='1,10,20', help='compare with the frame this many ticks old')
    args = ap.parse_args()

    import oracle_lift as ol
    import oracle_square as sq
    import robosuite as suite
    from evh_plant.env_factory import make_controller_config, set_control_delta
    controller = make_controller_config()
    set_control_delta(controller, False)
    env = suite.make(env_name='NutAssemblySquare', robots='Panda', has_renderer=False,
                     has_offscreen_renderer=True, use_camera_obs=False, control_freq=20.0,
                     horizon=args.ticks + 10, reward_shaping=False,
                     controller_configs=controller)
    cfg = sq.SquareConfig()

    def held(o):
        return sq.is_held(o['robot0_eef_pos'], o['robot0_gripper_qpos'], o['SquareNut_pos'],
                          o['SquareNut_quat'])

    def render(cam):
        return env.sim.render(camera_name=cam, width=args.size, height=args.size)[::-1]

    print(f'patch {args.patch}px of {args.size}px, threshold {args.thresh}/255; '
          f'fraction of patches changed per tick: median / p90 / mean')
    for v in [float(x) for x in args.speeds.split(',')]:
        world = ol.MovingCube(env, v, 20.0, joint='SquareNut_joint0', pos_key='SquareNut_pos',
                              held_fn=held, half_range=0.06)
        ages = [int(x) for x in args.ages.split(',')]
        frac = {c: [] for c in CAMERAS}
        aged = {(c, a): [] for c in CAMERAS for a in ages}
        phase_frac = {c: {} for c in CAMERAS}
        wins = 0
        for k in range(args.episodes):
            np.random.seed(k)
            o = world.reset()
            prev = {c: render(c) for c in CAMERAS}
            hist = {c: [prev[c]] for c in CAMERAS}
            chunk, i = None, args.replan
            for t in range(args.ticks):
                if i >= args.replan:
                    chunk, i = sq.plan_chunk(o['robot0_eef_pos'], o['robot0_eef_quat'],
                                             o['robot0_gripper_qpos'], o['SquareNut_pos'],
                                             o['SquareNut_quat'], cfg), 0
                phase = sq.start_phase(o['robot0_eef_pos'], o['robot0_eef_quat'],
                                       o['robot0_gripper_qpos'], o['SquareNut_pos'],
                                       o['SquareNut_quat'], cfg)
                o, *_ = world.step(chunk[i])
                i += 1
                for c in CAMERAS:
                    cur = render(c)
                    f = float(changed_patches(prev[c], cur, args.patch, args.thresh).mean())
                    frac[c].append(f)
                    phase_frac[c].setdefault(phase, []).append(f)
                    prev[c] = cur
                    hist[c].append(cur)
                    for a in ages:
                        if len(hist[c]) > a:
                            aged[(c, a)].append(float(changed_patches(
                                hist[c][-1 - a], cur, args.patch, args.thresh).mean()))
                if env._check_success():
                    wins += 1
                    break
        print(f'\nnut speed {v} m/s ({wins}/{args.episodes} solved)')
        for c in CAMERAS:
            a = np.array(frac[c])
            by = '  '.join(f'{p}:{np.mean(x):.3f}' for p, x in phase_frac[c].items())
            print(f'  {c:20s} {np.median(a):.3f} / {np.percentile(a, 90):.3f} / {a.mean():.3f}'
                  f'   by phase  {by}')
        print('  vs the frame k ticks old (mean fraction changed): ' + '  '.join(f'k={a}' for a in ages))
        for c in CAMERAS:
            print(f'  {c:20s} ' + '  '.join(f'{np.mean(aged[(c, a)]):.3f}' for a in ages))


if __name__ == '__main__':
    main()
