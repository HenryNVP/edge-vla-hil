"""Paper 2 mechanism, with DP in place of the scripted planner: Lift and Square, moving object.

The oracle testbeds (oracle_lift.py, oracle_square.py) showed what continuation, anchoring and
close gating do when every chunk is right for the observation it was planned from. This runs the
same executors and the same sweep (oracle_lift.run_sweep, the same conveyor, tracker model and
delay tokens) on the image DP checkpoints, so the open question is only whether a learned,
image-conditioned chunk behaves like the planner's.

Two ways to continue a DP plan from the end of the queue, since images cannot be rolled forward:

    continuation[_anchored]          proprio roll (VLASH-style): current images, the proprio
                                     history replaced by the queued plan's last targets
    continuation_inpaint[_anchored]  current observation as is, the queued actions as a hard
                                     RTC-style inpainting prefix (capped by the horizon, 15)

The proprio roll is deliberately cheap: a first-order model of the OSC's lag. Each queued target
moves the predicted EE pose `lag` (0.3) of the way toward it, position linearly and orientation by
normalised lerp (targets through the controller-frame offset, invariant 7). Setting the pose to the
last TARGET instead was wrong by up to 10 cm on Square, because DP's absolute targets lead the arm
when it moves fast, and it did worse than no roll (continuation 1/20 at d = 1). The quaternion is
kept on the measured sign: q and -q are the same rotation, but the policy sees the raw numbers.
Fingers <- open, or closed at `closed_qpos` once the plan closes them. The DP gets no object state, so nothing else is rolled. Anchoring and the
tracker read the true object position from the simulator through the tracker model; it rides in
the obs dict under 'object' and never reaches the policy.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys

import numpy as np

import oracle_lift as ol

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASKS = {   # env, checkpoint, object keys, conveyor half range, closed finger qpos
    'lift': ('Lift', 'dp_lift_ph_image_cnn.ckpt', 'cube_pos', 'cube_quat', 'cube_joint0',
             0.12, 0.02),
    'square': ('NutAssemblySquare', 'dp_square_ph_image_cnn.ckpt', 'SquareNut_pos',
               'SquareNut_quat', 'SquareNut_joint0', 0.06, 0.01),
}
OPEN_QPOS = 0.04


def make_frames(tracker, pos_key: str, quat_key: str):
    """(agentview, wrist, proprio, extra); extra = [object(3), quat(4), tracked(3), ok(1)]."""
    from sil_step import frames as image_frames

    def frames(obs: dict, needs_wrist: bool = True):
        agent, wrist, proprio = image_frames(obs, needs_wrist)
        obj = np.asarray(obs[pos_key], np.float32)
        est = obj if tracker is None else tracker.update(obj)
        ok = est is not None
        extra = np.concatenate([obj, np.asarray(obs[quat_key], np.float32),
                                np.asarray(est if ok else obj, np.float32), [float(ok)]])
        return agent, wrist, proprio, extra.astype(np.float32)
    return frames


def stack(hist, needs_wrist: bool = True) -> dict:
    from sil_step import stack as image_stack
    return {**image_stack(hist, needs_wrist), 'object': hist[-1][3]}


def object_tracker(obs: dict):
    o = obs['object']
    return o[7:10] if o[10] > 0.5 else None


def proprio_roll(obs: dict, queued: np.ndarray, closed_qpos: float = 0.02,
                 lag: float = 0.3) -> dict:
    """Predict the proprio history at the end of the queue: the arm follows each queued target
    with first-order lag `lag` per tick (1.0 = it reaches every target at once)."""
    from oracle_square import axisangle_to_quat, ctrl_to_eef_quat
    if len(queued) == 0:
        return obs
    s = np.asarray(obs['proprio'], np.float32).copy()
    now = s[-1]
    width_now = float(now[7] - now[8])
    p, q = now[0:3].astype(float), now[3:7].astype(float)
    states = []
    for a in queued:
        p = p + lag * (a[:3] - p)
        qt = ctrl_to_eef_quat(axisangle_to_quat(a[3:6]))
        if qt @ q < 0:
            qt = -qt
        q = q + lag * (qt - q)
        q = q / np.linalg.norm(q)
        states.append((p, q, a[6]))
    for row, (p, q, g) in zip(range(len(s) - 1, -1, -1), states[::-1]):
        s[row, 0:3] = p
        s[row, 3:7] = q
        if g < 0:
            s[row, 7:9] = [OPEN_QPOS, -OPEN_QPOS]
        elif width_now > 2 * OPEN_QPOS - 0.01:            # open now, the plan closes them
            s[row, 7:9] = [closed_qpos, -closed_qpos]
        else:
            s[row, 7:9] = now[7:9]
    return {**obs, 'proprio': s}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--task', default='lift', choices=sorted(TASKS))
    ap.add_argument('--strategies', default='synchronous,noskip_async,continuation,'
                                            'continuation_inpaint')
    ap.add_argument('--deadline-s', type=float, default=20.0)
    ap.add_argument('--denoise-steps', type=int, default=4, help='paper-1 cells used 4')
    ap.add_argument('--backend', default='dp', help='dp | act | any make_policy backend')
    ap.add_argument('--weights', default='', help='default: the task\'s DP checkpoint')
    ap.add_argument('--roll-lag', type=float, default=0.3,
                    help='first-order lag of the proprio roll per tick (1 = jump to targets)')
    ol.add_common_args(ap)
    args = ap.parse_args()
    if args.hz != 20.0:
        sys.exit('the DP checkpoints run at 20 Hz')

    import run_manifest
    from evh_controller.policy import make_policy
    from evh_plant.env_factory import EnvSpec, build_env
    run_manifest.record(args.out, {'kind': 'dp_mech', 'config': vars(args)})
    # (the name stays dp_mech: DP was first; --backend act runs the same sweep on ACT)

    env_name, ckpt, pos_key, quat_key, joint, half_range, closed = TASKS[args.task]
    if args.backend != 'dp' and not args.weights:
        sys.exit(f'--backend {args.backend} needs --weights')
    policy = make_policy(args.backend, args.weights or os.path.join(_REPO, 'checkpoints', ckpt),
                         denoise_steps=args.denoise_steps)
    if not policy.absolute_actions:
        sys.exit('the mechanism executors shift absolute targets; this checkpoint emits deltas')
    tracker = ol.tracker_from(args)
    deadline = int(args.deadline_s * args.hz)
    base = build_env(EnvSpec(env_name=env_name, action_hz=args.hz,
                             max_episode_s=args.deadline_s + 1.0, seed=args.seed,
                             absolute_actions=True)).env

    if args.task == 'square':
        import oracle_square as sq

        def held(o):
            return sq.is_held(o['robot0_eef_pos'], o['robot0_gripper_qpos'], o[pos_key],
                              o[quat_key])
    else:
        held = None                     # MovingCube's default: the Lift cube in the hand

    ol.run_sweep(args, policy=policy, tracker=tracker, deadline=deadline,
                 make_world=lambda v: ol.MovingCube(base, v, args.hz, joint=joint,
                                                    pos_key=pos_key, held_fn=held,
                                                    half_range=half_range),
                 frame_fn=make_frames(tracker, pos_key, quat_key), stack_fn=stack,
                 executor_kw={'roll_fn': functools.partial(proprio_roll, closed_qpos=closed,
                                                       lag=args.roll_lag),
                              'anchor_fn': object_tracker})


if __name__ == '__main__':
    main()
