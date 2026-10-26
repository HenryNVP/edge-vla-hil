"""Paper 2 mechanism, with DP in place of the scripted planner: Lift and Square, moving object.

The oracle testbeds (oracle_lift.py, oracle_square.py) showed what continuation, anchoring and
close gating do when every chunk is right for the observation it was planned from. This runs the
same executors and the same sweep (oracle_lift.run_sweep, the same conveyor, tracker model and
delay tokens) on the image DP checkpoints, so the open question is only whether a learned,
image-conditioned chunk behaves like the planner's.

`--task nut_goal` runs the goal-from-cloud task (nut_goal.py): both nuts are on the table and the
episode's goal — scene k gets goal k % 2, `--goal-switch TICK` flips it once — says which one to
seat. GoalWorld wraps the env OUTERMOST (it owns the goal and its switch) around the conveyor
(MovingCube on the episode's initial goal nut), and its keys ('goal', 'goal_nut_pos',
'goal_nut_quat') follow the current goal, so the gates and anchoring read the goal nut unchanged.
The goal rides in the policy's obs as obs['goal'] (an int, from the newest frame), so the cloud
half of a split policy reads it at request time — link-delayed, which is the point: the robot
learns of a switch only from a memory requested after it. Three extra CSV columns (goal,
switched, wrong_nut) feed the robot-only guess analysis; every other task's behaviour and CSV
are unchanged.

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

import nut_goal
import oracle_lift as ol

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASKS = {   # env, checkpoint, object keys, conveyor joint, half range, closed finger qpos
    'lift': ('Lift', 'dp_lift_ph_image_cnn.ckpt', 'cube_pos', 'cube_quat', 'cube_joint0',
             0.12, 0.02),
    'square': ('NutAssemblySquare', 'dp_square_ph_image_cnn.ckpt', 'SquareNut_pos',
               'SquareNut_quat', 'SquareNut_joint0', 0.06, 0.01),
    'nut_goal': (nut_goal.ENV_NAME, 'dp_square_ph_image_cnn.ckpt', 'goal_nut_pos',
                 'goal_nut_quat', None, 0.06, 0.01),
}   # nut_goal's joint is per-episode (the initial goal nut's, retargeted by nut_reset) and the
    # checkpoint default is the Square one — the closest there is; the split planners that the
    # task is for take --weights anyway
OPEN_QPOS = 0.04


def make_frames(tracker, pos_key: str, quat_key: str, top: bool = False, goal: bool = False,
                watch=None):
    """(agentview, wrist, proprio, extra[, goal[, top]]); extra = [object(3), quat(4),
    tracked(3), ok(1)]; top = the close top camera's upright image (top_camera.py), for policies
    with needs_top; goal (nut_goal) appends the observation's goal int for stack to hand the
    policy as obs['goal']; watch (nut_goal) sees every observation (NutWatch)."""
    from sil_step import frames as image_frames

    def frames(obs: dict, needs_wrist: bool = True):
        if watch is not None:
            watch.note(obs)
        agent, wrist, proprio = image_frames(obs, needs_wrist)
        obj = np.asarray(obs[pos_key], np.float32)
        est = obj if tracker is None else tracker.update(obj)
        ok = est is not None
        extra = np.concatenate([obj, np.asarray(obs[quat_key], np.float32),
                                np.asarray(est if ok else obj, np.float32), [float(ok)]])
        out = [agent, wrist, proprio, extra.astype(np.float32)]
        if goal:
            out.append(int(obs['goal']))
        if top:
            topimg = np.flipud(np.asarray(obs['birdview_image'])).astype(np.uint8)
            out.append(topimg)
        return tuple(out)
    return frames


def stack(hist, needs_wrist: bool = True) -> dict:
    """The policy obs from the frame history; anything make_frames appended after the extra
    vector rides along: an int is the goal (nut_goal; the NEWEST frame's, i.e. the current one),
    an ndarray is the top camera's image stacked over the history."""
    from sil_step import stack as image_stack
    out = {**image_stack(hist, needs_wrist), 'object': hist[-1][3]}
    for i, cell in enumerate(hist[-1][4:], start=4):
        if isinstance(cell, np.ndarray):
            out['top'] = np.stack([h[i] for h in hist])
        else:
            out['goal'] = int(cell)
    return out


class TopCam:
    """The env with the close top camera re-applied after every (hard) reset."""

    def __init__(self, env) -> None:
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        import top_camera
        self.env.reset()
        return top_camera.apply(self.env)


def object_tracker(obs: dict):
    o = obs['object']
    return o[7:10] if o[10] > 0.5 else None


def oracle_layout(obs: dict, task: str) -> dict:
    """The oracle testbeds' proprio layout, rebuilt from the policy's proprio and 'object', so
    their gates run unchanged. Square (and nut_goal, whose 'object' is the goal nut's):
    [proprio(9), nut(3), nut quat(4), tracked(3), ok(1)] is exactly proprio + object; Lift has
    no object quaternion."""
    o = obs['object']
    tail = o if task in ('square', 'nut_goal') else np.concatenate([o[0:3], o[7:11]])
    return {'proprio': np.concatenate([np.asarray(obs['proprio'])[-1], tail])[None]}


def task_gate(obs: dict, action, task: str, slack: float = 0.0) -> bool:
    if task in ('square', 'nut_goal'):        # nut_goal gates on the goal nut like Square's
        import oracle_square as sq
        return sq.square_gate(oracle_layout(obs, task), action, slack=slack)
    return ol.lift_gate(oracle_layout(obs, task), action)


class SquareLift:
    """Lift once the fingers have stopped closing: finger width changed by under `settle_m` since
    the previous tick, at least `min_ticks` after the close. Fingers shut on nothing (width under
    the oracle's empty_width) pass at once: holding cannot rescue an empty grasp. Keeps its own
    previous width, since ACT sees one observation step."""

    def __init__(self, settle_m: float = 0.0005, min_ticks: int = 3) -> None:
        self.settle_m, self.min_ticks = settle_m, min_ticks
        self._prev = None

    def __call__(self, obs: dict, since_close: int) -> bool:
        import oracle_square as sq
        row = np.asarray(obs['proprio'])[-1]
        w = float(row[7] - row[8])
        prev, self._prev = (None if since_close <= 1 else self._prev), w
        if w < sq.SquareConfig().empty_width:
            return True
        return (since_close >= self.min_ticks and prev is not None
                and abs(w - prev) < self.settle_m)


class NutWatch:
    """nut_goal's CSV extras, watched from the observation stream: the harness only ever shows the
    env to frames and the success check, so this rides along with them. `start(goal)` at reset,
    `note(obs)` on every frame, `row(env)` at episode end. wrong_nut marks the NON-goal nut
    (w.r.t. the final goal — a switch re-grades) lifted at any tick (the same start-height test
    GoalWorld uses) or seated on its peg at the end (robosuite's objects_on_pegs): the robot-only
    guess going to the nut the goal did not name."""

    def __init__(self) -> None:
        self.start(0)

    def start(self, goal: int) -> None:
        self.goal0, self._z0, self._lifted = goal, None, [False, False]

    def note(self, obs: dict) -> None:
        if self._z0 is None:
            self._z0 = [float(obs[f'{n}_pos'][2]) for n in nut_goal.NUTS]
        for i, n in enumerate(nut_goal.NUTS):
            self._lifted[i] |= float(obs[f'{n}_pos'][2]) > self._z0[i] + nut_goal.LIFTED_M

    def row(self, env) -> dict:
        env.success()                          # refreshes objects_on_pegs behind the goal check
        other = 1 - env.goal
        return {'goal': self.goal0, 'switched': int(env.switched),
                'wrong_nut': int(self._lifted[other] or bool(env.objects_on_pegs[other]))}


def nut_reset(env, k: int, switch_tick: int | None = None):
    """nut_goal's per-episode reset: scene k gets goal k % 2, the same in every cell, and the
    conveyor is pointed at that nut — its joint's qpos/qvel addresses and raw position key, which
    are per nut (GoalWorld's goal_ keys follow the CURRENT goal, but the conveyor tracks the
    episode's initial nut all along). Held = the nut in the gripper (oracle_square's test) for
    the square nut, GoalWorld's lift test for the round one."""
    goal = k % 2
    nut = nut_goal.NUTS[goal]
    mc = env.env                                # the MovingCube inside GoalWorld
    mc._qadr = env.sim.model.get_joint_qpos_addr(f'{nut}_joint0')[0]
    mc._vadr = env.sim.model.get_joint_qvel_addr(f'{nut}_joint0')[0]
    mc.pos_key = f'{nut}_pos'
    if goal == 0:
        import oracle_square as sq

        def held(o, nut=nut):
            return sq.is_held(o['robot0_eef_pos'], o['robot0_gripper_qpos'],
                              o[f'{nut}_pos'], o[f'{nut}_quat'])
    else:
        def held(o, goal=goal):
            return env.lifted(o, goal)
    mc.held_fn = held
    return env.reset(goal=goal, switch_tick=switch_tick)


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
    ap.add_argument('--backend', default='dp', help='dp | act | two_stream | two_stream_chunk | bev | any make_policy backend')
    ap.add_argument('--weights', default='', help='default: the task\'s DP checkpoint')
    ap.add_argument('--gate-k', type=float, default=2.0,
                    help='close gate position slack in tracker sigmas (Square)')
    ap.add_argument('--roll-lag', type=float, default=0.3,
                    help='first-order lag of the proprio roll per tick (1 = jump to targets)')
    ap.add_argument('--refiner', default='', help="refiner weights (refiner.py) for 'refined'")
    ap.add_argument('--refiner-log', default='', help="record the refiner's inputs (npz) for training")
    ap.add_argument('--probe-log', default='',
                    help='Square: counterfactual close probes (npz) for fuzzy_gate.py')
    ap.add_argument('--fail-log', default='',
                    help='Square: per-episode failure class and milestones (csv, square_failures.py)')
    ap.add_argument('--fuzzy-gate', default='', help='fuzzy gate weights: replace the hard gate')
    ap.add_argument('--label-log', default='',
                    help="label every tick with the policy's fresh chunk (npz), refiner training")
    ap.add_argument('--replan', type=int, default=4,
                    help="cloud_local: entries of each local plan played before replanning")
    ap.add_argument('--refiner-lead', type=int, default=0, help='predicted entry whose position is played')
    ap.add_argument('--drop-after', type=int, default=None,
                    help='cloud_local: plan with no cloud memory once it is older than this many '
                         'ticks (for policies trained with --cloud-dropout)')
    ap.add_argument('--goal-switch', type=int, default=None,
                    help='nut_goal: flip the goal once at that tick (cancelled if the goal nut '
                         'is already lifted; default: no switch)')
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
    if args.backend == 'two_stream':
        import two_stream
        policy = two_stream.TwoStreamPolicy(args.weights)
        ol.CLOUD_LOCAL_REPLAN = args.replan
    elif args.backend == 'bev':
        import bev_planner
        policy = bev_planner.BEVPolicy(args.weights)
        ol.CLOUD_LOCAL_REPLAN = args.replan
        ol.CLOUD_LOCAL_DROP_AFTER = args.drop_after
    elif args.backend == 'two_stream_chunk':
        import two_stream
        policy = two_stream.TwoStreamChunkPolicy(args.weights)
    else:
        policy = make_policy(args.backend,
                             args.weights or os.path.join(_REPO, 'checkpoints', ckpt),
                             denoise_steps=args.denoise_steps)
    if not policy.absolute_actions:
        sys.exit('the mechanism executors shift absolute targets; this checkpoint emits deltas')
    tracker = ol.tracker_from(args)
    deadline = int(args.deadline_s * args.hz)
    needs_top = bool(getattr(policy, 'needs_top', False))
    cams = ('agentview', 'robot0_eye_in_hand') + (('birdview',) if needs_top else ())
    base = build_env(EnvSpec(env_name=env_name, action_hz=args.hz,
                             max_episode_s=args.deadline_s + 1.0, seed=args.seed,
                             absolute_actions=True, cameras=cams)).env
    if needs_top:
        base = TopCam(base)

    refine_fn = None
    if args.refiner:
        import refiner
        refine_fn = refiner.make_refine_fn(refiner.Refiner(args.refiner), object_tracker,
                                            args.refiner_lead)
        if args.refiner_log:
            refine_fn = refiner.logging_refine_fn(refine_fn, args.refiner_log)

    if args.task == 'square':
        import oracle_square as sq

        def held(o):
            return sq.is_held(o['robot0_eef_pos'], o['robot0_gripper_qpos'], o[pos_key],
                              o[quat_key])
    else:
        held = None                     # MovingCube's default: the Lift cube in the hand

    gate_fn = functools.partial(task_gate, task=args.task,
                                slack=args.gate_k * tracker.spec.sigma_mm * 1e-3)
    if args.fuzzy_gate:
        import fuzzy_gate
        gate_fn = fuzzy_gate.make_gate_fn(fuzzy_gate.FuzzyGate(args.fuzzy_gate), object_tracker)
    nutgoal = args.task == 'nut_goal'
    nutwatch = NutWatch() if nutgoal else None
    probers, traced, tracers = [], [], []

    def make_world(v):
        if nutgoal:
            # GoalWorld outermost: it owns the goal and its switch, and its goal_ keys follow the
            # CURRENT goal (so frames, anchoring and the gate read the goal nut). The conveyor
            # rides the episode's INITIAL goal nut; nut_reset (via run_sweep's reset_fn) points
            # it there each episode — the keys here only have to cover construction
            return nut_goal.GoalWorld(ol.MovingCube(
                base, v, args.hz, joint=f'{nut_goal.NUTS[0]}_joint0',
                pos_key=f'{nut_goal.NUTS[0]}_pos', half_range=half_range))
        world = ol.MovingCube(base, v, args.hz, joint=joint, pos_key=pos_key, held_fn=held,
                              half_range=half_range)
        if args.fail_log:
            import square_failures
            if tracers:
                tracers[-1].finish()               # the previous speed's last episode
            world = square_failures.Tracer(world, traced, pos_key, quat_key)
            tracers.append(world)
        if args.probe_log:
            import fuzzy_gate
            probers.append(fuzzy_gate.Prober(world, args.probe_log.replace('.npz', f'_{v}.npz')))
            return probers[-1]
        return world

    stack_fn = stack
    if args.label_log:
        import refiner
        stack_fn = refiner.ChunkLabeler(policy, stack, args.label_log)
    ol.run_sweep(args, policy=policy, tracker=tracker, deadline=deadline,
                 make_world=make_world,
                 frame_fn=make_frames(tracker, pos_key, quat_key, needs_top,
                                      goal=nutgoal, watch=nutwatch), stack_fn=stack_fn,
                 executor_kw={'roll_fn': functools.partial(proprio_roll, closed_qpos=closed,
                                                       lag=args.roll_lag),
                              'anchor_fn': object_tracker,
                              'gate_fn': gate_fn, 'refine_fn': refine_fn,
                              'lift_fn': SquareLift() if args.task in ('square', 'nut_goal')
                              else None},
                 is_success=(lambda e: e.success()) if nutgoal else None,
                 reset_fn=(lambda e, k: (nutwatch.start(k % 2),
                                         nut_reset(e, k, args.goal_switch))[1]) if nutgoal
                 else None,
                 columns=['goal', 'switched', 'wrong_nut'] if nutgoal else None,
                 row_fn=nutwatch.row if nutgoal else None)
    if args.refiner_log:
        refine_fn.save()
    if args.label_log:
        stack_fn.save()
    for p in probers:
        p.save()
    if args.fail_log:
        import square_failures
        tracers[-1].finish()
        square_failures.write(args.fail_log, args.out, traced)


if __name__ == '__main__':
    main()
