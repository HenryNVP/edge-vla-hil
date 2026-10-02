"""Mechanism testbed: a scripted chunk policy on robosuite Lift, with an optional moving cube.

Paper 2 asks what a fast layer on the robot can do with a stale remote plan. Studying that with
the learned policy first would confound two things: whether a mechanism works, and whether the
policy copes with the states the mechanism puts it in. This module removes the second one. The
"policy" here plans from privileged state (EE pose, gripper, cube position) with a fixed speed
limit, so a chunk is exactly as good as the observation it was planned from. Every failure is
then the execution's, and the DP is brought back only to validate a mechanism that already works.

    plan_chunk(...)      the planner: H absolute EE targets toward approach -> descend -> close ->
                         lift, rolled out kinematically from the observed state. It uses the cube
                         position AT PLANNING TIME, so a stale chunk aims where the cube was.
    OraclePolicy         the ChunkPolicy face of it (deterministic, unguided), for the executors.
    advance(...)         the conveyor: a 1-D back-and-forth motion along one axis.
    MovingCube           env wrapper that moves the cube by `advance` every env step until the
                         cube is held or lifted, so "the world moves while the robot waits" is a
                         real, controlled quantity (paper 1's world speed, here in m/s).

Everything at module scope is numpy (the bare-box test suite imports it); robosuite is imported in
`make_env` only. Run the sweep inside the host image:

    docker run --rm -v ~/edge-vla-hil:/ws edge-vla-hil:host \\
        python3 /ws/scripts/oracle_lift.py --strategies synchronous,noskip_async \\
        --delays 1,10,20 --speeds 0,0.02 --episodes 20 --out /ws/outputs/mech/check.csv
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _pkg in ('evh_controller', 'evh_plant', 'evh_reactive'):
    _path = os.path.join(_REPO, 'ros2_ws', 'src', _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evh_plant.env_factory import eef_to_control_quat, quat_to_axisangle  # noqa: E402

GRIP_OPEN, GRIP_CLOSE = -1.0, 1.0
APPROACH, DESCEND, CLOSE, LIFT = 'approach', 'descend', 'close', 'lift'


@dataclasses.dataclass(frozen=True)
class OracleConfig:
    chunk_size: int = 15
    step_m: float = 0.008        # EE travel per control tick (0.16 m/s at 20 Hz)
    hover_m: float = 0.08        # approach height above the cube centre
    grasp_dz: float = 0.0        # grasp height relative to the cube centre
    close_ticks: int = 6         # ticks the gripper needs to close on the cube
    lift_m: float = 0.15         # lift target above the cube centre
    xy_tol: float = 0.006        # aligned enough above the cube to start descending
    descend_xy_tol: float = 0.02  # drift allowed during the descent before re-approaching
    grasp_xy_tol: float = 0.012  # aligned enough at grasp height to close
    z_tol: float = 0.006
    held_width: float = 0.07     # finger opening below which the cube counts as held
    held_dist: float = 0.03      # max EE-to-cube distance for a held cube


def _move(p: np.ndarray, target: np.ndarray, step: float) -> np.ndarray:
    d = target - p
    n = float(np.linalg.norm(d))
    return target.copy() if n <= step else p + d * (step / n)


def is_held(eef_pos, cube_pos, gripper_qpos, cfg: OracleConfig) -> bool:
    width = float(np.sum(np.abs(np.asarray(gripper_qpos))))
    dist = float(np.linalg.norm(np.asarray(cube_pos) - np.asarray(eef_pos)))
    return width < cfg.held_width and dist < cfg.held_dist


def start_phase(eef_pos, cube_pos, gripper_qpos, cfg: OracleConfig) -> str:
    """The phase the observed state is in; the plan rolls forward from here."""
    eef_pos, cube_pos = np.asarray(eef_pos), np.asarray(cube_pos)
    if is_held(eef_pos, cube_pos, gripper_qpos, cfg):
        return LIFT
    xy = float(np.linalg.norm(eef_pos[:2] - cube_pos[:2]))
    dz = float(eef_pos[2] - (cube_pos[2] + cfg.grasp_dz))
    if xy <= cfg.grasp_xy_tol and abs(dz) <= cfg.z_tol:
        return CLOSE
    below_hover = dz < cfg.hover_m - cfg.grasp_dz + cfg.z_tol
    if dz > 0 and (xy <= cfg.xy_tol or (below_hover and xy <= cfg.descend_xy_tol)):
        return DESCEND
    return APPROACH


def plan_chunk(eef_pos, cube_pos, gripper_qpos, rot_axisangle,
               cfg: OracleConfig = OracleConfig()) -> np.ndarray:
    """H absolute 7-dim targets [pos(3), axis-angle(3), gripper], rolled out from the observation.

    Orientation is held at the observed (controller-frame) value. The cube is assumed to stay
    where it was observed: that assumption is exactly what goes stale in a moving world.
    """
    p = np.asarray(eef_pos, dtype=float).copy()
    cube = np.asarray(cube_pos, dtype=float)
    phase = start_phase(p, cube, gripper_qpos, cfg)
    closed = 0
    out = np.zeros((cfg.chunk_size, 7), dtype=np.float32)
    for k in range(cfg.chunk_size):
        if phase == APPROACH:
            hover = cube + [0.0, 0.0, cfg.hover_m]
            # rise first if below the hover height and off to the side, so the fingers clear the cube
            target = hover if p[2] >= hover[2] - cfg.z_tol else np.array([p[0], p[1], hover[2]])
            p = _move(p, target, cfg.step_m)
            if np.linalg.norm(p[:2] - cube[:2]) <= cfg.xy_tol:
                phase = DESCEND
        elif phase == DESCEND:
            # track the cube's xy on the way down; this plan's cube is fixed, so a moving cube
            # only enters through the NEXT observation (or through anchoring)
            p = _move(p, cube + [0.0, 0.0, cfg.grasp_dz], cfg.step_m)
            if abs(p[2] - (cube[2] + cfg.grasp_dz)) <= cfg.z_tol:
                phase = CLOSE
        elif phase == CLOSE:
            closed += 1
            if closed >= cfg.close_ticks:
                phase = LIFT
        else:  # LIFT
            p = _move(p, np.array([p[0], p[1], cube[2] + cfg.lift_m]), cfg.step_m)
        out[k, :3] = p
        out[k, 3:6] = rot_axisangle
        out[k, 6] = GRIP_CLOSE if phase in (CLOSE, LIFT) else GRIP_OPEN
    return out


class OraclePolicy:
    """ChunkPolicy face of `plan_chunk`. obs['proprio'][-1] = [eef(3), quat(4), grip(2), cube(3)]."""
    action_dim = 7
    n_obs_steps = 1
    needs_wrist = False
    absolute_actions = True
    guided_resampling = False

    def __init__(self, cfg: OracleConfig = OracleConfig()) -> None:
        self.cfg = cfg
        self.chunk_size = cfg.chunk_size

    def predict(self, obs: dict) -> np.ndarray:
        s = np.asarray(obs['proprio'])[-1]
        rot = quat_to_axisangle(eef_to_control_quat(s[3:7]))
        return plan_chunk(s[0:3], s[9:12], s[7:9], rot, self.cfg)

    def predict_inpaint(self, obs, prefix, weights) -> np.ndarray:
        return self.predict(obs)          # deterministic: nothing to guide


def advance(pos: float, direction: float, step: float, lo: float, hi: float):
    """One conveyor step along an axis, reversing at the ends. Returns (pos, direction)."""
    nxt = pos + direction * step
    if nxt > hi:
        return hi - (nxt - hi), -1.0
    if nxt < lo:
        return lo + (lo - nxt), 1.0
    return nxt, direction


class MovingCube:
    """Moves the Lift cube along `axis` at `speed` m/s until it is held or lifted.

    The cube is placed kinematically each step (free joint qpos set, velocity zeroed), so its
    motion does not depend on friction or on the controller. speed = 0 is plain Lift.
    """

    def __init__(self, env, speed: float, control_hz: float, axis: int = 1,
                 half_range: float = 0.12, cfg: OracleConfig = OracleConfig()) -> None:
        self.env, self.speed, self.hz, self.axis = env, speed, control_hz, axis
        self.half_range, self.cfg = half_range, cfg
        jnt = 'cube_joint0'
        self._qadr = env.sim.model.get_joint_qpos_addr(jnt)[0]
        self._vadr = env.sim.model.get_joint_qvel_addr(jnt)[0]
        self._obs = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs = self.env.reset()
        self._start(obs)
        return obs

    def _start(self, obs) -> None:
        self._obs = obs
        self._z0 = float(obs['cube_pos'][2])
        c = float(obs['cube_pos'][self.axis])
        self._lo, self._hi = c - self.half_range, c + self.half_range
        self._dir = 1.0 if np.random.rand() < 0.5 else -1.0

    def _moving(self) -> bool:
        o = self._obs
        if self.speed <= 0 or o is None:
            return False
        if float(o['cube_pos'][2]) > self._z0 + 0.005:
            return False
        return not is_held(o['robot0_eef_pos'], o['cube_pos'], o['robot0_gripper_qpos'], self.cfg)

    def step(self, action):
        if self._moving():
            sim = self.env.sim
            i = self._qadr + self.axis
            sim.data.qpos[i], self._dir = advance(float(sim.data.qpos[i]), self._dir,
                                                  self.speed / self.hz, self._lo, self._hi)
            sim.data.qvel[self._vadr:self._vadr + 6] = 0.0
            sim.forward()
        out = self.env.step(action)
        self._obs = out[0]
        return out


# ------------------------------------------------------------------ tracker model
@dataclasses.dataclass(frozen=True)
class TrackerSpec:
    """How the robot's measurement of the cube degrades. All zero = the privileged tracker.

    sigma_mm   isotropic Gaussian noise per measurement
    latency    ticks between the scene and the measurement reaching the robot
    dropout    long-run fraction of ticks with no measurement (occlusion, detector miss)
    burst      mean length of a dropout, ticks (1 = independent misses)
    filtered   pass measurements through the alpha-beta filter with delay compensation
               (evh_reactive.object_filter); otherwise report the last measurement as is
    """
    sigma_mm: float = 0.0
    latency: int = 0
    dropout: float = 0.0
    burst: int = 1
    filtered: bool = False
    alpha: float = 0.5
    beta: float = 0.1

    @property
    def tag(self) -> str:
        return (f's{self.sigma_mm:g}_l{self.latency}_p{self.dropout:g}_b{self.burst}'
                f'{"_ab" if self.filtered else ""}')


class TrackerModel:
    """Turns the true cube position into what a tracker on the robot would report, one call per
    env tick. Its own RNG, seeded per episode, so the noise does not perturb scene generation."""

    def __init__(self, spec: TrackerSpec) -> None:
        from evh_reactive.object_filter import AlphaBetaFilter
        self.spec = spec
        self.filter = AlphaBetaFilter(spec.alpha, spec.beta, latency=spec.latency)
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(10_000 + seed)
        self._pipe: list = []          # measurements in flight (latency)
        self._out = 0                  # ticks left in the current dropout
        self._last = None
        self.filter.reset()

    def _measure(self, true_pos) -> np.ndarray | None:
        sp = self.spec
        if self._out > 0:
            self._out -= 1
            return None
        if sp.dropout > 0:
            start = sp.dropout / (sp.burst * (1.0 - sp.dropout) + sp.dropout)
            if self.rng.random() < start:
                self._out = max(sp.burst - 1, 0)
                return None
        return np.asarray(true_pos, float) + self.rng.normal(0.0, sp.sigma_mm * 1e-3, 3)

    def update(self, true_pos) -> np.ndarray | None:
        self._pipe.append(self._measure(true_pos))
        z = self._pipe.pop(0) if len(self._pipe) > self.spec.latency else None
        if self.spec.filtered:
            return self.filter.update(z)
        if z is not None:
            self._last = z
        return self._last


# ------------------------------------------------------------------ harness glue
def make_frames(tracker: TrackerModel | None = None):
    """frame_fn for run_episode. proprio = [eef(3), quat(4), grip(2), cube(3), tracked(3), ok(1)]:
    the planner reads the true cube (planning quality is not the variable); anchoring reads the
    tracked one."""
    def frames(obs: dict, needs_wrist: bool = False):
        cube = np.asarray(obs['cube_pos'], np.float32)
        est = cube if tracker is None else tracker.update(cube)
        ok = est is not None
        tracked = np.asarray(est if ok else cube, np.float32)
        proprio = np.concatenate([np.asarray(obs['robot0_eef_pos'], np.float32),
                                  np.asarray(obs['robot0_eef_quat'], np.float32),
                                  np.asarray(obs['robot0_gripper_qpos'], np.float32),
                                  cube, tracked, np.float32([ok])])
        return None, None, proprio
    return frames


frames = make_frames()


def stack(hist, needs_wrist: bool = False) -> dict:
    return {'proprio': np.stack([h[2] for h in hist])}


def make_env(control_hz: float, horizon: int):
    import robosuite as suite
    from evh_plant.env_factory import check_timebase, make_controller_config, set_control_delta
    controller = make_controller_config()
    set_control_delta(controller, False)
    env = suite.make(env_name='Lift', robots='Panda', has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False, control_freq=control_hz,
                     horizon=horizon, reward_shaping=False, controller_configs=controller)
    check_timebase(control_hz, float(env.model_timestep))
    return env


def make_executor_for(name: str, worker, policy):
    """The repo's executors, plus `aligned_skip`: time-aligned, unguided, tick-based join (the
    skipping baseline paper 1 identifies; RTC's join without its guidance)."""
    from evh_controller.chunk_executor import (AnchoredExecutor, CoverageAwareExecutor,
                                               make_executor)
    if name == 'aligned_skip':
        return CoverageAwareExecutor(worker, policy, rule='aligned', guided=False, reindex=False)
    if name.startswith('continuation'):            # M3: plan from the queue's end, append
        from evh_controller.chunk_executor.continuation import ContinuationExecutor
        roll = None if name == 'continuation_noroll' else oracle_roll
        anchor = cube_tracker if name == 'continuation_anchored' else None
        return ContinuationExecutor(worker, policy, roll_fn=roll, anchor_fn=anchor)
    if name in ('anchored', 'anchored_wait'):      # M1, with the privileged tracker
        rule = 'waiting' if name == 'anchored_wait' else 'aligned'
        return AnchoredExecutor(worker, policy, anchor_fn=cube_tracker, rule=rule)
    return make_executor(name, worker, policy)


def oracle_roll(obs: dict, queued: np.ndarray, cfg: OracleConfig = OracleConfig()) -> dict:
    """Roll the robot's part of the observation to the end of the queued actions.

    EE position <- the last queued target. Gripper and cube follow the plan's grasp: once the
    queue ends with at least `close_ticks` closed commands the cube is predicted to be in the
    hand (so it moves with the EE). Otherwise the fingers are predicted open and the cube stays
    where it was observed. This is a privileged model of the grasp, legitimate in a mechanism
    testbed; for DP only the proprio is rolled (paper 2, M3).
    """
    if len(queued) == 0:
        return obs
    s = np.asarray(obs['proprio'], dtype=np.float32).copy()
    last = s[-1]
    tail_closed = 0
    for a in queued[::-1]:
        if a[6] <= 0:
            break
        tail_closed += 1
    last[0:3] = queued[-1][:3]
    if tail_closed >= cfg.close_ticks:
        last[7:9] = [0.02, -0.02]
        last[9:12] = queued[-1][:3]
    else:
        last[7:9] = [0.04, -0.04]
    s[-1] = last
    return {**obs, 'proprio': s}


def cube_tracker(obs: dict) -> np.ndarray | None:
    """The robot's tracked cube position, appended to proprio by the frame function (the true
    position with the default, privileged tracker). None while the tracker has no fix yet."""
    s = np.asarray(obs['proprio'])[-1]
    return s[12:15] if s[15] > 0.5 else None


COLUMNS = ['strategy', 'delay', 'speed', 'world_speed', 'tracker', 'scene', 'success', 'ticks',
           'idle_frac', 'requests', 'path_mm']


def main() -> None:
    from evh_plant.env_factory import hold_action
    from sil_step import StepWorker, fixed_delay, run_episode

    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--strategies', default='synchronous,noskip_async,aligned_skip')
    ap.add_argument('--delays', default='1,5,10,15,20,30')
    ap.add_argument('--speeds', default='0', help='cube speeds, m/s')
    ap.add_argument('--world-speed', type=float, default=1.0)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--deadline-s', type=float, default=30.0)
    ap.add_argument('--hz', type=float, default=20.0)
    ap.add_argument('--chunk', type=int, default=15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tracker-sigma-mm', type=float, default=0.0)
    ap.add_argument('--tracker-latency', type=int, default=0)
    ap.add_argument('--tracker-dropout', type=float, default=0.0)
    ap.add_argument('--tracker-burst', type=int, default=1)
    ap.add_argument('--tracker-filter', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = OracleConfig(chunk_size=args.chunk)
    policy = OraclePolicy(cfg)
    spec = TrackerSpec(args.tracker_sigma_mm, args.tracker_latency, args.tracker_dropout,
                       args.tracker_burst, args.tracker_filter)
    tracker = TrackerModel(spec)
    frame_fn = make_frames(tracker)
    deadline = int(args.deadline_s * args.hz)
    base = make_env(args.hz, deadline + 10)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for speed in [float(x) for x in args.speeds.split(',')]:
            env = MovingCube(base, speed, args.hz, cfg=cfg)
            for d in [int(x) for x in args.delays.split(',')]:
                for strat in args.strategies.split(','):
                    ok = 0
                    for k in range(args.episodes):
                        np.random.seed(args.seed + k)      # scene k is the same in every cell
                        obs = env.reset()
                        tracker.reset(args.seed + k)
                        worker = StepWorker(policy, fixed_delay(d))
                        ex = make_executor_for(strat, worker, policy)
                        ep = run_episode(env, obs, policy, ex, worker, deadline_ticks=deadline,
                                         world_speed=args.world_speed, absolute=True,
                                         frame_fn=frame_fn, stack_fn=stack,
                                         hold_fn=lambda o: hold_action(o, True, 7))
                        ok += ep.success
                        w.writerow({'strategy': strat, 'delay': d, 'speed': speed,
                                    'world_speed': args.world_speed, 'tracker': spec.tag,
                                    'scene': k,
                                    'success': int(ep.success), 'ticks': ep.ticks,
                                    'idle_frac': round(ep.idle / max(ep.ticks, 1), 4),
                                    'requests': ep.requests, 'path_mm': round(ep.path_m * 1e3, 1)})
                    fh.flush()
                    print(f'{spec.tag} speed={speed} d={d} {strat}: {ok}/{args.episodes}', flush=True)


if __name__ == '__main__':
    main()
