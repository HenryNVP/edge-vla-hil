"""Mechanism testbed, hard task: a scripted chunk policy on robosuite NutAssemblySquare.

Lift forgives a 2 cm error (the fingers open ~8 cm around a 4 cm cube), so it cannot tell a good
tracker or join from a mediocre one (docs/paper2-edge-plan.md, 5b). Square cannot be forgiving
like that. The nut's square hole (4.5 cm) clears the 3.2 cm peg by ±6.5 mm, the nut must be
turned to the peg's orientation to within a few degrees, and success also needs the gripper to
let go and move away. The handle grasp needs the fingers across a 3.2 cm bar.

Same contract as oracle_lift.py. The planner rolls a chunk forward from privileged state (EE
pose, gripper, nut pose) and is exactly as good as the observation it was planned from:

    approach   hover over the handle, turning the EE so the fingers close across the handle
    descend    to handle height, tracking the handle's xy
    close      close the gripper for `close_ticks`
    lift       to carry height
    transport  move so the nut's centre is over the peg, turning it to the peg's orientation
               (square symmetry: the nearest multiple of 90 degrees)
    lower      until the nut sits low on the peg
    release    open for `open_ticks`
    retreat    move up, so the success check sees the gripper away from the nut

The nut can ride the conveyor (oracle_lift.MovingCube with the nut's joint) until it is held.
Anchoring tracks the nut's position. The peg is fixed, so only the grasp needs anchoring.

    docker run --rm -v ~/edge-vla-hil:/ws edge-vla-hil:host \\
        python3 /ws/scripts/oracle_square.py --strategies synchronous,noskip_async \\
        --delays 1,10 --episodes 10 --out /ws/outputs/mech/sq_check.csv
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np

import oracle_lift as ol
from evh_plant.env_factory import EEF_TO_CONTROL_QUAT, eef_to_control_quat, quat_to_axisangle

PEG = np.array([0.23, 0.10, 0.85])           # peg1 (the square peg), world-axis aligned
PEG_TOP = 0.95                               # peg1 is a 0.2 m post centred at 0.85
NUT_HALF_H = 0.01
HANDLE_LOCAL = np.array([0.054, 0.0, 0.0])   # handle site in the nut frame
TABLE_Z = 0.82
APPROACH, DESCEND, CLOSE, LIFT, TRANSPORT, LOWER, RELEASE, RETREAT = (
    'approach', 'descend', 'close', 'lift', 'transport', 'lower', 'release', 'retreat')
CLOSED_PHASES = (CLOSE, LIFT, TRANSPORT, LOWER)


@dataclasses.dataclass(frozen=True)
class SquareConfig:
    chunk_size: int = 15
    step_m: float = 0.008          # EE travel per tick
    rot_step: float = 0.06         # EE yaw per tick, rad
    hover_m: float = 0.08
    grasp_dz: float = 0.0
    carry_z: float = 1.02          # EE height while carrying (nut clears the 0.95 m peg top)
    clear_m: float = 0.008         # nut bottom above the peg top to count as clear of it
    entering_tol: float = 0.012    # once the nut is around the peg, keep lowering within this
    seat_z: float = 0.835          # nut-centre height at which it counts as seated
    retreat_m: float = 0.12
    close_ticks: int = 6
    open_ticks: int = 5
    xy_tol: float = 0.004
    descend_xy_tol: float = 0.015
    grasp_xy_tol: float = 0.008
    align_tol: float = 0.003       # nut centre over the peg, to start lowering
    yaw_tol: float = 0.03
    z_tol: float = 0.006


# ------------------------------------------------------------------ geometry
def quat_to_mat(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_of(q) -> float:
    """Heading of an (upright) body's x axis about world z."""
    r = quat_to_mat(q)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def finger_yaw(eef_quat) -> float:
    """Heading of the closing axis: the EE frame's y axis (measured on the Panda gripper)."""
    r = quat_to_mat(eef_quat)
    return float(np.arctan2(r[1, 1], r[0, 1]))


def wrap(a: float, period: float = 2 * np.pi) -> float:
    """Wrap to (-period/2, period/2]."""
    return float(-((-a + period / 2) % period - period / 2))


def rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def handle_pos(nut_pos, nut_quat) -> np.ndarray:
    return np.asarray(nut_pos, float) + rz(yaw_of(nut_quat)) @ HANDLE_LOCAL


def grasp_turn(eef_quat, nut_quat) -> float:
    """EE yaw change that puts the closing axis along the nut's y axis (across the handle),
    the smaller of the two equivalent ways."""
    return wrap(yaw_of(nut_quat) + np.pi / 2 - finger_yaw(eef_quat), np.pi)


def seat_turn(nut_yaw: float) -> float:
    """Nut yaw change to the peg's orientation (square: modulo 90 degrees)."""
    return wrap(-nut_yaw, np.pi / 2)


def is_held(eef, grip, nut_pos, nut_quat) -> bool:
    width = float(np.sum(np.abs(np.asarray(grip))))
    return 0.015 < width < 0.06 and float(np.linalg.norm(handle_pos(nut_pos, nut_quat)
                                                          - np.asarray(eef))) < 0.02


def on_peg(nut_pos) -> bool:
    nut_pos = np.asarray(nut_pos)
    return bool(np.linalg.norm(nut_pos[:2] - PEG[:2]) < 0.02 and nut_pos[2] < TABLE_Z + 0.04)


# ------------------------------------------------------------------ planner
def unpack(s):
    """proprio layout: eef(3) eef_quat(4) grip(2) nut(3) nut_quat(4) tracked(3) ok(1)."""
    return s[0:3], s[3:7], s[7:9], s[9:12], s[12:16]


def start_phase(eef, eq, grip, nut, nq, cfg: SquareConfig) -> str:
    held = is_held(eef, grip, nut, nq)
    if on_peg(nut):
        return RELEASE if held else RETREAT
    if held:
        err = float(np.linalg.norm(nut[:2] - PEG[:2]))
        yaw_ok = abs(seat_turn(yaw_of(nq))) <= cfg.yaw_tol
        clear = nut[2] - NUT_HALF_H >= PEG_TOP + cfg.clear_m
        if yaw_ok and (err <= 2 * cfg.align_tol or (not clear and err <= cfg.entering_tol)):
            return LOWER
        if not clear:
            return LIFT            # misaligned below the peg top: rise before realigning
        return TRANSPORT           # realign at the current height (lift first if near the table)
    h = handle_pos(nut, nq)
    xy = float(np.linalg.norm(eef[:2] - h[:2]))
    dz = float(eef[2] - (h[2] + cfg.grasp_dz))
    turned = abs(grasp_turn(eq, nq)) <= cfg.yaw_tol
    if turned and xy <= cfg.grasp_xy_tol and abs(dz) <= cfg.z_tol:
        return CLOSE
    below_hover = dz < cfg.hover_m - cfg.grasp_dz + cfg.z_tol
    if dz > 0 and turned and (xy <= cfg.xy_tol or (below_hover and xy <= cfg.descend_xy_tol)):
        return DESCEND
    return APPROACH


def clear_eef_z(rel0, cfg: SquareConfig) -> float:
    """Lowest EE height at which the held nut clears the peg top."""
    return PEG_TOP + cfg.clear_m + NUT_HALF_H + 0.005 - float(rel0[2])


def _turn(current: float, target: float, step: float) -> float:
    d = wrap(target - current)
    return target if abs(d) <= step else current + np.sign(d) * step


def plan_chunk(eef, eq, grip, nut, nq, cfg: SquareConfig = SquareConfig()) -> np.ndarray:
    """H absolute targets [pos, axis-angle (controller frame), gripper] from the observation."""
    eef, nut = np.asarray(eef, float), np.asarray(nut, float)
    phase = start_phase(eef, eq, grip, nut, nq, cfg)
    q_ctrl0 = eef_to_control_quat(eq)
    p, turn = eef.copy(), 0.0                      # EE position, EE yaw change since the obs
    rel0 = nut - eef                               # nut centre relative to the EE (if held)
    nut_yaw0 = yaw_of(nq)
    h = handle_pos(nut, nq)
    grasp_target = grasp_turn(eq, nq)
    count = 0
    seat = None                                    # (EE yaw target, EE xy target) for transport
    out = np.zeros((cfg.chunk_size, 7), dtype=np.float32)
    for k in range(cfg.chunk_size):
        if phase == APPROACH:
            hover = h + [0.0, 0.0, cfg.hover_m]
            target = hover if p[2] >= hover[2] - cfg.z_tol else np.array([p[0], p[1], hover[2]])
            p = ol._move(p, target, cfg.step_m)
            turn = _turn(turn, grasp_target, cfg.rot_step)
            if (np.linalg.norm(p[:2] - h[:2]) <= cfg.xy_tol
                    and abs(wrap(grasp_target - turn)) <= cfg.yaw_tol):
                phase = DESCEND
        elif phase == DESCEND:
            p = ol._move(p, h + [0.0, 0.0, cfg.grasp_dz], cfg.step_m)
            if abs(p[2] - (h[2] + cfg.grasp_dz)) <= cfg.z_tol:
                phase, count = CLOSE, 0
        elif phase == CLOSE:
            count += 1
            if count >= cfg.close_ticks:
                phase = LIFT
        elif phase == LIFT:
            p = ol._move(p, np.array([p[0], p[1], cfg.carry_z]), cfg.step_m)
            if abs(p[2] - cfg.carry_z) <= cfg.z_tol:
                phase = TRANSPORT
        elif phase == TRANSPORT:
            z_t = max(p[2], clear_eef_z(rel0, cfg)) if p[2] < cfg.carry_z else cfg.carry_z
            if seat is None:
                dyaw = seat_turn(nut_yaw0 + turn)
                rel = rz(turn) @ rel0                    # nut offset now
                rel_final = rz(dyaw) @ rel               # ... after the seating turn
                seat = (turn + dyaw, PEG[:2] - rel_final[:2])
            p = ol._move(p, np.array([seat[1][0], seat[1][1], z_t]), cfg.step_m)
            turn = _turn(turn, seat[0], cfg.rot_step)
            if (np.linalg.norm(p[:2] - seat[1]) <= cfg.align_tol
                    and abs(wrap(seat[0] - turn)) <= cfg.yaw_tol):
                phase = LOWER
        elif phase == LOWER:
            ez = cfg.seat_z - rel0[2]
            p = ol._move(p, np.array([p[0], p[1], ez]), cfg.step_m)
            if abs(p[2] - ez) <= cfg.z_tol:
                phase, count = RELEASE, 0
        elif phase == RELEASE:
            count += 1
            if count >= cfg.open_ticks:
                phase = RETREAT
        else:  # RETREAT
            p = ol._move(p, np.array([p[0], p[1], max(p[2], eef[2]) + cfg.step_m]), cfg.step_m)
        q = ol_quat_mul(axisangle_z(turn), q_ctrl0)
        out[k, :3] = p
        out[k, 3:6] = quat_to_axisangle(q)
        out[k, 6] = ol.GRIP_CLOSE if phase in CLOSED_PHASES else ol.GRIP_OPEN
    return out


def axisangle_z(a: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(a / 2), np.cos(a / 2)])


def ol_quat_mul(q1, q2) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


class SquarePolicy:
    action_dim = 7
    n_obs_steps = 1
    needs_wrist = False
    absolute_actions = True
    guided_resampling = False

    def __init__(self, cfg: SquareConfig = SquareConfig()) -> None:
        self.cfg = cfg
        self.chunk_size = cfg.chunk_size

    def predict(self, obs: dict) -> np.ndarray:
        return plan_chunk(*unpack(np.asarray(obs['proprio'])[-1]), self.cfg)

    def predict_inpaint(self, obs, prefix, weights) -> np.ndarray:
        return self.predict(obs)


# ------------------------------------------------------------------ harness glue
def make_frames(tracker=None):
    def frames(obs: dict, needs_wrist: bool = False):
        nut = np.asarray(obs['SquareNut_pos'], np.float32)
        est = nut if tracker is None else tracker.update(nut)
        ok = est is not None
        proprio = np.concatenate([np.asarray(obs['robot0_eef_pos'], np.float32),
                                  np.asarray(obs['robot0_eef_quat'], np.float32),
                                  np.asarray(obs['robot0_gripper_qpos'], np.float32),
                                  nut, np.asarray(obs['SquareNut_quat'], np.float32),
                                  np.asarray(est if ok else nut, np.float32),
                                  np.float32([ok])])
        return None, None, proprio
    return frames


def nut_tracker(obs: dict):
    s = np.asarray(obs['proprio'])[-1]
    return s[16:19] if s[19] > 0.5 else None


def ctrl_to_eef_quat(q_ctrl) -> np.ndarray:
    """Inverse of eef_to_control_quat: q_eef = q_ctrl * conj(EEF_TO_CONTROL_QUAT)."""
    x, y, z, w = EEF_TO_CONTROL_QUAT
    return ol_quat_mul(q_ctrl, np.array([-x, -y, -z, w]))


def axisangle_to_quat(aa) -> np.ndarray:
    aa = np.asarray(aa, float)
    a = float(np.linalg.norm(aa))
    if a < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    ax = aa / a
    return np.concatenate([ax * np.sin(a / 2), [np.cos(a / 2)]])


def square_roll(obs: dict, queued: np.ndarray, cfg: SquareConfig = SquareConfig()) -> dict:
    """Roll the robot's part of the observation to the end of the queue. EE pose <- the last
    target. The nut follows the EE rigidly up to the last closed command if it is held now, or
    is predicted grasped where it lies if the queue ends with a completed close."""
    if len(queued) == 0:
        return obs
    s = np.asarray(obs['proprio'], np.float32).copy()
    last = s[-1]
    eef, eq, grip, nut, nq = (np.array(x, float) for x in unpack(last))
    held = is_held(eef, grip, nut, nq)
    tail_closed = 0
    for a in queued[::-1]:
        if a[6] <= 0:
            break
        tail_closed += 1
    end = queued[-1]
    end_eq = ctrl_to_eef_quat(axisangle_to_quat(end[3:6]))
    if held:
        closed_idx = [i for i, a in enumerate(queued) if a[6] > 0]
        j = closed_idx[-1] if closed_idx else None
        if j is not None and (not closed_idx or closed_idx[0] == 0):
            ref = queued[j]
            ref_eq = ctrl_to_eef_quat(axisangle_to_quat(ref[3:6]))
            dyaw = wrap(finger_yaw(ref_eq) - finger_yaw(eq))
            nut = ref[:3] + rz(dyaw) @ (nut - eef)
            nq = ol_quat_mul(axisangle_z(dyaw), nq)
    last[0:3] = end[:3]
    last[3:7] = end_eq
    # held at the queue's end if the queue closes long enough, or if it is held now and the
    # whole queue keeps it closed (a short just-in-time queue holds fewer than close_ticks)
    stays_held = held and tail_closed == len(queued)
    last[7:9] = ([0.016, -0.016] if tail_closed >= cfg.close_ticks or stays_held
                 else [0.04, -0.04])
    last[9:12] = nut
    last[12:16] = nq
    s[-1] = last
    return {**obs, 'proprio': s}


def square_diverged(obs: dict, recent: np.ndarray, cfg: SquareConfig = SquareConfig()) -> bool:
    eef, eq, grip, nut, nq = unpack(np.asarray(obs['proprio'])[-1])
    if on_peg(nut):
        return False
    return ol.grasp_failed(recent, is_held(eef, grip, nut, nq), cfg.close_ticks)


def make_env(control_hz: float, horizon: int):
    import robosuite as suite
    from evh_plant.env_factory import check_timebase, make_controller_config, set_control_delta
    controller = make_controller_config()
    set_control_delta(controller, False)
    env = suite.make(env_name='NutAssemblySquare', robots='Panda', has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False,
                     control_freq=control_hz, horizon=horizon, reward_shaping=False,
                     controller_configs=controller)
    check_timebase(control_hz, float(env.model_timestep))
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--strategies', default='synchronous,noskip_async,aligned_skip')
    ap.add_argument('--deadline-s', type=float, default=40.0)
    ol.add_common_args(ap)
    args = ap.parse_args()
    cfg = SquareConfig(chunk_size=args.chunk)
    tracker = ol.tracker_from(args)
    deadline = int(args.deadline_s * args.hz)
    base = make_env(args.hz, deadline + 10)

    def held(o):
        return is_held(o['robot0_eef_pos'], o['robot0_gripper_qpos'], o['SquareNut_pos'],
                       o['SquareNut_quat'])

    ol.run_sweep(args, policy=SquarePolicy(cfg), tracker=tracker, deadline=deadline,
                 make_world=lambda v: ol.MovingCube(base, v, args.hz, joint='SquareNut_joint0',
                                                    pos_key='SquareNut_pos', held_fn=held),
                 frame_fn=make_frames(tracker),
                 executor_kw={'roll_fn': square_roll, 'anchor_fn': nut_tracker,
                              'diverged_fn': square_diverged})


if __name__ == '__main__':
    main()
