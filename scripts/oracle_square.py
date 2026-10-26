"""Mechanism testbed, hard task: a scripted chunk policy on robosuite NutAssemblySquare.

Lift forgives a 2 cm error (the fingers open ~8 cm around a 4 cm cube), so it cannot tell a good
tracker or join from a mediocre one (docs/paper2-edge-plan.md, 5b). Square cannot be forgiving
like that. The nut's square hole (4.5 cm) clears the 3.2 cm peg by ±6.5 mm, the nut must be
turned to the peg's orientation to within a few degrees, and success also needs the gripper to
let go and move away. The handle grasp needs the fingers across a 3.2 cm bar.

The planner is per-nut, not square-only: a NutSpec carries what differs between the two
robosuite nuts (obs keys, handle offset, target peg, placement yaw) and every planning function
takes one, defaulting to the square nut so the Square testbed is bit-for-bit unchanged. The
round nut (nut_goal.py's goal 1) seats on the round peg at any yaw — its hole (inradius 0.031)
clears the 0.02 radius peg with 11 mm to spare — so its only yaw rule is reach: turn just
enough that the EE target stays inside the arm's workspace. Planning on nut_goal follows
GoalWorld's goal (goal_nut_pos / goal_nut_quat, spec=None in make_frames): the planner is
stateless, so a mid-episode goal switch re-targets at the next replan — it opens the gripper
(not held on the new nut) and approaches the new nut's handle.

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
import functools

import numpy as np

import nut_goal as ng
import oracle_lift as ol
from evh_plant.env_factory import EEF_TO_CONTROL_QUAT, eef_to_control_quat, quat_to_axisangle


# ------------------------------------------------------------------ per-nut geometry
@dataclasses.dataclass(frozen=True, eq=False)
class NutSpec:
    """One nut's geometry and placement rule — everything that differs between the two
    robosuite nuts (models/assets/objects/{square,round}-nut.xml). The planner is otherwise
    identical for both: same phases, speeds and tolerances, because the nuts differ only in
    where the handle sits, which peg they go on, and whether their yaw must meet the peg.

    name          robosuite object name: obs keys '<name>_pos' / '<name>_quat', joint
                  '<name>_joint0'
    handle_local  handle site in the nut frame (square 0.054, round 0.06, along x)
    peg_xy        the peg this nut goes on; nut_goal.PEGS_XY owns the coordinates
    peg_top       top of the peg post: both posts are 0.2 m centred at z 0.85 -> 0.95
    yaw_period    the placement yaw must match the peg modulo this period: pi/2 for the square
                  hole over the square peg; 2*pi for the round nut (no yaw constraint)
    grasp_bias    added to cfg.grasp_dz for this nut's handle grasp. The round nut needs the EE
                  biased below the handle site: near the table the OSC settles ~1 cm above its
                  target, the bar then rides the fingertips and squirts out on lift — seen as an
                  endless descend/close/reopen loop after a goal switch re-target (seed 5000:
                  never at 0, first-try at -6 mm). 0 for the square nut, whose grasp was tuned
                  as is.
    """
    name: str
    handle_local: np.ndarray
    peg_xy: np.ndarray
    peg_top: float
    yaw_period: float
    grasp_bias: float = 0.0

    def grasp_dz(self, cfg: 'SquareConfig') -> float:
        """Height of the EE below the handle site when grasping, this nut's bias included."""
        return cfg.grasp_dz + self.grasp_bias


SQUARE = NutSpec('SquareNut', np.array([0.054, 0.0, 0.0]), np.asarray(ng.PEGS_XY[0], float),
                 0.95, np.pi / 2)
ROUND = NutSpec('RoundNut', np.array([0.06, 0.0, 0.0]), np.asarray(ng.PEGS_XY[1], float),
                0.95, 2 * np.pi, -0.006)
NUT_SPECS = (SQUARE, ROUND)
assert tuple(s.name for s in NUT_SPECS) == ng.NUTS      # spec i must be the nut goal i names


def spec_for(goal: int) -> NutSpec:
    """The nut a nut_goal goal id names (goal 0 = square -> peg1, goal 1 = round -> peg2)."""
    return NUT_SPECS[goal]


PEG = np.array([SQUARE.peg_xy[0], SQUARE.peg_xy[1], 0.85])   # peg1 (the square peg), world-axis aligned
PEG_TOP = SQUARE.peg_top                     # peg1 is a 0.2 m post centred at 0.85
NUT_HALF_H = 0.01
HANDLE_LOCAL = SQUARE.handle_local           # handle site in the nut frame
TABLE_Z = 0.82
ROBOT_BASE_XY = np.array([-0.56, 0.0])       # Panda base on the robosuite table
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
    empty_width: float = 0.012     # finger opening below which the gripper closed on nothing
    seat_rule: str = 'nearest'     # 'nearest' quarter turn, or 'reach' (tested: worse, see seat_choice)
    reach_max: float = 0.78        # EE distance from the base beyond which 'reach' turns once more


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


def handle_pos(nut_pos, nut_quat, spec: NutSpec = SQUARE) -> np.ndarray:
    return np.asarray(nut_pos, float) + rz(yaw_of(nut_quat)) @ spec.handle_local


def grasp_turn(eef_quat, nut_quat) -> float:
    """EE yaw change that puts the closing axis along the nut's y axis (across the handle),
    the smaller of the two equivalent ways."""
    return wrap(yaw_of(nut_quat) + np.pi / 2 - finger_yaw(eef_quat), np.pi)


def seat_turn(nut_yaw: float) -> float:
    """Nut yaw change to the peg's orientation (square: modulo 90 degrees)."""
    return wrap(-nut_yaw, np.pi / 2)


def yaw_err(nut_yaw: float, spec: NutSpec = SQUARE) -> float:
    """How far the nut's yaw is from a placement the peg accepts: the square hole must meet the
    square peg modulo 90 degrees; the round nut's hole clears the round peg at any yaw."""
    if spec.yaw_period >= 2 * np.pi:
        return 0.0
    return abs(wrap(-nut_yaw, spec.yaw_period))


def free_seat_turn(rel_xy, peg_xy, reach_max: float) -> tuple[float, np.ndarray]:
    """(nut yaw change, EE xy target) for seating a nut with no yaw constraint (the round nut).

    Any yaw seats, so the only reason to turn is reach: past ~0.84 m from the base the OSC
    settles centimetres short and the nut jams at the peg's mouth — the same fact that gave the
    square nut its 'reach' rule. The EE target is |peg_xy - rz(dyaw) @ rel| from the base, so
    the reachable dyaws form one contiguous cone around the base->peg line; take the least turn
    into it, and none when the current yaw already seats within reach."""
    rel = np.asarray(rel_xy, float)
    R = np.asarray(peg_xy, float) - ROBOT_BASE_XY
    r, n = float(np.linalg.norm(rel)), float(np.linalg.norm(R))
    if r < 1e-9 or n < 1e-9:
        return 0.0, np.asarray(peg_xy, float) - rel
    c = (r * r + n * n - reach_max * reach_max) / (2 * r * n)   # law of cosines: the cone edge
    centre = wrap(np.arctan2(R[1], R[0]) - np.arctan2(rel[1], rel[0]))
    if c <= -1:
        dyaw = 0.0                # every yaw seats within reach
    elif c >= 1:
        dyaw = centre             # unreachable even aligned: best effort, EE nearest the base
    else:
        phi = float(np.arccos(c))
        # centre is wrapped to (-pi, pi]; with |centre| <= pi, sign(centre)*(|centre| - phi)
        # is the cone endpoint nearest zero (the other endpoint is 2*pi further around)
        dyaw = 0.0 if abs(centre) <= phi else np.sign(centre) * (abs(centre) - phi)
    return dyaw, np.asarray(peg_xy, float) - (rz(dyaw) @ np.array([rel[0], rel[1], 0.0]))[:2]


def seat_choice(nut_yaw: float, rel_xy, rule: str = 'reach',
                reach_max: float = 0.78, spec: NutSpec = SQUARE) -> tuple[float, np.ndarray]:
    """(nut yaw change, EE xy target) for seating; `rel_xy` = nut centre minus EE, now.

    'nearest' takes the smallest quarter turn. That can point the handle away from the robot
    and ask for a reach at the edge of the Panda's workspace (~0.84 m from the base), where the
    OSC settles centimetres short and the nut jams. 'reach' keeps the smallest turn unless its EE
    target is beyond `reach_max`, and then takes the smallest turn that is within reach. Large
    turns (up to 180 degrees) were tried and are worse: they drive the wrist toward its joint
    limit while carrying the nut.

    The round nut has no yaw to match, so its options are continuous, not four: free_seat_turn
    takes the least turn that restores reach (none when already within it).
    """
    if spec.yaw_period >= 2 * np.pi:
        return free_seat_turn(rel_xy, spec.peg_xy, reach_max)
    base = seat_turn(nut_yaw)
    options = []
    for k in (-2, -1, 0, 1):
        dyaw = base + k * np.pi / 2
        target = spec.peg_xy - (rz(dyaw) @ np.array([rel_xy[0], rel_xy[1], 0.0]))[:2]
        options.append((abs(dyaw), float(np.linalg.norm(target - ROBOT_BASE_XY)), dyaw, target))
    options.sort(key=lambda o: o[0])
    if rule == 'nearest':
        return options[0][2], options[0][3]
    for o in options:
        if o[1] <= reach_max:
            return o[2], o[3]
    best = min(options, key=lambda o: o[1])
    return best[2], best[3]


def is_held(eef, grip, nut_pos, nut_quat, spec: NutSpec = SQUARE) -> bool:
    width = float(np.sum(np.abs(np.asarray(grip))))
    return 0.015 < width < 0.06 and float(np.linalg.norm(handle_pos(nut_pos, nut_quat, spec)
                                                          - np.asarray(eef))) < 0.02


def on_peg(nut_pos, spec: NutSpec = SQUARE) -> bool:
    nut_pos = np.asarray(nut_pos)
    return bool(np.linalg.norm(nut_pos[:2] - spec.peg_xy) < 0.02 and nut_pos[2] < TABLE_Z + 0.04)


# ------------------------------------------------------------------ planner
def unpack(s):
    """proprio layout: eef(3) eef_quat(4) grip(2) nut(3) nut_quat(4) tracked(3) ok(1)."""
    return s[0:3], s[3:7], s[7:9], s[9:12], s[12:16]


def start_phase(eef, eq, grip, nut, nq, cfg: SquareConfig, spec: NutSpec = SQUARE) -> str:
    held = is_held(eef, grip, nut, nq, spec)
    if on_peg(nut, spec):
        return RELEASE if held else RETREAT
    if held:
        err = float(np.linalg.norm(nut[:2] - spec.peg_xy))
        yaw_ok = yaw_err(yaw_of(nq), spec) <= cfg.yaw_tol
        clear = nut[2] - NUT_HALF_H >= spec.peg_top + cfg.clear_m
        if yaw_ok and (err <= 2 * cfg.align_tol or (not clear and err <= cfg.entering_tol)):
            return LOWER
        if not clear:
            return LIFT            # misaligned below the peg top: rise before realigning
        return TRANSPORT           # realign at the current height (lift first if near the table)
    if float(np.sum(np.abs(np.asarray(grip)))) < cfg.empty_width:
        return APPROACH            # closed on nothing (or on the nut a goal switch abandoned):
                                 # reopen and rise before trying again
    h = handle_pos(nut, nq, spec)
    gdz = spec.grasp_dz(cfg)
    xy = float(np.linalg.norm(eef[:2] - h[:2]))
    dz = float(eef[2] - (h[2] + gdz))
    turned = abs(grasp_turn(eq, nq)) <= cfg.yaw_tol
    if turned and xy <= cfg.grasp_xy_tol and abs(dz) <= cfg.z_tol:
        return CLOSE
    below_hover = dz < cfg.hover_m - gdz + cfg.z_tol
    if dz > 0 and turned and (xy <= cfg.xy_tol or (below_hover and xy <= cfg.descend_xy_tol)):
        return DESCEND
    return APPROACH


def clear_eef_z(rel0, cfg: SquareConfig, spec: NutSpec = SQUARE) -> float:
    """Lowest EE height at which the held nut clears the peg top."""
    return spec.peg_top + cfg.clear_m + NUT_HALF_H + 0.005 - float(rel0[2])


def _turn(current: float, target: float, step: float) -> float:
    d = wrap(target - current)
    return target if abs(d) <= step else current + np.sign(d) * step


def plan_chunk(eef, eq, grip, nut, nq, cfg: SquareConfig = SquareConfig(),
               spec: NutSpec = SQUARE) -> np.ndarray:
    """H absolute targets [pos, axis-angle (controller frame), gripper] from the observation."""
    eef, nut = np.asarray(eef, float), np.asarray(nut, float)
    phase = start_phase(eef, eq, grip, nut, nq, cfg, spec)
    q_ctrl0 = eef_to_control_quat(eq)
    p, turn = eef.copy(), 0.0                      # EE position, EE yaw change since the obs
    rel0 = nut - eef                               # nut centre relative to the EE (if held)
    nut_yaw0 = yaw_of(nq)
    h = handle_pos(nut, nq, spec)
    gdz = spec.grasp_dz(cfg)
    grasp_target = grasp_turn(eq, nq)
    count = 0
    seat = None                                    # (EE yaw target, EE xy target) for transport
    out = np.zeros((cfg.chunk_size, 7), dtype=np.float32)
    for k in range(cfg.chunk_size):
        if phase == APPROACH:
            # (start_phase sends a gripper that closed on nothing here: the open command below
            # releases it, and the rise to hover gives the fingers time to open)
            hover = h + [0.0, 0.0, cfg.hover_m]
            target = hover if p[2] >= hover[2] - cfg.z_tol else np.array([p[0], p[1], hover[2]])
            p = ol._move(p, target, cfg.step_m)
            turn = _turn(turn, grasp_target, cfg.rot_step)
            if (np.linalg.norm(p[:2] - h[:2]) <= cfg.xy_tol
                    and abs(wrap(grasp_target - turn)) <= cfg.yaw_tol):
                phase = DESCEND
        elif phase == DESCEND:
            p = ol._move(p, h + [0.0, 0.0, gdz], cfg.step_m)
            if abs(p[2] - (h[2] + gdz)) <= cfg.z_tol:
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
            z_t = max(p[2], clear_eef_z(rel0, cfg, spec)) if p[2] < cfg.carry_z else cfg.carry_z
            if seat is None:
                rel = rz(turn) @ rel0                    # nut offset now
                dyaw, target = seat_choice(nut_yaw0 + turn, rel[:2], cfg.seat_rule,
                                           cfg.reach_max, spec)
                seat = (turn + dyaw, target)
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

    def __init__(self, cfg: SquareConfig = SquareConfig(), spec: NutSpec = SQUARE) -> None:
        self.cfg = cfg
        self.spec = spec              # nut_goal re-targets by setting this at each replan
        self.chunk_size = cfg.chunk_size

    def predict(self, obs: dict) -> np.ndarray:
        return plan_chunk(*unpack(np.asarray(obs['proprio'])[-1]), self.cfg, self.spec)

    def predict_inpaint(self, obs, prefix, weights) -> np.ndarray:
        return self.predict(obs)


# ------------------------------------------------------------------ harness glue
def make_frames(tracker=None, spec: NutSpec | None = SQUARE):
    """frame_fn for run_episode. proprio = [eef(3), quat(4), grip(2), nut(3), quat(4),
    tracked(3), ok(1)]; `spec` picks the nut through its obs keys. spec=None follows the goal
    instead — the GoalWorld keys goal_nut_pos / goal_nut_quat — so the same frames track
    whichever nut the current goal names and a goal switch re-targets at the next replan (the
    planning spec is picked the same way, from obs['goal']).
    """
    pos_key, quat_key = ((f'{spec.name}_pos', f'{spec.name}_quat') if spec is not None
                         else ('goal_nut_pos', 'goal_nut_quat'))

    def frames(obs: dict, needs_wrist: bool = False):
        nut = np.asarray(obs[pos_key], np.float32)
        est = nut if tracker is None else tracker.update(nut)
        ok = est is not None
        proprio = np.concatenate([np.asarray(obs['robot0_eef_pos'], np.float32),
                                  np.asarray(obs['robot0_eef_quat'], np.float32),
                                  np.asarray(obs['robot0_gripper_qpos'], np.float32),
                                  nut, np.asarray(obs[quat_key], np.float32),
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


def square_roll(obs: dict, queued: np.ndarray, cfg: SquareConfig = SquareConfig(),
                spec: NutSpec = SQUARE) -> dict:
    """Roll the robot's part of the observation to the end of the queue. EE pose <- the last
    target. The nut follows the EE rigidly up to the last closed command if it is held now, or
    is predicted grasped where it lies if the queue ends with a completed close."""
    if len(queued) == 0:
        return obs
    s = np.asarray(obs['proprio'], np.float32).copy()
    last = s[-1]
    eef, eq, grip, nut, nq = (np.array(x, float) for x in unpack(last))
    held = is_held(eef, grip, nut, nq, spec)
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


def square_gate(obs: dict, action: np.ndarray, cfg: SquareConfig = SquareConfig(),
                slack: float = 0.0, spec: NutSpec = SQUARE) -> bool:
    """Close only when the measured EE is at the handle of the TRACKED nut, turned across it.
    `slack` (m) widens the position tolerances for a noisy tracker: with 10 mm of noise a fixed
    8 mm check almost never passes and every close waits out the gate's timeout (0/20 against
    7/20 ungated). The sweep sets it to k * sigma, so a tracker too noisy to verify the grasp
    degrades the gate toward no gate instead of below it."""
    eef, eq, grip, nut, nq = unpack(np.asarray(obs['proprio'])[-1])
    tracked = nut_tracker(obs)
    if tracked is None:
        return True
    h = handle_pos(tracked, nq, spec)
    return (float(np.linalg.norm(eef[:2] - h[:2])) <= cfg.grasp_xy_tol + slack
            and abs(float(eef[2] - h[2] - spec.grasp_dz(cfg))) <= cfg.z_tol + slack
            and abs(grasp_turn(eq, nq)) <= 2 * cfg.yaw_tol)


def square_diverged(obs: dict, recent: np.ndarray, cfg: SquareConfig = SquareConfig(),
                    spec: NutSpec = SQUARE) -> bool:
    eef, eq, grip, nut, nq = unpack(np.asarray(obs['proprio'])[-1])
    if on_peg(nut, spec):
        return False
    return ol.grasp_failed(recent, is_held(eef, grip, nut, nq, spec), cfg.close_ticks)


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
    ap.add_argument('--seat-rule', default='nearest', choices=['reach', 'nearest'])
    ap.add_argument('--gate-k', type=float, default=2.0,
                    help='gate position slack in tracker sigmas (0 = fixed tolerance)')
    ap.add_argument('--conveyor-range', type=float, default=0.06,
                    help='half range of the nut conveyor, m (keeps the nut in the task workspace)')
    ap.add_argument('--refiner', default='', help="refiner weights (refiner.py) for 'refined'")
    ap.add_argument('--refiner-lead', type=int, default=0, help='predicted entry whose position is played')
    ol.add_common_args(ap)
    args = ap.parse_args()
    cfg = SquareConfig(chunk_size=args.chunk, seat_rule=args.seat_rule)
    tracker = ol.tracker_from(args)
    deadline = int(args.deadline_s * args.hz)
    base = make_env(args.hz, deadline + 10)
    refine_fn = None
    if args.refiner:
        import refiner
        refine_fn = refiner.make_refine_fn(refiner.Refiner(args.refiner), nut_tracker,
                                            args.refiner_lead)

    def held(o):
        return is_held(o['robot0_eef_pos'], o['robot0_gripper_qpos'], o['SquareNut_pos'],
                       o['SquareNut_quat'])

    ol.run_sweep(args, policy=SquarePolicy(cfg), tracker=tracker, deadline=deadline,
                 make_world=lambda v: ol.MovingCube(base, v, args.hz, joint='SquareNut_joint0',
                                                    pos_key='SquareNut_pos', held_fn=held,
                                                    half_range=args.conveyor_range),
                 frame_fn=make_frames(tracker),
                 executor_kw={'roll_fn': square_roll, 'anchor_fn': nut_tracker,
                              'diverged_fn': square_diverged,
                              'gate_fn': functools.partial(
                                  square_gate, slack=args.gate_k * tracker.spec.sigma_mm * 1e-3),
                              'refine_fn': refine_fn})


if __name__ == '__main__':
    main()
