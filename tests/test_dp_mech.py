"""Tests for the DP mechanism testbed's glue — numpy only, no robosuite or torch.

Pinned: the proprio roll writes the queued plan's last targets into the history (newest last),
in the eef-quat frame (the controller-frame offset undone, invariant 7), and closes the fingers
only once the plan closes them; the tracked object rides outside the policy's keys. nut_goal:
the goal schedule (k % 2) and the per-episode conveyor retarget, obs['goal'] through
make_frames/stack, the wrong_nut bookkeeping, and the wiring through oracle_lift.run_sweep
(success = GoalWorld's goal-nut check, not robosuite's any-nut one).
"""
import numpy as np

import dp_mech
from evh_plant.env_factory import eef_to_control_quat, quat_to_axisangle

DOWN = np.array([1.0, 0.0, 0.0, 0.0])


def _obs(width=0.04):
    s = np.tile(np.concatenate([[0.0, 0.0, 1.0], DOWN, [width, -width]]), (2, 1))
    return {'proprio': s.astype(np.float32), 'object': np.zeros(11, np.float32)}


def _queue(n, grip=-1.0):
    q = np.zeros((n, 7), np.float32)
    q[:, 0] = np.arange(1, n + 1) * 0.01
    q[:, 2] = 1.0
    q[:, 3:6] = quat_to_axisangle(eef_to_control_quat(DOWN))
    q[:, 6] = grip
    return q


def test_roll_with_no_lag_writes_the_last_targets_newest_last():
    rolled = dp_mech.proprio_roll(_obs(), _queue(5), lag=1.0)['proprio']
    assert np.allclose(rolled[:, 0], [0.04, 0.05])
    assert np.allclose(np.abs(rolled[-1, 3:7] @ DOWN), 1.0, atol=1e-5)   # same orientation
    assert np.allclose(rolled[-1, 7:9], [0.04, -0.04])


def test_roll_lags_behind_the_targets_and_keeps_the_measured_quat_sign():
    obs = _obs()
    obs['proprio'][:, 3:7] = -DOWN                         # robosuite reported the other sign
    q = _queue(1)
    rolled = dp_mech.proprio_roll(obs, q, lag=0.3)['proprio']
    assert np.isclose(rolled[-1, 0], 0.3 * 0.01)
    assert rolled[-1, 3:7] @ -DOWN > 0.999


def test_roll_closes_the_fingers_only_when_the_plan_does():
    rolled = dp_mech.proprio_roll(_obs(), _queue(3, grip=1.0), closed_qpos=0.01)['proprio']
    assert np.allclose(rolled[-1, 7:9], [0.01, -0.01])
    held = dp_mech.proprio_roll(_obs(width=0.012), _queue(3, grip=1.0))['proprio']
    assert np.allclose(held[-1, 7:9], [0.012, -0.012])        # already closed: keep the measurement


def test_short_queue_rolls_only_the_newest_frame_and_empty_is_a_no_op():
    obs = _obs()
    rolled = dp_mech.proprio_roll(obs, _queue(1))['proprio']
    assert np.allclose(rolled[0], obs['proprio'][0]) and np.isclose(rolled[1, 0], 0.003)
    assert dp_mech.proprio_roll(obs, _queue(0)) is obs


def test_object_tracker_reads_the_extra_not_the_proprio():
    obs = _obs()
    assert dp_mech.object_tracker(obs) is None
    obs['object'][7:11] = [0.1, 0.2, 0.8, 1.0]
    assert np.allclose(dp_mech.object_tracker(obs), [0.1, 0.2, 0.8])


def test_oracle_layout_feeds_the_square_gate_unchanged():
    import oracle_square as sq
    nut = np.array([-0.1, 0.15, 0.83])
    q = np.array([0.0, 0.0, 0.0, 1.0])
    eq = sq.ol_quat_mul(sq.axisangle_z(sq.grasp_turn(DOWN, q)), DOWN)
    eef = sq.handle_pos(nut, q) + [0.0, 0.0, sq.SquareConfig().grasp_dz]
    obs = {'proprio': np.concatenate([eef, eq, [0.04, -0.04]])[None].astype(np.float32),
           'object': np.concatenate([nut, q, nut, [1.0]]).astype(np.float32)}
    assert dp_mech.oracle_layout(obs, 'square')['proprio'].shape == (1, 20)
    assert dp_mech.task_gate(obs, None, 'square')
    obs['object'][7] += 0.03                                   # tracked nut 3 cm away
    assert not dp_mech.task_gate(obs, None, 'square')


def test_square_lift_waits_for_the_fingers_to_settle():
    lift = dp_mech.SquareLift(settle_m=0.0005, min_ticks=3)
    widths = [0.03, 0.022, 0.0205, 0.0204, 0.0204]
    got = [lift(_obs(w / 2), k + 1) for k, w in enumerate(widths)]
    assert got == [False, False, False, True, True]
    assert dp_mech.SquareLift()(_obs(0.002), 1)          # shut on nothing: no point holding


# ------------------------------------------------------------------ nut_goal
import csv                                    # noqa: E402
import types                                  # noqa: E402

import nut_goal as ng                         # noqa: E402
import oracle_lift as ol                      # noqa: E402


class Sim:
    """Just enough MuJoCo for the conveyor: per-nut joint addresses."""

    def __init__(self):
        adr = {f'{n}_joint0': 7 * i for i, n in enumerate(ng.NUTS)}
        self.model = types.SimpleNamespace(
            get_joint_qpos_addr=lambda j: (adr[j], adr[j] + 7),
            get_joint_qvel_addr=lambda j: (adr[j], adr[j] + 6))
        self.data = types.SimpleNamespace(qpos=np.zeros(14), qvel=np.zeros(14))


class NutEnv:
    """Fake NutAssembly: both nuts always visible, the square nut's z follows the action's
    gripper channel (0 = stays on the table), and `pegs` says which nuts robosuite would call
    seated — default only the square one, so robosuite's any-nut success and GoalWorld's
    goal-nut success differ on odd scenes by construction."""

    def __init__(self):
        self.sim = Sim()
        self.pegs = np.array([1.0, 0.0])
        self.objects_on_pegs = np.zeros(2)
        self.z = 0.83

    def _obs(self):
        return {'SquareNut_pos': np.array([0.1, 0.15, self.z]),
                'SquareNut_quat': np.array([0.0, 0.0, 0.0, 1.0]),
                'RoundNut_pos': np.array([0.1, -0.15, 0.83]),
                'RoundNut_quat': np.array([0.0, 0.0, 0.0, 1.0]),
                'robot0_eef_pos': np.zeros(3), 'robot0_eef_quat': np.array([0.0, 0.0, 0.0, 1.0]),
                'robot0_gripper_qpos': np.array([0.04, -0.04]),
                'agentview_image': np.zeros((2, 2, 3), np.uint8),
                'robot0_eye_in_hand_image': np.zeros((2, 2, 3), np.uint8)}

    def reset(self):
        self.z = 0.83
        return self._obs()

    def step(self, action):
        self.z = 0.83 + float(action[6])
        return self._obs(), 0.0, False, {}

    def _check_success(self):
        self.objects_on_pegs = self.pegs.copy()
        return bool(self.pegs.any())


class HoldPolicy:
    """Absolute hold chunks, compatible with dp_mech's stacked obs; records the goal each
    request's obs carried, which is what the cloud half of a split policy would read."""

    action_dim = 7
    n_obs_steps = 1
    needs_wrist = False
    absolute_actions = True
    guided_resampling = False
    chunk_size = 2

    def __init__(self):
        self.goals = []

    def predict(self, obs):
        self.goals.append(obs.get('goal'))
        return np.zeros((self.chunk_size, 7), np.float32)

    def predict_inpaint(self, obs, prefix, weights):
        return self.predict(obs)


def _nut_world():
    env = NutEnv()
    mc = ol.MovingCube(env, 0.0, 20.0, joint=f'{ng.NUTS[0]}_joint0',
                       pos_key=f'{ng.NUTS[0]}_pos', held_fn=lambda o: False, half_range=0.06)
    return ng.GoalWorld(mc), mc, env


def _goal_obs(g, bird=False):
    o = dict(NutEnv()._obs())
    o.update(goal=g, goal_nut_pos=o[f'{ng.NUTS[g]}_pos'], goal_nut_quat=o[f'{ng.NUTS[g]}_quat'])
    if bird:
        o['birdview_image'] = np.full((2, 2, 3), 7, np.uint8)
    return o


def test_nut_reset_schedules_goal_k_mod_2_and_points_the_conveyor_there():
    gw, mc, env = _nut_world()
    for k, goal in ((0, 0), (1, 1), (4, 0)):
        obs = dp_mech.nut_reset(gw, k)
        assert obs['goal'] == goal and gw.goal == goal
        nut = ng.NUTS[goal]
        assert mc.pos_key == f'{nut}_pos'
        assert mc._qadr == env.sim.model.get_joint_qpos_addr(f'{nut}_joint0')[0]
        assert mc._vadr == env.sim.model.get_joint_qvel_addr(f'{nut}_joint0')[0]


def test_nut_reset_passes_the_goal_switch_through_to_goalworld():
    gw, mc, env = _nut_world()
    assert dp_mech.nut_reset(gw, 0, switch_tick=1)['goal'] == 0
    assert gw.step(np.zeros(7))[0]['goal'] == 1 and gw.switched    # not lifted: it fires
    dp_mech.nut_reset(gw, 1)                                       # default: no switch
    assert gw.step(np.zeros(7))[0]['goal'] == 1 and not gw.switched


def test_stack_hands_the_policy_the_newest_goal_and_leaves_other_tasks_alone():
    frames = dp_mech.make_frames(None, 'goal_nut_pos', 'goal_nut_quat', goal=True)
    out = dp_mech.stack([frames(_goal_obs(0), False), frames(_goal_obs(1), False)],
                        needs_wrist=False)
    assert out['goal'] == 1 and out['object'].shape == (11,)       # follows the newest frame
    plain = dp_mech.make_frames(None, 'goal_nut_pos', 'goal_nut_quat')
    out = dp_mech.stack([plain(_goal_obs(1), False)], needs_wrist=False)
    assert 'goal' not in out                                       # policies that ignore it: no key
    both = dp_mech.make_frames(None, 'goal_nut_pos', 'goal_nut_quat', top=True, goal=True)
    out = dp_mech.stack([both(_goal_obs(1, bird=True), False)] * 2, needs_wrist=False)
    assert out['goal'] == 1 and out['top'].shape == (2, 2, 2, 3)   # goal and top ride together


def test_nut_watch_marks_the_wrong_nut():
    gw, mc, env = _nut_world()
    w = dp_mech.NutWatch()
    w.start(1)                                    # goal: the round nut; the square one moves
    w.note({'SquareNut_pos': np.array([0.0, 0.0, 0.83]),
            'RoundNut_pos': np.array([0.0, 0.0, 0.83])})
    w.note({'SquareNut_pos': np.array([0.0, 0.0, 0.90]),
            'RoundNut_pos': np.array([0.0, 0.0, 0.83])})          # square nut lifted once
    dp_mech.nut_reset(gw, 1)
    assert w.row(gw) == {'goal': 1, 'switched': 0, 'wrong_nut': 1}
    env.pegs = np.array([0.0, 1.0])               # goal 0: the round nut ends on its peg
    w.start(0)
    w.note(env._obs())
    dp_mech.nut_reset(gw, 0)
    assert w.row(gw) == {'goal': 0, 'switched': 0, 'wrong_nut': 1}
    env.pegs = np.array([0.0, 0.0])               # nothing lifted, nothing seated
    w.start(1)
    w.note(env._obs())
    assert w.row(gw) == {'goal': 1, 'switched': 0, 'wrong_nut': 0}


def test_nut_goal_sweep_wiring(tmp_path):
    """dp_mech's nut wiring end to end through oracle_lift.run_sweep on the fake env: the goal
    schedule, the goal reaching the policy's obs at request time, success = GoalWorld's
    goal-nut check (robosuite's any-nut one says True for both scenes), and the extra columns."""
    gw, mc, env = _nut_world()
    watch, policy = dp_mech.NutWatch(), HoldPolicy()
    args = types.SimpleNamespace(speeds='0', delays='1', strategies='synchronous', episodes=2,
                                 seed=0, world_speed=1.0, out=str(tmp_path / 'nut.csv'))
    ol.run_sweep(args, policy=policy, tracker=ol.TrackerModel(ol.TrackerSpec()), deadline=2,
                 make_world=lambda v: gw,
                 frame_fn=dp_mech.make_frames(None, 'goal_nut_pos', 'goal_nut_quat',
                                              goal=True, watch=watch),
                 stack_fn=dp_mech.stack, is_success=lambda e: e.success(),
                 reset_fn=lambda e, k: (watch.start(k % 2), dp_mech.nut_reset(e, k))[1],
                 columns=['goal', 'switched', 'wrong_nut'], row_fn=watch.row)
    rows = list(csv.DictReader(open(args.out)))
    assert [r['scene'] for r in rows] == ['0', '1'] and [r['goal'] for r in rows] == ['0', '1']
    assert [r['success'] for r in rows] == ['1', '0']       # scene 1's any-nut says True
    assert [r['wrong_nut'] for r in rows] == ['0', '1']     # scene 1 seated the wrong nut
    assert [r['switched'] for r in rows] == ['0', '0']
    assert list(rows[0])[-3:] == ['goal', 'switched', 'wrong_nut']
    assert policy.goals[0] == 0 and policy.goals[-1] == 1   # each request's obs carried its goal
