#!/usr/bin/env python3
"""Interp step 0: roll out DP (or ACT, --backend act) on Square (or Lift, --task lift) and log, per control tick, what the policy saw next to the
sim's ground truth (docs/interp-plan.md).

Per tick: `global_cond` (the obs encoder's output, the only thing the denoiser is conditioned on;
for the robomimic Square checkpoint 2 obs steps x [agentview 64 | wrist 64 | proprio 9]), the
executed command, and the state: eef pose, gripper, nut pose, held, on_peg. Per episode: success
and the square_failures class (Lift: success / fail). One compressed .npz per episode in --out-dir.
nut_pos / nut_quat are the task object's pose (the cube on Lift); feats are saved for DP only.

The rollout is sil_step's own loop (synchronous, d = 1), so episodes match the paper-2 harness.
The encoder is run once more per tick on the same stacked history the executor gets; it does
not touch the rollout (eval-mode crop is a centre crop, so it is deterministic).

    python3 scripts/interp_collect.py --out-dir outputs/interp/dp_square --episodes 200
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sil_step  # noqa: E402  (puts the ROS-free packages on sys.path)
import oracle_square as sq  # noqa: E402
import square_failures  # noqa: E402
from interp_gainmap import TASKS  # noqa: E402


class Recorder:
    """Wraps the env: keeps the latest obs, and the command sent on each step."""

    def __init__(self, env, obs) -> None:
        self.env, self.obs, self.commands = env, obs, []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        self.commands.append(np.asarray(action, np.float32))
        out = self.env.step(action)
        self.obs = out[0]
        return out


def encode(policy, stacked: dict, encoder=None) -> np.ndarray:
    """global_cond for one stacked history, exactly as predict_inpaint builds it (`encoder`
    swaps in another obs encoder of the same shape, e.g. an untrained control)."""
    import torch
    dp = policy._policy
    encoder = encoder or dp.obs_encoder
    with torch.no_grad():
        nobs = dp.normalizer.normalize(policy._to_policy_obs(stacked))
        To = policy.n_obs_steps
        this = {k: v[:, :To, ...].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
        return encoder(this).reshape(-1).cpu().numpy().astype(np.float32)


def state(o, task=TASKS['square']) -> dict:
    s = {'eef_pos': o['robot0_eef_pos'], 'eef_quat': o['robot0_eef_quat'],
         'grip': o['robot0_gripper_qpos'], 'nut_pos': o[task['pos']],
         'nut_quat': o[task['quat']]}
    if task is TASKS['square']:
        s['held'] = sq.is_held(o['robot0_eef_pos'], o['robot0_gripper_qpos'],
                               o['SquareNut_pos'], o['SquareNut_quat'])
        s['on_peg'] = sq.on_peg(o['SquareNut_pos'])
    return s


def place(env, obs, shift, task=TASKS['square']):
    """The scene's nut moved by shift = (dx, dy[, dyaw_deg]); returns the fresh observation, so
    the policy's first frame already shows it there."""
    if not shift.any():
        return obs
    from dp_anchor_probe import observe, set_object
    set_object(env, task['joint'], np.asarray(obs[task['pos']][:2]) + shift[:2])
    if len(shift) > 2 and shift[2]:
        sim = env.sim
        adr = sim.model.get_joint_qpos_addr(task['joint'])[0] + 3   # wxyz
        h = np.radians(shift[2]) / 2
        w, x, y, z = sim.data.qpos[adr:adr + 4]
        c, s_ = np.cos(h), np.sin(h)                # q <- rz(dyaw) * q
        sim.data.qpos[adr:adr + 4] = [c * w - s_ * z, c * x - s_ * y, c * y + s_ * x,
                                      c * z + s_ * w]
        sim.forward()
    return observe(env)


def self_label(env, policy, obs, batch, task, delta: float = 0.01, full: bool = False) -> np.ndarray:
    """Label-free target chunk (abs10) for the current state, from the policy itself: with the
    object outside the demos' box, the policy's chunk with the object moved to the nearest point
    of the box (delta inside), its xy extended to the real object by the policy's own local slope
    there (central differences, ±delta per axis); inside the box, the policy's own chunk. The
    sim is restored bitwise after each counterfactual (as interp_gainmap.probe). full: rescale
    each axis's slope so the chunk's last row follows that axis fully (gain 1)."""
    import torch
    from dp_anchor_probe import observe, set_object
    from fuzzy_gate import _data_arrays
    from interp_act_sites import forward
    act = policy._model
    xy = np.asarray(obs[task['pos']][:2], float)
    lo, hi = np.asarray(task['lo']), np.asarray(task['hi'])
    if np.all((xy >= lo) & (xy <= hi)):
        with torch.no_grad():
            return forward(act, batch)[0].cpu().numpy()
    c = np.clip(xy, lo + delta, hi - delta)
    d = env.sim.data
    saved = _data_arrays(d)

    def chunk_at(p):
        set_object(env, task['joint'], p)
        b = policy._obs_to_batch(sil_step.stack([sil_step.frames(observe(env))], True))
        for k, v in saved.items():
            cur = getattr(d, k)
            if cur.shape == v.shape:
                cur[...] = v
        with torch.no_grad():
            return forward(act, b)[0].cpu().numpy()
    label = chunk_at(c)
    for k in (0, 1):
        e = np.eye(2)[k] * delta
        slope = (chunk_at(c + e)[:, :2] - chunk_at(c - e)[:, :2]) / (2 * delta)   # (16, 2)
        if full:
            slope /= np.clip(slope[-1, k], 0.2, 1.0)
        label[:, :2] += slope * (xy[k] - c[k])
    observe(env)
    return label


def cf_label(env, policy, nuis, backend: str = 'act') -> np.ndarray:
    """Exact label for an appearance shift (interp_nuisance): the policy's own chunk (abs10) on
    the same state rendered without the shift. DP: its full horizon ('action_pred', both history
    frames showing the shift-free render)."""
    import torch
    from dp_anchor_probe import observe
    if nuis is not None:
        nuis.off(env)
    stacked = sil_step.stack([sil_step.frames(observe(env))] * policy.n_obs_steps, True)
    if nuis is not None:
        nuis.on(env)
    with torch.no_grad():
        if backend == 'dp':
            out = policy._policy.predict_action(policy._to_policy_obs(stacked))
            return out['action_pred'][0].cpu().numpy()
        if backend == 'smolvla':                            # abs10, unnormalised
            return policy._model.predict_action_chunk(policy._obs_to_batch(stacked))[0].cpu().numpy()
        from interp_act_sites import forward
        return forward(policy._model, policy._obs_to_batch(stacked))[0].cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out-dir', required=True)
    p.add_argument('--task', default='square', choices=sorted(TASKS))
    p.add_argument('--backend', default='dp', help='make_policy backend (dp, act)')
    p.add_argument('--weights', default='', help="default: the task's DP checkpoint")
    p.add_argument('--denoise-steps', type=int, default=16)
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--first-scene', type=int, default=0, help='resume: skip scenes below this')
    p.add_argument('--deadline-s', type=float, default=20.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--shift', default='0,0',
                   help='dx,dy[,dyaw_deg]: move (m) and turn the nut at reset, before the first '
                        'frame (use --shift=-0.02,0: a leading minus reads as an option)')
    p.add_argument('--corrector', default='',
                   help='interp_corrector label files: run the corrected policy')
    p.add_argument('--teacher', default='',
                   help="ACT only: also save the policy's head inputs and this ACT's chunk at every "
                        'prediction (labels for interp_edit --onpolicy)')
    p.add_argument('--selflabel', nargs='?', const='slope', default='', choices=('slope', 'full', 'cf'),
                   help='ACT only: like --teacher, labels from self_label (no demos, no teacher); '
                        'full = slopes rescaled to gain 1; cf = cf_label (the --nuisance-free render)')
    p.add_argument('--nuisance', default='', help='interp_nuisance spec: appearance shift at every reset')
    p.add_argument('--label-weights', default='',
                   help='--selflabel cf: label with this checkpoint (the unedited expert) instead '
                        'of the acting policy (DAgger rounds after a repair)')
    p.add_argument('--save-images', action='store_true',
                   help='with --teacher / --selflabel: also save what the policy saw each tick '
                        '(agentview, wrist, proprio) for deeper repairs (interp_edit --images)')
    p.add_argument('--label-every', type=int, default=1,
                   help='DP --selflabel cf: label every n-th tick (rows say which: label_tick)')
    p.add_argument('--oracle', action='store_true',
                   help="correct with the sim's true mistake instead (upper bound)")
    args = p.parse_args()
    shift = np.array([float(v) for v in args.shift.split(',')])
    task = TASKS[args.task]
    weights = args.weights or os.path.join(sil_step._REPO, 'checkpoints', task['ckpt'])

    from evh_controller.policy import make_policy
    from evh_plant.env_factory import EnvSpec, build_env

    os.makedirs(args.out_dir, exist_ok=True)
    policy = make_policy(args.backend, weights, denoise_steps=args.denoise_steps)
    built = build_env(EnvSpec(env_name=task['env'], action_hz=sil_step.CONTROL_HZ,
                              max_episode_s=args.deadline_s + 1.0, seed=args.seed,
                              absolute_actions=policy.absolute_actions))
    env = built.env
    if args.corrector or args.oracle:
        from dp_anchor_probe import observe
        from interp_corrector import Corrected, load_fix, mistake
        inner = policy
        fix = ((lambda g: mistake(env, inner, observe(env), 0)[None]) if args.oracle
               else load_fix(args.corrector))
        policy = Corrected(policy, fix)
    if (args.teacher or args.selflabel) and args.backend == 'act':
        import torch
        from interp_act_sites import Sites, forward
        sites = Sites(policy._model)
        teacher = make_policy('act', args.teacher)._model if args.teacher else None
    from interp_nuisance import Nuisance
    nuis = Nuisance(args.nuisance) if args.nuisance else None
    labeller = (make_policy(args.backend, args.label_weights, denoise_steps=args.denoise_steps)
                if args.label_weights else policy)
    # SmolVLA with --teacher: an ACT expert labels the VLA's own states (same abs10 action space)
    teacher_pol = make_policy('act', args.teacher) if args.teacher and args.backend == 'smolvla' else None

    def fresh(o):
        """Reset observation with the nuisance applied (the model is rebuilt each reset)."""
        if nuis is not None:
            from dp_anchor_probe import observe
            nuis.on(env)
            o = observe(env)
        return place(env, o, shift, task)
    obs = fresh(built.obs)
    worker = sil_step.StepWorker(policy, sil_step.fixed_delay(1))
    executor = sil_step.make_strategy('synchronous', worker, policy)
    wins = 0
    for scene in range(args.episodes):
        if scene >= args.first_scene:
            rec = Recorder(env, obs)
            feats, states, heads, labels, label_ticks, seen = [], [], [], [], [], []

            def stack_fn(hist, needs_wrist):
                stacked = sil_step.stack(hist, needs_wrist)
                if args.backend == 'dp':
                    feats.append(encode(policy, stacked))
                if args.selflabel == 'cf' and args.backend == 'smolvla':
                    t = len(states)                                   # this tick
                    if t % args.label_every == 0:                    # VLA labels are costly
                        label_ticks.append(t)
                        if args.save_images:
                            seen.append((stacked['agentview'][-1], stacked['wrist'][-1],
                                         stacked['proprio'][-1]))
                        labels.append(cf_label(env, labeller, nuis, 'smolvla'))
                elif teacher_pol is not None:
                    t = len(states)
                    if t % args.label_every == 0:
                        import torch
                        label_ticks.append(t)
                        if args.save_images:
                            seen.append((stacked['agentview'][-1], stacked['wrist'][-1],
                                         stacked['proprio'][-1]))
                        with torch.no_grad():
                            labels.append(teacher_pol._model.predict_action_chunk(
                                teacher_pol._obs_to_batch(stacked))[0].cpu().numpy())
                elif args.selflabel == 'cf' and args.backend == 'dp':   # its head input: feats
                    if (len(feats) - 1) % args.label_every == 0:          # DP labels are costly
                        label_ticks.append(len(feats) - 1)
                        labels.append(cf_label(env, labeller, nuis, 'dp'))
                elif args.teacher or args.selflabel:
                    if args.save_images:
                        seen.append((stacked['agentview'][-1], stacked['wrist'][-1], stacked['proprio'][-1]))
                    b = policy._obs_to_batch(stacked)
                    with torch.no_grad():
                        forward(policy._model, b)
                        heads.append(sites.out['dec'][:, 0].float().cpu().numpy().astype(np.float16))
                        labels.append(teacher.predict_action_chunk(b)[0].cpu().numpy() if teacher
                                      is not None else None)
                    if teacher is None:
                        labels[-1] = (cf_label(env, labeller, nuis) if args.selflabel == 'cf' else
                                      self_label(env, policy, rec.obs, b, task,
                                                 full=args.selflabel == 'full'))
                states.append(state(rec.obs, task))
                return stacked

            ep = sil_step.run_episode(rec, obs, policy, executor, worker,
                                      deadline_ticks=int(args.deadline_s * sil_step.CONTROL_HZ),
                                      absolute=policy.absolute_actions, stack_fn=stack_fn)
            states.append(state(rec.obs, task))     # the state after the last step
            cols = {k: np.array([s[k] for s in states]) for k in states[0]}
            cmds = np.array(rec.commands)
            nut0 = cols['nut_pos'][0]
            cls = (square_failures.classify(cols['held'][1:], cols['nut_pos'][1:],
                                            cols['on_peg'][1:], ep.success, cmds[:, 6] > 0,
                                            nut0[2], sq.PEG) if args.task == 'square' else
                   {'cls': 'success' if ep.success else 'fail'})
            np.savez_compressed(os.path.join(args.out_dir, f'ep{scene:04d}.npz'),
                                feats=np.array(feats), commands=cmds, success=ep.success,
                                cls=cls['cls'], ticks=ep.ticks, shift=shift, task=args.task,
                                nuisance=args.nuisance,
                                **cols, **({'dec': np.array(heads)} if heads else {}),
                                **({'teacher': np.array(labels)} if labels else {}),
                                **({'label_tick': np.array(label_ticks)} if label_ticks else {}),
                                **({'img_agent': np.array([x[0] for x in seen]),
                                    'img_wrist': np.array([x[1] for x in seen]),
                                    'proprio': np.array([x[2] for x in seen], np.float32)}
                                   if seen else {}))
            wins += ep.success
            print(f'scene {scene}: {cls["cls"]} ticks={ep.ticks} '
                  f'corrected={getattr(policy, "applied", 0)} '
                  f'({wins}/{scene - args.first_scene + 1})', flush=True)
        obs = fresh(env.reset())
    env.close()


if __name__ == '__main__':
    main()
