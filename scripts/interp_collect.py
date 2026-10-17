#!/usr/bin/env python3
"""Interp step 0: roll out DP on Square and log, per control tick, what the policy saw next to the
sim's ground truth (docs/interp-plan.md).

Per tick: `global_cond` (the obs encoder's output, the only thing the denoiser is conditioned on;
for the robomimic Square checkpoint 2 obs steps x [agentview 64 | wrist 64 | proprio 9]), the
executed command, and the state: eef pose, gripper, nut pose, held, on_peg. Per episode: success
and the square_failures class. One compressed .npz per episode in --out-dir.

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


def state(o) -> dict:
    return {'eef_pos': o['robot0_eef_pos'], 'eef_quat': o['robot0_eef_quat'],
            'grip': o['robot0_gripper_qpos'], 'nut_pos': o['SquareNut_pos'],
            'nut_quat': o['SquareNut_quat'],
            'held': sq.is_held(o['robot0_eef_pos'], o['robot0_gripper_qpos'],
                               o['SquareNut_pos'], o['SquareNut_quat']),
            'on_peg': sq.on_peg(o['SquareNut_pos'])}


def place(env, obs, shift):
    """The scene's nut moved by shift = (dx, dy[, dyaw_deg]); returns the fresh observation, so
    the policy's first frame already shows it there."""
    if not shift.any():
        return obs
    from dp_anchor_probe import observe, set_object
    set_object(env, 'SquareNut_joint0', np.asarray(obs['SquareNut_pos'][:2]) + shift[:2])
    if len(shift) > 2 and shift[2]:
        sim = env.sim
        adr = sim.model.get_joint_qpos_addr('SquareNut_joint0')[0] + 3   # wxyz
        h = np.radians(shift[2]) / 2
        w, x, y, z = sim.data.qpos[adr:adr + 4]
        c, s_ = np.cos(h), np.sin(h)                # q <- rz(dyaw) * q
        sim.data.qpos[adr:adr + 4] = [c * w - s_ * z, c * x - s_ * y, c * y + s_ * x,
                                      c * z + s_ * w]
        sim.forward()
    return observe(env)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out-dir', required=True)
    p.add_argument('--weights', default=os.path.join(sil_step._REPO, 'checkpoints',
                                                     'dp_square_ph_image_cnn.ckpt'))
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
    args = p.parse_args()
    shift = np.array([float(v) for v in args.shift.split(',')])

    from evh_controller.policy import make_policy
    from evh_plant.env_factory import EnvSpec, build_env

    os.makedirs(args.out_dir, exist_ok=True)
    policy = make_policy('dp', args.weights, denoise_steps=args.denoise_steps)
    if args.corrector:
        from interp_corrector import Corrected, load_fix
        policy = Corrected(policy, load_fix(args.corrector))
    built = build_env(EnvSpec(env_name='NutAssemblySquare', action_hz=sil_step.CONTROL_HZ,
                              max_episode_s=args.deadline_s + 1.0, seed=args.seed,
                              absolute_actions=policy.absolute_actions))
    env = built.env
    obs = place(env, built.obs, shift)
    worker = sil_step.StepWorker(policy, sil_step.fixed_delay(1))
    executor = sil_step.make_strategy('synchronous', worker, policy)
    wins = 0
    for scene in range(args.episodes):
        if scene >= args.first_scene:
            rec = Recorder(env, obs)
            feats, states = [], []

            def stack_fn(hist, needs_wrist):
                stacked = sil_step.stack(hist, needs_wrist)
                feats.append(encode(policy, stacked))
                states.append(state(rec.obs))
                return stacked

            ep = sil_step.run_episode(rec, obs, policy, executor, worker,
                                      deadline_ticks=int(args.deadline_s * sil_step.CONTROL_HZ),
                                      absolute=policy.absolute_actions, stack_fn=stack_fn)
            states.append(state(rec.obs))           # the state after the last step
            cols = {k: np.array([s[k] for s in states]) for k in states[0]}
            cmds = np.array(rec.commands)
            nut0 = cols['nut_pos'][0]
            cls = square_failures.classify(cols['held'][1:], cols['nut_pos'][1:],
                                           cols['on_peg'][1:], ep.success, cmds[:, 6] > 0,
                                           nut0[2], sq.PEG)
            np.savez_compressed(os.path.join(args.out_dir, f'ep{scene:04d}.npz'),
                                feats=np.array(feats), commands=cmds, success=ep.success,
                                cls=cls['cls'], ticks=ep.ticks, shift=shift, **cols)
            wins += ep.success
            print(f'scene {scene}: {cls["cls"]} ticks={ep.ticks} '
                  f'corrected={getattr(policy, "applied", 0)} '
                  f'({wins}/{scene - args.first_scene + 1})', flush=True)
        obs = place(env, env.reset(), shift)
    env.close()


if __name__ == '__main__':
    main()
