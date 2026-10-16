#!/usr/bin/env python3
"""Interp: where does the nut's x get lost between DP's features and its action?

At the gain map's operating points (interp_gainmap.operating_points), the nut is moved ±2 cm in
x and in y. That gives the feature change each move causes, dg_x and dg_y (global_cond, per
2 cm). The denoiser is then run by hand with the noise fixed, so the chunk's grasp target is a
differentiable function of global_cond alone, and four things are measured per axis:

  |dg|      how much the features move (perception: does the encoder register the move?)
  gain      target(g0 + dg) - target(g0) along that axis / 20 mm (the gain map's diagonal,
            now driven through the features only)
  patch     the same with only the agentview part, or only the wrist part, of dg (activation
            patching; proprio does not change, the arm is frozen)
  chunk     mean |change| of the whole chunk's xyz targets / 20 mm (does dg move the action at
            all, or just not along its own axis?)
  FiLM      relative change of each FiLM block's scale/bias output, |F(g0 + dg) - F(g0)| /
            |F(g0)| (conditioning: does x get in?). The step embedding enters before the
            Mish, so the numerator is the same at every diffusion step; the first is shown
  J         the 2x274 Jacobian d target / d global_cond (autograd through DDIM); its gain
            along the unit dg, and |cos| between dg and J's top input direction

    python3 scripts/interp_sensitivity.py --out outputs/interp/sensitivity.npz --scenes 20
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_probe as ip  # noqa: E402
import sil_step  # noqa: E402
from dp_anchor_probe import observe, set_object  # noqa: E402
from fuzzy_gate import _data_arrays  # noqa: E402
from interp_collect import encode  # noqa: E402
from interp_gainmap import TASKS, make_env, operating_points  # noqa: E402

JOINT = TASKS['square']['joint']

D_M = 0.02
PARTS = {'agentview': ip.BLOCKS['agentview'], 'wrist': ip.BLOCKS['wrist']}


class Denoiser:
    """DP's sampling loop with the noise fixed, as a differentiable map global_cond -> target."""

    def __init__(self, policy, seed: int) -> None:
        import torch
        self.dp = policy._policy
        self.start = policy.n_obs_steps - 1          # trajectory row of chunk entry 0
        torch.manual_seed(seed)
        self.noise = torch.randn((1, int(self.dp.horizon), int(self.dp.action_dim)),
                                 device=self.dp.device, dtype=self.dp.dtype)
        self.idx = None

    def chunk(self, g):
        dp = self.dp
        sched = dp.noise_scheduler
        sched.set_timesteps(dp.num_inference_steps)
        traj = self.noise
        for t in sched.timesteps:
            out = dp.model(traj, t, local_cond=None, global_cond=g.reshape(1, -1))
            traj = sched.step(out, t, traj, **getattr(dp, 'kwargs', {})).prev_sample
        return dp.normalizer['action'].unnormalize(traj)[0, self.start:]

    def target(self, g):
        """xy of the chunk's last open-gripper entry; the entry is fixed by the first call."""
        c = self.chunk(g)
        if self.idx is None:
            open_ = np.flatnonzero(c[:, -1].detach().cpu().numpy() <= 0)
            self.idx = int(open_[-1]) if len(open_) else len(c) - 1
        return c[self.idx, :2] * 1e3                 # mm


def film(dp, g, step_t: int):
    """Every FiLM block's scale/bias output for global_cond g at diffusion step step_t."""
    import torch
    emb = dp.model.diffusion_step_encoder(torch.tensor([step_t], device=dp.device))
    cond = torch.cat([emb, g.reshape(1, -1)], dim=-1)
    return [m(cond).reshape(-1) for n, m in dp.model.named_modules() if n.endswith('cond_encoder')]


def probe(env, policy, o, seed: int) -> dict:
    import torch
    dp = policy._policy
    d = env.sim.data
    saved = _data_arrays(d)
    xy0 = np.array(o['SquareNut_pos'][:2], float)

    def g_at(dx, dy):
        set_object(env, JOINT, xy0 + [dx, dy])
        cur = sil_step.frames(observe(env))
        g = encode(policy, sil_step.stack(deque([cur] * policy.n_obs_steps), True))
        for k, v in saved.items():
            cur_ = getattr(d, k)
            if cur_.shape == v.shape:
                cur_[...] = v
        return torch.tensor(g, device=dp.device, dtype=dp.dtype)

    g0 = g_at(0, 0)
    dg = {'x': (g_at(D_M, 0) - g_at(-D_M, 0)) / 2, 'y': (g_at(0, D_M) - g_at(0, -D_M)) / 2}
    observe(env)
    den = Denoiser(policy, seed)
    out = {}
    with torch.no_grad():
        t0 = den.target(g0)
        full0 = den.chunk(g0)
        c0 = full0[:, :3]
        for ax, k in (('x', 0), ('y', 1)):
            v = dg[ax]
            out[f'norm_{ax}'] = float(v.norm())
            out[f'gain_{ax}'] = float((den.target(g0 + v) - t0)[k]) / (D_M * 1e3)
            out[f'cross_{ax}'] = float((den.target(g0 + v) - t0)[1 - k]) / (D_M * 1e3)
            full = den.chunk(g0 + v)
            out[f'chunk_{ax}'] = float((full[:, :3] - c0).norm(dim=1).mean()) / D_M
            # where the change goes: mean signed xyz shift (per mm), rotation and gripper parts
            out[f'dxyz_{ax}'] = ((full[:, :3] - c0).mean(0) / D_M).cpu().numpy()
            out[f'drot_{ax}'] = float((full[:, 3:9] - full0[:, 3:9]).norm(dim=1).mean())
            out[f'dgrip_{ax}'] = float((full[:, 9] - full0[:, 9]).abs().mean())
            for part, cols in PARTS.items():
                m = torch.zeros_like(v)
                m[cols] = 1.0
                out[f'patch_{part}_{ax}'] = float((den.target(g0 + m * v) - t0)[k]) / (D_M * 1e3)
            step_t = int(dp.noise_scheduler.timesteps[0])
            f0, f1 = film(dp, g0, step_t), film(dp, g0 + v, step_t)
            out[f'film_{ax}'] = np.array([float((b - a).norm() / a.norm()) for a, b in zip(f0, f1)])
    J = torch.autograd.functional.jacobian(den.target, g0).detach().cpu().numpy()   # (2, 274)
    _u, sv, vt = np.linalg.svd(J, full_matrices=False)
    for ax, k in (('x', 0), ('y', 1)):
        v = dg[ax].cpu().numpy()
        out[f'jgain_{ax}'] = float((J @ v)[k]) / (D_M * 1e3)
        out[f'jcos_{ax}'] = float(abs(vt[0] @ v) / np.linalg.norm(v))
    out['sv'] = sv
    return out


def check_replica(policy, obs) -> None:
    """The hand-run sampler must reproduce policy.predict exactly (same seed, same target)."""
    import torch
    from dp_anchor_probe import predict
    from interp_gainmap import target
    hist = deque([sil_step.frames(obs)] * policy.n_obs_steps, maxlen=policy.n_obs_steps)
    want = target(predict(policy, hist, 1234)) * 1e3
    den = Denoiser(policy, 1234)
    g = torch.tensor(encode(policy, sil_step.stack(hist, True)), device=policy._policy.device)
    with torch.no_grad():
        got = den.target(g).cpu().numpy()
    assert np.allclose(got, want, atol=1e-3), f'denoiser replica drifted: {got} vs {want}'


def collect(args) -> dict:
    from evh_controller.policy import make_policy
    policy = make_policy('dp', args.weights, denoise_steps=args.denoise_steps)
    env, obs = make_env()
    check_replica(policy, obs)
    rows = []
    for scene in range(args.scenes):
        for o, r, _t in operating_points(env, obs, policy, args.radii, scene):
            rows.append({'scene': scene, 'radius': r, **probe(env, policy, o, 1000 + scene)})
        print(f'scene {scene}: {sum(x["scene"] == scene for x in rows)} points', flush=True)
        obs = env.reset()
    env.close()
    return {k: np.array([x[k] for x in rows]) for k in rows[0]}


def report(d: dict) -> None:
    med = lambda k, m: float(np.median(d[k][m]))
    for r in sorted(set(d['radius']), reverse=True):
        m = d['radius'] == r
        print(f'\narm {r * 100:.0f} cm from the handle, {m.sum()} points (medians; gains per mm)')
        print(f'{"":22s} {"x":>7s} {"y":>7s}')
        for label, key in (('|dg| per 2 cm', 'norm'), ('gain via features', 'gain'),
                           ('  cross-axis', 'cross'), ('whole chunk moves', 'chunk'),
                           ('  agentview only', 'patch_agentview'),
                           ('  wrist only', 'patch_wrist'), ('gain via J (linear)', 'jgain'),
                           ('|cos(dg, top J dir)|', 'jcos')):
            print(f'{label:22s} {med(key + "_x", m):7.2f} {med(key + "_y", m):7.2f}')
        if 'dxyz_x' in d:
            for ax in 'xy':
                v = np.median(d[f'dxyz_{ax}'][m], 0)
                print(f'nut moved in {ax}: mean chunk shift xyz per mm [{v[0]:+.2f} {v[1]:+.2f} '
                      f'{v[2]:+.2f}], rot6d change {med("drot_" + ax, m):.3f}, '
                      f'gripper change {med("dgrip_" + ax, m):.3f}')
        fx, fy = d['film_x'][m], d['film_y'][m]
        print(f'FiLM rel. change: x {np.median(fx):.4f}  y {np.median(fy):.4f}  per block x/y: '
              + ' '.join(f'{a:.2f}' for a in np.median(fx, 0) / np.median(fy, 0)))
        sv = d['sv'][m]
        print(f'J singular values (median): {np.median(sv[:, 0]):.2f}, {np.median(sv[:, 1]):.2f}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', required=True)
    p.add_argument('--weights', default=os.path.join(sil_step._REPO, 'checkpoints',
                                                     'dp_square_ph_image_cnn.ckpt'))
    p.add_argument('--denoise-steps', type=int, default=16)
    p.add_argument('--scenes', type=int, default=20)
    p.add_argument('--radii', default='0.08,0.03')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    args.radii = [float(r) for r in str(args.radii).split(',')]
    if not args.report_only:
        np.savez_compressed(args.out, **collect(args))
    report(dict(np.load(args.out)))


if __name__ == '__main__':
    main()
