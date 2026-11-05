#!/usr/bin/env python3
"""Interp A for Diffusion Policy: repair a shift by refitting only the layers that read the
observation features.

DP's observation features (global_cond) enter the denoiser only through the FiLM conditioning
layers of its residual blocks (cond_encoder: Mish -> Linear). Those Linear layers are DP's
readout; everything else (image encoder, convolutions, final layer) stays frozen. They are
trained with DP's own denoising loss on recorded on-policy decisions: global_cond from
interp_collect --backend dp ('feats') and the label trajectory ('teacher': the full-horizon abs10
action_pred of the unedited DP on the shift-free render, interp_collect --selflabel cf).
Unshifted episodes labelled the same way keep in-range behaviour. Because the encoder is frozen,
the cached global_cond is exactly what the edited policy computes. Saved as a normal workspace
.ckpt (both 'model' and 'ema_model' replaced), so every interp script loads it with --weights.

    python3 scripts/interp_dp_repair.py --parent checkpoints/dp_lift_ph_image_cnn.ckpt \
        --data 'outputs/interp/nuis/dp_cf1_bothcam,outputs/interp/nuis/dp_cf1_none' \
        --out outputs/interp/edit/dp_rep_bothcam.ckpt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sil_step  # noqa: E402,F401  (puts the ROS-free packages on sys.path)


def load(patterns: str):
    """(global_cond (N, D), label trajectories abs10 (N, H, 10)) from every episode."""
    C, A = [], []
    for f in sorted(f for g in patterns.split(',') for d in sorted(glob.glob(g))
                    for f in glob.glob(f'{d}/ep*.npz')):
        e = np.load(f)
        if 'label_tick' in e:                              # labelled every n-th tick only
            C.append(e['feats'][e['label_tick']])
            A.append(e['teacher'])
            continue
        n = min(len(e['feats']), len(e['teacher']))
        C.append(e['feats'][:n])
        A.append(e['teacher'][:n])
    return np.concatenate(C).astype(np.float32), np.concatenate(A).astype(np.float32)


def readout_params(model):
    """The FiLM conditioning Linear layers: the only path from global_cond into the denoiser."""
    return [p for n, p in model.named_parameters() if 'cond_encoder' in n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--parent', required=True, help='DP workspace .ckpt')
    p.add_argument('--data', required=True, help='comma-separated globs of recorded episode dirs')
    p.add_argument('--out', required=True, help='edited .ckpt')
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    import dill
    import torch
    from evh_controller.policy import make_policy
    torch.manual_seed(args.seed)
    policy = make_policy('dp', args.parent)
    dp = policy._policy
    dev = next(dp.parameters()).device
    cond, traj = load(args.data)
    with torch.no_grad():
        target = dp.normalizer['action'].normalize(torch.from_numpy(traj).to(dev))
    cond = torch.from_numpy(cond).to(dev)
    sched = dp.noise_scheduler
    pred_type = sched.config.prediction_type
    dp.requires_grad_(False)
    params = readout_params(dp.model)
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)
    dp.model.train()
    for step in range(args.steps):
        i = torch.from_numpy(rng.integers(len(cond), size=args.batch)).to(dev)
        x0 = target[i]
        noise = torch.randn_like(x0)
        t = torch.randint(0, sched.config.num_train_timesteps, (len(i),), device=dev).long()
        pred = dp.model(sched.add_noise(x0, noise, t), t, global_cond=cond[i])
        loss = torch.nn.functional.mse_loss(pred, noise if pred_type == 'epsilon' else x0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f'step {step}: loss {loss.item():.4f}', flush=True)
    dp.model.eval()
    payload = torch.load(open(args.parent, 'rb'), pickle_module=dill, map_location='cpu')
    state = {k: v.detach().cpu() for k, v in dp.state_dict().items()}
    for key in ('model', 'ema_model'):
        if key in payload['state_dicts']:
            payload['state_dicts'][key] = state
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, open(args.out, 'wb'), pickle_module=dill)
    stats = dict(vars(args), decisions=len(cond), params=sum(q.numel() for q in params),
                 final_loss=float(loss.item()))
    with open(args.out + '.json', 'w') as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
