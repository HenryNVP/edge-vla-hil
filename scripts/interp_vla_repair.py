#!/usr/bin/env python3
"""Interp A for SmolVLA: repair a shift by refitting only the layer that turns the action
expert's output into actions (action_out_proj, the flow-matching velocity readout).

Trained with SmolVLA's own flow-matching loss (policy.forward) on recorded decisions: what the
policy saw ('img_agent', 'img_wrist', 'proprio') and the unedited policy's chunk on the shift-free
render ('teacher', abs10; interp_collect --backend smolvla --selflabel cf --save-images), the
unshifted episodes labelled the same way keeping in-range behaviour. --site expert also trains
the action expert (deeper). Saved as a pretrained_model dir (plus instruction.txt).

    python3 scripts/interp_vla_repair.py --parent <smolvla pretrained_model> \
        --data 'outputs/interp/nuis/vla_cf1_bothflip,outputs/interp/nuis/vla_cf1_none' --out ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sil_step  # noqa: E402,F401  (puts the ROS-free packages on sys.path)


def load(patterns: str):
    """(agentview, wrist, proprio, label chunks abs10) over every recorded labelled decision."""
    A, W, S, L = [], [], [], []
    for f in sorted(f for g in patterns.split(',') for d in sorted(glob.glob(g))
                    for f in glob.glob(f'{d}/ep*.npz')):
        e = np.load(f)
        A.append(e['img_agent'])
        W.append(e['img_wrist'])
        S.append(e['proprio'])
        L.append(e['teacher'])
    return (np.concatenate(A), np.concatenate(W), np.concatenate(S).astype(np.float32),
            np.concatenate(L).astype(np.float32))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--parent', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--site', choices=('out', 'expert'), default='out')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    import torch
    from evh_controller.policy import make_policy
    torch.manual_seed(args.seed)
    policy = make_policy('smolvla', args.parent)
    vla = policy._model
    dev = next(vla.parameters()).device
    agent, wrist, prop, lab = load(args.data)
    keys = list(vla.config.image_features)                   # agentview first, then the wrist
    H = vla.config.chunk_size
    vla.requires_grad_(False)
    mods = [vla.model.action_out_proj] + ([vla.model.vlm_with_expert.lm_expert] if args.site == 'expert' else [])
    params = [q for m in mods for q in m.parameters()]
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)
    vla.train()
    for step in range(args.steps):
        i = rng.integers(len(lab), size=args.batch)
        img = [torch.from_numpy(x[i]).to(dev).permute(0, 3, 1, 2).float() / 255 for x in (agent, wrist)]
        batch = {keys[0]: img[0], keys[1]: img[1],
                 'observation.state': torch.from_numpy(prop[i]).to(dev),
                 'action': torch.from_numpy(lab[i][:, :H]).to(dev),
                 'task': [policy.instruction] * len(i)}
        loss, _ = vla.forward(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == args.steps - 1:
            print(f'step {step}: loss {loss.item():.4f}', flush=True)
    vla.eval()
    vla.save_pretrained(args.out)
    for f in os.listdir(args.parent):
        if not os.path.exists(os.path.join(args.out, f)):
            shutil.copy(os.path.join(args.parent, f), args.out)
    stats = dict(vars(args), decisions=len(lab), params=sum(q.numel() for q in params),
                 final_loss=float(loss.item()))
    with open(os.path.join(args.out, 'edit.json'), 'w') as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
