#!/usr/bin/env python3
"""One-shot ACT inference benchmark (CUDA). Mirrors the DP docker one-liner.

Requires LeRobot ACT (Python 3.10+ via pip, or a v0.3.3 src install on Py3.8 — see Dockerfile note).
Default checkpoint: lerobot/act_aloha_sim_transfer_cube_human (bimanual ALOHA sim, single camera).

Usage (Jetson container, after lerobot is available):
    python3 scripts/bench_act.py --device cuda
    python3 scripts/bench_act.py --repo lerobot/act_aloha_sim_insertion_human --device cuda
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time


def _import_act_policy():
    tried: list[str] = []
    for modpath in (
        'lerobot.policies.act.modeling_act',
        'lerobot.common.policies.act.modeling_act',
    ):
        try:
            return importlib.import_module(modpath).ACTPolicy
        except Exception as exc:
            tried.append(f'  {modpath}: {type(exc).__name__}: {exc}')
    raise ImportError(
        'Could not import LeRobot ACTPolicy. Tried:\n'
        + '\n'.join(tried)
        + '\nInstall lerobot==0.3.3 (Python 3.10+) or clone v0.3.3 into PYTHONPATH.'
    )


def _dummy_batch(policy, device: str):
    import torch

    batch: dict[str, torch.Tensor] = {}
    for key, feat in policy.config.input_features.items():
        if 'image' in key:
            c, h, w = feat.shape
            batch[key] = torch.zeros(1, c, h, w, device=device)
        else:
            batch[key] = torch.zeros(1, feat.shape[0], device=device)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark LeRobot ACT chunk inference.')
    parser.add_argument('--repo', default='lerobot/act_aloha_sim_transfer_cube_human',
                        help='HuggingFace repo id or local pretrained dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iters', type=int, default=10)
    args = parser.parse_args()

    import torch

    ACTPolicy = _import_act_policy()
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA unavailable; falling back to CPU')
        device = 'cpu'

    print(f'loading {args.repo} on {device}')
    policy = ACTPolicy.from_pretrained(args.repo)
    policy.eval().to(device)

    cfg = policy.config
    print(f'chunk {cfg.chunk_size} n_action {cfg.n_action_steps} '
          f'action_dim {cfg.action_feature.shape[0]}')

    batch = _dummy_batch(policy, device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            policy.reset()
            policy.select_action(batch)

        times: list[float] = []
        for _ in range(args.iters):
            policy.reset()
            t0 = time.perf_counter()
            policy.select_action(batch)
            times.append((time.perf_counter() - t0) * 1e3)

    mean = sum(times) / len(times)
    print(f'inference mean {mean:.0f} ms  min {min(times):.0f}  max {max(times):.0f}')


if __name__ == '__main__':
    main()
