"""PushT sanity check: latency x chunk-strategy sweep with the pretrained LeRobot diffusion policy.

Throwaway, non-ROS rig. Confirms the latency-robust-chunking phenomenon (and the latency_chunking
core) on a real reactive task before investing in the robosuite/RoboMimic testbed. Uses the
pretrained `lerobot/diffusion_pusht` checkpoint — zero training.

    pip install -r requirements.txt
    python run_pusht.py --strategies synchronous,naive_async,temporal_ensemble,rtc_freeze \
                        --latencies 0,2,4,8,12 --jitter 1 --episodes 30 --out results/pusht.csv

Latency is expressed in CONTROL STEPS. PushT runs ~10 Hz, so 1 step ~= 100 ms of inference+network
delay; a diffusion policy on an edge box is genuinely in the multi-step regime.

NOTE (version-sensitive lines are marked ###): the LeRobot policy API and gym_pusht obs format
drift between releases. The three spots to check against your installed versions are the policy
import, the action-chunk call, and the observation-batch keys/size.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from latency_chunking import (
    DelayModel, make_strategy, run_episode, EpisodeResult,
)


# --------------------------------------------------------------- env adapter
class GymPushTEnv:
    """Adapt gym_pusht to the latency_chunking Env protocol: reset()->obs, step(a)->(obs,r,done).

    Keeps the raw gym observation dict so the policy adapter can build its input batch.
    """
    def __init__(self, render: bool = False):
        import gymnasium as gym
        import gym_pusht  # noqa: F401  (registers the env)
        self.env = gym.make(
            'gym_pusht/PushT-v0',
            obs_type='pixels_agent_pos',          ### obs format: image + agent position
            render_mode='rgb_array' if render else None,
        )
        self._raw = None

    def reset(self):
        obs, _info = self.env.reset()
        self._raw = obs
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, _info = self.env.step(np.asarray(action, np.float32))
        self._raw = obs
        return obs, float(reward), bool(terminated or truncated)


# ------------------------------------------------------------ policy adapter
def load_policy(device: str = 'cuda'):
    """Return (policy_fn, action_dim, horizon) for the pretrained diffusion_pusht checkpoint."""
    import torch
    import torch.nn.functional as F
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy  ### import

    policy = DiffusionPolicy.from_pretrained('lerobot/diffusion_pusht').to(device).eval()
    horizon = getattr(policy.config, 'horizon', 16)
    action_dim = 2

    def _batch(obs: dict):
        img = torch.from_numpy(np.asarray(obs['pixels'])).permute(2, 0, 1).float() / 255.0
        img = F.interpolate(img.unsqueeze(0), size=(96, 96), mode='bilinear',
                            align_corners=False)              ### checkpoint expects 96x96
        state = torch.from_numpy(np.asarray(obs['agent_pos'], np.float32)).unsqueeze(0)
        return {'observation.image': img.to(device),
                'observation.state': state.to(device)}

    @torch.no_grad()
    def policy_fn(obs: dict) -> np.ndarray:
        batch = _batch(obs)
        if hasattr(policy, 'predict_action_chunk'):
            chunk = policy.predict_action_chunk(batch)        ### preferred (returns [B,H,A])
        else:                                                  # older LeRobot fallback
            chunk = policy.diffusion.generate_actions(batch)
        return chunk[0].detach().cpu().numpy()

    return policy_fn, action_dim, horizon


# -------------------------------------------------------------------- sweep
def sweep(args) -> None:
    policy_fn, _action_dim, _horizon = load_policy(args.device)
    strategies = [s.strip() for s in args.strategies.split(',')]
    latencies = [float(x) for x in args.latencies.split(',')]

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['strategy', 'latency_steps', 'jitter_steps', 'drop_prob', 'episodes',
                    'success_rate', 'mean_max_coverage', 'mean_steps'])
        for strat in strategies:
            for lat in latencies:
                results: list[EpisodeResult] = []
                for ep in range(args.episodes):
                    env = GymPushTEnv()
                    delay = DelayModel(latency_steps=lat, jitter_steps=args.jitter,
                                       drop_prob=args.drop_prob, seed=ep)
                    results.append(run_episode(
                        env, policy_fn, make_strategy(strat), delay,
                        replan_period=args.replan_period, max_steps=args.max_steps))
                sr = float(np.mean([r.success for r in results]))
                cov = float(np.mean([r.max_reward for r in results]))
                steps = float(np.mean([r.steps for r in results]))
                w.writerow([strat, lat, args.jitter, args.drop_prob, args.episodes,
                            f'{sr:.3f}', f'{cov:.3f}', f'{steps:.1f}'])
                print(f'{strat:18s} lat={lat:5.1f}  success={sr*100:5.1f}%  '
                      f'coverage={cov:.3f}  steps={steps:.0f}')
    print(f'\nwrote {args.out}')
    if args.plot:
        _plot(args.out, args.out.replace('.csv', '.png'))


def _plot(csv_path: str, png_path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(csv_path)))
    strategies = sorted({r['strategy'] for r in rows})
    plt.figure(figsize=(7, 4.5))
    for strat in strategies:
        pts = [(float(r['latency_steps']), float(r['success_rate']))
               for r in rows if r['strategy'] == strat]
        pts.sort()
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker='o', label=strat)
    plt.xlabel('injected latency (control steps)')
    plt.ylabel('success rate')
    plt.title('PushT: latency-robust chunking strategies')
    plt.ylim(0, 1.02)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=130)
    print(f'wrote {png_path}')


def main() -> None:
    p = argparse.ArgumentParser(description='PushT latency x chunk-strategy sanity sweep')
    p.add_argument('--strategies',
                   default='synchronous,naive_async,temporal_ensemble,rtc_freeze')
    p.add_argument('--latencies', default='0,2,4,8,12', help='comma-separated, in control steps')
    p.add_argument('--jitter', type=float, default=1.0, help='delay std, control steps')
    p.add_argument('--drop_prob', type=float, default=0.0)
    p.add_argument('--episodes', type=int, default=30)
    p.add_argument('--replan_period', type=int, default=8)
    p.add_argument('--max_steps', type=int, default=300)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default='results/pusht.csv')
    p.add_argument('--plot', action='store_true')
    sweep(p.parse_args())


if __name__ == '__main__':
    main()
