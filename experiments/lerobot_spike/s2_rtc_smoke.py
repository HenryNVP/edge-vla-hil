"""Spike S2: LeRobot's own RTC on SmolVLA, on real LIBERO observations.

Builds env, policy and processors exactly as `lerobot-eval` does (same config parser, same
factories), switches RTC on through the policy config, and checks three things:

  * a chunk comes back with the expected shape, with and without RTC guidance;
  * with guidance, the new chunk's first `d` entries follow the previous chunk's leftover
    (distance to the leftover drops versus unguided sampling from the same noise);
  * latency and peak GPU memory per call on this laptop.

    cd external/lerobot-spike && MUJOCO_GL=egl .venv/bin/python \
        ../../experiments/lerobot_spike/s2_rtc_smoke.py
"""
import time

import sys

import torch

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.rtc import RTCConfig

sys.argv = [sys.argv[0], '--policy.path=HuggingFaceVLA/smolvla_libero', '--env.type=libero',
            '--env.task=libero_object', '--eval.batch_size=1', '--eval.n_episodes=1']


@parser.wrap()   # the same parser lerobot-eval uses; it resolves --policy.path
def _parse(cfg: EvalPipelineConfig) -> EvalPipelineConfig:
    return cfg


cfg = _parse()
envs = make_env(cfg.env, n_envs=1)
env = next(iter(next(iter(envs.values())).values()))
policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
policy.eval()
pre, _post = make_pre_post_processors(
    policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
    preprocessor_overrides={'device_processor': {'device': str(policy.config.device)}})
env_pre, _env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

H = policy.config.chunk_size
print(f'chunk_size={H} n_action_steps={policy.config.n_action_steps} '
      f'num_steps={policy.config.num_steps}')
policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=H, max_guidance_weight=10.0)
policy.init_rtc_processor()

obs, _ = env.reset(seed=0)
o = preprocess_observation(obs)
o['task'] = list(env.call('task_description'))
o = pre(env_pre(o))

torch.cuda.reset_peak_memory_stats()
# no inference_mode: RTC's guidance differentiates through the denoiser (a VJP), which
# inference_mode forbids even inside torch.enable_grad
if True:
    t0 = time.perf_counter()
    first = policy.predict_action_chunk(o)
    torch.cuda.synchronize()
    print(f'unguided chunk {tuple(first.shape)} in {1e3 * (time.perf_counter() - t0):.0f} ms')
    for d in (5, 25, 40):
        leftover = first[:, d:]           # what the previous chunk still had when this was asked
        # the model samples in its padded action space (max_action_dim), not the 7-D one
        noise = torch.randn(first.shape[0], H, policy.config.max_action_dim,
                            device=first.device, dtype=first.dtype)
        t0 = time.perf_counter()
        guided = policy.predict_action_chunk(o, noise=noise.clone(), inference_delay=d,
                                             prev_chunk_left_over=leftover)
        torch.cuda.synchronize()
        ms = 1e3 * (time.perf_counter() - t0)
        free = policy.predict_action_chunk(o, noise=noise.clone())
        n = min(d, leftover.shape[1])
        dg = (guided[:, :n] - leftover[:, :n]).norm(dim=-1).mean().item()
        df = (free[:, :n] - leftover[:, :n]).norm(dim=-1).mean().item()
        print(f'd={d:2d} (d/H={d / H:.2f}) guided {ms:.0f} ms; distance of first {n} entries to '
              f'leftover: guided {dg:.3f} vs unguided {df:.3f}')
print(f'peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
env.close()
