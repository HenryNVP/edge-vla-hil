"""SmolVLA + LeRobot's RTC on LIBERO, driven by the unchanged `chunk_executor` strategies.

Plug-and-play at both ends, customisation only in the plugs:

  * `LeRobotChunkPolicy` is a `ChunkPolicy` over a LeRobot policy. `predict` is
    `predict_action_chunk`; `predict_inpaint` (what the RTC executor calls) is
    `predict_action_chunk(inference_delay=d, prev_chunk_left_over=prefix)` with RTC enabled in
    the policy config, i.e. LeRobot's own guidance and weight schedule. The executor keeps chunks in
    the MODEL's action space, so the leftover it hands back is exactly what LeRobot expects.
  * `LiberoStepEnv` wraps LeRobot's LIBERO env (the single env, not the vector wrapper, so there
    is no auto-reset) and applies LeRobot's own postprocessors to each action as it is stepped.

The loop, the delay model and the strategies are `scripts/sil_step.py` and
`evh_controller.chunk_executor`, unchanged.

Two conventions, both forced by LIBERO's action space (delta pose + absolute gripper):
  * an idle tick sends zero motion and REPEATS the last gripper command. A zero gripper value
    would half-open the hand and drop whatever it holds;
  * the deadline is the suite's own episode limit (280 steps for libero_object), in ticks.

Scene pairing: each LIBERO env walks a fixed list of initial states, one per reset. Rewinding
`init_state_id` at the start of every cell makes episode k the same initial state in every cell,
and torch is reseeded per episode so the first chunk's noise is paired too.

Validation cell: synchronous, chunk 1, d = 1, frozen world. That is LeRobot's own evaluation
(replan every tick, the world waiting for the policy), so it must reproduce `lerobot-eval`.

    cd external/lerobot-spike && MUJOCO_GL=egl .venv/bin/python ../../experiments/lerobot/adapter.py \
        --strategies synchronous --delays 1 --chunk 1 --world-speed 0 --out ../../outputs/libero/val.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_REPO, 'ros2_ws', 'src', 'evh_controller'),
           os.path.join(_REPO, 'ros2_ws', 'src', 'evh_plant'), os.path.join(_REPO, 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def rtc_args(weights) -> tuple[int, int]:
    """Our executor's soft mask -> LeRobot's (inference_delay, attended length).

    RTCExecutor freezes the entries that will play while the request is in flight (weight 1)
    and decays the rest; LeRobot takes the frozen count as `inference_delay` and builds its own
    decay over the leftover it is given. Only the frozen count crosses over.
    """
    w = np.asarray(weights, dtype=float)
    return int(np.sum(w >= 1.0 - 1e-6)), int(np.sum(w > 0))


class LeRobotChunkPolicy:
    """`ChunkPolicy` over a LeRobot policy; chunks are in the model's action space."""
    n_obs_steps = 1
    needs_wrist = False
    absolute_actions = False       # LIBERO: delta end-effector actions
    guided_resampling = True       # LeRobot RTC guides inside the flow integration

    def __init__(self, policy, preprocess, max_chunk_actions: int = 0) -> None:
        self.policy = policy
        self.preprocess = preprocess                # raw env obs -> policy batch
        H = int(policy.config.chunk_size)
        self.chunk_size = min(max_chunk_actions, H) if max_chunk_actions else H
        self.action_dim = int(policy.config.output_features['action'].shape[0])
        self.task = ''
        self.rtc_attend = 10          # LeRobot's execution_horizon; set from --rtc-horizon

    def new_episode(self, task: str, seed: int) -> None:
        import torch
        self.task = task
        self.policy.reset()
        torch.manual_seed(seed)

    def _chunk(self, obs, **kw) -> np.ndarray:
        batch = self.preprocess(obs['raw'][-1], self.task)
        # no inference_mode: RTC's guidance differentiates through the denoiser
        out = self.policy.predict_action_chunk(batch, **kw)
        return out[0, :self.chunk_size].detach().float().cpu().numpy()

    def predict(self, obs) -> np.ndarray:
        return self._chunk(obs)

    def predict_inpaint(self, obs, prefix, weights) -> np.ndarray:
        import torch
        d, _paper_extent = rtc_args(weights)
        # Only the frozen count crosses over. LeRobot's `execution_horizon` is how many leftover
        # actions the new chunk stays consistent with (its docs: 8-12), NOT the RTC paper's
        # prefix-attention end H - s: passing H - s = 40 of a 50-action chunk made every new
        # chunk copy the old plan, executed actions never reflected new observations, and
        # success went to 0/6 (vs 6/6 at near-zero guidance), 2026-09-30.
        self.policy.rtc_processor.rtc_config.execution_horizon = max(self.rtc_attend, d)
        p = self.policy.config
        leftover = torch.as_tensor(np.asarray(prefix, np.float32)[None],
                                   device=p.device)
        return self._chunk(obs, inference_delay=d, prev_chunk_left_over=leftover)


class LiberoStepEnv:
    """LeRobot's LIBERO env with its postprocessors applied per step; `None` means hold."""

    def __init__(self, env, postprocess) -> None:
        self.env = env
        self.postprocess = postprocess            # model-space action (7,) -> env action (7,)
        self.success = False
        self._grip = -1.0                         # LIBERO: -1 open, +1 closed

    def reset_pairing(self) -> None:
        self.env.init_state_id = self.env.episode_index

    def reset(self):
        obs, _info = self.env.reset()
        self.success = False
        self._grip = -1.0
        self._over = False
        self._obs = obs
        return obs

    def step(self, action):
        # the underlying robosuite env has its own horizon and raises on any action after it;
        # once the env has ended the episode, later ticks are no-ops (it is a failure unless it
        # already succeeded) and the loop runs out its deadline
        if self._over:
            return self._obs, 0.0, True, {}
        if action is None:
            act = np.zeros(7, np.float32)
            act[6] = self._grip
        else:
            act = np.asarray(self.postprocess(action), np.float32).reshape(-1)
            self._grip = float(act[6])
        obs, reward, terminated, truncated, info = self.env.step(act)
        self.success = bool(info.get('is_success', False))
        self._over = bool(terminated or truncated)
        self._obs = obs
        return obs, reward, self._over, info


def batch1(tree):
    """Add a leading batch axis to every array in a nested observation dict: the single env
    returns what the vector env would, minus the batch axis `preprocess_observation` expects."""
    if isinstance(tree, dict):
        return {k: batch1(v) for k, v in tree.items()}
    return np.expand_dims(np.asarray(tree), 0)


def raw_frames(obs, _needs_wrist):
    return obs, None, None


def raw_stack(hist, _needs_wrist):
    return {'raw': [h[0] for h in hist]}


EPISODE_COLUMNS = ['condition', 'task', 'scene', 'success', 'wall_s', 'sim_s', 'env_steps',
                   'idle_frac', 'requests']


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', required=True)
    p.add_argument('--policy', default='HuggingFaceVLA/smolvla_libero')
    p.add_argument('--suite', default='libero_object')
    p.add_argument('--tasks', default='0,1,2,3,4,5,6,7,8,9')
    p.add_argument('--episodes', type=int, default=10, help='per task')
    p.add_argument('--strategies', default='synchronous,naive_async,rtc')
    p.add_argument('--delays', default='5,10,25',
                   help='ticks, or trace:<session>:<scale> to replay a recorded WiFi session')
    p.add_argument('--chunk', type=int, default=50, help='actions kept per chunk (<= 50)')
    p.add_argument('--world-speed', type=float, default=1.0)
    p.add_argument('--deadline', type=int, default=280, help='ticks; libero_object limit is 280')
    p.add_argument('--guidance', type=float, default=10.0, help='LeRobot docs: 10 for 10 steps')
    p.add_argument('--s-wait', type=int, default=10,
                   help='coverage-aware family: waiting-mode chunk length (the policy tolerates '
                        'about 10 open-loop steps)')
    p.add_argument('--rtc-horizon', type=int, default=10,
                   help='RTC: actions played between requests and the soft-mask horizon s '
                        '(LeRobot docs: 8-12)')
    args = p.parse_args()
    import run_manifest
    run_manifest.record(args.out, {'kind': 'libero_lerobot', 'config': vars(args)})

    import torch
    from evh_controller.chunk_executor import RTCExecutor, make_executor
    from lerobot.configs import parser
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.rtc import RTCConfig
    from lerobot.configs.types import RTCAttentionSchedule
    from sil_step import StepWorker, TraceDelay, make_delay, make_strategy, run_episode

    tasks = [int(x) for x in args.tasks.split(',')]
    sys.argv = [sys.argv[0], f'--policy.path={args.policy}', '--env.type=libero',
                f'--env.task={args.suite}', f'--env.task_ids={tasks}',
                # LIBERO's own limit (280 for libero_object) was set for zero delay; the env must
                # run as long as our tick deadline, and success at 280 is read off afterwards
                f'--env.episode_length={args.deadline}', '--eval.batch_size=1',
                '--eval.n_episodes=1']

    def _parse(cfg):
        return cfg

    # the parser lerobot-eval uses; it reads the annotation, which `from __future__ import
    # annotations` would otherwise leave as an unresolvable string for a locally imported class
    _parse.__annotations__ = {'cfg': EvalPipelineConfig, 'return': EvalPipelineConfig}
    cfg = parser.wrap()(_parse)()
    envs = make_env(cfg.env, n_envs=1)[args.suite]
    lr_policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    lr_policy.eval()
    lr_policy.config.rtc_config = RTCConfig(
        enabled=True, execution_horizon=int(lr_policy.config.chunk_size),
        max_guidance_weight=args.guidance, prefix_attention_schedule=RTCAttentionSchedule.EXP)
    lr_policy.init_rtc_processor()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={'device_processor': {'device': str(lr_policy.config.device)}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    def preprocess(raw, task):
        o = preprocess_observation(batch1(raw))
        o['task'] = [task]
        return pre(env_pre(o))

    def postprocess(model_action):
        a = post(torch.as_tensor(np.asarray(model_action, np.float32)[None],
                                 device=lr_policy.config.device))
        a = env_post({'action': a})['action']
        return a.detach().cpu().numpy()[0]

    policy = LeRobotChunkPolicy(lr_policy, preprocess, max_chunk_actions=args.chunk)
    policy.rtc_attend = args.rtc_horizon
    steppers = {t: LiberoStepEnv(envs[t].envs[0], postprocess) for t in tasks}

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(EPISODE_COLUMNS)
        for d in args.delays.split(','):
            for strategy in args.strategies.split(','):
                label = (f'strat={strategy}_d={d}_k={policy.chunk_size}_alpha={args.world_speed:g}'
                         f'_deadline={args.deadline}'
                         + (f'_s={args.rtc_horizon}' if strategy == 'rtc' else ''))
                wins = n = 0
                t0 = time.time()
                for t in tasks:
                    env = steppers[t]
                    env.reset_pairing()
                    task_text = env.env.task_description
                    for scene in range(args.episodes):
                        obs = env.reset()
                        policy.new_episode(task_text, seed=scene)
                        delay = make_delay(d)
                        if isinstance(delay, TraceDelay):
                            delay.for_episode(1000 * t + scene)   # paired across strategies
                        worker = StepWorker(policy, delay)
                        if strategy == 'rtc':
                            executor = RTCExecutor(worker, policy,
                                                   exec_horizon_min=args.rtc_horizon,
                                                   replan_every=args.rtc_horizon)
                        else:
                            executor = make_strategy(strategy, worker, policy, args.s_wait,
                                                     s_min=args.rtc_horizon)
                        ep = run_episode(env, obs, policy, executor, worker,
                                         deadline_ticks=args.deadline,
                                         world_speed=args.world_speed, absolute=False,
                                         is_success=lambda e: e.success,
                                         frame_fn=raw_frames, stack_fn=raw_stack,
                                         hold_fn=lambda _o: None)
                        wins += ep.success
                        n += 1
                        w.writerow([label, t, scene, int(ep.success), f'{ep.wall_s:.2f}',
                                    f'{ep.sim_s:.3f}', ep.env_steps,
                                    f'{ep.idle / max(ep.ticks, 1):.3f}', ep.requests])
                        f.flush()
                print(f'{label}: {wins}/{n} ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
