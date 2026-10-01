"""Chunk execution past the half-horizon bound, on the official RTC Kinetix code.

`eval_flow.eval` in Physical-Intelligence/real-time-chunking-kinetix evaluates chunk-execution
methods at inference delay `d` and execution horizon `s`, but only where `s >= d` and `d <= H - s`
(its main sweep, and an assertion). That is the published envelope, `d <= H/2`. This module is
the same evaluation with the envelope removed, plus the two things the paper needs that the
official loop does not have:

  * a SYNCHRONOUS arm: request, wait `d` ticks, play `s` actions from the chunk's start;
  * a WORLD-SPEED knob `alpha` for the ticks on which the robot has no action to play. The world
    advances `alpha` env steps per idle tick (a deterministic accumulator): 1 = Kinetix as
    published (the world keeps moving while you wait), 0 = the world is frozen (waiting is free,
    the quasi-static case). Every idle tick still counts toward the episode DEADLINE, which is
    measured in control ticks, so a frozen world still costs time.

Nothing in `model.py` is touched: RTC is `FlowPolicy.realtime_action` and the plain sampler is
`FlowPolicy.action`, called with the official arguments. Only the loop around them is ours.

Timing model (one request in flight, a request issued at tick t arrives at tick t + d):

    overlap methods (naive, rtc, hard)   issue every P = max(s, d) ticks (s when d <= s: the
        official schedule; every arrival when d > s: the slot is the bottleneck). A chunk issued
        at t0 is TIME-ALIGNED: its entry j is the action for tick t0 + j, as in the official
        code. While it is in flight the previous chunk keeps playing; if that runs out first the
        robot is starved and idles. Entries for ticks that passed before arrival are skipped.
    sync                                  issue at t0, idle d ticks, play entries 0..s-1, repeat:
        P = d + s. The chunk is played from its start, whatever the delay.

RTC's guidance is given only the previous chunk's actions that actually exist in the new
chunk's frame (`inference_delay = min(d, valid)`, `prefix_attention_horizon = H - P`). The
official loop pads with zeros past the end of the previous chunk; passing the padded zeros as
guidance targets would pull the new chunk toward a zero action, which is not RTC's intent. Where
`d <= s` and `d <= H - s` this reduces exactly to the official call.

An idle tick sends the zero action (motors and thrusters off) when the world steps. The official
NoisyActionWrapper still adds its noise, as on every step.

Outcome per episode: whether the FIRST episode was solved within `deadline` control ticks, and
at which tick it ended, so completion-time (survival) curves come out of the same run.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

import kinetix.environment.wrappers as wrappers
import model as _model
import train_expert

Method = Literal['sync', 'naive', 'rtc', 'hard']
OLD, NEW, IDLE = 0, 1, 2


@dataclasses.dataclass(frozen=True)
class RunConfig:
    method: Method = 'rtc'
    delay: int = 0                 # d: request -> arrival, in control ticks
    execute_horizon: int = 1       # s: actions played per chunk before the next request (min)
    world_speed: float = 1.0       # alpha: env steps per idle tick
    deadline: int = 256            # episode budget in control ticks (idle ticks included)
    # the episode clock starts when the first chunk is ready (the official convention, and what a
    # robot does: it does not start until the policy has answered once). False makes every chunk,
    # the first included, cost d; that adds a start-up stall which in a moving world costs more
    # than the delay being studied (pilot 2026-09-28: RTC 0.36 vs 0.62 at d = s = 4)
    free_first_chunk: bool = True
    num_evals: int = 2048
    num_flow_steps: int = 5
    max_guidance_weight: float = 5.0   # official RealtimeMethodConfig default
    # what an idle tick sends when the world steps: 'zero' (motors and thrusters off, the main
    # setting) or 'repeat' (the last action played, a robustness check on that choice)
    idle_action: Literal['zero', 'repeat'] = 'zero'
    model: _model.ModelConfig = _model.ModelConfig()


def cycle_plan(method: str, d: int, s: int, H: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Static per-cycle schedule: (period P, source per tick, chunk index per tick).

    Sources index into the chunk frame of the cycle: OLD = the previous chunk, already shifted so
    its entry k is this cycle's tick k; NEW = the chunk requested at this cycle's first tick.
    Whether an OLD entry exists is a runtime mask (the first cycle differs); a NEW index past the
    chunk is marked IDLE here.
    """
    if s < 1:
        raise ValueError(f'execute_horizon must be >= 1, got {s}')
    if method == 'sync':
        P = d + s
        src = [IDLE] * d + [NEW] * s
        idx = [0] * d + list(range(s))
    elif method in ('naive', 'rtc', 'hard'):
        P = max(s, d)
        src = [OLD if k < d else NEW for k in range(P)]
        idx = list(range(P))
    else:
        raise ValueError(f'unknown method {method!r}')
    src = [IDLE if (sr == NEW and i >= H) else sr for sr, i in zip(src, idx)]
    return P, np.asarray(src, np.int32), np.asarray(idx, np.int32)


def make_env(env, num_evals: int):
    """The official evaluation wrapper stack, verbatim from eval_flow.eval."""
    return train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env))),
        num_evals)


def run(config: RunConfig, env, rng, level, policy: _model.FlowPolicy, env_params):
    """Evaluate one level. Returns per-episode arrays: solved (bool), end_tick (int, -1 = never),
    plus the fraction of ticks the robot was idle."""
    env = make_env(env, config.num_evals)
    H, A, N = policy.action_chunk_size, policy.action_dim, config.num_evals
    d, s = config.delay, config.execute_horizon
    P, src_np, idx_np = cycle_plan(config.method, d, s, H)
    src, idx = jnp.asarray(src_np), jnp.asarray(idx_np)
    alpha = float(config.world_speed)

    def infer(key, obs, old, old_valid):
        if config.method in ('naive', 'sync'):
            return policy.action(key, obs, config.num_flow_steps)
        schedule = 'exp' if config.method == 'rtc' else 'zeros'
        n_valid = jnp.sum(old_valid).astype(jnp.int32)
        return policy.realtime_action(
            key, obs, config.num_flow_steps, old * old_valid[None, :, None],
            jnp.minimum(d, n_valid), max(H - P, 0), schedule, config.max_guidance_weight)

    def tick(carry, plan):
        rng, obs, state, old, old_valid, new, acc, t, finished, solved, end_tick, idle, last = carry
        k_src, k_idx = plan
        use_old = (k_src == OLD) & (k_idx < H) & old_valid[jnp.minimum(k_idx, H - 1)]
        use_new = k_src == NEW
        acting = use_old | use_new
        a_old = old[:, jnp.minimum(k_idx, H - 1)]
        a_new = new[:, jnp.minimum(k_idx, H - 1)]
        rest = last if config.idle_action == 'repeat' else jnp.zeros((N, A))
        action = jnp.where(use_old, a_old, jnp.where(use_new, a_new, rest))
        last = jnp.where(acting, action, last)
        # world advances on every acting tick, and alpha steps per idle tick
        acc_next = jnp.where(acting, acc, acc + alpha)
        world_steps = acting | (acc_next >= 1.0 - 1e-6)
        acc_next = jnp.where(~acting & world_steps, acc_next - 1.0, acc_next)

        rng, key = jax.random.split(rng)
        next_obs, next_state, _r, done, info = env.step(key, state, action, env_params)
        keep = lambda x, y: jnp.where(  # noqa: E731
            jnp.reshape(world_steps, (1,) * x.ndim), x, y)
        obs = keep(next_obs, obs)
        state = jax.tree.map(keep, next_state, state)
        done = done & world_steps & (t < config.deadline)

        newly = done & ~finished
        solved = jnp.where(newly, info['returned_episode_solved'] > 0, solved)
        end_tick = jnp.where(newly, t, end_tick)
        finished = finished | done
        idle = idle + jnp.where(~acting & (t < config.deadline), 1, 0)
        return (rng, obs, state, old, old_valid, new, acc_next, t + 1, finished, solved,
                end_tick, idle, last), None

    def cycle(carry, _):
        rng, obs, state, old, old_valid, acc, t, finished, solved, end_tick, idle, last = carry
        rng, key = jax.random.split(rng)
        new = infer(key, obs, old, old_valid)
        inner = (rng, obs, state, old, old_valid, new, acc, t, finished, solved, end_tick, idle,
                 last)
        inner, _ = jax.lax.scan(tick, inner, (src, idx))
        rng, obs, state, _, _, new, acc, t, finished, solved, end_tick, idle, last = inner
        # the new chunk, shifted into the next cycle's frame; entries past its end do not exist
        old = jnp.concatenate([new[:, P:], jnp.zeros((N, min(P, H), A))], axis=1)[:, :H]
        old_valid = jnp.arange(H) < (H - P)
        return (rng, obs, state, old, old_valid, acc, t, finished, solved, end_tick, idle,
                last), None

    rng, key = jax.random.split(rng)
    obs, state = env.reset_to_level(key, level, env_params)
    rng, key = jax.random.split(rng)
    first = policy.action(key, obs, config.num_flow_steps)       # as the official eval does
    zeros_state = (jnp.float32(0.0), jnp.int32(0), jnp.zeros(N, bool), jnp.zeros(N, bool),
                   jnp.full(N, -1, jnp.int32), jnp.int32(0), jnp.zeros((N, A)))
    if config.free_first_chunk and config.method == 'sync':
        # sync plays the first chunk from its start, then enters its request/wait/play cycle
        acc, t, finished, solved, end_tick, idle, last = zeros_state
        pre = (rng, obs, state, first, jnp.ones(H, bool), first, acc, t, finished, solved,
               end_tick, idle, last)
        pre, _ = jax.lax.scan(tick, pre, (jnp.full(min(s, H), NEW, jnp.int32),
                                          jnp.arange(min(s, H), dtype=jnp.int32)))
        rng, obs, state, _, _, _, *zeros_state = pre
        old, old_valid = jnp.zeros_like(first), jnp.zeros(H, bool)
    elif config.free_first_chunk:
        old, old_valid = first, jnp.ones(H, bool)
    else:
        old, old_valid = jnp.zeros_like(first), jnp.zeros(H, bool)

    n_cycles = math.ceil(config.deadline / P) + 1
    carry = (rng, obs, state, old, old_valid, *zeros_state)
    carry, _ = jax.lax.scan(cycle, carry, None, length=n_cycles)
    *_, finished, solved, end_tick, idle, _last = carry
    return {'solved': solved & finished, 'end_tick': end_tick,
            'idle_frac': idle / config.deadline}
