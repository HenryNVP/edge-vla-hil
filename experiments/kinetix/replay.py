"""Replay a strategy's tick schedule (`schedule.py`) on Kinetix with the real policy and env.

The per-tick world model is `executor.run`'s, unchanged: the world steps on every acting tick and
`alpha` times per idle tick, idle ticks send the zero (or repeated) action, the episode deadline
counts every tick, and only the first episode's outcome is kept. What differs is only where the
schedule comes from: `executor.run` builds it from a fixed cycle (`cycle_plan`); here it is the
Python executor's own, so any unguided strategy in `chunk_executor` runs on the map without a JAX
port. Equivalence with `executor.run` on its own schedules is checked in `test_replay.py`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

import executor
import schedule as _schedule


def run(config: executor.RunConfig, env, rng, level, policy, env_params,
        sched: _schedule.Schedule | None = None):
    """Same contract as `executor.run`. `sched` defaults to simulating `config.method`."""
    H, A, N = policy.action_chunk_size, policy.action_dim, config.num_evals
    if sched is None:
        sched = _schedule.simulate(config.method, config.delay, config.execute_horizon, H,
                                   config.deadline)
    env = executor.make_env(env, N)
    alpha = float(config.world_speed)
    xs = (jnp.asarray(sched.issue), jnp.asarray(sched.slot), jnp.asarray(sched.entry))

    def tick(carry, plan):
        rng, obs, state, chunks, acc, t, finished, solved, end_tick, idle, last = carry
        issue, slot, entry = plan

        def request(args):
            rng, chunks = args
            rng, key = jax.random.split(rng)
            new = policy.action(key, obs, config.num_flow_steps)
            return rng, chunks.at[jnp.maximum(issue, 0)].set(new)

        rng, chunks = jax.lax.cond(issue >= 0, request, lambda a: a, (rng, chunks))
        acting = slot >= 0
        rest = last if config.idle_action == 'repeat' else jnp.zeros((N, A))
        action = jnp.where(acting, chunks[jnp.maximum(slot, 0)][:, entry], rest)
        last = jnp.where(acting, action, last)
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
        return (rng, obs, state, chunks, acc_next, t + 1, finished, solved, end_tick, idle,
                last), None

    rng, key = jax.random.split(rng)
    obs, state = env.reset_to_level(key, level, env_params)
    rng, key = jax.random.split(rng)
    first = policy.action(key, obs, config.num_flow_steps)       # the free first chunk, slot 0
    chunks = jnp.zeros((_schedule.RING, N, H, A)).at[0].set(first)
    carry = (rng, obs, state, chunks, jnp.float32(0.0), jnp.int32(0), jnp.zeros(N, bool),
             jnp.zeros(N, bool), jnp.full(N, -1, jnp.int32), jnp.int32(0), jnp.zeros((N, A)))
    carry, _ = jax.lax.scan(tick, carry, xs)
    *_, finished, solved, end_tick, idle, _last = carry
    return {'solved': solved & finished, 'end_tick': end_tick,
            'idle_frac': idle / config.deadline}
