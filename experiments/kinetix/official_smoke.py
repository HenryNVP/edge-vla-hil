"""Smoke test: the OFFICIAL RTC Kinetix eval, unmodified, at a few (delay, horizon) points.

Runs `eval_flow.eval` from Physical-Intelligence/real-time-chunking-kinetix exactly as its own
`main` does (same levels, same checkpoint selection, same method configs), but at a handful of
settings and fewer trials, to confirm the install and the downloaded `bc/31` policies before any
extension is built on them.

    cd external/rtc-kinetix
    uv run --python 3.11 python ../../experiments/kinetix/official_smoke.py --num-evals 256
"""
import argparse
import dataclasses
import sys
import time

sys.path.insert(0, 'src')

import eval_flow  # noqa: E402  (official module, unmodified)
import jax  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--num-evals', type=int, default=256)
p.add_argument('--run-path', default='assets/bc')
args = p.parse_args()

captured = {}
real_cli_main = eval_flow.main


def run(points):
    # reuse the official main's setup by calling it with a patched sweep loop is not possible
    # (the loop is inline), so replicate its setup lines verbatim below.
    import functools
    import pathlib
    import pickle

    import flax.nnx as nnx
    import jax.numpy as jnp
    import kinetix.environment.env as kenv
    import kinetix.environment.env_state as kenv_state
    from jax.experimental import shard_map

    import model as _model
    import train_expert

    level_paths = eval_flow.main.__defaults__[1]
    config = eval_flow.EvalConfig(num_evals=args.num_evals)
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS,
                                                   frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)
    env = kenv.make_kinetix_env_from_name('Kinetix-Symbolic-Continuous-v1',
                                          static_env_params=static_env_params)
    state_dicts = []
    for level_path in level_paths:
        level_name = level_path.replace('/', '_').replace('.json', '')
        log_dirs = sorted((p for p in pathlib.Path(args.run_path).iterdir()
                           if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
        with (log_dirs[config.step] / 'policies' / f'{level_name}.pkl').open('rb') as f:
            state_dicts.append(pickle.load(f))
    state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))
    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0),
                             jax.tree.map(lambda x: x[0], levels), env_params)[0].shape[-1]
    action_dim = env.action_space(env_params).shape[0]
    mesh = jax.make_mesh((jax.local_device_count(),), ('x',))
    pspec = jax.sharding.PartitionSpec('x')
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    @functools.partial(jax.jit, static_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @functools.partial(shard_map.shard_map, mesh=mesh, in_specs=(None, pspec, pspec, pspec),
                       out_specs=pspec)
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def _eval(config, rng, level, state_dict):
        policy = _model.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=config.model,
                                   rngs=nnx.Rngs(rng))
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        policy = nnx.merge(graphdef, state)
        info, _ = eval_flow.eval(config, env, rng, level, policy, env_params, static_env_params)
        return info

    rngs = jax.random.split(jax.random.key(0), len(level_paths))
    for d, s, method in points:
        c = dataclasses.replace(config, inference_delay=d, execute_horizon=s, method=method)
        t0 = time.time()
        out = jax.device_get(_eval(c, rngs, levels, state_dicts))
        solved = out['returned_episode_solved']
        name = type(method).__name__.replace('MethodConfig', '')
        print(f'd={d} s={s} {name:9s} mean solve {solved.mean():.3f}  per level '
              + ' '.join(f'{v:.2f}' for v in solved) + f'  ({time.time() - t0:.0f}s)', flush=True)


run([(0, 1, eval_flow.NaiveMethodConfig()),
     (1, 4, eval_flow.NaiveMethodConfig()),
     (1, 4, eval_flow.RealtimeMethodConfig()),
     (4, 4, eval_flow.NaiveMethodConfig()),
     (4, 4, eval_flow.RealtimeMethodConfig())])
