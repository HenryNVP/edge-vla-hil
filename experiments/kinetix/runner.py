"""Load the official levels and `bc/` policies, and evaluate RunConfigs on all levels at once.

The setup lines are the official `eval_flow.main`'s, unchanged: same 12 levels, same env params,
same checkpoint selection (`step=-1`, the last epoch directory), same per-level vmap and sharding.
Run from `external/rtc-kinetix` (the level paths are relative to it) with its own environment:

    cd external/rtc-kinetix
    uv run --python 3.11 python ../../experiments/kinetix/runner.py --out ../../outputs/kinetix/x.csv \
        --methods sync,naive,rtc --delays 0,1,2,4,6,8,10 --horizons 4 --world-speeds 1
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import os
import pathlib
import pickle
import sys
import time

sys.path.insert(0, 'src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts'))

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import kinetix.environment.env as kenv  # noqa: E402
import kinetix.environment.env_state as kenv_state  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import shard_map  # noqa: E402

import eval_flow  # noqa: E402
import model as _model  # noqa: E402
import train_expert  # noqa: E402

import executor  # noqa: E402

LEVEL_PATHS = eval_flow.main.__defaults__[1]


@dataclasses.dataclass
class Setup:
    env: object
    env_params: object
    static_env_params: object
    levels: object
    state_dicts: object
    obs_dim: int
    action_dim: int
    level_paths: tuple


def load(run_path: str = 'assets/bc', level_paths=LEVEL_PATHS, step: int = -1) -> Setup:
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
        log_dirs = sorted((p for p in pathlib.Path(run_path).iterdir()
                           if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
        with (log_dirs[step] / 'policies' / f'{level_name}.pkl').open('rb') as f:
            state_dicts.append(pickle.load(f))
    state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))
    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0),
                             jax.tree.map(lambda x: x[0], levels), env_params)[0].shape[-1]
    action_dim = env.action_space(env_params).shape[0]
    return Setup(env, env_params, static_env_params, levels, state_dicts, obs_dim, action_dim,
                 tuple(level_paths))


def _sharded(fn, check_rep: bool = True):
    # check_rep=False only for our loop: its carry starts from constant scalars (tick counter,
    # world-speed accumulator), which shard_map's replication check cannot type. It is a static
    # check, not a numerical change; the official path keeps the official setting.
    mesh = jax.make_mesh((jax.local_device_count(),), ('x',))
    pspec = jax.sharding.PartitionSpec('x')
    sharding = jax.sharding.NamedSharding(mesh, pspec)
    return functools.partial(jax.jit, static_argnums=(0,), in_shardings=sharding,
                             out_shardings=sharding)(
        functools.partial(shard_map.shard_map, mesh=mesh, in_specs=(None, pspec, pspec, pspec),
                          out_specs=pspec, check_rep=check_rep)(fn))


def _policy(setup: Setup, model_config, rng, state_dict):
    policy = _model.FlowPolicy(obs_dim=setup.obs_dim, action_dim=setup.action_dim,
                               config=model_config, rngs=nnx.Rngs(rng))
    graphdef, state = nnx.split(policy)
    state.replace_by_pure_dict(state_dict)
    return nnx.merge(graphdef, state)


def make_ours(setup: Setup):
    @functools.partial(_sharded, check_rep=False)
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def _run(config: executor.RunConfig, rng, level, state_dict):
        params = setup.env_params
        if config.deadline > params.max_timesteps:
            params = params.replace(max_timesteps=config.deadline)
        return executor.run(config, setup.env, rng, level,
                            _policy(setup, config.model, rng, state_dict), params)
    return _run


def make_official(setup: Setup):
    @_sharded
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def _eval(config: eval_flow.EvalConfig, rng, level, state_dict):
        info, _ = eval_flow.eval(config, setup.env, rng, level,
                                 _policy(setup, config.model, rng, state_dict),
                                 setup.env_params, setup.static_env_params)
        return info
    return _eval


def evaluate(fn, config, setup: Setup, rngs, group: int = 12):
    """Run `fn` over all levels, `group` levels at a time.

    RTC's guidance differentiates through the denoiser, and at 2048 episodes on all 12 levels at
    once it asks for ~13 GB, more than an 8 GB laptop GPU has. Each level carries its own key and
    state, so evaluating them in groups gives the same per-level result as all at once.
    """
    n = len(setup.level_paths)
    outs = []
    for lo in range(0, n, group):
        sl = slice(lo, min(lo + group, n))
        levels = jax.tree.map(lambda x: x[sl], setup.levels)
        sds = jax.tree.map(lambda x: x[sl], setup.state_dicts)
        outs.append(jax.device_get(fn(config, rngs[sl], levels, sds)))
    return jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *outs)


def group_for(config) -> int:
    """Levels per call that fit in 8 GB: RTC's VJP scales with episodes x levels."""
    heavy = getattr(config, 'method', None) in ('rtc', 'hard') or type(
        getattr(config, 'method', None)).__name__ == 'RealtimeMethodConfig'
    n = getattr(config, 'num_evals', 2048)
    return 12 if not heavy or n <= 512 else max(1, 12 * 512 // n)


COLUMNS = ['method', 'delay', 'execute_horizon', 'world_speed', 'deadline', 'free_first_chunk',
           'num_evals', 'seed', 'level', 'solve_rate', 'idle_frac', 'median_end_tick']


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--methods', default='sync,naive,rtc')
    p.add_argument('--delays', default='0,1,2,3,4,5,6,8,10')
    p.add_argument('--horizons', default='4')
    p.add_argument('--world-speeds', default='1')
    p.add_argument('--deadline', type=int, default=256)
    p.add_argument('--num-evals', type=int, default=2048)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--episodes-npz', default='', help='also save per-episode end ticks here')
    p.add_argument('--idle-action', choices=['zero', 'repeat'], default='zero')
    p.add_argument('--resume', action='store_true',
                   help='skip configurations already in --out with a row for every level')
    args = p.parse_args()
    import run_manifest
    run_manifest.record(args.out, {'kind': 'kinetix', 'config': vars(args)})

    setup = load()
    run = make_ours(setup)
    rngs = jax.random.split(jax.random.key(args.seed), len(setup.level_paths))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    new = not os.path.exists(args.out)
    done = set()
    if args.resume and not new:
        counts: dict = {}
        with open(args.out) as fr:
            for r in csv.DictReader(fr):
                key = (r['method'], int(r['delay']), int(r['execute_horizon']),
                       float(r['world_speed']), int(r['deadline']), int(r['num_evals']),
                       int(r['seed']))
                counts[key] = counts.get(key, 0) + 1
        done = {k for k, c in counts.items() if c >= len(setup.level_paths)}
    per_episode = {}
    with open(args.out, 'a', newline='') as f:
        w = csv.DictWriter(f, COLUMNS)
        if new:
            w.writeheader()
        for alpha in [float(x) for x in args.world_speeds.split(',')]:
            for s in [int(x) for x in args.horizons.split(',')]:
                for d in [int(x) for x in args.delays.split(',')]:
                    for method in args.methods.split(','):
                        if (method, d, s, alpha, args.deadline, args.num_evals,
                                args.seed) in done:
                            continue
                        cfg = executor.RunConfig(method=method, delay=d, execute_horizon=s,
                                                 world_speed=alpha, deadline=args.deadline,
                                                 num_evals=args.num_evals,
                                                 idle_action=args.idle_action)
                        t0 = time.time()
                        out = evaluate(run, cfg, setup, rngs, group_for(cfg))
                        for i, lp in enumerate(setup.level_paths):
                            ends = out['end_tick'][i][out['solved'][i]]
                            w.writerow(dict(
                                method=method, delay=d, execute_horizon=s, world_speed=alpha,
                                deadline=args.deadline, free_first_chunk=cfg.free_first_chunk,
                                num_evals=args.num_evals, seed=args.seed, level=lp,
                                solve_rate=f"{out['solved'][i].mean():.4f}",
                                idle_frac=f"{float(out['idle_frac'][i]):.4f}",
                                median_end_tick=(f'{np.median(ends):.1f}' if len(ends) else '')))
                        f.flush()
                        per_episode[f'{method}|d{d}|s{s}|a{alpha}'] = out['end_tick'] * np.where(
                            out['solved'], 1, -1)
                        print(f'alpha={alpha} s={s} d={d:2d} {method:5s} '
                              f"solve {out['solved'].mean():.3f} idle "
                              f"{float(out['idle_frac'].mean()):.2f} ({time.time() - t0:.0f}s)",
                              flush=True)
    if args.episodes_npz:
        np.savez_compressed(args.episodes_npz, **per_episode)


if __name__ == '__main__':
    main()
