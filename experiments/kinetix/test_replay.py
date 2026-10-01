"""The replay engine is `executor.run` with the schedule taken from outside.

Fed `executor.run`'s own schedule (`schedule.from_cycle_plan`), it must give the same per-level
result, to the XLA-compilation noise measured in `test_executor.py`; it also draws the same
random keys in the same order, so in practice it agrees almost bitwise. That licenses running
any `chunk_executor` strategy's schedule through it. Run in the official environment:

    cd external/rtc-kinetix
    uv run --python 3.11 --with pytest python -m pytest -q ../../experiments/kinetix/test_replay.py
"""
import functools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _official_available():
    return os.path.isdir('assets/bc') and os.path.isdir('src')


@pytest.mark.skipif(not _official_available(), reason='run from external/rtc-kinetix')
@pytest.mark.parametrize('method,d,s,alpha', [('sync', 4, 8, 1.0), ('sync', 8, 2, 0.0),
                                               ('naive', 2, 4, 1.0), ('naive', 6, 2, 0.5),
                                               ('naive', 12, 8, 0.0)])
def test_replaying_the_cycle_plan_reproduces_executor_run(method, d, s, alpha):
    import jax

    import executor
    import replay
    import runner
    import schedule

    setup = runner.load()
    n = 64
    cfg = executor.RunConfig(method=method, delay=d, execute_horizon=s, world_speed=alpha,
                             deadline=256, num_evals=n)
    sched = schedule.from_cycle_plan(method, d, s, 8, cfg.deadline)

    @functools.partial(runner._sharded, check_rep=False)
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def _replayed(config, rng, level, state_dict):
        return replay.run(config, setup.env, rng, level,
                          runner._policy(setup, config.model, rng, state_dict),
                          setup.env_params, sched=sched)

    rngs = jax.random.split(jax.random.key(0), len(setup.level_paths))
    theirs = jax.device_get(runner.make_ours(setup)(cfg, rngs, setup.levels, setup.state_dicts))
    ours = jax.device_get(_replayed(cfg, rngs, setup.levels, setup.state_dicts))
    np.testing.assert_allclose(ours['idle_frac'], theirs['idle_frac'], atol=1e-6)
    diff = np.abs(ours['solved'].mean(axis=1) - theirs['solved'].mean(axis=1)) * n
    assert diff.max() <= 6, f'per-level episode differences {diff}'
    assert diff.sum() <= 0.015 * n * len(diff), f'per-level episode differences {diff}'


@pytest.mark.skipif(not _official_available(), reason='run from external/rtc-kinetix')
def test_noskip_async_runs_through_the_runner():
    import jax

    import executor
    import runner

    setup = runner.load()
    cfg = executor.RunConfig(method='noskip_async', delay=6, execute_horizon=2, world_speed=1.0,
                             deadline=256, num_evals=32)
    rngs = jax.random.split(jax.random.key(0), len(setup.level_paths))
    out = jax.device_get(runner.make_ours(setup)(cfg, rngs, setup.levels, setup.state_dicts))
    # d = 6, H = 8: coverage H / 2d (test_schedule.py), so idle about 1/3 of the deadline
    assert np.allclose(out['idle_frac'], 1 - 8 / 12, atol=0.03)
    assert 0 < out['solved'].mean() < 1
