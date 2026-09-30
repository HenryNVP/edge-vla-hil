"""Tests for the extended executor. Two kinds:

1. The schedule (`cycle_plan`), pure numpy: the timing model the paper states in words.
2. Equivalence with the OFFICIAL `eval_flow.eval`: wherever the official loop is defined
   (`s >= d`, `d <= H - s`) and with its conventions (free first chunk, world always moving,
   256-step episodes), our loop must agree with it per level. That is what licenses calling the
   extension "the official evaluation, extended".

   Agreement cannot be bitwise. Kinetix physics is chaotic, and two XLA compilations of the same
   program round differently, so a few episodes diverge. Measured 2026-09-28: the official eval
   compiled with shard_map vs plain jit(vmap), same keys, 64 episodes per level, disagrees on 2 of
   12 levels by one episode each, at (d=0, s=1, naive) and (d=1, s=4, RTC). Ours vs official
   differs by the same mechanism, and more on the most chaotic level: h17_unicycle once differed
   by 5/64 in a full run while the same case passed with 0-1 when run alone, i.e. the gap moves
   with compilation order, not with the loop. So the bound is on the whole: at most 1.5% of all
   episodes, and no level off by more than 6/64. The full-scale check (2048 per level, against
   the official numbers) is in reproduce.py.

Run in the official environment (needs JAX, the Kinetix submodule and `assets/bc/31`):

    cd external/rtc-kinetix
    uv run --python 3.11 --with pytest python -m pytest -q ../../experiments/kinetix/test_executor.py
"""
import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from executor import IDLE, NEW, OLD, cycle_plan  # noqa: E402


# ------------------------------------------------------------------ schedule
def test_official_regime_is_the_official_schedule():
    """d <= s: play d actions of the previous chunk, then the new one from entry d to s-1."""
    P, src, idx = cycle_plan('rtc', 2, 4, 8)
    assert P == 4
    assert list(src) == [OLD, OLD, NEW, NEW]
    assert list(idx) == [0, 1, 2, 3]


def test_sync_waits_then_plays_from_the_start():
    P, src, idx = cycle_plan('sync', 3, 8, 8)
    assert P == 11
    assert list(src) == [IDLE] * 3 + [NEW] * 8
    assert list(idx[3:]) == list(range(8))


def test_past_the_half_horizon_the_slot_sets_the_period_and_entries_are_skipped():
    """d = 6 > s = 4 with H = 8: one request per arrival (P = 6). The chunk issued at t0 lands at
    t0 + 6 and is played from its entry 6 in the NEXT cycle's frame; its entries 0-5 were for
    ticks already gone."""
    P, src, idx = cycle_plan('rtc', 6, 4, 8)
    assert P == 6
    assert list(src) == [OLD] * 6          # nothing new arrives inside the cycle
    assert list(idx) == list(range(6))


def test_new_entries_past_the_chunk_are_idle():
    P, src, _ = cycle_plan('naive', 0, 10, 8)
    assert P == 10
    assert list(src[8:]) == [IDLE, IDLE]


def test_unknown_method_and_empty_horizon_are_refused():
    with pytest.raises(ValueError):
        cycle_plan('telepathy', 1, 4, 8)
    with pytest.raises(ValueError):
        cycle_plan('rtc', 1, 0, 8)


# ------------------------------------------------------------ equivalence
def _official_available():
    return os.path.isdir('assets/bc') and os.path.isdir('src')


@pytest.mark.skipif(not _official_available(), reason='run from external/rtc-kinetix')
@pytest.mark.parametrize('method,d,s', [('naive', 0, 1), ('naive', 1, 4), ('rtc', 1, 4),
                                         ('rtc', 3, 5), ('hard', 2, 4)])
def test_matches_the_official_eval_where_it_is_defined(method, d, s):
    import jax

    import eval_flow
    import executor
    import runner

    setup = runner.load()
    n = 64
    official_method = {'naive': eval_flow.NaiveMethodConfig(),
                       'rtc': eval_flow.RealtimeMethodConfig(),
                       'hard': eval_flow.RealtimeMethodConfig(prefix_attention_schedule='zeros')}
    ocfg = dataclasses.replace(eval_flow.EvalConfig(num_evals=n), inference_delay=d,
                               execute_horizon=s, method=official_method[method])
    rngs = jax.random.split(jax.random.key(0), len(setup.level_paths))
    theirs = jax.device_get(runner.make_official(setup)(ocfg, rngs, setup.levels,
                                                        setup.state_dicts))
    cfg = executor.RunConfig(method=method, delay=d, execute_horizon=s, world_speed=1.0,
                             deadline=256, free_first_chunk=True, num_evals=n)
    ours = jax.device_get(runner.make_ours(setup)(cfg, rngs, setup.levels, setup.state_dicts))
    diff = np.abs(ours['solved'].mean(axis=1) - theirs['returned_episode_solved']) * n
    assert diff.max() <= 6, f'per-level episode differences {diff}'
    assert diff.sum() <= 0.015 * n * len(diff), f'per-level episode differences {diff}'
