"""Full-scale check that the extended loop IS the official evaluation where both are defined.

For each (method, d, s) inside the official envelope, run the official `eval_flow.eval` and our
loop with the official conventions (free first chunk, world always moving, 256-step episodes) at
the official trial count, same keys, and write both to one CSV. The paper's claim "we extend the
official evaluation" rests on these rows agreeing to within sampling noise; bitwise agreement is
not attainable on chaotic physics (see test_executor.py).

    cd external/rtc-kinetix
    uv run --python 3.11 python ../../experiments/kinetix/reproduce.py \
        --out ../../outputs/kinetix/reproduce.csv
"""
import argparse
import csv
import dataclasses
import os
import sys
import time

sys.path.insert(0, 'src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts'))

import jax  # noqa: E402

import eval_flow  # noqa: E402
import executor  # noqa: E402
import runner  # noqa: E402

POINTS = [('naive', 0, 1), ('naive', 1, 4), ('rtc', 1, 4), ('naive', 2, 4), ('rtc', 2, 4),
          ('naive', 3, 4), ('rtc', 3, 4), ('naive', 4, 4), ('rtc', 4, 4)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--num-evals', type=int, default=2048)
    p.add_argument('--start', type=int, default=0, help='resume from this POINTS index')
    args = p.parse_args()
    import run_manifest
    run_manifest.record(args.out, {'kind': 'kinetix_reproduce', 'config': vars(args)})
    setup = runner.load()
    ours_fn, theirs_fn = runner.make_ours(setup), runner.make_official(setup)
    rngs = jax.random.split(jax.random.key(0), len(setup.level_paths))
    official = {'naive': eval_flow.NaiveMethodConfig(), 'rtc': eval_flow.RealtimeMethodConfig()}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    new = args.start == 0 or not os.path.exists(args.out)
    with open(args.out, 'w' if new else 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['method', 'delay', 'execute_horizon', 'level', 'official', 'ours'])
        for method, d, s in POINTS[args.start:]:
            t0 = time.time()
            oc = dataclasses.replace(eval_flow.EvalConfig(num_evals=args.num_evals),
                                     inference_delay=d, execute_horizon=s, method=official[method])
            theirs = runner.evaluate(theirs_fn, oc, setup, rngs, runner.group_for(oc))
            cfg = executor.RunConfig(method=method, delay=d, execute_horizon=s, world_speed=1.0,
                                     deadline=256, free_first_chunk=True,
                                     num_evals=args.num_evals)
            ours = runner.evaluate(ours_fn, cfg, setup, rngs, runner.group_for(cfg))
            o = ours['solved'].mean(axis=1)
            t = theirs['returned_episode_solved']
            for i, lp in enumerate(setup.level_paths):
                w.writerow([method, d, s, lp, f'{t[i]:.4f}', f'{o[i]:.4f}'])
            f.flush()
            print(f'{method:5s} d={d} s={s} official {t.mean():.4f} ours {o.mean():.4f} '
                  f'diff {o.mean() - t.mean():+.4f} ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
