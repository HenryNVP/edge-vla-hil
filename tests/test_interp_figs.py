"""Smoke: the four interp figures build from a minimal synthetic ci.json and gain maps."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from interp_figs import main  # noqa: E402

FIGS = ('fig_seen_vs_action', 'fig_swap', 'fig_closepoint', 'fig_gain_space')
RADII = {'square DP': (0.08, 0.03), 'lift DP': (0.06, 0.02), 'lift ACT': (0.06, 0.02)}
GAIN_LABELS = ('action inside', 'action outside', 'action drop in-out', 'seen inside',
               'seen outside')
SWAP_LABELS = ('head in, dir in', 'head in, dir out', 'head out, dir in', 'head out, dir out',
               'direction effect', 'head effect')


def write_ci(path) -> None:
    rows = []
    for pol, radii in RADII.items():
        for r in radii:                       # both radii: the figures must pick the smaller
            for i, label in enumerate(GAIN_LABELS):
                for a, axis in enumerate(('x', 'y')):
                    est = 0.3 + 0.04 * ((3 * i + 5 * a) % 9)
                    rows.append(dict(section='gains', policy=pol, radius=r, label=label,
                                     axis=axis, est=est, lo=est - 0.05, hi=est + 0.06))
            if pol == 'lift ACT':
                continue                      # the swap section only covers the DP policies
            for i, label in enumerate(SWAP_LABELS):
                for a, axis in enumerate(('x', 'y')):
                    est = 0.25 + 0.04 * ((2 * i + 4 * a) % 8)
                    rows.append(dict(section='swap', policy=pol, radius=r, label=label,
                                     axis=axis, est=est, lo=est - 0.05, hi=est + 0.06))
    for j, s in enumerate(((-0.02, 0.0), (0.02, 0.0), (0.0, -0.02), (0.0, 0.02), (0.0, 0.0))):
        rows.append(dict(section='rollouts', shift=list(s), n=10, success=0.4 + 0.1 * j,
                         success_lo=0.1 + 0.1 * j, success_hi=0.65 + 0.1 * j,
                         followed=0.3 + 0.1 * j, followed_lo=0.05 + 0.1 * j,
                         followed_hi=0.6 + 0.1 * j, nearer_ghost=0.2, cross_mm=1.5))
    with open(path, 'w') as fh:
        json.dump(rows, fh)


def write_gainmap(path, radii, nut_xy, seed=0) -> None:
    """P = one operating point per radius, G = 5; response of a mild gain-0.8 linear map."""
    g, grid = 5, np.linspace(-0.04, 0.04, 5)
    dxy = np.stack(np.meshgrid(grid, grid, indexing='ij'), -1) * 1e3      # mm offsets
    A = np.array([[0.5, 0.05], [0.02, 0.8]])
    base = np.einsum('ab,ijb->ija', A, dxy)[None]
    resp = base + np.random.default_rng(seed).normal(0, 2.0, (len(radii), g, g, 2))
    np.savez(path, resp=resp.astype(np.float32), radius=np.array(radii),
             nut_xy=np.repeat(np.asarray(nut_xy)[None], len(radii), 0), grid=grid)


def test_figures_build_from_synthetic_data(tmp_path):
    data, out = tmp_path / 'data', tmp_path / 'figs'
    data.mkdir()
    write_ci(data / 'ci.json')
    write_gainmap(data / 'gainmap.npz', RADII['square DP'], (-0.112, 0.17))
    write_gainmap(data / 'gainmap_lift_dp.npz', RADII['lift DP'], (0.01, -0.005))
    write_gainmap(data / 'gainmap_lift_act.npz', RADII['lift ACT'], (-0.02, 0.01), seed=1)
    main(['--data', str(data), '--out', str(out)])
    for name in FIGS:
        for ext in ('pdf', 'png'):
            f = out / f'{name}.{ext}'
            assert f.is_file() and f.stat().st_size > 100, f'{f} missing or empty'
