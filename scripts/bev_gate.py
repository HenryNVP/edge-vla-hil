"""Where does the BEV planner's per-cell staleness gate open? (bev_model.py, `gate`)

Nothing supervises the gate, so this checks it against what actually moved, on held-out top-camera
demos (they carry the nut pose): for plans at age a (cloud frame t0 = t - a), each grid cell is
put in one region by priority

    nut_now     within RADIUS of the nut at t           (the cloud's map is wrong here...)
    nut_then    within RADIUS of the nut at t0          (...and here, if the nut moved)
    gripper     within RADIUS of the end effector at t  (the arm moves too)
    pegs        within RADIUS of either peg             (static: the cloud's map stays right)
    rest        everything else on the table

and the gate is averaged per region. The hoped-for answer: nut_now / nut_then / gripper > rest
~ pegs, and growing with the age and with how far the nut moved. Also writes a few gate maps.

With a `corr` model the gate is its staleness (1 - the attention a cell keeps on itself), and a
second check asks where the cell under the nut fetched its cloud feature from: if the
correspondence works, the attention centroid of that cell sits nearer the nut's t0 position than
the cell itself does, once the nut has moved.

    python3 scripts/bev_gate.py --weights outputs/gate/g0.pt --data data/oracle_square_top/heldout \
        --out outputs/gate/analysis
"""
from __future__ import annotations

import argparse
import os

import numpy as np

RADIUS = 0.06
PEGS = ((0.227, 0.101), (0.227, -0.101))     # Square's two pegs (world x, y); fixed
REGIONS = ('nut_now', 'nut_then', 'gripper', 'pegs', 'rest')


def cell_xy(grid: int, x_range, y_range) -> np.ndarray:
    """[grid*grid, 2] cell centres, row-major over (x, y), as bev_model.pillar_points."""
    xs, ys = np.linspace(*x_range, grid), np.linspace(*y_range, grid)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([gx, gy], -1).reshape(-1, 2)


def regions(cells: np.ndarray, nut_now, nut_then, eef, radius: float = RADIUS) -> np.ndarray:
    """[N] region index per cell (REGIONS order, first match wins)."""
    def near(p):
        return np.linalg.norm(cells - np.asarray(p)[:2], axis=1) <= radius
    out = np.full(len(cells), REGIONS.index('rest'))
    masks = [near(nut_now), near(nut_then), near(eef),
             np.logical_or.reduce([near(p) for p in PEGS])]
    for i in reversed(range(len(masks))):
        out[masks[i]] = i
    return out


def region_means(gate: np.ndarray, reg: np.ndarray) -> np.ndarray:
    """[len(REGIONS)] mean gate per region, nan where a region is empty."""
    return np.array([gate[reg == i].mean() if (reg == i).any() else np.nan
                     for i in range(len(REGIONS))])


def attn_centroid(attn_row: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """[2] attention-weighted source position of one query cell (its row of [N, N])."""
    return (attn_row[:, None] * cells).sum(0) / attn_row.sum()


def main() -> None:
    import torch
    from bev_planner import BEVPolicy, KEYS
    import glob
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ages', default='0,10,30,48')
    ap.add_argument('--stride', type=int, default=4)
    args = ap.parse_args()
    pol = BEVPolicy(args.weights)
    m, dev, st = pol.model, pol.dev, pol.st
    from bev_model import X_RANGE, Y_RANGE
    cells = cell_xy(m.grid, X_RANGE, Y_RANGE)
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    rows = []      # (age, nut displacement m, region means...)
    fetch = []     # corr: (age, disp, |cell - nut_then|, |centroid - nut_then|)
    maps = []
    for f in files:
        z = np.load(f)
        e = {k: z[k] for k in KEYS + ('nut',)}
        T = len(e['action'])
        for age in map(int, args.ages.split(',')):
            t = np.arange(age, T, args.stride)
            t0 = t - age
            g = lambda k, i: torch.from_numpy(e[k][i]).to(dev)
            s = lambda i: (g('state', i).float() - st['state_mean']) / st['state_std']
            with torch.no_grad():
                mem = m.memory(g('agentview', t0), g('top', t0), s(t0))
                local0 = m.local_map(g('agentview', t0), g('top', t0)) if m.use_corr else None
                _, gate, attn = m.plan(mem, g('agentview', t), g('top', t), g('wrist', t), s(t),
                                       torch.from_numpy(t - t0).long().to(dev), return_gate=True,
                                       local0=local0, return_attn=True)
            gate = gate.float().cpu().numpy()
            attn = attn.float().cpu().numpy() if attn is not None else None
            for j in range(len(t)):
                nut_now, nut_then = e['nut'][t[j], :3], e['nut'][t0[j], :3]
                reg = regions(cells, nut_now, nut_then, e['state'][t[j], :3])
                disp = float(np.linalg.norm(nut_now[:2] - nut_then[:2]))
                rows.append([age, disp, *region_means(gate[j], reg)])
                if attn is not None and disp > 0.02:
                    c = np.argmin(np.linalg.norm(cells - nut_now[:2], axis=1))
                    cen = attn_centroid(attn[j, c], cells)
                    fetch.append([age, disp, np.linalg.norm(cells[c] - nut_then[:2]),
                                  np.linalg.norm(cen - nut_then[:2])])
            if age == max(map(int, args.ages.split(','))) and len(maps) < 6:
                j = len(t) // 2
                maps.append((os.path.basename(f), age, gate[j].reshape(m.grid, m.grid),
                             e['nut'][t[j], :2], e['nut'][t0[j], :2], e['state'][t[j], :2]))
    rows = np.array(rows)
    np.save(os.path.join(args.out, 'rows.npy'), rows)
    print(f'{len(files)} episodes, {len(rows)} plans; mean gate per region (higher = trusts the '
          f'fresh view)')
    print(f'{"":22s}' + ''.join(f'{r:>10s}' for r in REGIONS))
    for age in sorted(set(rows[:, 0])):
        for lo, hi, lab in ((0, 0.02, 'nut moved <2cm'), (0.02, 9, 'nut moved >2cm')):
            sel = (rows[:, 0] == age) & (rows[:, 1] >= lo) & (rows[:, 1] < hi)
            if sel.any():
                mu = np.nanmean(rows[sel, 2:], 0)
                print(f'age {int(age):2d} {lab:15s}' + ''.join(f'{v:10.3f}' for v in mu)
                      + f'   n={sel.sum()}')
    if fetch:
        fetch = np.array(fetch)
        print('corr: where the cell under the nut fetched from (nut moved > 2 cm), metres to the '
              "nut's t0 position")
        for age in sorted(set(fetch[:, 0])):
            sel = fetch[:, 0] == age
            print(f'age {int(age):2d}  cell {fetch[sel, 2].mean():.3f}   attention centroid '
                  f'{fetch[sel, 3].mean():.3f}   closer in {np.mean(fetch[sel, 3] < fetch[sel, 2]):.0%}'
                  f'   n={sel.sum()}')
    _plot(maps, cells, os.path.join(args.out, 'gate_maps.png'))


def _plot(maps, cells, path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from bev_model import X_RANGE, Y_RANGE
    if not maps:
        return
    fig, axs = plt.subplots(1, len(maps), figsize=(3 * len(maps), 3.2))
    for ax, (name, age, gm, nn, nt, ee) in zip(np.atleast_1d(axs), maps):
        # rows are x, columns y; show y horizontal, x vertical (robot base at the bottom)
        ax.imshow(gm, origin='lower', vmin=0, vmax=1, cmap='viridis',
                  extent=(Y_RANGE[0], Y_RANGE[1], X_RANGE[0], X_RANGE[1]))
        ax.plot(nt[1], nt[0], 'wx', ms=8, label='nut at t0')
        ax.plot(nn[1], nn[0], 'w+', ms=10, mew=2, label='nut at t')
        ax.plot(ee[1], ee[0], 'r.', ms=8, label='gripper')
        for px, py in PEGS:
            ax.plot(py, px, 'ks', ms=5, mfc='none')
        ax.set_title(f'{name} age {age}', fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
    np.atleast_1d(axs)[0].legend(fontsize=6, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == '__main__':
    main()
