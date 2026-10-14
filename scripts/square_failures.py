"""Where do Square episodes fail? Per-episode milestones and a failure class (paper 2).

The close gate was the obvious suspect for ACT's Square failures, but a strict and a permissive
gate scored the same (docs/paper2-progress.md). This records what each episode got as far as,
so the remaining failures can be put where they happen:

    no_grasp   the nut was never held
    slip       held, but the nut never rose 1 cm (the grasp did not survive the lift)
    drop       lifted, then lost before the nut came within 3 cm of the peg (xy)
    peg_miss   carried to within 3 cm of the peg, never seated
    no_release seated on the peg at some tick, but no success (gripper not clear, or unseated)
    success

`Tracer` wraps the (MovingCube) world and keeps a few numbers per tick; `classify` is pure and
works on those arrays. dp_mech.py --fail-log <csv> writes one row per episode, in the sweep csv's
row order, with the sweep's own columns joined on.
"""
from __future__ import annotations

import csv

import numpy as np

CLASSES = ('success', 'no_grasp', 'slip', 'drop', 'peg_miss', 'no_release')
LIFT_M = 0.01
NEAR_PEG_M = 0.03


def classify(held, nut, on_peg, success, closing, z0: float, peg_xy) -> dict:
    """Per-tick arrays of one episode -> {'cls', milestones}. `closing` = gripper command closed."""
    held, on_peg, closing = (np.asarray(a, bool) for a in (held, on_peg, closing))
    nut = np.asarray(nut, float).reshape(-1, 3)
    lifted = held & (nut[:, 2] > z0 + LIFT_M)
    peg_d = np.linalg.norm(nut[:, :2] - np.asarray(peg_xy)[:2], axis=1)
    near = lifted & (peg_d < NEAR_PEG_M)
    closes = int(np.sum(closing[1:] & ~closing[:-1]) + (closing[0] if len(closing) else 0))
    first = lambda m: int(np.argmax(m)) if m.any() else -1
    out = {'closes': closes, 't_held': first(held), 't_lift': first(lifted),
           't_near_peg': first(near), 't_on_peg': first(on_peg),
           'max_lift_mm': round(float((nut[:, 2].max() - z0) * 1e3), 1) if len(nut) else 0.0,
           'min_peg_mm': round(float(peg_d[held].min() * 1e3), 1) if held.any() else -1.0,
           'drops': int(np.sum(lifted[:-1] & ~held[1:]))}
    if success:
        cls = 'success'
    elif on_peg.any():
        cls = 'no_release'
    elif near.any():
        cls = 'peg_miss'
    elif lifted.any():
        cls = 'drop'
    elif held.any():
        cls = 'slip'
    else:
        cls = 'no_grasp'
    out['cls'] = cls
    return out


class Tracer:
    """Wraps the world; records held / nut / on_peg / success / gripper command each step.
    Finished episodes land in the shared `episodes` list (one dict per episode)."""

    def __init__(self, world, episodes: list, pos_key: str = 'SquareNut_pos',
                 quat_key: str = 'SquareNut_quat') -> None:
        self.world, self.episodes = world, episodes
        self.pos_key, self.quat_key = pos_key, quat_key
        self._rec = None

    def __getattr__(self, name):
        return getattr(self.world, name)

    def _close(self) -> None:
        r, self._rec = self._rec, None
        if r is None:
            return
        self.episodes.append(classify(r['held'], r['nut'], r['on_peg'], r['success'],
                                      r['closing'], r['z0'], self._peg))

    def reset(self):
        import oracle_square as sq
        self._close()
        obs = self.world.reset()
        self._peg = sq.PEG
        self._rec = {'held': [], 'nut': [], 'on_peg': [], 'closing': [], 'success': False,
                     'z0': float(obs[self.pos_key][2])}
        return obs

    def step(self, action):
        import oracle_square as sq
        out = self.world.step(action)
        o, r = out[0], self._rec
        if r is not None:
            r['held'].append(sq.is_held(o['robot0_eef_pos'], o['robot0_gripper_qpos'],
                                        o[self.pos_key], o[self.quat_key]))
            r['nut'].append(np.array(o[self.pos_key], float))
            r['on_peg'].append(sq.on_peg(o[self.pos_key]))
            r['closing'].append(float(action[6]) > 0)
            r['success'] = r['success'] or bool(self.world._check_success())
        return out

    def finish(self) -> None:
        self._close()


def write(path: str, sweep_csv: str, episodes: list) -> None:
    """Join the sweep csv's rows (same order) with each episode's class and milestones."""
    with open(sweep_csv) as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != len(episodes):
        raise RuntimeError(f'{len(rows)} sweep rows but {len(episodes)} traced episodes')
    keys = list(rows[0]) + [k for k in episodes[0] if k not in rows[0]]
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row, ep in zip(rows, episodes):
            w.writerow({**row, **ep})


def summarize(paths: list[str]) -> None:
    """Class counts per (strategy, delay, speed)."""
    cells = {}
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                cells.setdefault((r['strategy'], r['delay'], r['speed']), []).append(r)
    print(f"{'strategy':32s} {'d':>3s} {'v':>5s}  " + ' '.join(f'{c:>10s}' for c in CLASSES))
    for (s, d, v), rs in sorted(cells.items()):
        n = [sum(r['cls'] == c for r in rs) for c in CLASSES]
        print(f'{s:32s} {d:>3s} {v:>5s}  ' + ' '.join(f'{x:>10d}' for x in n))


if __name__ == '__main__':
    import sys
    summarize(sys.argv[1:])
