import numpy as np
import pytest

import bev_planner as bp


def test_goal_consistent_t0_moves_a_pre_switch_memory_to_the_switch():
    goals = np.array([0, 0, 0, 1, 1, 1])                 # switch at frame 3
    assert bp.goal_consistent_t0(goals, 5, 1) == 3
    assert bp.goal_consistent_t0(goals, 2, 0) == 0       # before the switch: unchanged
    assert bp.goal_consistent_t0(goals, 5, 4) == 4       # after it: unchanged


def _episodes(d, n=2, T=30):
    rng = np.random.default_rng(0)
    P = np.array([[1., 0, 0, 42], [0, 1, 0, 42], [0, 0, 0, 1]], np.float32)
    for i in range(n):
        np.savez(f'{d}/ep_{i}.npz', P_agentview=P, P_birdview=P,
                 **{k: rng.integers(0, 255, (T, 84, 84, 3), dtype=np.uint8)
                    for k in ('agentview', 'top', 'wrist')},
                 state=rng.standard_normal((T, 9)).astype(np.float32),
                 action=rng.standard_normal((T, 10)).astype(np.float32))


def test_resume_continues_exactly_where_an_interrupted_run_stopped(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    _episodes(tmp_path)
    run = ['bev_planner.py', '--data', str(tmp_path), '--steps', '4', '--save-every', '2',
           '--batch', '2', '--local-enc', 'tiny', '--no-amp']
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr('sys.argv', run + ['--out', str(tmp_path / 'full.pt')])
    bp.main()

    calls, sample = [0], bp.sample_index

    def crash_at_step_3(*a, **k):
        calls[0] += 1
        if calls[0] == 3:
            raise KeyboardInterrupt
        return sample(*a, **k)
    monkeypatch.setattr(bp, 'sample_index', crash_at_step_3)
    monkeypatch.setattr('sys.argv', run + ['--out', str(tmp_path / 'cut.pt')])
    with pytest.raises(KeyboardInterrupt):
        bp.main()
    assert (tmp_path / 'cut.pt.resume').exists()
    steps = [0]

    def count(*a, **k):
        steps[0] += 1
        return sample(*a, **k)
    monkeypatch.setattr(bp, 'sample_index', count)
    monkeypatch.setattr('sys.argv', run + ['--out', str(tmp_path / 'cut.pt'), '--resume'])
    bp.main()

    assert steps[0] == 2                  # steps 3-4 only (a restart from scratch would match too)
    assert not (tmp_path / 'cut.pt.resume').exists()
    a, b = (torch.load(tmp_path / f, weights_only=False) for f in ('full.pt', 'cut.pt'))
    assert b['step'] == 4
    for k in a['model']:
        assert torch.equal(a['model'][k], b['model'][k]), k
