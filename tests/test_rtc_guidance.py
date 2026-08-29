"""Tests for RTC's guidance path — the mechanism, not just the scheduling.

Two defects this locks down, both of which let RTC look implemented while doing something else:

  1. `predict_inpaint` used to sample a chunk and overwrite the frozen entries AFTERWARDS. The
     model never saw the prefix, so the actions just past the frozen region continued a plan
     generated as if the arm were elsewhere — exactly the splice discontinuity RTC removes.
  2. RTCExecutor sent only the frozen slice as guidance while `freeze_weights` assigned decaying
     weights across the whole soft region. Those weights had nothing to pull toward, so the
     paper's continuity term was computed and then silently discarded.

The rotation round-trip matters because guidance arrives in the 7-dim action contract but the
diffusion trajectory lives in the checkpoint's native 10-dim space. Guiding in the wrong space
steers the sample toward nonsense while every array stays well-formed.
"""
import numpy as np
import pytest

from evh_controller.chunk_executor import RTCExecutor
from evh_controller.dp_repo_policy import (
    DiffusionPolicyRepoBackend,
    _axisangle_to_matrix,
    _matrix_to_axisangle,
    _matrix_to_rotation_6d,
    _rotation_6d_to_matrix,
)

ROTATIONS = [
    [0.0, 0.0, 0.0],
    [0.3, -0.1, 0.7],
    [np.pi / 2, 0.0, 0.0],
    [0.0, 2.5, 0.0],
    [-1.1, 0.4, -0.2],
]


# ------------------------------------------------------------- rotation round-trip
@pytest.mark.parametrize('aa', ROTATIONS)
def test_axisangle_matrix_round_trip(aa):
    R = _axisangle_to_matrix(np.asarray(aa))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), 'not a rotation matrix'
    assert np.isclose(np.linalg.det(R), 1.0), 'improper rotation (reflection)'
    assert np.allclose(_matrix_to_axisangle(R), aa, atol=1e-9)


@pytest.mark.parametrize('aa', ROTATIONS)
def test_rotation_6d_round_trip(aa):
    """_matrix_to_rotation_6d must invert _rotation_6d_to_matrix — same row convention."""
    R = _axisangle_to_matrix(np.asarray(aa))
    d6 = _matrix_to_rotation_6d(R)
    assert d6.shape == (6,)
    assert np.allclose(_rotation_6d_to_matrix(d6), R, atol=1e-9)


def test_action_space_round_trips_through_the_checkpoint_layout():
    """7-dim contract -> 10-dim native -> back. This is the conversion guidance passes through."""
    chunk7 = np.array([
        [0.4, -0.1, 1.05, *ROTATIONS[1], 1.0],
        [0.41, -0.11, 1.04, *ROTATIONS[3], -1.0],
        [0.39, -0.09, 1.06, *ROTATIONS[0], 0.0],
    ], dtype=np.float32)

    chunk10 = DiffusionPolicyRepoBackend._redo_abs_transform(chunk7)
    assert chunk10.shape == (3, 10)
    assert np.allclose(chunk10[:, :3], chunk7[:, :3])       # position passes through
    assert np.allclose(chunk10[:, 9], chunk7[:, 6])         # gripper passes through

    back = DiffusionPolicyRepoBackend._undo_abs_transform(chunk10)
    assert np.allclose(back, chunk7, atol=1e-6)


# ----------------------------------------------------------------- the soft mask
def test_freeze_weights_has_three_regions():
    """1.0 frozen, decaying soft, 0.0 free — the paper's W_i."""
    H, s, d = 15, 3, 5
    w = RTCExecutor.freeze_weights(H, s, d)

    assert w.shape == (H,)
    assert np.all(w[:d] == 1.0), 'frozen region must be hard'
    assert np.all(w[H - s:] == 0.0), 'tail must be free for the policy to regenerate'
    soft = w[d:H - s]
    assert np.all((soft >= 0.0) & (soft <= 1.0))
    assert np.all(np.diff(soft) < 0), 'soft region must decay away from the frozen prefix'


# ------------------------------------------------------- what the executor sends
class _Policy:
    chunk_size = 15
    action_dim = 7


class _CapturingWorker:
    """Records every request, and never delivers, so one guidance call can be inspected."""

    def __init__(self):
        self.requests = []

    def try_request(self, obs, t_issue, epoch, prefix=None, weights=None):
        self.requests.append({'t': t_issue, 'prefix': prefix, 'weights': weights})
        return True

    def poll(self):
        return None


def _obs():
    return {'agentview': np.zeros((84, 84, 3), np.uint8), 'proprio': np.zeros(9, np.float32)}


def _primed_executor(delays=(4,)):
    """An RTC executor holding a plan, with a delay history so the forecast is non-trivial."""
    worker = _CapturingWorker()
    ex = RTCExecutor(worker, _Policy())
    ex._chunk = np.arange(15 * 7, dtype=np.float32).reshape(15, 7)
    ex._i = 0
    ex._delays = list(delays)
    ex._pending_t = None
    worker.requests.clear()
    return ex, worker


def test_guidance_covers_the_whole_soft_region_not_just_the_frozen_prefix():
    """The regression: a guide shorter than the weighted region leaves the paper's continuity
    term with nothing to pull toward, and the new chunk is free to jump right after the freeze."""
    ex, worker = _primed_executor()
    ex.step(_obs(), t=0)

    req = worker.requests[-1]
    weights = req['weights']
    guide = req['prefix']
    weighted = int(np.count_nonzero(weights))

    assert len(guide) >= weighted, (
        f'guide covers {len(guide)} steps but {weighted} carry a non-zero weight — '
        'the soft region has no target')


def test_guidance_is_the_plan_from_the_action_about_to_execute():
    """Alignment: guide[0] must be the action emitted at this tick, or the freeze is off by one
    and the policy is told to hold a pose the arm has already left."""
    ex, worker = _primed_executor()
    ex._i = 3
    ex.step(_obs(), t=0)

    guide = worker.requests[-1]['prefix']
    assert np.allclose(guide[0], ex._chunk[3]), 'guidance does not start at the current action'
    assert len(guide) == len(ex._chunk) - 3


def test_guidance_never_runs_past_the_end_of_the_plan():
    """Late in a chunk there is less plan left than the horizon; the guide must just be shorter."""
    ex, worker = _primed_executor()
    ex._i = 13
    ex.step(_obs(), t=0)

    guide = worker.requests[-1]['prefix']
    assert len(guide) == 2
    assert np.allclose(guide[0], ex._chunk[13])


def test_bootstrap_issues_without_guidance():
    """No plan in hand: there is nothing to stay continuous with, so a plain sample is correct."""
    worker = _CapturingWorker()
    ex = RTCExecutor(worker, _Policy())
    ex.step(_obs(), t=0)

    assert worker.requests[-1]['prefix'] is None


# ------------------------------------------------- the fallback is still coherent
def test_base_fallback_respects_weights_it_is_given():
    """Backends without a sampler hook keep the post-hoc blend. It is a weaker approximation, but
    it must at least honour a hard freeze so a non-DP backend is not silently discontinuous."""
    from evh_controller.policy import PyTorchBackend

    policy = PyTorchBackend('', device='cpu')
    H, A = policy.chunk_size, policy.action_dim
    prefix = np.ones((H, A), dtype=np.float32)
    weights = RTCExecutor.freeze_weights(H, s=2, d=4)

    chunk = policy.predict_inpaint(
        {'agentview': np.zeros((2, 84, 84, 3), np.uint8),
         'proprio': np.zeros((2, 9), np.float32)}, prefix, weights)

    assert chunk.shape == (H, A)
    assert np.allclose(chunk[:4], 1.0), 'hard-frozen entries were not honoured'
