"""Pure-Python tests for the async chunk-execution strategies — no ROS2 required.

Guards the Wedge-A/B seam. A FakeWorker delivers each requested chunk after a configurable
number of polls (= control steps), so every strategy's splice behavior under a *known* delay is
asserted deterministically: synchronous pauses, naive-async replays the past, RTC continues
time-aligned and measures the delay it saw, resets discard in-flight chunks.
"""
import numpy as np
import pytest

from evh_controller.chunk_executor import (
    BIDExecutor,
    NaiveAsyncExecutor,
    NetworkAwareExecutor,
    RTCExecutor,
    SynchronousExecutor,
    TemporalEnsembleExecutor,
    make_executor,
)
from evh_controller.inference_worker import Arrival

STRATEGIES = ['synchronous', 'naive_async', 'temporal_ensemble', 'bid', 'rtc', 'network_aware']

H, A = 8, 7


def _obs():
    return {'agentview': np.zeros((84, 84, 3), np.uint8), 'proprio': np.zeros(9, np.float32)}


class FakePolicy:
    """Each predicted chunk is constant-valued with its call index: chunk k is all-(k+1)."""
    chunk_size = H
    action_dim = A

    def __init__(self):
        self.calls = 0

    def next_chunk(self):
        self.calls += 1
        return np.full((H, A), float(self.calls), dtype=np.float32)


class FakeWorker:
    """Deterministic worker: a requested chunk 'arrives' after `delay_steps` polls."""

    def __init__(self, policy, delay_steps=1):
        self.policy = policy
        self.delay_steps = delay_steps
        self._pending = None          # (polls_remaining, Arrival)
        self.requests = []            # (t_issue, prefix, weights) for assertions

    def try_request(self, obs, t_issue, epoch, prefix=None, weights=None):
        if self._pending is not None:
            return False
        self.requests.append((t_issue, prefix, weights))
        arrival = Arrival(self.policy.next_chunk(), t_issue, epoch, compute_s=0.001)
        self._pending = [max(1, self.delay_steps), arrival]
        return True

    def poll(self):
        if self._pending is None:
            return None
        self._pending[0] -= 1
        if self._pending[0] > 0:
            return None
        arrival = self._pending[1]
        self._pending = None
        return arrival


def _run(ex, steps, start=0):
    """Drive the executor; returns the list of emitted actions (None = hold)."""
    out = []
    for t in range(start, start + steps):
        a = ex.step(_obs(), t)
        out.append(None if a is None else float(np.asarray(a).flat[0]))
    return out


def _make(name, delay=1, **kwargs):
    policy = FakePolicy()
    worker = FakeWorker(policy, delay_steps=delay)
    ex = make_executor(name, worker, policy) if not kwargs else None
    if kwargs:
        cls = {c.name: c for c in (SynchronousExecutor, NaiveAsyncExecutor,
                                   TemporalEnsembleExecutor, BIDExecutor,
                                   RTCExecutor, NetworkAwareExecutor)}[name]
        ex = cls(worker, policy, **kwargs)
    return ex, worker, policy


# ---------------------------------------------------------------- construction
@pytest.mark.parametrize('name', STRATEGIES)
def test_make_executor(name):
    ex, _, _ = _make(name)
    assert ex.name == name


def test_make_executor_rejects_unknown():
    policy = FakePolicy()
    with pytest.raises(ValueError):
        make_executor('telepathy', FakeWorker(policy), policy)


@pytest.mark.parametrize('name', STRATEGIES)
def test_streams_actions_or_holds(name):
    ex, _, policy = _make(name, delay=2)
    for t in range(4 * H):
        a = ex.step(_obs(), t)
        assert a is None or np.asarray(a).shape == (policy.action_dim,)


# ----------------------------------------------------------------- synchronous
def test_synchronous_pauses_exactly_delay_steps():
    ex, _, _ = _make('synchronous', delay=3)
    out = _run(ex, 3 + H + 3 + 2)
    # bootstrap: request at t=0, arrives after 3 polls -> 3 holds
    assert out[:3] == [None, None, None]
    # chunk 1 streams fully...
    assert out[3:3 + H] == [1.0] * H
    # ...then pauses again for the next inference (requested as the last action went out)
    assert out[3 + H:3 + H + 2] == [None, None]
    assert out[3 + H + 2] == 2.0


# ----------------------------------------------------------------- naive async
def test_naive_async_jumps_to_new_chunk_start():
    ex, _, _ = _make('naive_async', delay=2, replan_every=4)
    out = _run(ex, 16)
    i2 = out.index(2.0)
    # the tick before switching it was still executing chunk 1: the jump is the discontinuity
    assert out[i2 - 1] == 1.0
    # after the jump it plays chunk 2 from index 0 (replaying the past): stays on 2.0
    assert out[i2:i2 + 2] == [2.0, 2.0]


def test_naive_async_repeats_last_action_when_starved():
    # replan cadence far longer than the chunk: it runs out and repeats open-loop
    ex, _, _ = _make('naive_async', delay=1, replan_every=100)
    out = _run(ex, 2 * H)
    assert out[0] is None                      # bootstrap
    assert all(v == 1.0 for v in out[1:])      # chunk 1 then stale repeats of its last action


# ------------------------------------------------------------ temporal ensemble
def test_temporal_ensemble_averages_overlapping_chunks():
    ex, _, _ = _make('temporal_ensemble', delay=1, m=0.01, replan_every=2)
    out = _run(ex, 12)
    numeric = [v for v in out if v is not None]
    # with several live chunks the vote is a strict blend, not any single chunk's value
    assert any(v not in (1.0, 2.0, 3.0, 4.0, 5.0) for v in numeric)


def test_temporal_ensemble_weights_favor_oldest():
    """ACT convention: exp(-m*i) ranked oldest-first -> older prediction dominates."""
    ex, _, _ = _make('temporal_ensemble', delay=1, m=10.0, replan_every=1)
    out = _run(ex, 6)
    numeric = [v for v in out if v is not None]
    # with a huge m the vote collapses onto the OLDEST covering chunk (value 1.0 while it lives)
    assert numeric[0] == pytest.approx(1.0)
    assert numeric[-1] == pytest.approx(1.0, abs=0.05)


# ------------------------------------------------------------------------- RTC
def test_rtc_continues_time_aligned_no_replay():
    ex, worker, _ = _make('rtc', delay=3)
    out = _run(ex, 3 + H)
    assert out[:3] == [None, None, None]       # bootstrap wait
    first2 = out.index(2.0)
    # chunk 2 was issued at the tick chunk 1 started executing; it arrives 3 steps later and
    # continues at index 3 — so chunk 1 executed exactly 3 actions (the frozen overlap)
    assert out[3:first2] == [1.0, 1.0, 1.0]
    assert ex._delays and ex._delays[-1] == 3  # measured, not assumed


def test_rtc_passes_the_remaining_plan_as_guidance():
    """RTC sends the plan in hand from the current action onward, and the MASK says which of it
    is frozen. The frozen length is read off the weights, not off len(guide): the guide also
    covers the soft region, which is what keeps the new chunk continuous with the old plan."""
    ex, worker, _ = _make('rtc', delay=2)
    _run(ex, 10)
    with_guide = [r for r in worker.requests if r[1] is not None and len(r[1])]
    assert with_guide, 'RTC never requested an inpainted chunk'
    _t_issue, guide, weights = with_guide[0]

    assert len(weights) == H                          # full-chunk soft mask
    d = int(np.count_nonzero(weights == 1.0))
    assert d > 0, 'nothing was frozen, so inference has no protected overlap'
    assert np.allclose(weights[:d], 1.0)              # frozen region is a prefix of the mask
    assert len(guide) >= int(np.count_nonzero(weights)), 'weighted steps with no target'
    # the guide is the executing chunk's continuation: constant chunk value, not zeros
    assert np.all(guide == guide.flat[0]) and guide.flat[0] > 0


def test_rtc_survives_delay_longer_than_chunk():
    """Degenerate regime d >= H (inference slower than a whole chunk): the arrival covers only
    the past. RTC must fall back to executing the stale chunk (synchronous semantics), never
    freeze the robot forever, and never freeze the ENTIRE next chunk in the request."""
    ex, worker, _ = _make('rtc', delay=12)     # > H=8
    out = _run(ex, 26)
    assert out[:12] == [None] * 12             # waiting for the first chunk
    assert out[12:20] == [1.0] * 8             # stale chunk executed from 0, not dropped
    assert any(v is not None and v >= 2.0 for v in out[20:]), 'second chunk never executed'
    for _t, guide, w in worker.requests:
        if guide is not None:
            # the invariant is about the MASK, not the guide's length: s_min actions must stay
            # unfrozen or the policy is asked to reproduce a plan it cannot improve on
            assert int(np.count_nonzero(w == 1.0)) < H, (
                'a fully-frozen request can never plan anything new')


def test_rtc_forecast_tracks_measured_delay():
    ex, _, _ = _make('rtc', delay=4)
    _run(ex, 30)
    assert ex.forecast_delay() == 4


def test_rtc_freeze_weights_structure():
    Hh, s, d = 16, 4, 3
    w = RTCExecutor.freeze_weights(Hh, s, d)
    assert w.shape == (Hh,)
    assert np.allclose(w[:d], 1.0)             # first d frozen
    assert np.allclose(w[Hh - s:], 0.0)        # tail unconstrained
    assert np.all((w >= 0.0) & (w <= 1.0))
    mid = w[d:Hh - s]
    assert np.all(np.diff(mid) <= 1e-9)        # soft middle decays toward the free tail


def test_network_aware_uses_quantile_not_max():
    """Wedge B: a quantile forecast should be <= RTC's max on a heavy-tailed sample."""
    delays = [1, 1, 1, 1, 1, 1, 1, 1, 1, 20]   # one outlier (jitter spike)
    rtc, _, _ = _make('rtc')
    na, _, _ = _make('network_aware', quantile=0.8)
    rtc._delays = list(delays)
    na._delays = list(delays)
    assert na.forecast_delay() < rtc.forecast_delay()


# ------------------------------------------------------------------ reset/epoch
def test_reset_discards_in_flight_chunk():
    ex, worker, _ = _make('synchronous', delay=5)
    ex.step(_obs(), 0)                 # request issued, in flight
    ex.reset()                         # episode boundary mid-inference
    out = _run(ex, 12, start=0)
    # the pre-reset chunk (value 1.0) must never be executed post-reset
    numeric = [v for v in out if v is not None]
    assert numeric and all(v >= 2.0 for v in numeric)


def test_reset_clears_chunk_state():
    ex, _, _ = _make('synchronous', delay=1)
    _run(ex, 4)
    assert ex._chunk is not None
    ex.reset()
    assert ex._chunk is None


def test_arrival_metrics_reported_once():
    ex, _, _ = _make('synchronous', delay=2)
    seen = []
    for t in range(6):
        ex.step(_obs(), t)
        m = ex.take_arrival_metrics()
        if m is not None:
            seen.append(m)
    assert len(seen) == 1              # exactly one arrival in this window
    compute_ms, delay_steps = seen[0]
    assert delay_steps == 2


# ------------------------------------------------- the guided-resampling capability check
class _Deterministic:
    """A backend with no guided generation — ACT, ONNX-ACT and the PyTorch fallback all are."""
    chunk_size = 16
    action_dim = 7
    n_obs_steps = 1
    guided_resampling = False


class _Generative(_Deterministic):
    guided_resampling = True


def test_rtc_on_a_deterministic_backend_is_flagged():
    """THE silent one: the run completes and writes a full CSV row labelled `rtc`, but
    predict_inpaint fell back to a post-hoc soft blend — which is what RTC exists to beat."""
    from evh_controller.chunk_executor import RTCExecutor, guidance_warning

    msg = guidance_warning(RTCExecutor, _Deterministic())

    assert msg is not None
    assert 'SOFT BLEND' in msg
    assert 'rtc' in msg
    assert 'dp_onnx' in msg, 'the warning should name a backend that would work'


def test_bid_is_flagged_too():
    from evh_controller.chunk_executor import BIDExecutor, guidance_warning

    assert guidance_warning(BIDExecutor, _Deterministic()) is not None


def test_network_aware_inherits_the_requirement_from_rtc():
    """Wedge B is RTC with a different forecast; it needs the same guidance and must not slip
    through by virtue of being a subclass."""
    from evh_controller.chunk_executor import NetworkAwareExecutor, guidance_warning

    assert guidance_warning(NetworkAwareExecutor, _Deterministic()) is not None


def test_a_generative_backend_is_not_flagged():
    from evh_controller.chunk_executor import RTCExecutor, guidance_warning

    assert guidance_warning(RTCExecutor, _Generative()) is None


@pytest.mark.parametrize('strategy', ['synchronous', 'naive_async', 'temporal_ensemble'])
def test_strategies_that_do_not_resample_are_never_flagged(strategy):
    """These splice actions they already have; a deterministic backend is a legitimate pairing
    and warning about it would train people to ignore the warning."""
    from evh_controller.chunk_executor import _REGISTRY, guidance_warning

    assert guidance_warning(_REGISTRY[strategy], _Deterministic()) is None


def test_the_diffusion_backends_declare_the_capability():
    """The flag is only worth having if the backends that DO guide actually set it."""
    from evh_controller.dp_onnx_policy import DiffusionONNXBackend
    from evh_controller.dp_repo_policy import DiffusionPolicyRepoBackend

    assert DiffusionPolicyRepoBackend.guided_resampling is True
    assert DiffusionONNXBackend.guided_resampling is True


def test_the_base_policy_does_not_claim_the_capability():
    from evh_controller.policy import ACTBackend, ChunkPolicy, ONNXBackend

    assert ChunkPolicy.guided_resampling is False
    assert ACTBackend.guided_resampling is False
    assert ONNXBackend.guided_resampling is False
