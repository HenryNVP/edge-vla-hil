"""Pure-Python tests for chunk-execution strategies — no ROS2 required.

Guards the Wedge-A/B seam: every strategy is constructible, streams correctly-shaped actions, and
RTC's soft-mask weights match the paper's structure. These hold against the stub policy and survive
the swap to the real diffusion/flow backend.
"""
import numpy as np
import pytest

from evh_controller.policy import make_policy
from evh_controller.chunk_executor import (
    make_executor, RTCExecutor, NetworkAwareExecutor,
    SynchronousExecutor, NaiveAsyncExecutor, TemporalEnsembleExecutor, BIDExecutor,
)

STRATEGIES = ['synchronous', 'naive_async', 'temporal_ensemble', 'bid', 'rtc', 'network_aware']


def _obs():
    return (np.zeros((84, 84, 3), np.uint8), np.zeros(7, np.float32))


@pytest.mark.parametrize('name', STRATEGIES)
def test_make_executor(name):
    ex = make_executor(name)
    assert ex.name == name


def test_make_executor_rejects_unknown():
    with pytest.raises(ValueError):
        make_executor('telepathy')


@pytest.mark.parametrize('name', STRATEGIES)
def test_streams_actions_of_right_shape(name):
    ex = make_executor(name)
    policy = make_policy('pytorch', '')
    obs = _obs()
    for t in range(40):                       # span multiple chunk boundaries (chunk_size=16)
        a = ex.select_action(obs, policy, t)
        assert a.shape == (policy.action_dim,)


def test_reset_clears_state():
    ex = SynchronousExecutor()
    policy = make_policy('pytorch', '')
    ex.select_action(_obs(), policy, 0)
    assert ex._chunk is not None
    ex.reset()
    assert ex._chunk is None


def test_rtc_freeze_weights_structure():
    H, s, d = 16, 4, 3
    w = RTCExecutor.freeze_weights(H, s, d)
    assert w.shape == (H,)
    assert np.allclose(w[:d], 1.0)            # first d frozen
    assert np.allclose(w[H - s:], 0.0)        # tail unconstrained
    assert np.all((w >= 0.0) & (w <= 1.0))
    # soft middle region is monotonic decreasing toward the unconstrained tail
    mid = w[d:H - s]
    assert np.all(np.diff(mid) <= 1e-9)


def test_network_aware_uses_quantile_not_max():
    """Wedge B: a quantile forecast should be <= RTC's max on a heavy-tailed sample."""
    delays = [1, 1, 1, 1, 1, 1, 1, 1, 1, 20]   # one outlier (jitter spike)
    rtc, na = RTCExecutor(), NetworkAwareExecutor(quantile=0.8)
    rtc._delays = list(delays)
    na._delays = list(delays)
    assert na.forecast_delay() < rtc.forecast_delay()
