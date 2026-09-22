"""Tests for the relay's channel model — pure Python, no ROS, so they can afford large samples.

These are the statistics RQ2 is about: whether the delay distribution has a tail, and whether
loss comes spread thinly (iid) or in outages (Gilbert–Elliott) at the same average rate. If the
channel does not produce what a condition's label says, the curves are plotted against a fiction.
"""
import pytest

from evh_latency.channel import Channel, GilbertElliott


# ------------------------------------------------------------------------- delay
def test_delay_sampling_is_seeded():
    first = [Channel(50.0, 10.0, seed=42).delay_ms() for _ in range(1)]
    a, b = Channel(50.0, 10.0, seed=42), Channel(50.0, 10.0, seed=42)
    assert [a.delay_ms() for _ in range(10)] == [b.delay_ms() for _ in range(10)]
    assert all(d >= 0.0 for d in first)


def test_lognormal_jitter_is_heavy_tailed_where_the_others_are_not():
    """A quantile delay forecast can only beat a max when the delay distribution has a TAIL.
    gaussian and uniform do not have one, so a sweep using them cannot separate the two."""
    tails = {}
    for model in ('gaussian', 'uniform', 'lognormal'):
        ch = Channel(100.0, 20.0, jitter_model=model, seed=3)
        d = sorted(ch.delay_ms() for _ in range(20000))
        tails[model] = d[-1] / max(d[int(0.95 * len(d))], 1e-9)

    assert tails['uniform'] < 1.2, 'uniform is bounded; it has no tail by construction'
    assert tails['lognormal'] > 2 * tails['gaussian'], f'not heavier-tailed: {tails}'


@pytest.mark.parametrize('model', ['gaussian', 'uniform', 'lognormal'])
def test_every_jitter_model_stays_non_negative_and_near_the_nominal_delay(model):
    ch = Channel(100.0, 20.0, jitter_model=model, seed=5)
    d = [ch.delay_ms() for _ in range(5000)]
    assert min(d) >= 0.0, f'{model} produced a negative delay'
    assert 90.0 <= sum(d) / len(d) <= 130.0


def test_an_unknown_jitter_model_falls_back_to_gaussian_rather_than_raising():
    """A typo'd model must not kill the relay mid-sweep; gaussian is the documented default."""
    d = [Channel(50.0, 5.0, jitter_model='definitely-not-a-model', seed=1).delay_ms()
         for _ in range(200)]
    assert min(d) >= 0.0


# -------------------------------------------------------------------------- loss
def _loss_series(ch, seconds=600.0, hz=20.0, start=1000.0):
    return [ch.dropped(start + i / hz) for i in range(int(seconds * hz))]


def _bursts(series):
    runs, n = [], 0
    for lost in series:
        if lost:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    return runs


def test_no_loss_configured_drops_nothing():
    assert not any(_loss_series(Channel(drop_prob=0.0, loss_model='gilbert')))


@pytest.mark.parametrize('loss_model', ['iid', 'gilbert'])
def test_both_loss_models_hit_the_same_average_rate(loss_model):
    """Equal average loss is the whole point of the comparison."""
    ch = Channel(drop_prob=0.15, loss_model=loss_model, burst_ms=250.0, seed=2)
    series = _loss_series(ch, seconds=3000.0)
    assert sum(series) / len(series) == pytest.approx(0.15, abs=0.02)


def test_gilbert_concentrates_the_same_loss_into_long_bursts():
    iid = _bursts(_loss_series(Channel(drop_prob=0.15, loss_model='iid', seed=2), 3000.0))
    ge = _bursts(_loss_series(
        Channel(drop_prob=0.15, loss_model='gilbert', burst_ms=500.0, seed=2), 3000.0))
    mean_iid, mean_ge = sum(iid) / len(iid), sum(ge) / len(ge)

    assert mean_iid < 1.3, 'iid losses at 15% should mostly be singletons'
    # 500 ms at 20 Hz is ~10 messages per outage
    assert 7.0 < mean_ge < 13.0, f'mean burst {mean_ge:.1f} messages, expected ~10'


def test_every_relay_sees_the_same_outages():
    """A WiFi outage hits images, proprio and actions together; two relays with the same seed
    must agree on the state at every instant without talking to each other, even when they
    sample at different rates and started at different times."""
    image = GilbertElliott(0.1, 300.0, seed=7)
    action = GilbertElliott(0.1, 300.0, seed=7)
    _ = [action.bad(1000.0 + i * 0.013) for i in range(500)]   # a head start, another rate
    times = [1000.0 + i * 0.05 for i in range(4000)]
    assert [image.bad(t) for t in times] == [action.bad(t) for t in times]


def test_a_different_seed_gives_a_different_schedule():
    a, b = GilbertElliott(0.1, 300.0, seed=1), GilbertElliott(0.1, 300.0, seed=2)
    times = [1000.0 + i * 0.05 for i in range(4000)]
    assert [a.bad(t) for t in times] != [b.bad(t) for t in times]


def test_the_state_can_be_queried_out_of_order():
    """Messages from different relays arrive interleaved; asking about an earlier time after a
    later one must give the same answer as asking in order."""
    ge = GilbertElliott(0.2, 200.0, seed=3)
    times = [500.0 + i * 0.037 for i in range(2000)]
    forward = [ge.bad(t) for t in times]
    fresh = GilbertElliott(0.2, 200.0, seed=3)
    assert [fresh.bad(t) for t in reversed(times)][::-1] == forward


@pytest.mark.parametrize('loss,burst', [(0.0, 100.0), (1.0, 100.0), (0.1, 0.0)])
def test_gilbert_refuses_degenerate_parameters(loss, burst):
    with pytest.raises(ValueError):
        GilbertElliott(loss, burst)


def test_an_unknown_loss_model_is_refused():
    """Unlike a jitter typo, a loss-model typo would silently turn a burst condition into iid."""
    with pytest.raises(ValueError):
        Channel(drop_prob=0.1, loss_model='burst')
