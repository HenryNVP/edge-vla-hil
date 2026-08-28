"""Tests for the controller's observation history — pure Python, no ROS.

This is half of the policy contract (policy.py's docstring is the other half): values stacked over
the last n_obs_steps CONTROL TICKS, oldest first. The subtle one is that sampling is driven by the
tick and not by message arrival — under heavy delay the same frame legitimately repeats in
consecutive slots, and that repetition is the signal the latency-robust strategies react to.
"""
import numpy as np
import pytest

from evh_controller.obs_buffer import ObsBuffer

H = W = 4


def _img(fill):
    return np.full((H, W, 3), fill, np.uint8)


def _prop(fill):
    return np.full(9, fill, np.float32)


def _filled(n_obs_steps=2, needs_wrist=False, fill=1):
    buf = ObsBuffer(n_obs_steps, needs_wrist=needs_wrist)
    buf.image = _img(fill)
    buf.proprio = _prop(fill)
    if needs_wrist:
        buf.wrist = _img(fill)
    return buf


def test_nothing_is_sampled_until_every_required_stream_has_arrived():
    """The first chunk must never be computed against a half-populated observation."""
    buf = ObsBuffer(2, needs_wrist=False)
    assert buf.ready() is False and buf.sample() is None

    buf.image = _img(1)
    assert buf.sample() is None, 'image alone is not an observation'

    buf.proprio = _prop(1)
    assert buf.ready() is True and buf.sample() is not None


def test_a_wrist_policy_waits_for_the_wrist_camera():
    buf = ObsBuffer(2, needs_wrist=True)
    buf.image, buf.proprio = _img(1), _prop(1)
    assert buf.sample() is None, 'needs_wrist policy sampled without a wrist frame'

    buf.wrist = _img(1)
    assert buf.sample() is not None


def test_history_is_stacked_oldest_first():
    buf = _filled(n_obs_steps=3)
    for fill in (1, 2, 3):
        buf.image, buf.proprio = _img(fill), _prop(fill)
        obs = buf.sample()

    assert obs['agentview'].shape == (3, H, W, 3)
    assert [int(f[0, 0, 0]) for f in obs['agentview']] == [1, 2, 3], 'not oldest-first'
    assert obs['proprio'].shape == (3, 9)


def test_history_is_capped_at_n_obs_steps():
    """n_obs_steps is a checkpoint property; sending more frames than the policy was trained on
    would fail deep inside the model with a shape error."""
    buf = _filled(n_obs_steps=2)
    for fill in range(5):
        buf.image, buf.proprio = _img(fill), _prop(fill)
        obs = buf.sample()

    assert obs['agentview'].shape[0] == 2
    assert [int(f[0, 0, 0]) for f in obs['agentview']] == [3, 4], 'kept the wrong window'


def test_a_frame_repeats_when_no_new_message_arrived():
    """Sampling is per control tick, not per message. Under delay the policy sees the same frame
    twice — that is the observation the strategies are supposed to cope with, not an error."""
    buf = _filled(n_obs_steps=2, fill=7)
    buf.sample()
    obs = buf.sample()          # tick again with nothing new delivered

    assert [int(f[0, 0, 0]) for f in obs['agentview']] == [7, 7]


def test_wrist_is_absent_from_the_dict_when_there_is_no_wrist_camera():
    obs = _filled(n_obs_steps=1, needs_wrist=False).sample()
    assert 'wrist' not in obs
    assert set(obs) == {'agentview', 'proprio'}


def test_wrist_is_stacked_alongside_the_other_streams():
    buf = _filled(n_obs_steps=2, needs_wrist=True)
    buf.sample()
    obs = buf.sample()

    assert obs['wrist'].shape == (2, H, W, 3)


def test_clear_drops_the_history_but_keeps_the_latest_messages():
    """On /episode/reset the history must not bleed across the boundary, but the latest frames are
    still valid — the plant has already reset and keeps publishing."""
    buf = _filled(n_obs_steps=3, fill=1)
    buf.sample()
    buf.sample()
    buf.clear()

    assert buf.ready() is True, 'clear() must not un-ready the buffer'
    obs = buf.sample()
    assert obs['agentview'].shape[0] == 1, 'history survived the episode boundary'


@pytest.mark.parametrize('n_obs_steps', [0, -1])
def test_a_degenerate_history_length_still_keeps_one_step(n_obs_steps):
    """A zero-length deque would silently stack nothing and hand the policy an empty batch."""
    buf = _filled(n_obs_steps=n_obs_steps)
    assert buf.sample()['agentview'].shape[0] == 1
