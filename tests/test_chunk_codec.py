"""Tests for the robot-side execution wire format — pure numpy, no ROS.

A layout slip here is silent in the worst way: a chunk decoded with the wrong horizon or action
width still produces numbers, just not the ones the policy computed.
"""
import numpy as np
import pytest

from evh_controller.chunk_codec import (
    Chunk,
    CodecError,
    Request,
    decode_chunk,
    decode_request,
    encode_chunk,
    encode_request,
)


def test_a_bootstrap_request_round_trips_without_guidance():
    out = decode_request(encode_request(Request(3, 40, 2, None, None), action_dim=7))
    assert (out.req_id, out.t_issue, out.epoch) == (3, 40, 2)
    assert out.prefix is None and out.weights is None


def test_an_rtc_request_round_trips_its_prefix_and_weights():
    prefix = np.arange(5 * 7, dtype=float).reshape(5, 7)
    weights = np.linspace(1.0, 0.0, 16)
    out = decode_request(encode_request(Request(0, 1, 1, prefix, weights), action_dim=7))
    assert np.array_equal(out.prefix, prefix)
    assert np.allclose(out.weights, weights)


def test_a_chunk_round_trips_shape_values_and_timing():
    actions = np.random.default_rng(0).normal(size=(16, 7)).astype(np.float32)
    out = decode_chunk(encode_chunk(Chunk(9, 120, 4, 0.068, actions)))
    assert (out.req_id, out.t_issue, out.epoch) == (9, 120, 4)
    assert out.compute_s == pytest.approx(0.068)
    assert out.actions.shape == (16, 7) and out.actions.dtype == np.float32
    assert np.allclose(out.actions, actions)


def test_large_tick_counts_survive_the_float_encoding():
    out = decode_chunk(encode_chunk(Chunk(2 ** 40, 2 ** 40 + 1, 7, 0.0, np.zeros((1, 7)))))
    assert out.req_id == 2 ** 40 and out.t_issue == 2 ** 40 + 1


@pytest.mark.parametrize('values', [[], [1.0, 2.0], [0, 0, 0, 0, 2, 7] + [0.0] * 13])
def test_a_chunk_whose_length_does_not_match_its_header_is_refused(values):
    with pytest.raises(CodecError):
        decode_chunk(values)


def test_a_request_whose_length_does_not_match_its_header_is_refused():
    good = encode_request(Request(0, 0, 0, np.zeros((2, 7)), np.ones(4)), action_dim=7)
    with pytest.raises(CodecError):
        decode_request(good[:-1])


def test_a_chunk_must_be_two_dimensional():
    with pytest.raises(CodecError):
        encode_chunk(Chunk(0, 0, 0, 0.0, np.zeros(7)))
