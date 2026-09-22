"""Wire format for robot-side chunk execution: inference requests up, action chunks down.

Robot-side execution splits the controller in two across the network (see remote_worker.py): the
chunk executor runs next to the robot and the policy server runs next to the GPU. Two messages
cross the link, both carried in a `sensor_msgs/JointState` so no custom message package is needed
(the same trade-off `/cmd/waypoint` makes): the header stamp is the send time, which is what the
receiving side's age metric reads, and `position` is a flat float64 vector laid out below.

    request  (robot -> policy, /policy/request)
        [req_id, t_issue, epoch, action_dim, n_prefix, n_weights,
         prefix (n_prefix * action_dim) ..., weights (n_weights) ...]

    chunk    (policy -> robot, /cmd/chunk)
        [req_id, t_issue, epoch, compute_s, horizon, action_dim, chunk (horizon * action_dim) ...]

`req_id` pairs a reply with its request, so a reply that arrives after the robot side gave up on
it (a timeout, see RemoteWorker) is recognised as stale and dropped rather than spliced in.
`t_issue` and `epoch` are the robot side's own bookkeeping, echoed back untouched: the delay a
strategy measures is then entirely in robot-side ticks, and a chunk computed before an episode
reset is discarded exactly as the local worker's are.

Integers ride in float64, which is exact below 2**53. Everything here is pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_REQ_HEADER = 6
_CHUNK_HEADER = 6


@dataclass
class Request:
    req_id: int
    t_issue: int
    epoch: int
    prefix: np.ndarray | None    # [n_prefix, action_dim] guide actions (RTC), or None
    weights: np.ndarray | None   # [n_weights] soft mask over the chunk (RTC), or None


@dataclass
class Chunk:
    req_id: int
    t_issue: int
    epoch: int
    compute_s: float
    actions: np.ndarray          # [horizon, action_dim]


class CodecError(ValueError):
    """A message whose layout does not add up; never guessed at."""


def encode_request(req: Request, action_dim: int) -> list[float]:
    prefix = (np.zeros((0, action_dim)) if req.prefix is None
              else np.asarray(req.prefix, dtype=np.float64).reshape(-1, action_dim))
    weights = (np.zeros(0) if req.weights is None
               else np.asarray(req.weights, dtype=np.float64).reshape(-1))
    head = [req.req_id, req.t_issue, req.epoch, action_dim, len(prefix), len(weights)]
    return [float(v) for v in head] + prefix.reshape(-1).tolist() + weights.tolist()


def decode_request(values) -> Request:
    v = np.asarray(values, dtype=np.float64)
    if v.size < _REQ_HEADER:
        raise CodecError(f'request of {v.size} values is shorter than its header')
    req_id, t_issue, epoch, a_dim, n_prefix, n_w = (int(x) for x in v[:_REQ_HEADER])
    expected = _REQ_HEADER + n_prefix * a_dim + n_w
    if v.size != expected:
        raise CodecError(f'request carries {v.size} values, header says {expected}')
    body = v[_REQ_HEADER:]
    prefix = body[:n_prefix * a_dim].reshape(n_prefix, a_dim) if n_prefix else None
    weights = body[n_prefix * a_dim:] if n_w else None
    return Request(req_id, t_issue, epoch, prefix, weights)


def encode_chunk(chunk: Chunk) -> list[float]:
    a = np.asarray(chunk.actions, dtype=np.float64)
    if a.ndim != 2:
        raise CodecError(f'a chunk is [horizon, action_dim], got shape {a.shape}')
    head = [chunk.req_id, chunk.t_issue, chunk.epoch, chunk.compute_s, a.shape[0], a.shape[1]]
    return [float(v) for v in head] + a.reshape(-1).tolist()


def decode_chunk(values) -> Chunk:
    v = np.asarray(values, dtype=np.float64)
    if v.size < _CHUNK_HEADER:
        raise CodecError(f'chunk of {v.size} values is shorter than its header')
    req_id, t_issue, epoch = (int(x) for x in v[:3])
    compute_s = float(v[3])
    horizon, a_dim = int(v[4]), int(v[5])
    if v.size != _CHUNK_HEADER + horizon * a_dim:
        raise CodecError(
            f'chunk carries {v.size} values, header says {_CHUNK_HEADER + horizon * a_dim}')
    actions = v[_CHUNK_HEADER:].reshape(horizon, a_dim).astype(np.float32)
    return Chunk(req_id, t_issue, epoch, compute_s, actions)
