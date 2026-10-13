"""Tests for the learned-refiner executor — pure Python, no ROS, no robosuite.

Pinned: every played action is the refiner's output; the refiner is handed the rest of the
current chunk from the entry being played and the chunk's (request tick, object at request);
a gate passed in is ignored, since the refiner times the close itself.
"""
import numpy as np

from evh_controller.chunk_executor.refined import RefinedExecutor
from test_anchored import H, Policy, Worker, _obj


def test_plays_refiner_output_with_remaining_chunk_and_anchor():
    calls = []

    def refine_fn(obs, remaining, chunk_anchor, t):
        calls.append((len(remaining), float(remaining[0, 0]), chunk_anchor, t))
        out = np.array(remaining[0], np.float32)
        out[1] = 9.0
        return out

    policy = Policy()
    ex = RefinedExecutor(Worker(policy, 2), policy, anchor_fn=lambda o: o.get('obj'),
                         refine_fn=refine_fn, gate_fn=lambda o, a: False)
    assert ex.gate_fn is None
    played = [ex.step(_obj(0.01 * t), t) for t in range(6)]
    first = next(k for k, a in enumerate(played) if a is not None)
    assert all(a[1] == 9.0 for a in played[first:])
    n, x0, anchor, t = calls[0]
    assert n == H - (x0 / 0.01)                 # remaining starts at the entry played now
    assert anchor[0] == 0 and np.allclose(anchor[1], [0, 0, 0])
    assert t == first
