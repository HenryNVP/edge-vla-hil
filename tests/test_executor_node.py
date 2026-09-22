"""Unit tests for executor_node.py — the robot-side tick, driven through a stub, no graph spun up.

The strategies and RemoteWorker have their own tests; what the node itself decides is: nothing
runs until the policy's shape is known, metrics are published under the same names as the
policy-side placement (so the recorder reads both the same way), and a hold publishes nothing.
"""
import types

import numpy as np
import pytest
from conftest import requires_ros2


def _stub(executor=None):
    from builtin_interfaces.msg import Time

    published = {k: [] for k in ('waypoint', 'latency', 'delay', 'lost', 'request', 'age')}

    def _pub(key):
        return types.SimpleNamespace(publish=lambda m: published[key].append(m))

    stub = types.SimpleNamespace(
        chunk_executor=executor, worker=None, info=None, _t=0,
        pub_waypoint=_pub('waypoint'), pub_latency=_pub('latency'), pub_delay=_pub('delay'),
        pub_lost=_pub('lost'), pub_request=_pub('request'), pub_chunk_age=_pub('age'),
        get_clock=lambda: types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(to_msg=lambda: Time(sec=1, nanosec=0),
                                              nanoseconds=1_200_000_000)),
        published=published)
    from evh_controller.executor_node import ExecutorNode
    stub._to_waypoint = lambda a: ExecutorNode._to_waypoint(stub, a)
    return stub


class FakeExecutor:
    def __init__(self, action=None, metrics=None, lost=0):
        self.action, self.metrics, self.lost, self.steps = action, metrics, lost, []

    def step(self, obs, t):
        self.steps.append(t)
        return self.action

    def take_arrival_metrics(self):
        m, self.metrics = self.metrics, None
        return m

    def take_lost(self):
        n, self.lost = self.lost, 0
        return n


@requires_ros2
def test_nothing_runs_before_the_policy_info_arrives():
    from evh_controller.executor_node import ExecutorNode

    stub = _stub(executor=None)
    ExecutorNode._tick(stub)
    assert stub._t == 0 and stub.published['waypoint'] == []


@requires_ros2
def test_an_action_goes_out_as_a_local_waypoint():
    from evh_controller.executor_node import ExecutorNode

    stub = _stub(FakeExecutor(action=np.arange(7, dtype=float)))
    ExecutorNode._tick(stub)
    assert list(stub.published['waypoint'][0].position) == [0, 1, 2, 3, 4, 5, 6]
    assert stub._t == 1


@requires_ros2
def test_a_hold_publishes_nothing():
    from evh_controller.executor_node import ExecutorNode

    stub = _stub(FakeExecutor(action=None))
    ExecutorNode._tick(stub)
    assert stub.published['waypoint'] == [] and stub._t == 1


@requires_ros2
def test_arrival_and_loss_metrics_use_the_policy_side_names():
    from evh_controller.executor_node import ExecutorNode

    stub = _stub(FakeExecutor(action=np.zeros(7), metrics=(68.0, 9), lost=2))
    ExecutorNode._tick(stub)
    assert [m.data for m in stub.published['latency']] == pytest.approx([68.0])
    assert [m.data for m in stub.published['delay']] == pytest.approx([9.0])
    assert len(stub.published['lost']) == 2


@requires_ros2
def test_policy_info_decoding_refuses_a_wrong_width():
    from evh_controller.executor_node import policy_info_from

    with pytest.raises(ValueError):
        policy_info_from([16.0, 7.0, 1.0])


@requires_ros2
def test_the_info_builds_the_strategy_over_a_remote_worker():
    from sensor_msgs.msg import JointState

    from evh_controller.executor_node import ExecutorNode
    from evh_controller.remote_worker import RemoteWorker

    stub = _stub(executor=None)
    stub.strategy = 'rtc'
    stub.get_parameter = lambda name: types.SimpleNamespace(
        value={'timeout_factor': 2.0, 'min_timeout_s': 0.25, 'first_timeout_s': 10.0}[name])
    stub._send_request = lambda req: None
    stub.get_logger = lambda: types.SimpleNamespace(info=lambda *_: None)
    ExecutorNode._on_info(stub, JointState(position=[16.0, 7.0, 1.0, 1.0]))

    assert isinstance(stub.worker, RemoteWorker)
    assert stub.chunk_executor.name == 'rtc' and stub.chunk_executor.policy.chunk_size == 16


@requires_ros2
def test_a_chunk_reports_its_downlink_age():
    from sensor_msgs.msg import JointState

    from evh_controller.chunk_codec import Chunk, encode_chunk
    from evh_controller.executor_node import ExecutorNode

    delivered = []
    stub = _stub(executor=None)
    stub.worker = types.SimpleNamespace(deliver=delivered.append)
    msg = JointState(position=encode_chunk(Chunk(0, 0, 1, 0.05, np.zeros((2, 7)))))
    msg.header.stamp.sec, msg.header.stamp.nanosec = 1, 0      # sent 200 ms before "now"
    ExecutorNode._on_chunk(stub, msg)

    assert [m.data for m in stub.published['age']] == pytest.approx([200.0])
    assert len(delivered) == 1
