"""Controller node: runs the diffusion/flow policy under a pluggable chunk-execution strategy.

HiL "Controller" side (Jetson Orin Nano). Subscribes to the (latency-injected) observation topics
and streams the policy's raw OSC_POSE actions on /cmd/waypoint (JointState.position =
[dpos(3), axis-angle drot(3), gripper]) at the policy control rate. The full 7-dim action is
forwarded — the reactive layer anchors it into an absolute task-space target using zero-delay
local state (or forwards it scaled, in the passthrough baseline). *How* the action chunk is
executed under inference latency is delegated to a ChunkExecutor strategy
(synchronous | naive_async | temporal_ensemble | bid | rtc | network_aware), which is the seam for
the Wedge-A baseline comparison and the Wedge-B extension.

Inference is ASYNCHRONOUS: the policy runs on a background InferenceWorker, the tick only
streams actions from the strategy (which may return None = hold, e.g. the synchronous strategy's
pause). The delay a chunk experiences — request to arrival, in control steps — is measured
honestly and published, and is what feeds RTC's delay forecast.

Resets the executor (chunk buffers, timestep counter) on /episode/reset from the plant so chunks
never bleed across episode boundaries.

Wiring only. The pieces with behaviour of their own live beside it: `chunk_executor/` (the
strategies), `inference_worker.py` (the background GPU slot), `obs_buffer.py` (the observation
history contract), `policy.py` (the backends).

Metrics (published on the tick a chunk arrives):
  /metrics/inference_ms   true wall-clock inference time of that chunk
  /metrics/delay_steps    request->arrival delay in control steps (what the strategies fight)
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Empty, Float32

from evh_controller.chunk_executor import make_executor
from evh_controller.inference_worker import InferenceWorker
from evh_controller.obs_buffer import ObsBuffer
from evh_controller.policy import make_policy

# Latched: published once at startup, but the plant must receive it whenever it joins — the
# controller often comes up much later (checkpoint load / HF download). See PlantNode._on_policy_mode.
MODE_QOS = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


# The command path, and the one topic that crosses the network in the split deployment. It is
# BEST_EFFORT with a depth of 1 ON PURPOSE, and the three packages that touch it must agree or DDS
# silently refuses to pair them.
#
# Reliable delivery is the wrong contract here. A waypoint is an ABSOLUTE target and the reactive
# layer latches it, so a lost one costs nothing — it simply keeps tracking the previous target.
# A LATE one costs plenty: reliable QoS retransmits and delivers in order, so a stale waypoint
# arrives after a fresher one was already available and the arm is commanded backwards. That is
# precisely the "re-apply an old command" behaviour the latched-absolute-target design exists to
# prevent. Newest-wins, no retransmit, no head-of-line blocking.
WAYPOINT_QOS = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                          history=QoSHistoryPolicy.KEEP_LAST)


class ControllerNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__('evh_controller', **kwargs)

        self.declare_parameter('backend', 'pytorch')        # pytorch | dp | tensorrt
        self.declare_parameter('weights_path', '')           # ckpt (dir/.ckpt) or .engine
        self.declare_parameter('strategy', 'synchronous')    # chunk-execution strategy
        self.declare_parameter('control_hz', 20.0)   # action stream rate = policy training rate
        self.declare_parameter('denoise_steps', 16)  # dp backend: DDIM steps (0=ckpt default)
        self.declare_parameter('prompt', 'pick up the block')

        backend = self.get_parameter('backend').value
        weights = self.get_parameter('weights_path').value
        strategy = self.get_parameter('strategy').value
        self.control_hz = self.get_parameter('control_hz').value
        denoise_steps = int(self.get_parameter('denoise_steps').value)

        self.policy = make_policy(backend, weights, denoise_steps=denoise_steps)
        self.worker = InferenceWorker(self.policy)
        self.chunk_executor = make_executor(strategy, self.worker, self.policy)
        self.get_logger().info(
            f'evh_controller: backend={backend} strategy={strategy} ctrl={self.control_hz}Hz '
            f'chunk={self.policy.chunk_size} n_obs={self.policy.n_obs_steps} '
            f'absolute={self.policy.absolute_actions}')

        # latest observations (overwritten by callbacks) + per-tick history for the policy
        self.obs = ObsBuffer(self.policy.n_obs_steps, self.policy.needs_wrist)
        self._t = 0   # control timestep counter

        self.create_subscription(Image, '/obs/image', self._on_image, qos_profile_sensor_data)
        self.create_subscription(
            Image, '/obs/image_wrist', self._on_wrist, qos_profile_sensor_data)
        self.create_subscription(
            JointState, '/obs/proprio', self._on_proprio, qos_profile_sensor_data)
        # eval-plane signal from the plant; deliberately NOT routed through the latency relay
        self.create_subscription(Empty, '/episode/reset', self._on_episode_reset, 10)

        self.pub_waypoint = self.create_publisher(JointState, '/cmd/waypoint', WAYPOINT_QOS)
        self.pub_latency = self.create_publisher(Float32, '/metrics/inference_ms', 10)
        self.pub_delay = self.create_publisher(Float32, '/metrics/delay_steps', 10)

        # announce the mode the CHECKPOINT dictates so the plant can cross-check its launch arg
        self.pub_mode = self.create_publisher(Bool, '/policy/absolute', MODE_QOS)
        self.pub_mode.publish(Bool(data=bool(self.policy.absolute_actions)))

        self.create_timer(1.0 / self.control_hz, self._tick)

    # ------------------------------------------------------------- callbacks
    def _on_image(self, msg: Image) -> None:
        self.obs.image = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)

    def _on_wrist(self, msg: Image) -> None:
        self.obs.wrist = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)

    def _on_proprio(self, msg: JointState) -> None:
        self.obs.proprio = np.asarray(msg.position, dtype=np.float32)

    def _on_episode_reset(self, _msg: Empty) -> None:
        self.chunk_executor.reset()
        self.obs.clear()
        self._t = 0

    # --------------------------------------------------------------- control
    def _tick(self) -> None:
        obs = self.obs.sample()
        if obs is None:
            return  # wait for first observations

        action = self.chunk_executor.step(obs, self._t)
        self._t += 1

        metrics = self.chunk_executor.take_arrival_metrics()
        if metrics is not None:
            compute_ms, delay_steps = metrics
            self.pub_latency.publish(Float32(data=float(compute_ms)))
            self.pub_delay.publish(Float32(data=float(delay_steps)))

        if action is not None:   # None = hold: no new waypoint, the reactive layer keeps tracking
            self.pub_waypoint.publish(self._to_waypoint(action))

    # --------------------------------------------------------------- helpers
    def _to_waypoint(self, action: np.ndarray) -> JointState:
        """Pack the full OSC_POSE action [dpos(3), drot(3), gripper] into the waypoint message.

        The delta is anchored downstream by the reactive layer against zero-delay local EE state
        (robosuite's own per-step goal-update convention), so no pose composition happens here.
        """
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size < 7:
            a = np.pad(a, (0, 7 - a.size))
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = [float(v) for v in a[:7]]
        return msg

    def destroy_node(self) -> None:
        self.worker.shutdown()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
