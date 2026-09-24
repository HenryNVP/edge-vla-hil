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

`executor` chooses where chunks are executed, one of the paper's factors:
  policy  (default) the executor runs here and streams one waypoint per tick over the network.
  robot   the executor runs next to the robot (executor_node.py) and this node only SERVES
          chunks: a request arrives on /policy/request, is answered from the newest observation
          history on /cmd/chunk (chunk_server.py). `strategy` is then the executor node's
          parameter, not this one's.
Either way the node announces the policy's shape on the latched /policy/info, which is all the
robot side needs to run a strategy without loading the model.

Wiring only. The pieces with behaviour of their own live beside it: `chunk_executor/` (the
strategies), `inference_worker.py` (the background GPU slot), `obs_buffer.py` (the observation
history contract), `policy.py` (the backends).

Metrics:
  /metrics/inference_ms   true wall-clock inference time of a chunk        (on its arrival tick)
  /metrics/delay_steps    request->arrival delay in control steps: d_inf   (on its arrival tick)
  /metrics/obs_age_ms     age of the observation the policy uses: d_obs    (every control tick)

d_obs compares the plant's capture stamp with this node's clock, so across machines it is only as
good as their clock sync (chrony); on one host it is exact. The third component, d_act, is
measured where the waypoint lands, by the reactive layer.
"""
from __future__ import annotations

import time

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
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Bool, Empty, Float32

from evh_controller.chunk_codec import CodecError, decode_request, encode_chunk
from evh_controller.chunk_executor import make_executor
from evh_controller.chunk_server import ChunkServer
from evh_controller.image_codec import RAW_QUALITY, decode_image
from evh_controller.inference_worker import InferenceWorker
from evh_controller.obs_buffer import ObsBuffer
from evh_controller.phase import seconds_to_phase
from evh_controller.policy import make_policy

# Latched: published once at startup, but the plant must receive it whenever it joins — the
# controller often comes up much later (checkpoint load / HF download). See PlantNode._on_policy_mode.
MODE_QOS = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


# The command path, and the one topic that crosses the network in the split deployment. It is
# BEST_EFFORT with a depth of 1 ON PURPOSE, and the three packages that touch it must agree or DDS
# silently refuses to pair them. (The latency relay between them is best-effort both ways, via
# qos_profile_sensor_data, so it pairs with this profile on either side.)
#
# Reliable delivery is the wrong contract here. A waypoint is an ABSOLUTE target and the reactive
# layer latches it, so a lost one costs nothing — it simply keeps tracking the previous target.
# A LATE one costs plenty: reliable QoS retransmits and delivers in order, so a stale waypoint
# arrives after a fresher one was already available and the arm is commanded backwards. That is
# precisely the "re-apply an old command" behaviour the latched-absolute-target design exists to
# prevent. Newest-wins, no retransmit, no head-of-line blocking.
WAYPOINT_QOS = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                          history=QoSHistoryPolicy.KEEP_LAST)
# The chunk request/reply pair (robot-side execution) reuses it for the same reasons: a retry or a
# newer chunk supersedes an old one, and a reliable resend would only deliver something stale.

# How often the chunk server checks for a finished chunk (robot-side execution only).
SERVE_HZ = 200.0


def _parse_absolute(value: str) -> bool | None:
    """`auto` lets the backend derive its action convention; true/false forces it.

    The forced settings exist for deliberately mismatched runs (pair with the plant's
    `strict_mode_check:=false`) and for checkpoints predating the mode stamp — not for routine
    use, where guessing is exactly what invariant 1 is about.
    """
    text = str(value).strip().lower()
    if text in ('', 'auto'):
        return None
    if text in ('true', '1', 'yes'):
        return True
    if text in ('false', '0', 'no'):
        return False
    raise ValueError(f"policy_absolute must be auto|true|false, got {value!r}")


def _stamp_s(msg) -> float | None:
    """A header stamp in seconds; None for an unstamped (zero) header."""
    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return t if t > 0.0 else None


class ControllerNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__('evh_controller', **kwargs)

        self.declare_parameter('backend', 'pytorch')        # pytorch | act | onnx | dp | dp_onnx
        self.declare_parameter('weights_path', '')           # ckpt (dir/.ckpt) or .onnx
        self.declare_parameter('strategy', 'synchronous')    # chunk-execution strategy
        self.declare_parameter('control_hz', 20.0)   # action stream rate = policy training rate
        self.declare_parameter('denoise_steps', 16)  # dp backend: DDIM steps (0=ckpt default)
        self.declare_parameter('prompt', 'pick up the block')
        self.declare_parameter('policy_absolute', 'auto')    # auto (from checkpoint) | true | false
        self.declare_parameter('executor', 'policy')         # policy | robot (see docstring)
        self.declare_parameter('tick_phase_ms', 10.0)  # tick this long after the plant publishes
        # must match the plant's image_quality: 0 = raw Image, 1-100 = JPEG CompressedImage. DDS
        # pairs nothing across mismatched types, so a disagreement silences the image topics.
        self.declare_parameter('image_quality', 0)

        backend = self.get_parameter('backend').value
        weights = self.get_parameter('weights_path').value
        strategy = self.get_parameter('strategy').value
        self.control_hz = self.get_parameter('control_hz').value
        self.img_quality = int(self.get_parameter('image_quality').value)
        denoise_steps = int(self.get_parameter('denoise_steps').value)
        absolute = _parse_absolute(self.get_parameter('policy_absolute').value)
        self.placement = str(self.get_parameter('executor').value).strip().lower()
        if self.placement not in ('policy', 'robot'):
            raise ValueError(f"executor must be policy|robot, got {self.placement!r}")

        self.policy = make_policy(backend, weights, denoise_steps=denoise_steps,
                                  absolute=absolute)
        self.worker = InferenceWorker(self.policy)
        if self.placement == 'policy':
            self.chunk_executor = make_executor(strategy, self.worker, self.policy)
            self.server = None
        else:
            self.chunk_executor = None
            self.server = ChunkServer(self.worker)
        self._latest_obs: dict | None = None
        self.get_logger().info(
            f'evh_controller: backend={backend} executor={self.placement} '
            f'strategy={strategy if self.placement == "policy" else "(robot side)"} '
            f'ctrl={self.control_hz}Hz '
            f'chunk={self.policy.chunk_size} n_obs={self.policy.n_obs_steps} '
            f'absolute={self.policy.absolute_actions} '
            f'guided={self.policy.guided_resampling}')

        # latest observations (overwritten by callbacks) + per-tick history for the policy
        self.obs = ObsBuffer(self.policy.n_obs_steps, self.policy.needs_wrist)
        self._t = 0   # control timestep counter

        img_cls = Image if self.img_quality <= RAW_QUALITY else CompressedImage
        self.create_subscription(img_cls, '/obs/image', self._on_image, qos_profile_sensor_data)
        self.create_subscription(
            img_cls, '/obs/image_wrist', self._on_wrist, qos_profile_sensor_data)
        self.create_subscription(
            JointState, '/obs/proprio', self._on_proprio, qos_profile_sensor_data)
        # eval-plane signal from the plant; deliberately NOT routed through the latency relay
        self.create_subscription(Empty, '/episode/reset', self._on_episode_reset, 10)

        self.pub_waypoint = self.create_publisher(JointState, '/cmd/waypoint', WAYPOINT_QOS)
        self.pub_latency = self.create_publisher(Float32, '/metrics/inference_ms', 10)
        self.pub_delay = self.create_publisher(Float32, '/metrics/delay_steps', 10)
        self.pub_obs_age = self.create_publisher(Float32, '/metrics/obs_age_ms', 10)

        # announce the mode the CHECKPOINT dictates so the plant can cross-check its launch arg
        self.pub_mode = self.create_publisher(Bool, '/policy/absolute', MODE_QOS)
        self.pub_mode.publish(Bool(data=bool(self.policy.absolute_actions)))
        # the policy's shape, for a robot-side executor that never loads the model
        self.pub_info = self.create_publisher(JointState, '/policy/info', MODE_QOS)
        self.pub_info.publish(self._policy_info_msg())

        if self.server is not None:
            self.create_subscription(
                JointState, '/policy/request', self._on_request, WAYPOINT_QOS)
            self.pub_chunk = self.create_publisher(JointState, '/cmd/chunk', WAYPOINT_QOS)
            # faster than the control tick so a finished chunk is not held back up to 50 ms
            self.create_timer(1.0 / SERVE_HZ, self._serve)

        # tick a fixed few ms after the plant's observations go out (phase.py), not at whatever
        # phase this node happened to start at
        time.sleep(seconds_to_phase(time.time(), 1.0 / self.control_hz,
                                    float(self.get_parameter('tick_phase_ms').value) / 1e3))
        self.create_timer(1.0 / self.control_hz, self._tick)

    # ------------------------------------------------------------- callbacks
    def _on_image(self, msg) -> None:
        self.obs.put('image', decode_image(msg), _stamp_s(msg))

    def _on_wrist(self, msg) -> None:
        self.obs.put('wrist', decode_image(msg), _stamp_s(msg))

    def _on_proprio(self, msg: JointState) -> None:
        self.obs.put('proprio', np.asarray(msg.position, dtype=np.float32), _stamp_s(msg))

    def _on_episode_reset(self, _msg: Empty) -> None:
        if self.chunk_executor is not None:
            self.chunk_executor.reset()
        if self.server is not None:
            self.server.reset()
        # frames captured before the reset show the previous episode's scene; only newer ones count
        self.obs.clear(since_s=self.get_clock().now().nanoseconds / 1e9)
        self._latest_obs = None
        self._t = 0

    def _on_request(self, msg: JointState) -> None:
        try:
            self.server.on_request(decode_request(msg.position))
        except CodecError as exc:
            self.get_logger().warn(f'dropping a malformed chunk request: {exc}')

    def _serve(self) -> None:
        chunk = self.server.step(self._latest_obs)
        if chunk is None:
            return
        self.pub_latency.publish(Float32(data=float(chunk.compute_s * 1e3)))
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = encode_chunk(chunk)
        self.pub_chunk.publish(msg)

    # --------------------------------------------------------------- control
    def _tick(self) -> None:
        obs = self.obs.sample()
        if obs is None:
            return  # wait for first observations
        age = self.obs.age(self.get_clock().now().nanoseconds / 1e9)
        if age is not None:
            self.pub_obs_age.publish(Float32(data=float(age * 1e3)))
        if self.chunk_executor is None:
            self._latest_obs = obs     # serving: requests are answered from this history
            return

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
    def _policy_info_msg(self) -> JointState:
        """[chunk_size, action_dim, guided_resampling, absolute_actions] — see executor_node."""
        msg = JointState()
        msg.position = [float(self.policy.chunk_size), float(self.policy.action_dim),
                        float(bool(self.policy.guided_resampling)),
                        float(bool(self.policy.absolute_actions))]
        return msg

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
