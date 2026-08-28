"""Simulation Plant node: wraps a robosuite/MuJoCo environment as a ROS2 node.

Responsibilities (HiL "Plant" side):
  * step the physics at a fixed control frequency (wall-clock throttled),
  * publish observations  -> /obs/image (sensor_msgs/Image), /obs/joint_state (JointState),
                             /obs/ee_pose (PoseStamped; the reactive layer's zero-delay anchor),
  * apply incoming low-level actions <- /cmd/action (JointState) from the reactive layer,
  * manage episodes: on task success OR horizon timeout, publish the outcome on /eval/success
    (std_msgs/Bool, True/False — the recorder needs BOTH for an honest success rate), reset the
    env, and announce the boundary on /episode/reset so downstream nodes clear their state,
  * cross-check the policy's action mode against ours and abort on a disagreement.

The node is deliberately unaware of the network boundary; latency is injected downstream by
evh_latency via topic remapping. If robosuite is unavailable it degrades to synthetic
observations (no physics, no episodes) so the graph and tests still run.

Sibling modules hold what is not HiL logic: `env_factory` (robosuite construction and its
version quirks), `messages` (the observation contract and ROS packing), `video` (mp4 recording).
"""
from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Empty

from evh_plant.env_factory import EnvSpec, build_env
from evh_plant.messages import PlantObservation
from evh_plant.video import VideoRecorder

# must match the controller's publisher QoS (latched) — the controller announces its mode once,
# at startup, and the plant is usually already running by then.
MODE_QOS = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


def _quat_to_axisangle(q: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] -> axis-angle (axis * angle). Canonical (shortest-path) hemisphere.

    Duplicated from evh_reactive.transforms rather than shared: the two packages are deployed on
    different machines and neither depends on the other. test_plant_node.py asserts they agree.
    """
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.zeros(3)
    q = q / n
    if q[3] < 0.0:
        q = -q
    s = np.linalg.norm(q[:3])
    if s < 1e-12:
        return np.zeros(3)
    return (q[:3] / s) * (2.0 * np.arctan2(s, q[3]))


def mode_mismatch_message(plant_absolute: bool, policy_absolute: bool) -> str | None:
    """Describe an action-mode disagreement, or None when the two modes agree.

    Split out from the node so the wording (the thing a confused user actually reads) is
    unit-testable without ROS.
    """
    if bool(plant_absolute) == bool(policy_absolute):
        return None
    misread = ('a world-frame pose read as a delta' if policy_absolute
               else 'a delta read as a world-frame pose')
    return (f'ACTION MODE MISMATCH: the plant/reactive layer is running with '
            f'absolute_actions={plant_absolute}, but the loaded policy emits '
            f'absolute={policy_absolute} actions — every command would be misread ({misread}), '
            f'and the run would still produce plausible-looking metrics. '
            f'Relaunch with absolute:={"true" if policy_absolute else "false"}.')


class PlantNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__('evh_plant', **kwargs)

        # --- parameters ---
        self.declare_parameter('env_name', 'Lift')          # robosuite task
        self.declare_parameter('robot', 'Panda')
        self.declare_parameter('control_hz', 20.0)          # observation publish rate
        self.declare_parameter('action_hz', 200.0)          # physics / action apply rate
        # comma-separated; first camera -> /obs/image, second (if any) -> /obs/image_wrist
        self.declare_parameter('camera', 'agentview,robot0_eye_in_hand')
        self.declare_parameter('image_size', 84)       # DP checkpoints are trained at 84x84
        self.declare_parameter('seed', 0)
        self.declare_parameter('max_episode_s', 20.0)  # episode horizon; timeout counts as failure
        # True -> OSC control_delta=False: /cmd/action is an absolute EE pose target
        # [pos(3), axis-angle(3), gripper], matching the abs-action DP checkpoints
        self.declare_parameter('absolute_actions', True)
        # abort instead of running on if the policy's mode disagrees with ours (see
        # _on_policy_mode); false only to deliberately run a mismatched config for debugging
        self.declare_parameter('strict_mode_check', True)
        self.declare_parameter('video_path', '')       # headless mp4; empty disables recording
        self.declare_parameter('video_duration', 0.0)  # seconds; 0 = record until shutdown

        self.control_hz = self.get_parameter('control_hz').value
        self.action_hz = self.get_parameter('action_hz').value
        self.img_size = int(self.get_parameter('image_size').value)
        self.cameras = [c.strip() for c in str(self.get_parameter('camera').value).split(',')
                        if c.strip()]
        self.camera = self.cameras[0]
        self.absolute_actions = bool(self.get_parameter('absolute_actions').value)
        self.strict_mode_check = bool(self.get_parameter('strict_mode_check').value)
        self.mode_mismatch = False

        self.recorder = VideoRecorder.from_duration(
            str(self.get_parameter('video_path').value), self.control_hz,
            float(self.get_parameter('video_duration').value))

        # --- publishers ---
        self.pub_image = self.create_publisher(Image, '/obs/image', 10)
        self.pub_wrist = (self.create_publisher(Image, '/obs/image_wrist', 10)
                          if len(self.cameras) > 1 else None)
        self.pub_joint = self.create_publisher(JointState, '/obs/joint_state', 10)
        self.pub_proprio = self.create_publisher(JointState, '/obs/proprio', 10)
        self.pub_ee_pose = self.create_publisher(PoseStamped, '/obs/ee_pose', 10)
        self.pub_success = self.create_publisher(Bool, '/eval/success', 10)
        self.pub_reset = self.create_publisher(Empty, '/episode/reset', 10)

        # --- subscribers ---
        self.sub_action = self.create_subscription(
            JointState, '/cmd/action', self._on_action, 10)
        self.sub_policy_mode = self.create_subscription(
            Bool, '/policy/absolute', self._on_policy_mode, MODE_QOS)

        self._env = None
        self._obs: dict | None = None
        self._last_action: np.ndarray | None = None
        self._action_dim = 7
        self._build_env()

        # separate callback groups: with a MultiThreadedExecutor the high-rate physics timer
        # can never starve the obs publisher (observed under load with a single thread).
        # _publish_observation only reads self._obs (replaced atomically) — no env calls.
        self._cb_physics = MutuallyExclusiveCallbackGroup()
        self._cb_io = MutuallyExclusiveCallbackGroup()
        self.create_timer(1.0 / self.control_hz, self._publish_observation,
                          callback_group=self._cb_io)
        self.create_timer(1.0 / self.action_hz, self._step_physics,
                          callback_group=self._cb_physics)

        self.get_logger().info(
            f'evh_plant up: env={self.get_parameter("env_name").value} '
            f'control={self.control_hz}Hz action={self.action_hz}Hz')

    # ------------------------------------------------------------------ env
    def _build_env(self) -> None:
        """Construct the sim, or fall back to synthetic observations if robosuite is missing."""
        spec = EnvSpec(
            env_name=self.get_parameter('env_name').value,
            robot=self.get_parameter('robot').value,
            cameras=tuple(self.cameras),
            image_size=self.img_size,
            action_hz=self.action_hz,
            max_episode_s=float(self.get_parameter('max_episode_s').value),
            seed=int(self.get_parameter('seed').value),
            absolute_actions=self.absolute_actions,
        )
        try:
            built = build_env(spec)
        except Exception as exc:
            self.get_logger().warn(
                f'robosuite unavailable ({type(exc).__name__}: {exc}); '
                'publishing synthetic observations (no physics, no episodes)')
            return

        for note in built.notes:
            self.get_logger().warn(f'evh_plant: {note}')
        self._env, self._obs, self._action_dim = built.env, built.obs, built.action_dim
        self.get_logger().info(
            f'evh_plant: robosuite env ready (action_dim={self._action_dim})')

    # ------------------------------------------------------------- callbacks
    def _on_action(self, msg: JointState) -> None:
        """Cache the latest low-level action from the reactive layer."""
        self._last_action = np.asarray(msg.position, dtype=np.float32)

    def _on_policy_mode(self, msg: Bool) -> None:
        """Cross-check the policy's action mode against ours — invariant: they MUST agree.

        The plant and the reactive layer take `absolute` from a launch arg; the policy derives it
        from the checkpoint. Nothing else compares the two, and a disagreement raises no error
        anywhere — it just produces garbage motion and a CSV full of plausible numbers. This is
        the one place both values meet, so the check lives here.

        The reactive layer is not checked separately on purpose: every launch file feeds it and
        the plant the SAME `absolute` arg, so it cannot disagree with us.
        """
        problem = mode_mismatch_message(self.absolute_actions, bool(msg.data))
        if problem is None:
            self.get_logger().info(
                f'evh_plant: action mode agrees with the policy (absolute={bool(msg.data)})')
            return
        self.mode_mismatch = True
        self.get_logger().error(problem)
        if not self.strict_mode_check:
            self.get_logger().error(
                'evh_plant: continuing anyway (strict_mode_check=false) — metrics from this run '
                'are NOT trustworthy')
            return
        self.get_logger().error('evh_plant: aborting; pass strict_mode_check:=false to override')
        rclpy.shutdown()   # ends executor.spin(); main() then exits non-zero

    # ---------------------------------------------------------------- actions
    def _current_action(self) -> np.ndarray:
        if self._last_action is None:
            return self._hold_action()
        action = self._last_action.reshape(-1)
        if action.size < self._action_dim:
            action = np.pad(action, (0, self._action_dim - action.size))
        return action[:self._action_dim].astype(np.float32)

    def _hold_action(self) -> np.ndarray:
        """Neutral action for 'no command yet' (episode start / just after reset).

        Delta mode: zeros = don't move. Absolute mode (OSC control_delta=False): zeros would be an
        ABSOLUTE target at the world origin (0,0,0), yanking the arm off the table until the
        reactive layer's first /cmd/action lands (~one inference latency) — so command the CURRENT
        EE pose instead, i.e. a genuine hold. Gripper stays 0 (neutral/open at episode start).
        """
        action = np.zeros(self._action_dim, dtype=np.float32)
        if self.absolute_actions and self._obs is not None and self._action_dim >= 6:
            try:
                ee_pos = np.asarray(self._obs['robot0_eef_pos'], dtype=np.float32)
                ee_quat = self._obs['robot0_eef_quat']
            except KeyError as exc:
                # a .get() default here would silently reintroduce the origin lurch this
                # function exists to prevent; never happens for a robosuite robot env
                raise RuntimeError(
                    f'absolute-mode hold needs {exc} in the obs dict — refusing to fall back '
                    'to a zero action, which OSC would read as the world origin') from None
            action[:3] = ee_pos
            action[3:6] = _quat_to_axisangle(ee_quat).astype(np.float32)
        return action

    # --------------------------------------------------------------- episodes
    def _step_physics(self) -> None:
        """Advance the simulator one action step; close the episode on success or timeout."""
        if self._env is None:
            return

        self._obs, _reward, done, _info = self._env.step(self._current_action())

        success = (bool(self._env._check_success())
                   if hasattr(self._env, '_check_success') else False)
        if success or done:
            self.pub_success.publish(Bool(data=success))
            self._reset_episode()

    def _reset_episode(self) -> None:
        """Reset the env and tell downstream nodes to drop episode-scoped state."""
        self._obs = self._env.reset()
        self._last_action = None   # don't carry the last command into the new episode
        self.pub_reset.publish(Empty())

    # ----------------------------------------------------------- observations
    def _publish_observation(self) -> None:
        now = self.get_clock().now().to_msg()
        want_wrist = self.pub_wrist is not None

        if self._env is None or self._obs is None:
            obs = PlantObservation.synthetic(self.img_size, want_wrist)
        else:
            # snapshot ONE obs reference and hand it over whole: _step_physics (200 Hz on a
            # separate thread) and _reset_episode both rebind self._obs, so reading it per-field
            # would pair an image with proprio/ee_pose from a different sim step.
            obs = PlantObservation.from_robosuite(self._obs, self.cameras, want_wrist)
            if obs is None:
                return   # obs dict carried no image; nothing to publish this tick

        self.pub_image.publish(obs.image_msg(now))
        wrist_msg = obs.wrist_msg(now)
        if self.pub_wrist is not None and wrist_msg is not None:
            self.pub_wrist.publish(wrist_msg)
        self.recorder.record(obs.frame)

        self.pub_joint.publish(obs.joint_state_msg(now))
        self.pub_proprio.publish(obs.proprio_msg(now))
        self.pub_ee_pose.publish(obs.ee_pose_msg(now))

    def destroy_node(self) -> None:
        self.recorder.close()
        if self._env is not None:
            self._env.close()
            self._env = None
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlantNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        aborted = node.mode_mismatch and node.strict_mode_check
        node.destroy_node()
        if rclpy.ok():        # _on_policy_mode may already have shut rclpy down
            rclpy.shutdown()
    if aborted:
        # non-zero so `ros2 launch` and the benchmark sweep surface it instead of recording
        # an empty row that looks like a merely unlucky condition
        raise SystemExit(1)


if __name__ == '__main__':
    main()
