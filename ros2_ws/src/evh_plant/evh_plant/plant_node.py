"""Simulation Plant node: wraps a robosuite/MuJoCo environment as a ROS2 node.

Responsibilities (HiL "Plant" side):
  * step the physics at a fixed control frequency (wall-clock throttled),
  * publish observations  -> /obs/image (sensor_msgs/Image), /obs/joint_state (JointState),
                             /obs/ee_pose (PoseStamped; the reactive layer's zero-delay anchor),
  * apply incoming low-level actions <- /cmd/action (JointState) from the reactive layer,
  * manage episodes: on task success OR horizon timeout, publish the outcome on /eval/success
    (std_msgs/Bool, True/False — the recorder needs BOTH for an honest success rate), reset the
    env, and announce the boundary on /episode/reset so downstream nodes clear their state.

The node is deliberately unaware of the network boundary; latency is injected downstream by
evh_latency via topic remapping. If robosuite is unavailable it degrades to synthetic
observations (no physics, no episodes) so the graph and tests still run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Empty

# must match the controller's publisher QoS (latched) — the controller announces its mode once,
# at startup, and the plant is usually already running by then.
MODE_QOS = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


def _make_controller_config():
    """OSC_POSE-style controller config across robosuite 1.4 / 1.5 APIs."""
    try:
        from robosuite.controllers import load_controller_config
        return load_controller_config(default_controller='OSC_POSE')
    except Exception:
        pass
    try:
        from robosuite.controllers import load_composite_controller_config
        return load_composite_controller_config(controller='BASIC')
    except Exception:
        return None


def _set_control_delta(config: dict, value: bool) -> None:
    """Set OSC control_delta across robosuite config shapes (1.4 flat / 1.5 composite)."""
    if 'control_delta' in config:
        config['control_delta'] = value
        return
    for part_cfg in config.get('body_parts', {}).values():
        if isinstance(part_cfg, dict) and part_cfg.get('type', '').startswith('OSC'):
            part_cfg['control_delta'] = value


def _quat_to_axisangle(q: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] -> axis-angle (axis * angle). Canonical (shortest-path) hemisphere."""
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
        self._video_writer = None
        self._video_frames = 0
        video_duration = float(self.get_parameter('video_duration').value)
        self._video_max_frames = (
            int(video_duration * self.control_hz) if video_duration > 0.0 else 0)
        self._open_video_writer()

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
        try:
            self._build_env()
        except Exception as exc:
            self.get_logger().warn(
                f'robosuite unavailable ({type(exc).__name__}: {exc}); '
                'publishing synthetic observations (no physics, no episodes)')

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
        """Construct the robosuite env."""
        import robosuite as suite

        seed = int(self.get_parameter('seed').value)
        np.random.seed(seed)

        kwargs = {
            'env_name': self.get_parameter('env_name').value,
            'robots': self.get_parameter('robot').value,   # robosuite >=1.5 uses `robots`
            'has_renderer': False,
            'has_offscreen_renderer': True,
            'use_camera_obs': True,
            'camera_names': self.cameras,
            'camera_heights': [self.img_size] * len(self.cameras),
            'camera_widths': [self.img_size] * len(self.cameras),
            'control_freq': self.action_hz,
            # horizon is in control steps; hitting it = episode timeout = recorded failure
            'horizon': int(float(self.get_parameter('max_episode_s').value) * self.action_hz),
            'reward_shaping': False,
            'seed': seed,
        }
        controller = _make_controller_config()
        if controller is not None:
            if self.absolute_actions:
                _set_control_delta(controller, False)
            kwargs['controller_configs'] = controller
        elif self.absolute_actions:
            raise RuntimeError('absolute_actions needs an OSC controller config')
        else:
            self.get_logger().warn('evh_plant: using robosuite default controller config')

        try:
            self._env = suite.make(**kwargs)
        except TypeError:            # robosuite 1.4 has no seed kwarg (np.random covers it)
            kwargs.pop('seed', None)
            self._env = suite.make(**kwargs)
        self._obs = self._env.reset()
        low, _high = self._env.action_spec
        self._action_dim = len(low)
        self.get_logger().info(
            f'evh_plant: robosuite env ready (action_dim={self._action_dim})')

    def _open_video_writer(self) -> None:
        video_path = str(self.get_parameter('video_path').value).strip()
        if not video_path:
            return
        import imageio

        path = Path(video_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._video_writer = imageio.get_writer(str(path), fps=int(self.control_hz))
        limit = (
            f', max {self._video_max_frames} frames'
            if self._video_max_frames else ', until shutdown')
        self.get_logger().info(
            f'evh_plant: recording {path} @ {int(self.control_hz)} Hz{limit}')

    def _close_video_writer(self) -> None:
        if self._video_writer is None:
            return
        self._video_writer.close()
        self.get_logger().info(
            f'evh_plant: wrote {self._video_frames} frames to '
            f'{self.get_parameter("video_path").value}')
        self._video_writer = None

    def _maybe_record_frame(self, frame: np.ndarray) -> None:
        if self._video_writer is None:
            return
        self._video_writer.append_data(frame)
        self._video_frames += 1
        if self._video_max_frames and self._video_frames >= self._video_max_frames:
            self._close_video_writer()

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

    def _publish_observation(self) -> None:
        now = self.get_clock().now().to_msg()

        if self._env is None or self._obs is None:
            frame = np.random.randint(0, 255, (self.img_size, self.img_size, 3), np.uint8)
            wrist = frame
            joint_pos = np.zeros(7, dtype=float)
            joint_vel = np.zeros(7, dtype=float)
            ee_pos = np.zeros(3, dtype=float)
            ee_quat = np.array([0.0, 0.0, 0.0, 1.0])
            gripper_qpos = np.zeros(2, dtype=float)
        else:
            # snapshot ONE obs reference and read every field from it: _step_physics (200 Hz on a
            # separate thread) and _reset_episode both rebind self._obs, so re-reading self._obs
            # per-field would pair an image with proprio/ee_pose from a different sim step.
            obs = self._obs
            frame = self._upright(obs.get(f'{self.camera}_image'))
            wrist = (self._upright(obs.get(f'{self.cameras[1]}_image'))
                     if self.pub_wrist is not None else None)
            if frame is None:
                return   # obs dict without images (shouldn't happen with use_camera_obs)
            joint_pos = np.asarray(obs.get('robot0_joint_pos', np.zeros(7)), dtype=float)
            joint_vel = np.asarray(obs.get('robot0_joint_vel', np.zeros(7)), dtype=float)
            ee_pos = np.asarray(obs.get('robot0_eef_pos', np.zeros(3)), dtype=float)
            ee_quat = np.asarray(   # robosuite convention: [x, y, z, w]
                obs.get('robot0_eef_quat', [0.0, 0.0, 0.0, 1.0]), dtype=float)
            gripper_qpos = np.asarray(
                obs.get('robot0_gripper_qpos', np.zeros(2)), dtype=float)

        self.pub_image.publish(self._to_image_msg(frame, now))
        if self.pub_wrist is not None and wrist is not None:
            self.pub_wrist.publish(self._to_image_msg(wrist, now))
        self._maybe_record_frame(frame)

        js = JointState()
        js.header.stamp = now
        js.position = list(joint_pos)
        js.velocity = list(joint_vel)
        self.pub_joint.publish(js)

        prop = JointState()   # [eef_pos(3), eef_quat(4, xyzw), gripper_qpos(2)]
        prop.header.stamp = now
        prop.position = list(ee_pos) + list(ee_quat) + list(gripper_qpos)
        self.pub_proprio.publish(prop)

        ee = PoseStamped()
        ee.header.stamp = now
        ee.header.frame_id = 'base'
        ee.pose.position.x, ee.pose.position.y, ee.pose.position.z = map(float, ee_pos)
        (ee.pose.orientation.x, ee.pose.orientation.y,
         ee.pose.orientation.z, ee.pose.orientation.w) = map(float, ee_quat)
        self.pub_ee_pose.publish(ee)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _upright(frame) -> np.ndarray | None:
        """robosuite camera obs are bottom-up; flip to upright uint8."""
        if frame is None:
            return None
        return np.flipud(np.asarray(frame)).astype(np.uint8).copy()

    def _to_image_msg(self, frame: np.ndarray, stamp) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.height, msg.width = frame.shape[0], frame.shape[1]
        msg.encoding = 'rgb8'
        msg.step = msg.width * 3
        msg.data = frame.tobytes()
        return msg

    def destroy_node(self) -> None:
        self._close_video_writer()
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
