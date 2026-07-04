"""Simulation Plant node: wraps a robosuite/MuJoCo environment as a ROS2 node.

Responsibilities (HiL "Plant" side):
  * step the physics at a fixed control frequency (wall-clock throttled),
  * publish observations  -> /obs/image (sensor_msgs/Image), /obs/joint_state (JointState),
  * apply incoming low-level actions <- /cmd/action (JointState) from the reactive layer,
  * report task success on /eval/success (std_msgs/Bool) for the benchmark recorder.

The node is deliberately unaware of the network boundary; latency is injected downstream by
evh_latency via topic remapping.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped  # noqa: F401  (reserved: direct-waypoint debug mode)
from std_msgs.msg import Bool


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


class PlantNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__('evh_plant', **kwargs)

        # --- parameters ---
        self.declare_parameter('env_name', 'Lift')          # robosuite task
        self.declare_parameter('robot', 'Panda')
        self.declare_parameter('control_hz', 20.0)          # observation publish rate
        self.declare_parameter('action_hz', 200.0)          # physics / action apply rate
        self.declare_parameter('camera', 'agentview')
        self.declare_parameter('image_size', 224)
        self.declare_parameter('seed', 0)
        self.declare_parameter('video_path', '')       # headless mp4; empty disables recording
        self.declare_parameter('video_duration', 0.0)  # seconds; 0 = record until shutdown

        self.control_hz = self.get_parameter('control_hz').value
        self.action_hz = self.get_parameter('action_hz').value
        self.img_size = int(self.get_parameter('image_size').value)
        self.camera = self.get_parameter('camera').value
        self._video_writer = None
        self._video_frames = 0
        video_duration = float(self.get_parameter('video_duration').value)
        self._video_max_frames = (
            int(video_duration * self.control_hz) if video_duration > 0.0 else 0)
        self._open_video_writer()

        # --- publishers ---
        self.pub_image = self.create_publisher(Image, '/obs/image', 10)
        self.pub_joint = self.create_publisher(JointState, '/obs/joint_state', 10)
        self.pub_success = self.create_publisher(Bool, '/eval/success', 10)

        # --- subscribers ---
        self.sub_action = self.create_subscription(
            JointState, '/cmd/action', self._on_action, 10)

        self._env = None
        self._obs: dict | None = None
        self._last_action: np.ndarray | None = None
        self._action_dim = 7
        self._build_env()

        self.create_timer(1.0 / self.control_hz, self._publish_observation)
        self.create_timer(1.0 / self.action_hz, self._step_physics)

        self.get_logger().info(
            f'evh_plant up: env={self.get_parameter("env_name").value} '
            f'control={self.control_hz}Hz action={self.action_hz}Hz')

    # ------------------------------------------------------------------ env
    def _build_env(self) -> None:
        """Construct the robosuite env."""
        import robosuite as suite

        seed = int(self.get_parameter('seed').value)
        np.random.seed(seed)

        kwargs = dict(
            env_name=self.get_parameter('env_name').value,
            robots=self.get_parameter('robot').value,   # robosuite >=1.5 uses `robots`
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=self.camera,
            camera_heights=self.img_size,
            camera_widths=self.img_size,
            control_freq=self.action_hz,
            seed=seed,
        )
        controller = _make_controller_config()
        if controller is not None:
            kwargs['controller_configs'] = controller
        else:
            self.get_logger().warn('evh_plant: using robosuite default controller config')

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

    def _current_action(self) -> np.ndarray:
        if self._last_action is None:
            return np.zeros(self._action_dim, dtype=np.float32)
        action = self._last_action.reshape(-1)
        if action.size < self._action_dim:
            action = np.pad(action, (0, self._action_dim - action.size))
        return action[:self._action_dim].astype(np.float32)

    def _step_physics(self) -> None:
        """Advance the simulator by one action step using the cached action."""
        if self._env is None:
            return

        self._obs, _reward, done, _info = self._env.step(self._current_action())

        if hasattr(self._env, '_check_success') and self._env._check_success():
            self.pub_success.publish(Bool(data=True))
            self._obs = self._env.reset()

        if done:
            self._obs = self._env.reset()

    def _publish_observation(self) -> None:
        now = self.get_clock().now().to_msg()
        camera_key = f'{self.camera}_image'

        if self._env is None or self._obs is None:
            frame = np.random.randint(0, 255, (self.img_size, self.img_size, 3), np.uint8)
            joint_pos = np.zeros(7, dtype=float)
            joint_vel = np.zeros(7, dtype=float)
        else:
            frame = self._obs.get(camera_key)
            if frame is None:
                frame = self._env.sim.render(
                    camera_name=self.camera, height=self.img_size, width=self.img_size)
            frame = np.flipud(np.asarray(frame)).astype(np.uint8)
            joint_pos = np.asarray(self._obs.get('robot0_joint_pos', np.zeros(7)), dtype=float)
            joint_vel = np.asarray(self._obs.get('robot0_joint_vel', np.zeros(7)), dtype=float)

        self.pub_image.publish(self._to_image_msg(frame, now))
        self._maybe_record_frame(frame)

        js = JointState()
        js.header.stamp = now
        js.position = list(joint_pos)
        js.velocity = list(joint_vel)
        self.pub_joint.publish(js)

    # -------------------------------------------------------------- helpers
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
