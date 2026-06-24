"""Simulation Plant node: wraps a robosuite/MuJoCo environment as a ROS2 node.

Responsibilities (HiL "Plant" side):
  * step the physics at a fixed control frequency (wall-clock throttled),
  * publish observations  -> /obs/image (sensor_msgs/Image), /obs/joint_state (JointState),
  * apply incoming low-level actions <- /cmd/action (JointState) from the reactive layer,
  * report task success on /eval/success (std_msgs/Bool) for the benchmark recorder.

The node is deliberately unaware of the network boundary; latency is injected downstream by
evh_latency via topic remapping.

Phase 1 status: ROS2 plumbing complete; the robosuite env calls are TODO stubs so the graph
runs end-to-end with synthetic observations before the simulator is wired in.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped  # noqa: F401  (reserved: direct-waypoint debug mode)
from std_msgs.msg import Bool


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

        self.control_hz = self.get_parameter('control_hz').value
        self.action_hz = self.get_parameter('action_hz').value
        self.img_size = int(self.get_parameter('image_size').value)

        # --- publishers ---
        self.pub_image = self.create_publisher(Image, '/obs/image', 10)
        self.pub_joint = self.create_publisher(JointState, '/obs/joint_state', 10)
        self.pub_success = self.create_publisher(Bool, '/eval/success', 10)

        # --- subscribers ---
        self.sub_action = self.create_subscription(
            JointState, '/cmd/action', self._on_action, 10)

        # --- sim handle (TODO: construct robosuite env) ---
        self._env = None
        self._last_action: np.ndarray | None = None
        self._build_env()

        # observation timer (wall-clock throttled to control_hz)
        self.create_timer(1.0 / self.control_hz, self._publish_observation)
        # physics-step timer (apply most recent action at action_hz)
        self.create_timer(1.0 / self.action_hz, self._step_physics)

        self.get_logger().info(
            f'evh_plant up: env={self.get_parameter("env_name").value} '
            f'control={self.control_hz}Hz action={self.action_hz}Hz')

    # ------------------------------------------------------------------ env
    def _build_env(self) -> None:
        """TODO: construct the robosuite env.

        Example:
            import robosuite as suite
            self._env = suite.make(
                env_name=self.get_parameter('env_name').value,
                robots=self.get_parameter('robot').value,
                has_renderer=False, has_offscreen_renderer=True,
                use_camera_obs=True, camera_names=self.get_parameter('camera').value,
                camera_heights=self.img_size, camera_widths=self.img_size,
                control_freq=self.action_hz)
            self._obs = self._env.reset()
        """
        self.get_logger().warn('evh_plant: robosuite env is a STUB — publishing synthetic obs.')

    # ------------------------------------------------------------- callbacks
    def _on_action(self, msg: JointState) -> None:
        """Cache the latest low-level action from the reactive layer."""
        self._last_action = np.asarray(msg.position, dtype=np.float32)

    def _step_physics(self) -> None:
        """Advance the simulator by one action step using the cached action."""
        if self._env is None:
            return  # stub
        # TODO: self._obs, _, done, _ = self._env.step(self._last_action or zeros)
        # TODO: on success/done -> publish Bool(True) on /eval/success and reset.

    def _publish_observation(self) -> None:
        now = self.get_clock().now().to_msg()

        # ---- image ----
        if self._env is None:
            frame = np.random.randint(0, 255, (self.img_size, self.img_size, 3), np.uint8)
        else:
            frame = None  # TODO: self._obs[f'{camera}_image']
        self.pub_image.publish(self._to_image_msg(frame, now))

        # ---- joint state ----
        js = JointState()
        js.header.stamp = now
        if self._env is None:
            js.position = list(np.zeros(7, dtype=float))
            js.velocity = list(np.zeros(7, dtype=float))
        else:
            pass  # TODO: fill from self._obs['robot0_joint_pos'] etc.
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
