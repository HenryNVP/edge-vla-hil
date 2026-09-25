"""Full HiL loop on one host (development / baseline).

  plant --> [relay x3: agentview, wrist, proprio] --> controller --> [relay: waypoint]
        --> reactive --> plant

Relays sit on every networked link: the three observation topics and the waypoint (action) path.
Consumers subscribe to the /delayed topics via remap, so they are unaware of the relays. The
reactive->plant link (/cmd/action) and /obs/ee_pose are local and never relayed (invariant 8). For the true cross-machine run, launch
controller.launch.py on the Jetson and host.launch.py (plant + reactive + relays) here.

Launch args:
  latency_ms, jitter_ms, drop_prob : network-condition knobs (passed to all relays)
  executor    : policy (default) -> the chunk executor runs in the controller and streams one
                waypoint per tick over the action path; robot -> it runs next to the robot
                (executor_node), whole chunks cross the action path and requests the uplink.
  delay_obs, delay_act : which path the condition applies to. The other path's relays still run,
                in pass-through, so every placement has the same graph and relay floor.
                Defaults (true, false) reproduce the observation-only runs made before the
                action relay existed.
  loss_model  : iid | gilbert. gilbert drops whole outages of `burst_ms` on average at the same
                average rate drop_prob, one shared state across every relay (see
                evh_latency/channel.py)
  jitter_model : gaussian | uniform | lognormal. The first two are light-tailed, so a
                delay forecast based on a quantile cannot differ from one based on a
                max; lognormal supplies the heavy tail that separates them.
  backend     : pytorch | act | onnx | dp | dp_onnx   (dp = diffusion_policy-repo checkpoint,
                the real one; onnx = ACT exported via scripts/export_onnx.py, the Jetson fast
                path -- see policy.py module docstring)
  weights     : checkpoint path (e.g. /ws/checkpoints/dp_lift_ph_image_cnn.ckpt) or .onnx
  strategy    : synchronous | naive_async | temporal_ensemble | bid | rtc | network_aware
  denoise_steps : dp backend DDIM steps (0 = checkpoint default DDPM-100)
  absolute    : true -> abs-action policy: plant OSC control_delta=False, reactive latches
                the waypoint as the target directly (default; matches the DP Lift checkpoint)
  strict_mode_check : true (default) -> the plant aborts if `absolute` disagrees with the mode
                the loaded checkpoint dictates; false to run a mismatched config anyway
  passthrough : true -> disable reactive layer (monolithic baseline ablation)
  image_size  : camera resolution (84 = DP training resolution)
  env_name    : robosuite task, e.g. Lift or NutAssemblySquare (robomimic's Square)
  max_episode_s : episode horizon in seconds; timeout is a recorded failure (Square: 20 = the
                400 control steps its DP checkpoint was evaluated with)
  video       : headless mp4 path on the plant (empty = off); e.g. /ws/outputs/hil.mp4
  video_duration : seconds to record (0 = until shutdown)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from evh_bringup.launch_utils import degrade_when, image_msg_type, typed


def generate_launch_description() -> LaunchDescription:
    latency_ms = typed('latency_ms', float)
    jitter_ms = typed('jitter_ms', float)
    jitter_burst_ms = typed('jitter_burst_ms', float)
    jitter_bad_frac = typed('jitter_bad_frac', float)
    drop_prob = typed('drop_prob', float)
    delay_obs = typed('delay_obs', bool)
    delay_act = typed('delay_act', bool)
    jitter_model = LaunchConfiguration('jitter_model')
    loss_model = LaunchConfiguration('loss_model')
    burst_ms = typed('burst_ms', float)
    backend = LaunchConfiguration('backend')
    weights = LaunchConfiguration('weights')
    strategy = LaunchConfiguration('strategy')
    denoise_steps = typed('denoise_steps', int)
    absolute = typed('absolute', bool)
    strict_mode_check = typed('strict_mode_check', bool)
    passthrough = typed('passthrough', bool)
    executor = LaunchConfiguration('executor')
    image_size = typed('image_size', int)
    image_quality = typed('image_quality', int)
    env_name = LaunchConfiguration('env_name')
    max_episode_s = typed('max_episode_s', float)
    video = LaunchConfiguration('video')
    video_duration = typed('video_duration', float)

    args = [
        DeclareLaunchArgument('latency_ms', default_value='0.0'),
        DeclareLaunchArgument('jitter_ms', default_value='0.0'),
        DeclareLaunchArgument('drop_prob', default_value='0.0'),
        DeclareLaunchArgument('delay_obs', default_value='true'),
        DeclareLaunchArgument('delay_act', default_value='false'),
        DeclareLaunchArgument('executor', default_value='policy'),
        DeclareLaunchArgument('jitter_model', default_value='gaussian'),
        DeclareLaunchArgument('jitter_burst_ms', default_value='150.0'),
        DeclareLaunchArgument('jitter_bad_frac', default_value='0.05'),
        DeclareLaunchArgument('loss_model', default_value='iid'),
        DeclareLaunchArgument('burst_ms', default_value='100.0'),
        DeclareLaunchArgument('backend', default_value='pytorch'),
        DeclareLaunchArgument('weights', default_value=''),
        DeclareLaunchArgument('strategy', default_value='synchronous'),
        DeclareLaunchArgument('denoise_steps', default_value='16'),
        DeclareLaunchArgument('absolute', default_value='true'),
        DeclareLaunchArgument('strict_mode_check', default_value='true'),
        DeclareLaunchArgument('passthrough', default_value='false'),
        DeclareLaunchArgument('image_size', default_value='84'),
        DeclareLaunchArgument('image_quality', default_value='90'),
        DeclareLaunchArgument('env_name', default_value='Lift'),
        DeclareLaunchArgument('max_episode_s', default_value='20.0'),
        DeclareLaunchArgument('video', default_value=''),
        DeclareLaunchArgument('video_duration', default_value='0.0'),
    ]

    plant = Node(
        package='evh_plant', executable='plant_node', name='evh_plant', output='screen',
        parameters=[{
            'image_size': image_size, 'absolute_actions': absolute,
            'image_quality': image_quality,
            'env_name': env_name, 'max_episode_s': max_episode_s,
            'strict_mode_check': strict_mode_check,
            'video_path': video, 'video_duration': video_duration,
        }])

    # one relay per observation topic the controller consumes
    def relay(name, topic, msg_type, enabled):
        return Node(
            package='evh_latency', executable='latency_node', name=name, output='screen',
            parameters=[{
                'input_topic': topic, 'output_topic': f'{topic}/delayed',
                'msg_type': msg_type, 'enabled': enabled,
                'latency_ms': latency_ms, 'jitter_ms': jitter_ms,
                'drop_prob': drop_prob, 'jitter_model': jitter_model,
                'loss_model': loss_model, 'burst_ms': burst_ms,
                'jitter_burst_ms': jitter_burst_ms, 'jitter_bad_frac': jitter_bad_frac,
            }])

    relay_img = relay('latency_image', '/obs/image', image_msg_type(), delay_obs)
    relay_wrist = relay('latency_wrist', '/obs/image_wrist', image_msg_type(), delay_obs)
    relay_proprio = relay(
        'latency_proprio', '/obs/proprio', 'sensor_msgs/msg/JointState', delay_obs)
    # The action path. Which link carries the actions depends on the executor's placement:
    # streamed waypoints (executor:=policy) or whole chunks (executor:=robot). All three relays
    # are always in the graph so the hop count never changes; each degrades only its own link.
    relay_waypoint = relay('latency_waypoint', '/cmd/waypoint', 'sensor_msgs/msg/JointState',
                           degrade_when('delay_act', 'policy'))
    relay_chunk = relay('latency_chunk', '/cmd/chunk', 'sensor_msgs/msg/JointState', delay_act)
    # the chunk request is uplink traffic, so it rides with the observations
    relay_request = relay(
        'latency_request', '/policy/request', 'sensor_msgs/msg/JointState', delay_obs)

    robot_side = IfCondition(PythonExpression(["'", executor, "' == 'robot'"]))
    executor_node = Node(
        package='evh_controller', executable='executor_node', name='evh_executor',
        output='screen', condition=robot_side,
        parameters=[{'strategy': strategy}],
        remappings=[('/cmd/chunk', '/cmd/chunk/delayed')])

    # controller consumes the DELAYED observations
    controller = Node(
        package='evh_controller', executable='controller_node', name='evh_controller',
        output='screen',
        parameters=[{
            'backend': backend, 'weights_path': weights, 'strategy': strategy,
            'denoise_steps': denoise_steps, 'executor': executor,
            'image_quality': image_quality,
        }],
        remappings=[
            ('/obs/image', '/obs/image/delayed'),
            ('/obs/image_wrist', '/obs/image_wrist/delayed'),
            ('/obs/proprio', '/obs/proprio/delayed'),
            ('/policy/request', '/policy/request/delayed'),
        ])

    # reactive layer reads LOCAL (zero-delay) EE state, tracks delayed waypoints
    reactive = Node(
        package='evh_reactive', executable='reactive_node', name='evh_reactive', output='screen',
        parameters=[{'passthrough': passthrough, 'absolute_waypoints': absolute}],
        remappings=[('/cmd/waypoint', '/cmd/waypoint/delayed')])

    return LaunchDescription(
        args + [plant, relay_img, relay_wrist, relay_proprio, relay_waypoint, relay_chunk,
                relay_request, controller, executor_node, reactive])
