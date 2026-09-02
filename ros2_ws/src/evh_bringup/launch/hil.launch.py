"""Full HiL loop on one host (development / baseline).

  plant --> [latency relay x3: agentview, wrist, proprio] --> controller --> reactive --> plant

Latency relays sit on the observation path. The controller subscribes to the /delayed topics via
remap, so it is unaware of the injected delay. For the true cross-machine run, launch
controller.launch.py on the Jetson and host.launch.py (plant + reactive + relays) here.

Launch args:
  latency_ms, jitter_ms, drop_prob : network-condition knobs (passed to all relays)
  jitter_model : gaussian | uniform | lognormal. The first two are light-tailed, so a
                delay forecast based on a quantile cannot differ from one based on a
                max; lognormal supplies the heavy tail that separates them.
  backend     : pytorch | act | onnx | dp | tensorrt   (dp = diffusion_policy-repo checkpoint,
                the real one; onnx = ACT exported via scripts/export_onnx.py, the Jetson fast
                path -- see policy.py module docstring)
  weights     : checkpoint path (e.g. /ws/checkpoints/dp_lift_ph_image_cnn.ckpt), .onnx, or .engine
  strategy    : synchronous | naive_async | temporal_ensemble | bid | rtc | network_aware
  denoise_steps : dp backend DDIM steps (0 = checkpoint default DDPM-100)
  absolute    : true -> abs-action policy: plant OSC control_delta=False, reactive latches
                the waypoint as the target directly (default; matches the DP Lift checkpoint)
  strict_mode_check : true (default) -> the plant aborts if `absolute` disagrees with the mode
                the loaded checkpoint dictates; false to run a mismatched config anyway
  passthrough : true -> disable reactive layer (monolithic baseline ablation)
  image_size  : camera resolution (84 = DP training resolution)
  video       : headless mp4 path on the plant (empty = off); e.g. /ws/outputs/hil.mp4
  video_duration : seconds to record (0 = until shutdown)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from evh_bringup.launch_utils import typed


def generate_launch_description() -> LaunchDescription:
    latency_ms = typed('latency_ms', float)
    jitter_ms = typed('jitter_ms', float)
    drop_prob = typed('drop_prob', float)
    jitter_model = LaunchConfiguration('jitter_model')
    backend = LaunchConfiguration('backend')
    weights = LaunchConfiguration('weights')
    strategy = LaunchConfiguration('strategy')
    denoise_steps = typed('denoise_steps', int)
    absolute = typed('absolute', bool)
    strict_mode_check = typed('strict_mode_check', bool)
    passthrough = typed('passthrough', bool)
    image_size = typed('image_size', int)
    video = LaunchConfiguration('video')
    video_duration = typed('video_duration', float)

    args = [
        DeclareLaunchArgument('latency_ms', default_value='0.0'),
        DeclareLaunchArgument('jitter_ms', default_value='0.0'),
        DeclareLaunchArgument('drop_prob', default_value='0.0'),
        DeclareLaunchArgument('jitter_model', default_value='gaussian'),
        DeclareLaunchArgument('backend', default_value='pytorch'),
        DeclareLaunchArgument('weights', default_value=''),
        DeclareLaunchArgument('strategy', default_value='synchronous'),
        DeclareLaunchArgument('denoise_steps', default_value='16'),
        DeclareLaunchArgument('absolute', default_value='true'),
        DeclareLaunchArgument('strict_mode_check', default_value='true'),
        DeclareLaunchArgument('passthrough', default_value='false'),
        DeclareLaunchArgument('image_size', default_value='84'),
        DeclareLaunchArgument('video', default_value=''),
        DeclareLaunchArgument('video_duration', default_value='0.0'),
    ]

    plant = Node(
        package='evh_plant', executable='plant_node', name='evh_plant', output='screen',
        parameters=[{
            'image_size': image_size, 'absolute_actions': absolute,
            'strict_mode_check': strict_mode_check,
            'video_path': video, 'video_duration': video_duration,
        }])

    # one relay per observation topic the controller consumes
    def relay(name, topic, msg_type):
        return Node(
            package='evh_latency', executable='latency_node', name=name, output='screen',
            parameters=[{
                'input_topic': topic, 'output_topic': f'{topic}/delayed',
                'msg_type': msg_type,
                'latency_ms': latency_ms, 'jitter_ms': jitter_ms,
                'drop_prob': drop_prob, 'jitter_model': jitter_model,
            }])

    relay_img = relay('latency_image', '/obs/image', 'sensor_msgs/msg/Image')
    relay_wrist = relay('latency_wrist', '/obs/image_wrist', 'sensor_msgs/msg/Image')
    relay_proprio = relay('latency_proprio', '/obs/proprio', 'sensor_msgs/msg/JointState')

    # controller consumes the DELAYED observations
    controller = Node(
        package='evh_controller', executable='controller_node', name='evh_controller',
        output='screen',
        parameters=[{
            'backend': backend, 'weights_path': weights, 'strategy': strategy,
            'denoise_steps': denoise_steps,
        }],
        remappings=[
            ('/obs/image', '/obs/image/delayed'),
            ('/obs/image_wrist', '/obs/image_wrist/delayed'),
            ('/obs/proprio', '/obs/proprio/delayed'),
        ])

    # reactive layer reads LOCAL (zero-delay) EE state, tracks delayed waypoints
    reactive = Node(
        package='evh_reactive', executable='reactive_node', name='evh_reactive', output='screen',
        parameters=[{'passthrough': passthrough, 'absolute_waypoints': absolute}])

    return LaunchDescription(
        args + [plant, relay_img, relay_wrist, relay_proprio, controller, reactive])
