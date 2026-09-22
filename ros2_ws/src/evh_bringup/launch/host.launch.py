"""Host-side launch for the cross-machine HiL run: plant + reactive + latency relays.

Pair with controller.launch.py running on the Jetson. Set a matching ROS_DOMAIN_ID on both
machines. The relays live here, on the robot side, for both directions: the observation topics
on their way out and /cmd/waypoint on its way in (see hil.launch.py for delay_obs / delay_act). Keep the software latency relays for *controlled* injection on top of the physical link,
or set latency_ms=0 to measure the raw network only.
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
    delay_obs = typed('delay_obs', bool)
    delay_act = typed('delay_act', bool)
    jitter_model = LaunchConfiguration('jitter_model')
    absolute = typed('absolute', bool)
    strict_mode_check = typed('strict_mode_check', bool)
    passthrough = typed('passthrough', bool)
    image_size = typed('image_size', int)
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
        DeclareLaunchArgument('jitter_model', default_value='gaussian'),
        DeclareLaunchArgument('absolute', default_value='true'),
        DeclareLaunchArgument('strict_mode_check', default_value='true'),
        DeclareLaunchArgument('passthrough', default_value='false'),
        DeclareLaunchArgument('image_size', default_value='84'),
        DeclareLaunchArgument('env_name', default_value='Lift'),
        DeclareLaunchArgument('max_episode_s', default_value='20.0'),
        DeclareLaunchArgument('video', default_value=''),
        DeclareLaunchArgument('video_duration', default_value='0.0'),
    ]

    plant = Node(
        package='evh_plant', executable='plant_node', name='evh_plant', output='screen',
        parameters=[{
            'image_size': image_size, 'absolute_actions': absolute,
            'env_name': env_name, 'max_episode_s': max_episode_s,
            'strict_mode_check': strict_mode_check,
            'video_path': video, 'video_duration': video_duration,
        }])

    def relay(name, topic, msg_type, enabled):
        return Node(
            package='evh_latency', executable='latency_node', name=name, output='screen',
            parameters=[{
                'input_topic': topic, 'output_topic': f'{topic}/delayed',
                'msg_type': msg_type, 'enabled': enabled,
                'latency_ms': latency_ms, 'jitter_ms': jitter_ms,
                'drop_prob': drop_prob, 'jitter_model': jitter_model,
            }])

    relay_img = relay('latency_image', '/obs/image', 'sensor_msgs/msg/Image', delay_obs)
    relay_wrist = relay('latency_wrist', '/obs/image_wrist', 'sensor_msgs/msg/Image', delay_obs)
    relay_proprio = relay(
        'latency_proprio', '/obs/proprio', 'sensor_msgs/msg/JointState', delay_obs)
    # the action path: always relayed, degraded only when delay_act (see the docstring)
    relay_waypoint = relay(
        'latency_waypoint', '/cmd/waypoint', 'sensor_msgs/msg/JointState', delay_act)

    reactive = Node(
        package='evh_reactive', executable='reactive_node', name='evh_reactive', output='screen',
        parameters=[{'passthrough': passthrough, 'absolute_waypoints': absolute}],
        remappings=[('/cmd/waypoint', '/cmd/waypoint/delayed')])

    return LaunchDescription(args + [plant, relay_img, relay_wrist, relay_proprio, relay_waypoint, reactive])
