"""Host-side launch for the cross-machine HiL run: plant + reactive + latency relays.

Pair with controller.launch.py running on the Jetson. Set a matching ROS_DOMAIN_ID on both
machines. Keep the software latency relays for *controlled* injection on top of the physical link,
or set latency_ms=0 to measure the raw network only.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    latency_ms = LaunchConfiguration('latency_ms')
    jitter_ms = LaunchConfiguration('jitter_ms')
    drop_prob = LaunchConfiguration('drop_prob')
    absolute = LaunchConfiguration('absolute')
    passthrough = LaunchConfiguration('passthrough')
    image_size = LaunchConfiguration('image_size')
    video = LaunchConfiguration('video')
    video_duration = LaunchConfiguration('video_duration')

    args = [
        DeclareLaunchArgument('latency_ms', default_value='0.0'),
        DeclareLaunchArgument('jitter_ms', default_value='0.0'),
        DeclareLaunchArgument('drop_prob', default_value='0.0'),
        DeclareLaunchArgument('absolute', default_value='true'),
        DeclareLaunchArgument('passthrough', default_value='false'),
        DeclareLaunchArgument('image_size', default_value='84'),
        DeclareLaunchArgument('video', default_value=''),
        DeclareLaunchArgument('video_duration', default_value='0.0'),
    ]

    plant = Node(
        package='evh_plant', executable='plant_node', name='evh_plant', output='screen',
        parameters=[{
            'image_size': image_size, 'absolute_actions': absolute,
            'video_path': video, 'video_duration': video_duration,
        }])

    def relay(name, topic, msg_type):
        return Node(
            package='evh_latency', executable='latency_node', name=name, output='screen',
            parameters=[{
                'input_topic': topic, 'output_topic': f'{topic}/delayed',
                'msg_type': msg_type,
                'latency_ms': latency_ms, 'jitter_ms': jitter_ms, 'drop_prob': drop_prob,
            }])

    relay_img = relay('latency_image', '/obs/image', 'sensor_msgs/msg/Image')
    relay_wrist = relay('latency_wrist', '/obs/image_wrist', 'sensor_msgs/msg/Image')
    relay_proprio = relay('latency_proprio', '/obs/proprio', 'sensor_msgs/msg/JointState')

    reactive = Node(
        package='evh_reactive', executable='reactive_node', name='evh_reactive', output='screen',
        parameters=[{'passthrough': passthrough, 'absolute_waypoints': absolute}])

    return LaunchDescription(args + [plant, relay_img, relay_wrist, relay_proprio, reactive])
