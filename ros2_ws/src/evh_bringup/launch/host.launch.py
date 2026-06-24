"""Host-side launch for the cross-machine HiL run: plant + reactive + (optional) latency relays.

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
    passthrough = LaunchConfiguration('passthrough')

    args = [
        DeclareLaunchArgument('latency_ms', default_value='0.0'),
        DeclareLaunchArgument('jitter_ms', default_value='0.0'),
        DeclareLaunchArgument('drop_prob', default_value='0.0'),
        DeclareLaunchArgument('passthrough', default_value='false'),
    ]

    plant = Node(
        package='evh_plant', executable='plant_node', name='evh_plant', output='screen')
    relay_img = Node(
        package='evh_latency', executable='latency_node', name='latency_image', output='screen',
        parameters=[{
            'input_topic': '/obs/image', 'output_topic': '/obs/image/delayed',
            'msg_type': 'sensor_msgs/msg/Image',
            'latency_ms': latency_ms, 'jitter_ms': jitter_ms, 'drop_prob': drop_prob,
        }])
    relay_joint = Node(
        package='evh_latency', executable='latency_node', name='latency_joint', output='screen',
        parameters=[{
            'input_topic': '/obs/joint_state', 'output_topic': '/obs/joint_state/delayed',
            'msg_type': 'sensor_msgs/msg/JointState',
            'latency_ms': latency_ms, 'jitter_ms': jitter_ms, 'drop_prob': drop_prob,
        }])
    reactive = Node(
        package='evh_reactive', executable='reactive_node', name='evh_reactive', output='screen',
        parameters=[{'passthrough': passthrough}])

    return LaunchDescription(args + [plant, relay_img, relay_joint, reactive])
