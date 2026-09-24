"""Controller-only launch — run this ON THE JETSON for the true cross-machine HiL run.

The plant, reactive layer, and latency relays run on the x86 host (host.launch.py). With both
machines on the same ROS_DOMAIN_ID over Ethernet, DDS discovers the topics automatically and the
*physical* network replaces the software latency relay (or stacks with it).

Args:
  backend : pytorch | act | onnx | dp | dp_onnx
  weights : checkpoint path (dir/.ckpt) or .onnx path
  strategy, denoise_steps, executor : see hil.launch.py (with executor:=robot, pass the
                strategy to host.launch.py instead; this side only serves chunks)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from evh_bringup.launch_utils import typed


def generate_launch_description() -> LaunchDescription:
    backend = LaunchConfiguration('backend')
    weights = LaunchConfiguration('weights')
    strategy = LaunchConfiguration('strategy')
    denoise_steps = typed('denoise_steps', int)
    image_quality = typed('image_quality', int)
    executor = LaunchConfiguration('executor')

    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='onnx'),
        DeclareLaunchArgument('weights', default_value=''),
        DeclareLaunchArgument('strategy', default_value='rtc'),
        DeclareLaunchArgument('denoise_steps', default_value='16'),
        DeclareLaunchArgument('image_quality', default_value='0'),
        DeclareLaunchArgument('executor', default_value='policy'),
        Node(
            package='evh_controller', executable='controller_node', name='evh_controller',
            output='screen',
            parameters=[{'backend': backend, 'weights_path': weights, 'strategy': strategy,
                         'image_quality': image_quality,
                         'denoise_steps': denoise_steps, 'executor': executor}],
            remappings=[
                ('/obs/image', '/obs/image/delayed'),
                ('/obs/image_wrist', '/obs/image_wrist/delayed'),
                ('/obs/proprio', '/obs/proprio/delayed'),
                ('/policy/request', '/policy/request/delayed'),
            ]),
    ])
