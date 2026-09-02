"""Controller-only launch — run this ON THE JETSON for the true cross-machine HiL run.

The plant, reactive layer, and latency relays run on the x86 host (host.launch.py). With both
machines on the same ROS_DOMAIN_ID over Ethernet, DDS discovers the topics automatically and the
*physical* network replaces the software latency relay (or stacks with it).

Args:
  backend : pytorch | act | onnx | dp | tensorrt
  weights : checkpoint path (dir/.ckpt), .onnx, or .engine path
  strategy, denoise_steps : see hil.launch.py
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

    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='tensorrt'),
        DeclareLaunchArgument('weights', default_value=''),
        DeclareLaunchArgument('strategy', default_value='rtc'),
        DeclareLaunchArgument('denoise_steps', default_value='16'),
        Node(
            package='evh_controller', executable='controller_node', name='evh_controller',
            output='screen',
            parameters=[{'backend': backend, 'weights_path': weights, 'strategy': strategy,
                         'denoise_steps': denoise_steps}],
            remappings=[
                ('/obs/image', '/obs/image/delayed'),
                ('/obs/image_wrist', '/obs/image_wrist/delayed'),
                ('/obs/proprio', '/obs/proprio/delayed'),
            ]),
    ])
