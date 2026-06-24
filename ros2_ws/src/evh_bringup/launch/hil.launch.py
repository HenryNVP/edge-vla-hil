"""Full HiL loop on one host (development / baseline).

  plant --> [latency relay x2] --> controller --> reactive --> plant

Latency relays sit on the observation path. The controller subscribes to the /delayed topics via
remap, so it is unaware of the injected delay. For the true cross-machine run, launch
controller.launch.py on the Jetson and this file (minus the controller node) on the host.

Launch args:
  latency_ms, jitter_ms, drop_prob : network-condition knobs (passed to both relays)
  backend     : pytorch | tensorrt   (controller inference path)
  weights     : ACT checkpoint dir or .engine path
  passthrough : true -> disable reactive layer (monolithic baseline ablation)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    latency_ms = LaunchConfiguration('latency_ms')
    jitter_ms = LaunchConfiguration('jitter_ms')
    drop_prob = LaunchConfiguration('drop_prob')
    backend = LaunchConfiguration('backend')
    weights = LaunchConfiguration('weights')
    passthrough = LaunchConfiguration('passthrough')

    args = [
        DeclareLaunchArgument('latency_ms', default_value='0.0'),
        DeclareLaunchArgument('jitter_ms', default_value='0.0'),
        DeclareLaunchArgument('drop_prob', default_value='0.0'),
        DeclareLaunchArgument('backend', default_value='pytorch'),
        DeclareLaunchArgument('weights', default_value=''),
        DeclareLaunchArgument('passthrough', default_value='false'),
    ]

    plant = Node(
        package='evh_plant', executable='plant_node', name='evh_plant', output='screen')

    # one relay per observation topic
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

    # controller consumes the DELAYED observations
    controller = Node(
        package='evh_controller', executable='controller_node', name='evh_controller',
        output='screen',
        parameters=[{'backend': backend, 'weights_path': weights}],
        remappings=[
            ('/obs/image', '/obs/image/delayed'),
            ('/obs/joint_state', '/obs/joint_state/delayed'),
        ])

    # reactive layer reads LOCAL (zero-delay) joint state, tracks delayed waypoints
    reactive = Node(
        package='evh_reactive', executable='reactive_node', name='evh_reactive', output='screen',
        parameters=[{'passthrough': passthrough}])

    return LaunchDescription(args + [plant, relay_img, relay_joint, controller, reactive])
