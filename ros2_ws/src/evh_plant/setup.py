from setuptools import find_packages, setup

package_name = 'evh_plant'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='henry',
    maintainer_email='henrynguyen.vp@gmail.com',
    description='robosuite/MuJoCo simulation plant wrapped as a ROS2 node (HiL Plant side).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'plant_node = evh_plant.plant_node:main',
        ],
    },
)
