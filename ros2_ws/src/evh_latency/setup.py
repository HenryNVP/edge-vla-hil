from setuptools import find_packages, setup

package_name = 'evh_latency'

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
    description='Programmable latency / jitter / packet-drop relay at the DDS boundary.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'latency_node = evh_latency.latency_node:main',
        ],
    },
)
