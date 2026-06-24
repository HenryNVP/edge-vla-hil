import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'evh_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='henry',
    maintainer_email='henrynguyen.vp@gmail.com',
    description='Launch files, configs, and the benchmark recorder for the EdgeVLA-HiL testbed.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'benchmark = evh_bringup.benchmark:main',
        ],
    },
)
