from setuptools import find_packages, setup

package_name = 'evh_reactive'

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
    description='High-rate operational-space impedance controller (the reactive "spinal cord").',
    license='MIT',
    entry_points={
        'console_scripts': [
            'reactive_node = evh_reactive.reactive_node:main',
        ],
    },
)
