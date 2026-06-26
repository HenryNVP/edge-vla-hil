#!/usr/bin/env bash
# Source ROS + ensure colcon install/ exists (volume mounts hide the image's pre-built tree).
set -e
source /opt/ros/humble/setup.bash
cd /ws/ros2_ws
if [[ ! -f install/setup.bash ]]; then
  echo "[entrypoint-host] install/ missing — running colcon build (volume mount or first run)..."
  colcon build --symlink-install
fi
source install/setup.bash
exec "$@"
