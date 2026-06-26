#!/usr/bin/env bash
# Source ROS + build controller packages when install/ is missing (volume mount).
set -e
source /opt/ros/humble/setup.bash
cd /ws/ros2_ws
if [[ ! -f install/setup.bash ]]; then
  echo "[entrypoint-jetson] install/ missing — running colcon build..."
  colcon build --symlink-install --packages-up-to evh_controller evh_bringup
fi
source install/setup.bash
exec "$@"
