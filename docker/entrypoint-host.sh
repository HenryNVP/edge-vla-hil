#!/usr/bin/env bash
# Source ROS + build when the mounted install/ is missing or stale.
# See entrypoint-jetson.sh for why staleness is checked and not just existence: --symlink-install
# copies launch files rather than symlinking them, so a stale install/ silently runs old launch
# files (old remappings, old params) against current node code.
set -e
source /opt/ros/humble/setup.bash
cd /ws/ros2_ws
if [[ ! -f install/setup.bash ]]; then
  echo "[entrypoint-host] install/ missing — running colcon build (volume mount or first run)..."
  colcon build --symlink-install
elif [[ -n "$(find src -newer install/setup.bash -print -quit)" ]]; then
  echo "[entrypoint-host] src/ newer than install/ — rebuilding..."
  colcon build --symlink-install
fi
source install/setup.bash
exec "$@"
