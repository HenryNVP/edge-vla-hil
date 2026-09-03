#!/usr/bin/env bash
# Source ROS + build the controller packages when the mounted install/ is missing or stale.
#
# Staleness matters more than it looks: `colcon build --symlink-install` symlinks the Python
# packages (so node edits are live) but *copies* everything under share/ -- launch files included.
# Mount the repo over /ws with an install/ tree from an earlier build and ROS happily launches
# months-old launch files against current node code, so remappings, params and launch args silently
# revert. That is how the Jetson controller ended up subscribed to the undelayed /obs/* topics,
# bypassing the latency relay, in a run whose logs looked entirely normal. Rebuilding on any source
# file newer than the last build costs a `find` on the common path and ~20s when it actually fires.
set -e
source /ros_source.sh
cd /ws/ros2_ws
if [[ ! -f install/setup.bash ]]; then
  echo "[entrypoint-jetson] install/ missing — running colcon build..."
  colcon build --symlink-install --packages-up-to evh_controller evh_bringup
elif [[ -n "$(find src -newer install/setup.bash -print -quit)" ]]; then
  echo "[entrypoint-jetson] src/ newer than install/ — rebuilding..."
  colcon build --symlink-install --packages-up-to evh_controller evh_bringup
fi
source install/setup.bash
exec "$@"
