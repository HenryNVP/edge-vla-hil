#!/usr/bin/env bash
# Source ROS2 Humble — dustynv images use install/setup.bash; osrf apt images use setup.bash.
if [[ -f /opt/ros/humble/install/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/install/setup.bash
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
else
  echo "ERROR: ROS Humble setup.bash not found under /opt/ros/humble" >&2
  exit 1
fi
