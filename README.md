# EdgeVLA-HiL

**Closing the observation–action delay loop for edge-deployed VLA/ACT policies.**

A Hardware-in-the-Loop (HiL) testbed that physically decouples a robosuite/MuJoCo physics
simulation (Plant, x86 host) from an ACT policy inference engine (Controller, NVIDIA Jetson
Orin Nano) across a real ROS2 / Gigabit Ethernet boundary. We quantify how edge-induced
latency and jitter destabilize policy control, and show a high-rate local reactive layer
recovers task success.

## Repository layout

```
ros2_ws/src/
├── evh_plant/        # robosuite (MuJoCo) sim wrapped as a ROS2 node  [PC host]
├── evh_controller/   # ACT policy inference node (PyTorch + TensorRT)  [Jetson]
├── evh_reactive/     # high-rate operational-space impedance controller
├── evh_latency/      # programmable latency / jitter / drop injection at the DDS boundary
└── evh_bringup/      # launch files, configs, metrics recorder / benchmark
docker/               # Dockerfile.host (x86) and Dockerfile.jetson (arm64)
scripts/              # one-off tooling (ONNX export, TRT build)
```

## Node graph

```
  [evh_plant]  --/obs/image, /obs/joint_state-->  [evh_latency]  -->  [evh_controller (ACT)]
       ^                                                                      |
       |                                                              /cmd/waypoint (~10 Hz)
       |                                                                      v
       +-----------/cmd/action (~200-500 Hz)------------------------  [evh_reactive]
```

The `evh_latency` relay sits on the observation path (and optionally the command path) to emulate
edge network conditions. The `evh_reactive` controller runs co-located with the plant and tracks
the delayed waypoints using zero-delay local state.

## Topics (contract)

| Topic                | Type                          | From → To              | Rate       |
|----------------------|-------------------------------|------------------------|------------|
| `/obs/image`         | `sensor_msgs/Image`           | plant → controller     | sim rate   |
| `/obs/joint_state`   | `sensor_msgs/JointState`      | plant → controller     | sim rate   |
| `/cmd/waypoint`      | `geometry_msgs/PoseStamped`   | controller → reactive  | ~10 Hz     |
| `/cmd/action`        | `sensor_msgs/JointState`      | reactive → plant       | ~200-500 Hz|

Topics are remapped through `evh_latency` (e.g. `/obs/image` → `/obs/image/delayed`) via launch
arguments; nodes themselves are unaware of the injected delay.

## Quick start (PC host, simulation only)

```bash
# 1. system deps: ROS2 Humble + Python 3.10
# 2. python deps
pip install -r requirements-host.txt

# 3. build the workspace
cd ros2_ws
colcon build --symlink-install
source install/setup.bash

# 4. run the full HiL loop locally (controller in PyTorch fallback mode)
ros2 launch evh_bringup hil.launch.py latency_ms:=0.0 jitter_ms:=0.0
```

## Jetson deployment

```bash
docker build -f docker/Dockerfile.jetson -t edge-vla-hil:jetson .
# build the TRT engine from an exported ONNX policy (see scripts/build_trt_engine.py)
# then run only the controller node on the Jetson, plant + reactive on the host
ros2 launch evh_bringup controller.launch.py backend:=tensorrt
```

## Benchmark sweep

```bash
# sweeps injected latency; logs success rate, loop Hz, inference latency to CSV
ros2 run evh_bringup benchmark --sweep latency --values 0,25,50,100,200 --trials 20
```

## Status

Phase 1 skeleton. Nodes are runnable stubs with the ROS2 plumbing in place and `TODO` markers for
the physics, inference, and control logic. See per-package docstrings.
