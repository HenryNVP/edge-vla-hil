# EdgeVLA-HiL

**Benchmarking latency-robust action chunking over a real edge-network boundary.**

A Hardware-in-the-Loop (HiL) testbed that physically decouples a robosuite/MuJoCo physics
simulation (Plant, x86 host) from a small diffusion/flow policy inference engine (Controller,
NVIDIA Jetson Orin Nano) across a real ROS2 / Gigabit Ethernet boundary. We reproduce SOTA
latency-robust chunk-execution strategies (synchronous, naive-async, Temporal Ensembling, BID,
**RTC**) and measure how each holds up under *physically-injected* latency, jitter, and packet
loss — the stochastic regime that inference-time methods like RTC explicitly do not model — then
show a high-rate local reactive layer recovers task success they lose. See `proposal.md`.

## Repository layout

```
ros2_ws/src/
├── evh_plant/        # robosuite (MuJoCo) sim wrapped as a ROS2 node  [PC host]
├── evh_controller/   # diffusion/flow policy + pluggable chunk-execution strategy  [Jetson]
│   ├── policy.py          # diffusion/flow backends (PyTorch + TensorRT)
│   └── chunk_executor.py  # synchronous|naive_async|temporal_ensemble|bid|rtc|network_aware
├── evh_reactive/     # high-rate operational-space impedance controller
├── evh_latency/      # programmable latency / jitter / drop / reorder at the DDS boundary
└── evh_bringup/      # launch files, configs, metrics recorder / benchmark
docker/               # Dockerfile.host (x86) and Dockerfile.jetson (arm64)
scripts/              # one-off tooling (ONNX export, TRT build)
```

The **chunk-execution strategy** (`chunk_executor.py`) is the experiment's core seam: Wedge A
reproduces the baselines; Wedge B drops in `network_aware` (RTC with a measured-RTT/jitter delay
forecast) without touching the ROS2 node.

## Node graph

```
  [evh_plant]  --/obs/image, /obs/joint_state-->  [evh_latency]  -->  [evh_controller: policy + strategy]
       ^                                                                      |
       |                                              /cmd/waypoint (OSC delta + gripper, policy rate)
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
| `/obs/ee_pose`       | `geometry_msgs/PoseStamped`   | plant → reactive (local, zero-delay) | sim rate |
| `/cmd/waypoint`      | `sensor_msgs/JointState` — `position` = 7-dim OSC_POSE action `[dpos, drot(axis-angle), gripper]` | controller → reactive | ~20 Hz |
| `/cmd/action`        | `sensor_msgs/JointState`      | reactive → plant       | ~200-500 Hz|
| `/eval/success`      | `std_msgs/Bool` (True/False)  | plant → benchmark      | episode end|
| `/episode/reset`     | `std_msgs/Empty`              | plant → controller, reactive | episode end|

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

## Docker (host)

```bash
docker build -f docker/Dockerfile.host -t edge-vla-hil:host .

# Interactive shell (no volume — uses pre-built workspace inside the image)
docker run -it --rm --network host -e ROS_DOMAIN_ID=42 edge-vla-hil:host

# Dev mode: mount the repo (overrides /ws; entrypoint rebuilds ros2_ws/install/ once)
docker run -it --rm --network host -e ROS_DOMAIN_ID=42 \
  -v ~/edge-vla-hil:/ws edge-vla-hil:host

# Full local HiL loop inside the container
docker run -it --rm --gpus all --network host -e ROS_DOMAIN_ID=42 \
  -v ~/edge-vla-hil:/ws edge-vla-hil:host \
  ros2 launch evh_bringup hil.launch.py latency_ms:=0.0 jitter_ms:=0.0

# Record headless mp4 (agentview @ 20 Hz, no display needed)
docker run -it --rm --gpus all --network host -e ROS_DOMAIN_ID=42 \
  -v ~/edge-vla-hil:/ws edge-vla-hil:host \
  ros2 launch evh_bringup hil.launch.py \
    weights:=lerobot/diffusion_pusht video:=/ws/outputs/hil.mp4 video_duration:=30.0

# Host-only (pair with controller.launch.py on the Jetson)
docker run -it --rm --network host -e ROS_DOMAIN_ID=42 \
  -v ~/edge-vla-hil:/ws edge-vla-hil:host \
  ros2 launch evh_bringup host.launch.py latency_ms:=0.0
```

If you mount the repo to `/ws`, the image's baked-in `ros2_ws/install/` is hidden — the
entrypoint runs `colcon build` automatically when `install/setup.bash` is missing.

## Jetson deployment

```bash
# On the Jetson (L4T R35.x / JetPack 5.1.x). Pull base image first to verify connectivity:
docker pull dustynv/ros:humble-desktop-pytorch-l4t-r35.4.1

docker build -f docker/Dockerfile.jetson -t edge-vla-hil:jetson .

# Controller only (pair with host.launch.py on the desktop; same ROS_DOMAIN_ID)
docker run -it --rm --network host --runtime nvidia \
  -e ROS_DOMAIN_ID=42 \
  -v ~/edge-vla-hil:/ws \
  edge-vla-hil:jetson \
  ros2 launch evh_bringup controller.launch.py backend:=pytorch strategy:=rtc
```

Build the TRT engine on-device from ONNX (`scripts/build_trt_engine.py`), then pass
`backend:=tensorrt weights:=/ws/checkpoints/policy.engine`.

## Benchmark sweep (Wedge A)

```bash
# sweep chunk-execution strategy x injected latency, reactive layer on/off;
# logs success rate, loop Hz, inference latency to CSV (the headline curves).
ros2 run evh_bringup benchmark --sweep latency \
  --strategies synchronous,temporal_ensemble,rtc \
  --values 0,25,50,100,200 --jitter_ms 0 --duration 60
```

## Status

Phase 2 in progress. In place: the HiL plumbing, the latency harness, the episode/metrics plane
(success AND timeout recorded, `/episode/reset` boundary signal), the full-action contract
(gripper included; reactive layer tracks absolute targets from zero-delay local EE state), and
**asynchronous chunk execution** — policy inference on a background worker, arrival-based
strategies, honestly measured request→arrival delay feeding RTC's forecast
(`/metrics/inference_ms`, `/metrics/delay_steps`). Still to come: a trained policy for the
robosuite task (Phase 2 gate), true guided inpainting for RTC + real BID (Phase 3), TensorRT
backend (Phase 4). See per-package docstrings.
