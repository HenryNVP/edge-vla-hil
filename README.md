# EdgeVLA-HiL

**Benchmarking latency-robust action chunking over a real edge-network boundary.**

A testbed that runs the policy on the deployment hardware and puts a real network inside the
control loop: a robosuite/MuJoCo physics simulation (Plant, x86 host) is decoupled from a small
diffusion/flow policy inference engine (Controller, NVIDIA Jetson Orin Nano) across a real ROS2 /
Gigabit Ethernet boundary. We reproduce SOTA
latency-robust chunk-execution strategies (synchronous, naive-async, Temporal Ensembling, BID,
**RTC**) and measure how each holds up under *physically-injected* latency, jitter, and packet
loss — the stochastic regime that inference-time methods like RTC explicitly do not model — then
show a high-rate local reactive layer recovers task success they lose. See `proposal.md`.

**What is real here, and what is not.** The controller is real target hardware running the real
deployment artifact, and the observation/action path crosses a real wire. In the usual V-model
taxonomy that is **processor-in-the-loop**, plus network-in-the-loop — not hardware-in-the-loop,
which would need the *plant* to be a physical robot or a real-time plant emulator rather than
MuJoCo in Python. `HiL` stays in the project name; the claim made in the results is the narrower,
checkable one. The loop is likewise soft real-time by construction: `_step_physics` advances the
simulator once per ROS wall-clock timer fire, with no catch-up and no deadline accounting, so
simulated time is *defined* by when the timer happens to fire. That is only acceptable because the
timing floor is small next to the effect under study — see [Timing floor](#timing-floor), where it
is measured rather than assumed.

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
| `/obs/image`         | `sensor_msgs/Image` (agentview) | plant → controller   | sim rate   |
| `/obs/image_wrist`   | `sensor_msgs/Image` (eye-in-hand) | plant → controller | sim rate   |
| `/obs/proprio`       | `sensor_msgs/JointState` — `position` = `[eef_pos(3), eef_quat(4, xyzw), gripper_qpos(2)]` | plant → controller | sim rate |
| `/obs/joint_state`   | `sensor_msgs/JointState`      | plant → (debug)        | sim rate   |
| `/obs/ee_pose`       | `geometry_msgs/PoseStamped`   | plant → reactive (local, zero-delay) | sim rate |
| `/cmd/waypoint`      | `sensor_msgs/JointState` — `position` = 7-dim action `[pos(3), axis-angle(3), gripper]`; absolute EE target with the DP policy (`absolute:=true`), OSC delta otherwise | controller → reactive | ~20 Hz |
| `/cmd/action`        | `sensor_msgs/JointState`      | reactive → plant       | ~200-500 Hz|
| `/eval/success`      | `std_msgs/Bool` (True/False)  | plant → benchmark      | episode end|
| `/episode/reset`     | `std_msgs/Empty`              | plant → controller, reactive | episode end|
| `/metrics/inference_ms`, `/metrics/delay_steps` | `std_msgs/Float32` | controller → benchmark | per chunk |

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

If you mount the repo to `/ws`, the image's baked-in `ros2_ws/install/` is hidden — the entrypoint
runs `colcon build` automatically when `install/setup.bash` is missing *or* when anything under
`src/` is newer than it. The staleness half matters: `--symlink-install` symlinks the Python
packages but copies everything under `share/`, so a leftover `install/` tree launches old launch
files (old remappings, old params) against current node code, and says nothing about it.

## Jetson deployment

```bash
# On the Jetson (L4T R35.x / JetPack 5.1.x). Pull base image first to verify connectivity:
docker pull dustynv/ros:humble-desktop-pytorch-l4t-r35.4.1

docker build -f docker/Dockerfile.jetson -t edge-vla-hil:jetson .

# Copy the checkpoint to the device first (~4.6 GB) — it is not baked into the image:
#   scp checkpoints/dp_lift_ph_image_cnn.ckpt jetson:~/edge-vla-hil/checkpoints/

# Controller only (pair with host.launch.py on the desktop; same ROS_DOMAIN_ID).
# For the real two-machine run add -e CYCLONEDDS_URI=... first: see "Cross-machine HiL" below.
docker run -it --rm --network host --runtime nvidia \
  -e ROS_DOMAIN_ID=42 \
  -v ~/edge-vla-hil:/ws \
  edge-vla-hil:jetson \
  ros2 launch evh_bringup controller.launch.py strategy:=rtc \
    backend:=dp weights:=/ws/checkpoints/dp_lift_ph_image_cnn.ckpt
```

`backend:=dp` runs the real policy on the L4T torch wheels in the base image; `tensorrt` is a stub
(zeros/unimplemented), only useful for plumbing smoke tests.

First thing to measure on-device is **inference time per chunk** — the controller logs it and
publishes it on `/metrics/inference_ms`. On an RTX 5060 it is ~262 ms (16 DDIM steps). That single
number decides whether an Orin Nano is a viable controller or whether inference belongs elsewhere:
on-device it was too slow, so the fast path is ACT (single forward, no denoise loop) exported to
ONNX and run with `onnxruntime-gpu` (`backend:=onnx`, see `scripts/export_onnx.py` and
`scripts/bench_onnx.py`) — not LeRobot's `act` backend directly, since that needs Python 3.10+ and
the Jetson controller image is Python 3.8.

## Cross-machine HiL over a direct link

The real split — plant on the desktop, controller on the Jetson — needs the two DDS participants to
find each other. Over WiFi they will not: RTT was 80–330 ms and multicast discovery never completed.
Use a direct Ethernet cable between the two NICs and give it static addresses (there is no DHCP
server on a cable):

```bash
sudo ip addr add 10.10.10.1/24 dev eth0 && sudo ip link set eth0 up   # on the Jetson
sudo ip addr add 10.10.10.2/24 dev eno1 && sudo ip link set eno1 up   # on the desktop
```

Substitute your own interface names (`ip -br link`); the *addresses* are what the DDS configs below
pin, so keep those. These commands do not survive a reboot — make them permanent in NetworkManager
or netplan once the link is proven. Check the desktop NIC has no leftover `169.254.x.x` link-local
address from a failed DHCP attempt (`ip -o addr show dev eno1`) and delete it if so: DDS will
happily advertise it, and the other end cannot route to it.

Then point each side at its own CycloneDDS config, which pins the link's interface, switches
discovery to unicast peers, and caps the datagram size below the path MTU. All three matter; the
long comment in `docker/cyclonedds-jetson.xml` records what each one fixes and how it fails without
it (silently, in every case — the graph simply never connects).

```bash
# Desktop: plant + latency relay + reactive + recorder
docker run -it --rm --gpus all --network host \
  -e ROS_DOMAIN_ID=42 -e CYCLONEDDS_URI=file:///ws/docker/cyclonedds-host.xml \
  -v ~/edge-vla-hil:/ws edge-vla-hil:host \
  ros2 launch evh_bringup host.launch.py latency_ms:=0.0

# Jetson: controller only (same ROS_DOMAIN_ID)
docker run -it --rm --network host --runtime nvidia \
  -e ROS_DOMAIN_ID=42 -e CYCLONEDDS_URI=file:///ws/docker/cyclonedds-jetson.xml \
  -v ~/edge-vla-hil:/ws edge-vla-hil:jetson \
  ros2 launch evh_bringup controller.launch.py strategy:=rtc backend:=onnx \
    weights:=/ws/outputs/act_aloha.onnx
```

Both images use `rmw_cyclonedds_cpp` (the desktop image installs it explicitly — `osrf/ros` defaults
to FastRTPS, and two different RMWs do not talk to each other). `--network host` is required.

With both sides up, check the graph from a third shell on either machine rather than reading
`ros2 node list` by eye — a half-formed graph starts cleanly, prints no error, and produces a CSV
of plausible numbers with no policy in the loop:

```bash
docker run --rm --network host -v ~/edge-vla-hil:/ws \
  -e ROS_DOMAIN_ID=42 -e CYCLONEDDS_URI=file:///ws/docker/cyclonedds-jetson.xml \
  --entrypoint bash edge-vla-hil:jetson -lc \
  'source /ros_source.sh && python3 /ws/scripts/check_hil_link.py'
```

It names every expected node per machine and whether the latched `/policy/absolute` crossed the
link. Exit 0 is a full graph; exit 2 is the abs/delta cross-check firing (the graph formed and
`evh_plant` then exited — a working link, a mismatched config); exit 1 is a real discovery
failure. Note that when `evh_plant` aborts, `ros2 launch` tears down the rest of the desktop side
with it, so a stale check run a minute later reports the whole machine missing.

If the graph still does not connect, note that a node aborting at startup with `failed to initialize
rcl node` is the *good* failure: it means the pinned address is missing on that machine, so fix the
static IP. The silent failures are the ones to hunt, and the tool for it is the tracing overlay in
`docker/cyclonedds-debug.xml` — layer it on (`CYCLONEDDS_URI="file://...-jetson.xml,file://...-debug.xml"`)
and read the locators each side advertises in `outputs/cyclonedds-trace.log`. Also suspect a zombie
container on the same `ROS_DOMAIN_ID`; a fresh domain is cheaper than a wrong metric.

## Timing floor

The headline curves are success rate vs *injected* latency, so the testbed's own jitter has to be
small next to the smallest condition being resolved. Measured on the live cross-machine loop —
60 s window, observer on the Jetson, `latency_ms:=0`, ONNX ACT controller at `control_hz=20`:

| topic (publisher) | nominal | p50 | p99 | jitter @ p99 | skipped periods |
|---|---|---|---|---|---|
| `/obs/proprio` (plant, desktop) | 50.0 ms | 50.00 | 50.31 | +0.31 ms | 0.42% |
| `/cmd/action` (reactive, desktop) | 4.0 ms | 4.00 | 4.15 | +0.15 ms | 0.03% |
| `/cmd/waypoint` (controller, Jetson) | 50.0 ms | 50.00 | 50.34 | +0.34 ms | 1.27% |

Relay cost at `latency_ms=0` is **p50 0.97 ms, p99 1.23 ms, max 1.90 ms** — the floor sitting under
every value the sweep injects on top.

So timer jitter is sub-millisecond at p99, on the order of 1% of the smallest 25 ms sweep step: the
injected-latency axis is safe. **The skipped periods are the number to watch, and they are not
noise.** `/cmd/waypoint` is published and observed on the same machine, so its 1.27% is a real
missed deadline inside the controller rather than wire loss — in-loop inference measures 50–68 ms
against a 50 ms tick budget at `control_hz=20`, so roughly one tick in eighty has nothing new to
send. Harmless while a chunk covers 100 steps; it would dominate at a short chunk, and it is the
first thing to re-measure after changing the policy or the chunk length. The desktop-published rows
are observed across the wire, so their skip rates are an upper bound that includes best-effort
loss; run the script on the desktop to separate the two.

```bash
docker run --rm --network host -v ~/edge-vla-hil:/ws \
  -e ROS_DOMAIN_ID=42 -e CYCLONEDDS_URI=file:///ws/docker/cyclonedds-jetson.xml \
  --entrypoint bash edge-vla-hil:jetson -lc \
  'source /ros_source.sh && python3 /ws/scripts/measure_timing_floor.py --seconds 60'
```

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
