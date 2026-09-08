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
scripts/              # one-off tooling (dataset conversion, ONNX export, TRT build)
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
| `/cmd/action`        | `sensor_msgs/JointState` — carries a 50 ms **deadline QoS**: the plant re-applies the last action at `action_hz`, so it needs DDS to tell it when this layer has gone silent (it then holds instead) | reactive → plant       | ~200-500 Hz|
| `/eval/success`      | `std_msgs/Bool` (True/False)  | plant → benchmark      | episode end|
| `/episode/reset`     | `std_msgs/Empty`              | plant → controller, reactive | episode end|
| `/metrics/inference_ms`, `/metrics/delay_steps` | `std_msgs/Float32` | controller → benchmark | per chunk |

Topics are remapped through `evh_latency` (e.g. `/obs/image` → `/obs/image/delayed`) via launch
arguments; nodes themselves are unaware of the injected delay.

UML diagrams of all of this — deployment, the node graph with its QoS, per-package internals, and
sequence diagrams for the control cycle, async chunk execution, the episode boundary and bringup —
live in [`docs/diagrams/`](docs/diagrams/README.md).

The plant only conditionally trusts `/cmd/action`, because it re-applies whichever action it holds
until another arrives. It drops the cached command when the deadline above is missed, and ignores
the stream for 20 ms after an `/episode/reset` (the reactive layer learns about a reset up to one
of its own ticks late, and what it sends in that window was computed for the episode that just
ended). Both fall through to a mode-aware hold. Killing `evh_reactive` mid-episode in delta mode
moved the arm 189 mm in 4 s without the deadline and 8 mm with it — a cached delta is a *velocity*
command, since OSC re-derives `goal = eef + delta * output_max` every step.

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

### Diffusion Policy through ONNX (`backend:=dp_onnx`)

DP can also take the ONNX path, which is worth doing before concluding that torch was the problem.
It is not one graph: a prediction is an encoder pass plus `num_inference_steps` UNet passes with a
DDIM update between them, so `scripts/export_dp_onnx.py` writes **two** graphs and leaves the loop
in numpy (`evh_controller/dp_onnx_policy.py`). Unrolling the loop into a single graph would also
break RTC, whose guidance is applied *between* denoising steps.

```bash
# host: export (needs torch + the diffusion_policy repo), verify against torch, then benchmark
python scripts/export_dp_onnx.py --ckpt checkpoints/dp_lift_ph_image_cnn.ckpt \
  --out outputs/dp_lift_onnx --steps 16 --check
python scripts/bench_onnx.py outputs/dp_lift_onnx

# Jetson: numpy + onnxruntime only, no torch
ros2 launch evh_bringup controller.launch.py backend:=dp_onnx weights:=/ws/outputs/dp_lift_onnx
```

`--check` re-runs the whole sampler through ONNX Runtime against torch **from the same initial
noise** and fails the export if they diverge; anything less proves nothing, because the sampler
starts from `randn` and two correct implementations disagree completely on independent draws.
Measured for the Lift checkpoint: encoder output exact, final action within 0.099% of full scale.

What it buys, on an RTX 5060 at 16 DDIM steps. Both rows are `eval_dp_colocated.py --episodes 5
--seed 0`, so it is the same harness, protocol and seed on both sides:

| backend | success | per action (steady state) |
|---|---|---|
| `dp` (torch) | 5/5 | 252 ms |
| `dp_onnx` | 5/5 | **210 ms** |

Same behaviour (episode lengths within a few steps of each other), ~1.2x faster. Isolated,
`bench_onnx.py` puts the split at encoder 2.8 ms + UNet 12.3 ms x 16 steps = 200 ms; the first
call is ~4.9 s of ORT/CUDA warmup, so read the steady state, not the mean the eval prints.

Still 4x over the 50 ms budget at `control_hz=20`, on a 5060, before any Orin Nano penalty. The
UNet is 98% of it and scales linearly with `--steps`, so step count is the only lever with real
leverage; ONNX alone does not make a 16-step diffusion policy a 20 Hz controller. The exported
UNet is 1.0 GB fp32 (encoder 90 MB), which fits an 8 GB Orin but argues for fp16 first.

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

If the graph still does not connect, note that a node aborting at startup with `failed to initialize
rcl node` is the *good* failure: it means the pinned address is missing on that machine, so fix the
static IP. The silent failures are the ones to hunt, and the tool for it is the tracing overlay in
`docker/cyclonedds-debug.xml` — layer it on (`CYCLONEDDS_URI="file://...-jetson.xml,file://...-debug.xml"`)
and read the locators each side advertises in `outputs/cyclonedds-trace.log`. Also suspect a zombie
container on the same `ROS_DOMAIN_ID`; a fresh domain is cheaper than a wrong metric.

## ACT training data (robomimic -> LeRobot)

Per-task ACT policies are trained from the robomimic proficient-human demonstrations. The
pretrained LeRobot ACT checkpoints are bimanual ALOHA (14-dim joint actions), so nothing
transfers to a 7-dim Panda OSC action space — these are trained from scratch per task, or
pretrained across tasks and finetuned per task, which works because all four share an action
space.

```bash
# 1. fetch the demos. These are the v1.2-era image sets (agentview + robot0_eye_in_hand, 84x84,
#    20 Hz, 200 demos) — robomimic 0.5's own downloader now points at v1.5 raw states with no
#    image sets at all, and tool_hang's packaged image set is 66 GB of 240x240, so render that
#    one from states instead (robomimic/scripts/dataset_states_to_obs.py).
BASE=http://downloads.cs.stanford.edu/downloads/rt_benchmark
for task in lift can square; do
  mkdir -p data/robomimic/$task/ph
  curl -fL --retry 5 --retry-all-errors -C - \
    -o data/robomimic/$task/ph/image.hdf5 $BASE/$task/ph/image.hdf5
done

# 2. convert to a LeRobotDataset (delta actions, lossless PNG frames, ~20 s per task)
.venv-local/bin/python scripts/robomimic_to_lerobot.py \
  --dataset data/robomimic/lift/ph/image.hdf5 --out data/lerobot/lift_ph

# absolute-action variant, which is what `absolute:=true` (the default) needs. The action
# becomes 10-dim [pos, rot_6d, gripper] — the DP checkpoints' layout, and mandatory rather than
# cosmetic: every absolute orientation target in these demos sits on the pi wrap, where
# axis-angle flips sign between neighbouring frames (see scripts/robomimic_to_lerobot.py).
.venv-local/bin/python scripts/robomimic_to_lerobot.py \
  --dataset data/robomimic/lift/ph/image.hdf5 --out data/lerobot/lift_ph_abs6 \
  --actions abs-derived

# 3. train (in the host container; ~52M params, ~35 ms/step on an RTX 5060 -> ~1 h for 100k)
lerobot-train --dataset.repo_id=local/lift_ph_abs6 --dataset.root=/ws/data/lerobot/lift_ph_abs6 \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=16 --policy.n_action_steps=16 --policy.temporal_ensemble_coeff=null \
  --batch_size=8 --steps=100000 --save_freq=25000 --output_dir=/ws/outputs/train/act_lift_ph_abs6

# 4. run it in the loop
ros2 launch evh_bringup hil.launch.py backend:=act absolute:=true \
  weights:=/ws/outputs/train/act_lift_ph_abs6/checkpoints/last/pretrained_model

# a DELTA-trained checkpoint has a 7-dim head, which says nothing about its mode — stamp it,
# then run the plant in delta mode
python scripts/stamp_act_checkpoint.py \
  outputs/train/act_lift_ph_delta/checkpoints/last/pretrained_model \
  --dataset-root data/lerobot/lift_ph
ros2 launch evh_bringup hil.launch.py backend:=act absolute:=false \
  weights:=/ws/outputs/train/act_lift_ph_delta/checkpoints/last/pretrained_model
```

The action convention has to reach the plant, which cross-checks it (invariant 1). A 10-dim ACT
head announces itself — it can only be the abs `[pos, rot_6d, gripper]` layout, exactly as for
the DP checkpoints — and `ACTBackend` converts it to the 7-dim contract. A 7-dim head is
ambiguous, so it needs the stamp: `scripts/stamp_act_checkpoint.py` reads the mode from the
dataset LeRobot recorded in `train_config.json`, so it is derived rather than typed. Unstamped
means delta, which makes an abs policy abort the plant rather than run wrong. The controller's
`policy_absolute` parameter (`auto|true|false`, default `auto`) forces it for deliberately
mismatched runs, `benchmark --absolute auto` reads the same stamp, and `scripts/export_onnx.py`
carries it into the ONNX sidecar for the Jetson backend.

Measured at zero latency, `synchronous`, reactive layer on, 100k steps on Lift PH (200 demos):

| policy | mode | success | inference (mean / p95) | waypoint_hz |
|---|---|---|---|---|
| ACT, absolute (10-dim rot_6d) | `absolute:=true` | 9/10 | 6.5 / 8.0 ms | 19.8 |
| ACT, delta | `absolute:=false` | 7/10 | 6.6 / 8.0 ms | 19.9 |
| DP repo checkpoint (reference) | `absolute:=true` | 5/5 | 290 / 316 ms | 14.5 |

The ~45x inference gap is the point of keeping ACT as the deterministic baseline: at 6.5 ms the
policy is no longer what limits the loop, so the degradation curves isolate the network.

Leave `temporal_ensemble_coeff` null — `TemporalEnsembleExecutor` does the ensembling, and
letting ACT also do it internally hides the strategy being measured. ACT is deterministic at
inference (the VAE latent is zeroed), so it serves `synchronous`, `naive_async`,
`temporal_ensemble` and `network_aware`, but not RTC or BID, which need a denoising loop.

The converter self-checks each conversion against the source HDF5 and can cross-check against
another port of the same demos (`--verify-against`); the community ports
`ankile/robomimic-ph-*-image` agree with ours exactly on actions and state, but their video
encoding puts agentview at 17 dB PSNR — below the frame-to-frame motion in the same scene —
which is why this writes PNG.

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
(`/metrics/inference_ms`, `/metrics/delay_steps`), and **ACT policies trained on robomimic Lift**
in both action modes (9/10 absolute, 7/10 delta at zero latency — see the table above). Still to
come: true guided inpainting for RTC + real BID (Phase 3), TensorRT backend (Phase 4). See
per-package docstrings.
