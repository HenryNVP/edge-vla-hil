# CLAUDE.md

Guidance for working in this repo. Read `proposal.md` for the full research framing; this file is
the working map and the non-obvious context.

## What this project is

A Hardware-in-the-Loop (HiL) testbed that splits a robot policy across a real network boundary:
a **Plant** (robosuite/MuJoCo sim) on an x86 host, and a **Controller** (diffusion/flow policy)
on a Jetson Orin Nano, talking over ROS2/Ethernet. We inject latency/jitter/packet-loss at the
DDS boundary and measure how SOTA **latency-robust chunk-execution strategies** (synchronous,
naive-async, Temporal Ensembling, BID, **RTC**) hold up — then show a high-rate local **reactive
layer** recovers what they lose.

**The contribution is the edge-network regime, not a new policy.** RTC (NeurIPS 2025,
arXiv:2506.07339) and friends assume a *deterministic delay over a reliable channel*; they
explicitly do not model jitter, packet loss, or reordering. This testbed targets exactly that gap.
- **Wedge A** (current scope): reproduce the strategies as baselines, benchmark them under injected
  network degradation, add the reactive layer.
- **Wedge B** (reserved, future): `network_aware` chunking — replace RTC's `max(past delays)`
  forecast with a distribution-aware one. The code already has the seam for it.

## Architecture

```
[evh_plant] --img,state--> [evh_latency relay] --> [evh_controller: policy + chunk strategy]
     ^  (host, sim)          (inject lat/jitter/loss)        (Jetson)        |
     |                                                   /cmd/waypoint (EE target, control rate)
     +------ /cmd/action (~200-500 Hz) ------ [evh_reactive: impedance ctrl] <+
```

| Topic | Type | From → To |
|---|---|---|
| `/obs/image` | `sensor_msgs/Image` | plant → controller |
| `/obs/joint_state` | `sensor_msgs/JointState` | plant → controller |
| `/cmd/waypoint` | `geometry_msgs/PoseStamped` | controller → reactive (EE target) |
| `/cmd/action` | `sensor_msgs/JointState` | reactive → plant |
| `/eval/success`, `/metrics/inference_ms` | `std_msgs/Bool`, `Float32` | → benchmark recorder |

The latency relay remaps topics (`/obs/image` → `/obs/image/delayed`); nodes are unaware of the
injection. The controller subscribes to the `/delayed` topics via launch remap.

## Repo layout

```
ros2_ws/src/
  evh_plant/       robosuite/MuJoCo sim as a ROS2 node                [host]
  evh_controller/  policy.py (diffusion/flow backends) +
                   chunk_executor.py (strategy registry) +
                   controller_node.py                                 [Jetson]
  evh_reactive/    high-rate operational-space impedance controller   [host, co-located w/ plant]
  evh_latency/     generic relay: seeded latency/jitter/drop/reorder  (the instrument)
  evh_bringup/     launch/, config/default.yaml, benchmark.py
docker/            Dockerfile.host (x86), Dockerfile.jetson (arm64, pin to JetPack)
scripts/           export_onnx.py, build_trt_engine.py
tests/             pytest: pure-Python logic tests + ros2-marked node tests
```

## The two seams that matter most

1. **`evh_controller/chunk_executor.py`** — the heart of the experiment. A `ChunkExecutor`
   strategy interface + registry (`make_executor(name)`). To add a baseline or Wedge B, add a
   class here; nothing else changes. `RTCExecutor.freeze_weights()` implements the paper's
   soft-mask formula; `NetworkAwareExecutor` is the Wedge-B slot (subclasses RTC, overrides
   `forecast_delay`). Selected at runtime via the controller's `strategy` parameter.

2. **`evh_controller/policy.py`** — `make_policy(backend, weights)` returns a `ChunkPolicy` with
   `predict()` and `predict_inpaint()` (the latter is what RTC needs). **PyTorch backend is the
   fallback built first; TensorRT is the Jetson fast path.** Keep this backend-agnostic so a TRT
   stall never blocks the science.

## Build / test / run

```bash
# build (needs ROS2 Humble sourced)
cd ros2_ws && colcon build --symlink-install && source install/setup.bash && cd ..

# tests — pure-Python logic tests run anywhere; ros2-marked tests skip without rclpy
pytest -q                       # 22 pass + 8 skip on a bare box; all run once ROS2 is sourced

# full HiL loop locally (PyTorch fallback, synchronous strategy, no latency)
ros2 launch evh_bringup hil.launch.py strategy:=synchronous latency_ms:=0

# cross-machine: controller.launch.py on Jetson, host.launch.py on the PC (same ROS_DOMAIN_ID)

# Wedge-A sweep (strategy x latency x reactive on/off -> CSV)
ros2 run evh_bringup benchmark --sweep latency \
  --strategies synchronous,temporal_ensemble,rtc --values 0,25,50,100,200
```

Tests need `pytest` + `numpy`. There is **no system pip** on this box; logic tests were validated
in a throwaway venv. Don't install into the user's system Python.

## THE hard part: async generation

The control loop must emit an action every `Δt` (~20 ms @ 50 Hz); the policy takes hundreds of ms.
`select_action` currently calls `policy.predict()` **synchronously** — fine for sim/correctness,
but it would stall the real control loop and freeze the robot. A faithful deployment (and a
faithful RTC reproduction) requires:
- running inference on a **separate ROS2 callback group / worker thread** so the control timer
  never blocks;
- **measuring the real delay `d`** (timestamp `o_t` sent vs. chunk received) and feeding it to the
  strategy — see the `TODO: record the real measured delay` in `RTCExecutor`;
- **splicing** a freshly-arrived chunk onto in-flight actions (freeze-`d` + inpaint for RTC).

The `# TODO: model the inference stall` / async hooks in `chunk_executor.py` mark this. This is the
genuinely novel-on-a-real-link work — budget Phase 3 accordingly.

## Conventions & status

- **Status:** Phase-1 skeleton. Nodes are runnable stubs (synthetic obs, zero-action policy) with
  ROS2 plumbing complete and `TODO`/`STUB` markers for the real physics, inference, and control.
- **Action space:** end-effector (Cartesian) pose, OSC_POSE convention (`action_dim = 7`:
  6-DoF EE delta + gripper). The reactive layer tracks EE targets with Jacobian-transpose impedance
  (no IK). Confirm absolute-vs-delta convention when wiring the real policy (`_to_pose` TODO).
- **Reactive baseline:** `passthrough:=true` disables the reactive layer (the monolithic baseline)
  — it's a launch arg, not a code change.
- **Backbone:** small diffusion/flow policy (Diffusion Policy on RoboMimic is the path of least
  resistance, and it exposes the denoiser RTC's `predict_inpaint` needs). ACT+TE is a deterministic
  baseline only; AWE (sparse waypoints) is an optional ablation, not core.
- **Style:** match existing nodes — `rclpy` boilerplate, `**kwargs` forwarded to `super().__init__`
  (so tests can inject `parameter_overrides`), docstrings stating each node's role and stub state.
- **Version watch:** AWE/original-ACT pin MuJoCo 2.1; the proposal otherwise assumes newer. Pin the
  Plant's MuJoCo/robosuite to whatever the chosen policy was trained against to avoid sim mismatch.
- **CI:** `.github/workflows/ci.yml` — a fast pure-Python job + a `ros:humble` job that builds the
  workspace and runs the full suite.

## Git

Branch `master` (note: PRs usually target `main`). Commit/push only when asked.
