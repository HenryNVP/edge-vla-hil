# EdgeVLA-HiL: Closing the Observation–Action Delay Loop for Edge-Deployed VLA Policies

GitHub Repository Name: edge-vla-hil
Project Name: EdgeVLA-HiL (Edge Vision-Language-Action Hardware-in-the-Loop)

## 1. Executive Summary

Vision-Language-Action (VLA) foundation models reason and generalize impressively, but they
are slow. When a VLA is deployed onto resource-constrained edge hardware and asked to control a
robot over a network boundary, inference latency and network jitter open an **observation–action
delay loop**: the policy acts on stale state, deviations compound, and the control loop
destabilizes.

This project builds a **Hardware-in-the-Loop (HiL)** testbed that physically decouples the
physics simulation (Plant, on an x86 host) from the AI inference engine (Controller, on an NVIDIA
Jetson Orin Nano) across a real Gigabit Ethernet / ROS2 boundary. Using this testbed, the project
**quantifies how edge-induced delay degrades VLA control** and demonstrates that a lightweight
**high-rate local reactive layer** recovers task success and stability under injected latency and
jitter — without modifying the VLA itself.

The contribution is a systems + empirical one: a reproducible characterization of VLA degradation
under realistic edge conditions, and a hierarchical async control scheme that mitigates it.

## 2. Motivation & Relevance

Foundation-model policies are increasingly deployed to physical robots, but the gap between
"works in a notebook" and "stable on edge hardware over a network" is large and under-measured.
This project targets that gap directly:

- **Edge AI deployment:** Compile and run a pretrained VLA policy natively on a Jetson Orin Nano
  via ONNX → TensorRT, and report the *honest* achievable inference rate.
- **Hardware-in-the-Loop validation:** Prove (or break) closed-loop stability across a physical
  network boundary, with reproducible, programmatically injected latency and jitter.
- **Async hierarchical control:** Show that splitting a slow cognitive layer from a fast reactive
  layer recovers performance the monolithic loop loses.

## 3. System Architecture

A strict separation of concerns communicating asynchronously via ROS2 DDS over physical Gigabit
Ethernet.

### A. The Simulation Plant (PC Host)
- **Engine:** robosuite (MuJoCo) on x86 Ubuntu.
- **Role:** Simulates physics and contact dynamics, renders camera observations.
- **ROS2 interface:** A wrapper node that steps the sim at a fixed control frequency, publishes
  `sensor_msgs/Image` and `sensor_msgs/JointState`, and subscribes to task-space waypoint
  commands. A **latency-injection harness** programmatically delays/jitters/drops messages at the
  DDS boundary to emulate edge network conditions.

### B. The Edge AI Controller (NVIDIA Jetson Orin Nano)
- **Hardware:** Jetson Orin Nano (ARM Cortex-A78AE, Ampere GPU) — already in hand.
- **Brain:** A pretrained **ACT (Action Chunking with Transformers)** policy from LeRobot. ACT is
  chosen over a heavier VLA (e.g. OpenVLA) because it deploys cleanly on the 8 GB Orin Nano and
  converts to TensorRT with far less friction, while still exhibiting the observation–action delay
  loop this project studies. Training from scratch is out of scope; the contribution is
  deployment + control, not the model. (Language conditioning is light in ACT; the prompt selects
  the task/checkpoint rather than driving a language encoder — sufficient for the delay-loop study.)
- **Optimization:** Exported to ONNX and compiled via TensorRT (FP16) for maximum real-time
  factor. Expected honest inference rate is ~10 Hz on the Orin Nano, not >30 Hz.
- **Role:** Subscribes to the simulated topics, computes task-space waypoints from the language
  prompt + image, and publishes them back to the Plant.

### C. The Reactive Local Layer
A lightweight, high-rate (target ~200–500 Hz) **operational-space / impedance controller** runs
local to the actuators. It receives the delayed waypoints from the VLA and tracks them using
zero-delay local state, smoothing and stabilizing execution between (and through the latency of)
cognitive updates. Gains are **fixed** in this work — learned, language-conditioned gains are
deferred to future work (Section 6).

> Note: This is an operational-space impedance controller, deliberately *not* a receding-horizon
> QP-MPC. A true high-rate MPC is a separate project and is not required for the contribution.

## 4. Implementation Phases (≈12 weeks, solo, full-time)

**Phase 1 — HiL Testbed & Latency Harness (Weeks 1–2).**
Build the robosuite ROS2 wrapper and the programmatic latency/jitter/drop injection at the DDS
boundary. Getting this experimental instrument right early is the priority.

**Phase 2 — Baseline Policy In-Loop on PC (Weeks 3–5).**
Stand up the pretrained policy and validate baseline task success *before* the network boundary,
on a single well-chosen task (e.g., pick-and-place of a target block). Establish the
no-latency reference success rate.

**Phase 3 — Jetson Deployment (Weeks 5–8).**
ONNX → TensorRT on the Orin Nano; reach a stable, honest inference rate. **Risk-managed:** a
PyTorch GPU fallback is built first so a TensorRT stall never blocks downstream phases. The
project succeeds even if TRT underperforms — we simply report the rate achieved.

**Phase 4 — Reactive Local Layer (Weeks 8–10).**
Add the high-rate operational-space impedance controller that tracks the delayed VLA waypoints
using local state.

**Phase 5 — Benchmark Sweep & Writeup (Weeks 10–12).**
The money plot: **Task Success Rate, Control Loop Frequency, and Inference Latency vs. injected
latency/jitter**, measured *with vs. without* the reactive layer. Polish README, plots, and a
demo video.

## 5. Evaluation

Single task, fully characterized, beats many half-tasks. Core metrics:

- **Task Success Rate (%)** as a function of injected one-way latency and jitter.
- **Control Loop Frequency (Hz)** and end-to-end **Inference Latency (ms)** on the Orin Nano.
- **Stability margin:** the latency at which the monolithic loop fails vs. the hierarchical loop.

The headline result is the gap between the two curves: how much edge-induced delay the reactive
layer buys back.

## 6. Future Work: Language-Conditioned Impedance Tuning

A natural extension is to make the reactive layer's stiffness matrix ($K_p, K_d$) a function of
semantic intent — "wipe the table" → low Z-axis stiffness (compliant); "insert the peg firmly" →
high stiffness (rigid disturbance rejection). This is deferred because it requires demonstration
data labeled with impedance/contact-wrench targets paired with language, which standard datasets
(Robomimic, etc.) do not provide; generating that dataset is a project in its own right. The
architecture here is built to accept such gains directly, making this a clean follow-on.

## References

[1] Zhao, T. Z., et al. (2023). "Learning Fine-Grained Bimanual Manipulation with Low-Cost
Hardware." Robotics: Science and Systems (RSS).
[2] Jiang, et al. (2026). "Adaptive Action Chunking at Inference-time for Vision-Language-Action
Models." arXiv:2604.04161.
[3] Zhou, et al. (2026). "Terrain-Reactive Locomotion Policies from Vision-Language-Action Priors:
Bridging Semantic Commands and Whole-Body Control."
