# EdgeVLA-HiL: Benchmarking Latency-Robust Action Chunking over a Real Edge-Network Boundary

GitHub Repository Name: edge-vla-hil
Project Name: EdgeVLA-HiL (Edge Vision-Language-Action Hardware-in-the-Loop)

## 1. Executive Summary

Action-chunking policies (ACT, Diffusion Policy) and VLAs are slow to run, so a growing body of
work makes their *execution* robust to inference latency — Temporal Ensembling, Bidirectional
Decoding (BID), and most recently **Real-Time Chunking (RTC, NeurIPS 2025)**. But essentially all
of this work models latency as a **known, deterministic delay** and assumes a **reliable
communication channel**: RTC, for example, explicitly states it does not model sub-timestep
delays, stochastic jitter, packet loss, or out-of-order delivery. Real edge deployments — a policy
on a Jetson talking to a robot over a network — violate every one of those assumptions.

This project builds a **Hardware-in-the-Loop (HiL)** testbed that physically decouples the physics
simulation (Plant, x86 host) from the policy inference engine (Controller, NVIDIA Jetson Orin
Nano) across a real ROS2 / Ethernet boundary, with a programmable harness that injects
**latency, jitter, packet loss, and reordering** at the DDS layer. On this testbed we deliver:

1. **The first edge-network benchmark** of SOTA latency-robust chunking strategies
   (synchronous, naive-async, Temporal Ensembling, BID, **RTC**) under *physically realistic*
   network degradation — not just a deterministic delay constant.
2. **A high-rate local reactive layer** (operational-space impedance controller) that recovers
   task success the inference-time methods lose when delay becomes stochastic, and that composes
   with them rather than replacing them.

The contribution is a systems + empirical one, and it positions RTC/BID as **baselines we
reproduce and characterize**, not competitors. (A natural algorithmic extension — *network-aware
chunking* — is scoped as follow-on work in Section 7.)

## 2. Motivation & Relevance

The latency-robust-execution subfield grew quickly in 2024–2026, but it is almost entirely
evaluated in sim or with the policy co-located on the robot, where "delay" is a deterministic
number of timesteps. The gap between that and a real edge deployment is exactly:

- **Stochastic, heavy-tailed delay** (jitter) instead of a fixed `d`.
- **Packet loss and out-of-order delivery** over DDS/UDP, which no chunking method models.
- **A separate compute node** (Jetson) with its own honest inference rate and a real wire.

This project targets that gap directly:

- **Edge AI deployment:** compile and run a small diffusion/flow policy natively on a Jetson Orin
  Nano (ONNX → TensorRT) and report the honest achievable inference rate.
- **HiL validation:** reproduce latency-robust chunking strategies and measure where each breaks
  under injected jitter/loss/reorder — reproducibly, with a seeded harness.
- **Async hierarchical control:** show a fast local reactive layer recovers performance the
  cognitive loop loses, and complements RTC-style execution.

## 3. Policy Backbone

**A small diffusion / flow-matching policy** (e.g., Diffusion Policy on RoboMimic, or a compact
flow policy; few-step / consistency sampling for the Jetson). This choice is deliberate:

- **RTC and BID require a diffusion/flow policy** — their "inpainting" / guided resampling does not
  apply to deterministic transformers. Using a flow policy makes them first-class baselines.
- **Diffusion inference is genuinely slow** (iterative denoising, hundreds of ms), which makes the
  latency motivation realistic rather than contrived. We use a few-step sampler to keep the Orin
  Nano honest while preserving the effect under study.
- **Action space:** end-effector (Cartesian) pose deltas via RoboMimic's `OSC_POSE` controller, so
  the cognitive layer emits task-space targets and the reactive layer tracks them with Cartesian
  impedance (Jacobian-transpose, no IK).

**ACT + Temporal Ensembling** is retained as a deterministic baseline. **AWE** (sparse waypoints)
is an *optional sparsity ablation*, not the centerpiece.

## 4. System Architecture

Asynchronous ROS2 DDS over physical Ethernet, strict separation of concerns.

### A. Simulation Plant (PC Host)
robosuite/MuJoCo (RoboMimic task) wrapped as a ROS2 node: steps physics, publishes
`sensor_msgs/Image` + `JointState`, applies low-level actions, reports task success.

### B. Edge Controller (Jetson Orin Nano)
The diffusion/flow policy + a **pluggable chunk-execution strategy**. The strategy is the heart of
the experiment and the extensibility seam: `{synchronous, naive_async, temporal_ensemble, bid,
rtc}` for Wedge A, with `network_aware` reserved for Wedge B (Section 7). The policy backend is
swappable PyTorch ↔ TensorRT so a TRT stall never blocks the science.

### C. Reactive Local Layer
High-rate (~200–500 Hz) operational-space impedance controller, co-located with the plant
(zero-delay local state). Tracks the (delayed) task-space targets between cognitive updates. Fixed
gains in this work. A `passthrough` mode reproduces the monolithic (no-reactive-layer) baseline.

### D. Latency Harness (the instrument)
Per-link relay at the DDS boundary injecting seeded **latency + jitter + drop + reorder**. This is
what lets us test the regime RTC explicitly excludes. Software injection works whether or not the
Jetson is physically attached, and stacks on top of the real wire when it is.

```
[plant] --img,state--> [latency harness] --> [controller: policy + chunk strategy] --target-->
   ^                                                                                     |
   +------------------- [reactive layer @ high rate] <----------------------------------+
```

## 5. Experimental Design (Wedge A)

**Independent variables:** injected latency, jitter (std), packet-loss prob, chunk strategy
(5 levels), reactive layer on/off.

**Baselines reproduced** (chunk-execution strategies): synchronous (pause between chunks),
naive-async, Temporal Ensembling, BID, RTC.

**Metrics:** Task Success Rate (%), task throughput (progress/time), achieved control-loop Hz,
end-to-end inference latency (ms), and the **stability margin** — the latency/jitter at which each
strategy fails, with vs. without the reactive layer.

**Headline result:** success-rate curves for each strategy vs. injected jitter and packet loss,
showing (a) where deterministic-delay methods (esp. RTC) degrade once delay is stochastic/lossy,
and (b) how much the reactive layer buys back.

## 6. Implementation Phases (≈12 weeks, solo, full-time)

**Phase 1 — HiL testbed & latency/jitter/loss/reorder harness (Weeks 1–2).** Built first; it is
the experimental instrument.

**Phase 2 — Diffusion/flow policy in-loop on PC (Weeks 3–5).** Train/adopt a small diffusion
policy on one RoboMimic task (start with `Lift` or `Can`), validate baseline success before the
network boundary. Implement the chunk-execution strategy interface with the simple strategies
(synchronous, naive-async, TE).

**Phase 3 — RTC + BID baselines (Weeks 5–7).** Implement RTC (freeze-`d` + soft-masked guided
inpainting) and BID against the flow policy. Validate they match published behavior at
deterministic delay before adding network effects.

**Phase 4 — Jetson deployment (Weeks 6–8, overlaps).** ONNX → TensorRT; honest inference rate.
PyTorch fallback built first as insurance.

**Phase 5 — Reactive layer + benchmark sweep (Weeks 8–12).** Add the impedance layer; run the full
sweep over latency/jitter/loss × strategy × reactive on/off; produce the headline plots; writeup,
README, demo video.

## 7. Future Work

**Wedge B — Network-aware chunking (the algorithmic extension).** RTC forecasts the delay `d`
conservatively as the max of a buffer of past delays, assuming a reliable channel. Under
heavy-tailed jitter and loss this is exactly where it should break. The extension: drive the
freeze horizon and execution horizon from a *measured RTT/jitter distribution* (e.g., a quantile
or loss-aware estimate) rather than a single forecast `d`. This is a genuine, scoped algorithmic
contribution that drops into RTC's inference loop and is provable only on a testbed like this one.
The codebase reserves a `network_aware` strategy slot for it.

**Language-conditioned impedance.** Making the reactive layer's stiffness a function of semantic
intent (compliant "wipe" vs. rigid "insert"), gated on collecting impedance-labeled data.

## References

[1] Black, K., Galliker, M. Y., Levine, S. (2025). "Real-Time Execution of Action Chunking Flow
Policies." NeurIPS 2025. arXiv:2506.07339.
[2] Liu, Y., et al. (2024). "Bidirectional Decoding: Improving Action Chunking via Guided
Test-Time Sampling." arXiv:2408.17355.
[3] "FASTER: Rethinking Real-Time Flow VLAs." (2026). arXiv:2603.19199.
[4] Chi, C., et al. (2023). "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion."
Robotics: Science and Systems (RSS).
[5] Zhao, T. Z., et al. (2023). "Learning Fine-Grained Bimanual Manipulation with Low-Cost
Hardware" (ACT). RSS.
[6] Shi, L. X., Sharma, A., Zhao, T. Z., Finn, C. (2023). "Waypoint-Based Imitation Learning for
Robotic Manipulation" (AWE). CoRL. arXiv:2307.14326.
[7] "Leave No Observation Behind: Real-Time Correction for VLA Action Chunks." (2025).
arXiv:2509.23224.
