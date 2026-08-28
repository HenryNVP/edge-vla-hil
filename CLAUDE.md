# CLAUDE.md

Guidance for AI coding agents working in this repo. For the project narrative, node graph, topic
table, and run commands see `README.md` and `proposal.md` — this file covers what you need to
change code safely: the build/test loop, the invariants that fail *silently* if you break them,
and the conventions to match.

## What this is

A Hardware-in-the-Loop testbed that benchmarks latency-robust action-chunking strategies. A
robosuite/MuJoCo sim (**Plant**) is decoupled from a diffusion-policy inference engine
(**Controller**) across a ROS2 boundary with a programmable latency/jitter/drop relay in between.
A high-rate local **reactive** layer tracks the delayed waypoints. The headline result is
success-rate-vs-network-degradation curves per strategy, with and without the reactive layer.

The one seam that matters: `evh_controller/chunk_executor.py`. Strategies
(`synchronous | naive_async | temporal_ensemble | bid | rtc | network_aware`) are pluggable; the
ROS node never changes when you add or modify one. New execution-strategy work goes here.

## Source lives in `ros2_ws/src/` — everything else is scaffolding

| Package | File(s) | Role |
|---|---|---|
| `evh_plant` | `plant_node.py` | robosuite env as a ROS node; steps physics, publishes obs, applies actions, manages episodes |
| `evh_controller` | `controller_node.py`, `policy.py`, `dp_repo_policy.py`, `inference_worker.py`, `chunk_executor.py` | policy backends + async inference + chunk-execution strategies |
| `evh_reactive` | `reactive_node.py`, `transforms.py` | high-rate local tracking of delayed waypoints; pure-numpy quaternion helpers |
| `evh_latency` | `latency_node.py` | type-generic delay/jitter/drop/reorder relay |
| `evh_bringup` | `launch/*.launch.py`, `config/default.yaml`, `benchmark.py` | launch graphs, params, metrics recorder + sweep driver |

`external/diffusion_policy/` is a vendored third-party repo (real-stanford/diffusion_policy) — the
Lift checkpoint's model code. **Do not edit it or lint it as ours.** Per-package docstrings are
thorough and kept current; read the module docstring before changing a file.

## Build & test

Everything runs inside the `edge-vla-hil:host` Docker image (ROS2 Humble + Python 3.10 + all
deps). See README for `docker build`/`docker run`. Inside the container or a sourced host:

```bash
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
```

Tests — the fast logic suite needs no ROS build (conftest puts each package on `sys.path`); ROS
node-boot tests auto-skip when rclpy isn't sourced:

```bash
# apt pytest clashes with a pip anyio plugin — this flag is REQUIRED or collection errors out
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

Pure-Python modules (`chunk_executor`, `inference_worker`, `transforms`, rotation math in
`dp_repo_policy`, latency relay logic) are unit-tested and importable without ROS — prefer adding
coverage there. Node-level tests need a sourced ROS2 env.

Loading the real DP checkpoint needs extra deps: `scripts/setup_dp_deps.sh` (already baked into the
host image). The co-located eval `scripts/eval_dp_colocated.py` is the **regression gate** — it
runs the policy against robosuite directly (no ROS), so it isolates policy correctness from HiL
plumbing. Run it after touching `policy.py` / `dp_repo_policy.py`.

## Invariants that fail silently — check these before you change behavior

These are the traps. Except where noted, none of them raise an error; they just produce a robot
that misbehaves and metrics that look plausible but are wrong.

1. **`absolute` mode must agree across three places.** The DP Lift checkpoint is the *abs-action*
   variant: the policy emits 10-dim `[pos, rot_6d, gripper]` absolute EE targets, converted to
   7-dim `[pos(3), axis-angle(3), gripper]`. The plant (`absolute_actions` → OSC
   `control_delta=False`), the reactive layer (`absolute_waypoints`), and the policy
   (`ChunkPolicy.absolute_actions`, derived from the checkpoint) must all be on the same mode. The
   plant/reactive take it from a launch arg; the policy derives it from the checkpoint. A mismatch
   (e.g. `backend:=dp` with `absolute:=false`, or `pytorch` with `absolute:=true`) yields garbage
   motion. **This one is now enforced, not silent**: the controller announces the checkpoint's mode
   on a latched `/policy/absolute`, and the plant (`_on_policy_mode`) aborts with exit code 1 on a
   disagreement. Set `strict_mode_check:=false` to run a mismatched config deliberately. The
   reactive layer is not checked separately — every launch file feeds it and the plant the *same*
   `absolute` arg, so it cannot diverge from the plant.

2. **Delta mode vs absolute mode change what a "zero" or "hold" action means.** In delta mode a
   zero action = "don't move." In absolute mode a zero action = "go to world origin (0,0,0)" — a
   hard lurch. Anything that emits a default/placeholder action must be mode-aware — see the
   plant's `_hold_action`, which commands the current EE pose instead of zeros before the first
   `/cmd/action` of an episode arrives.

3. **`pos_scale` / `rot_scale` (reactive) must mirror the plant OSC's `output_max`** (robosuite
   default 0.05 m / 0.5 rad) — how far a unit action moves the OSC goal per step. They are
   duplicated params, not shared; drift between them breaks delta-mode tracking gains.

4. **robosuite is pinned to 1.4.1 + mujoco 2.3.7** (`requirements-host.txt`). robosuite 1.5.x
   multi-camera offscreen rendering is broken (obs images swap/corrupt after `env.step`). Do **not**
   bump without re-running a two-camera step-render check; the policy silently consumes noise
   otherwise.

5. **Async inference is epoch-guarded.** One request is in flight at a time
   (`InferenceWorker`, single-slot). A chunk carries the issue tick + an epoch tag; `reset()` bumps
   the epoch so in-flight chunks computed against pre-reset observations are discarded on arrival.
   Delay is measured in *control steps* (request→arrival) and feeds RTC's forecast — keep that the
   honest, measured value; don't shortcut it.

6. **`/obs/ee_pose` is deliberately NOT routed through the latency relay** — it's the reactive
   layer's zero-delay local anchor. Only `/obs/image`, `/obs/image_wrist`, `/obs/proprio` are
   delayed. Don't add ee_pose to the relay set.

## Contracts

- **Action (7-dim)** on `/cmd/waypoint` and `/cmd/action`: `sensor_msgs/JointState.position` =
  `[pos/dpos(3), axis-angle/drot(3), gripper]`. A `JointState` is reused instead of a custom msg to
  keep the build light — revisit only if the contract grows.
- **Observation dict** (controller ↔ policy), values stacked over the last `n_obs_steps` ticks,
  oldest first: `agentview` uint8 `[To,H,W,3]`, `wrist` uint8 `[To,H,W,3]` (only if
  `policy.needs_wrist`), `proprio` float `[To, D]` = `[eef_pos(3), eef_quat(4, xyzw),
  gripper_qpos(2)]`.
- **Quaternions are `[x, y, z, w]`** everywhere (robosuite / geometry_msgs order). Rotation deltas
  are axis-angle applied in the world frame: `R_goal = R(delta) @ R_current`.
- **Episode outcomes**: the plant publishes `/eval/success` `Bool` on *both* task success and
  horizon timeout (timeout = `False`) — the recorder needs both for an honest rate. Episode
  horizon is enforced via robosuite `horizon = max_episode_s * action_hz`.

## Conventions

- Match the surrounding style: dense, purposeful module docstrings that explain *why*; terse inline
  comments only where the code is subtle. New execution strategies subclass `ChunkExecutor`, set a
  `name`, and register in `_REGISTRY`.
- ROS entry points are `console_scripts` in each package's `setup.py` (e.g. `controller_node =
  evh_controller.controller_node:main`). Add new nodes the same way and rebuild with colcon.
- Rates: `control_hz` (20) = policy/observation cognitive rate; `action_hz` (200) = plant physics /
  reactive tracking rate. Keep `control_hz` equal to the policy's training rate.
- When metrics look self-contradictory, suspect ROS domain pollution from a zombie
  container/process — use a fresh `ROS_DOMAIN_ID` before trusting numbers.

## Roadmap (where things are headed)

Phase 2 done: async chunk execution + validated DP Lift policy in the loop. Next: true guided
inpainting for RTC + real BID (currently `predict_inpaint` is a soft-blend fallback and `BIDExecutor`
is a naive-async stub — both marked in-code as Step 4), then reactive tuning + the full benchmark
sweep, then the Jetson TensorRT backend (`TensorRTBackend` is a stub today).
</content>
</invoke>
