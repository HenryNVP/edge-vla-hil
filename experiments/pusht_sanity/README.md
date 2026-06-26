# PushT sanity check

A **throwaway, non-ROS** rig to confirm the latency-robust-chunking phenomenon — and validate the
project's experiment design — *before* building the robosuite/RoboMimic HiL testbed. It uses the
pretrained `lerobot/diffusion_pusht` checkpoint, so **zero training**.

## Why this exists

The real contribution lives in the ROS edge-network testbed (`../../ros2_ws`). But that's a lot to
build before knowing the core effect reproduces. PushT is extremely reactive, latency bites hard,
and a pretrained diffusion policy is one download away — so it's the cheapest possible way to
answer: *does injected latency degrade success, and do the chunk-execution strategies rank the way
the RTC paper says?* If yes, the robosuite investment is de-risked.

## What's here

| File | Role |
|---|---|
| `latency_chunking.py` | **pure-numpy core**: delayed-chunk scheduler + the 4 strategies + a mock env/policy. No torch/gym. |
| `test_latency_chunking.py` | pytest for the core machinery — runs anywhere (`pip install numpy pytest`). |
| `run_pusht.py` | the real sweep: pretrained diffusion_pusht x latency x strategy -> CSV (+ optional plot). |
| `requirements.txt` | deps for the real run. |

The strategy names mirror `evh_controller/chunk_executor.py`. `rtc_freeze` is the **freeze-only
approximation** of RTC (skip the d overlapped actions, continue from the time-aligned index); true
RTC also *guided-inpaints* the unfrozen tail during denoising, which needs the policy's denoiser
and is out of scope for this black-box rig. That's fine — freezing vs. naive switching is the
effect we're checking.

## Latency model

Latency is in **control steps**. PushT runs ~10 Hz, so 1 step ≈ 100 ms of inference+network delay
— a diffusion policy on an edge box is genuinely multi-step. A replan issued at step `t` observes
`o_t` but the chunk only arrives at `t+d`; `d = max(0, round(latency + jitter·N(0,1)))`, with an
optional `drop_prob` (chunk lost). Each strategy splices the late chunk differently.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# core machinery tests (no GPU/gym needed)
pytest test_latency_chunking.py -q

# the sweep (GPU recommended; --device cpu works but is slow)
python run_pusht.py \
  --strategies synchronous,naive_async,temporal_ensemble,rtc_freeze \
  --latencies 0,2,4,8,12 --jitter 1 --episodes 30 \
  --out results/pusht.csv --plot
```

## Expected result (the thing we're checking)

As injected latency grows you should see, qualitatively matching RTC:
- **synchronous** keeps success but throughput collapses (it pauses between chunks);
- **naive_async** loses success and gets jerky (chunk-boundary discontinuities);
- **temporal_ensemble** is smoother but reactivity-limited;
- **rtc_freeze** stays the most robust, gap widening with latency.

The pure-numpy core already reproduces exactly this ranking on the toy task (run the snippet in the
project root chat log, or `pytest` then inspect). The PushT run confirms it on a real policy.

## Version notes

`run_pusht.py` marks the 3 version-sensitive lines with `###`: the LeRobot policy import, the
action-chunk call (`predict_action_chunk` vs. older `diffusion.generate_actions`), and the
observation batch (keys `observation.image`/`observation.state`, 96×96 image). Check these against
your installed `lerobot` / `gym_pusht` versions — the rest is stable.

## Caveats

- Not the headline result — a 2-D pusher, not the robosuite manipulator. It validates the
  *phenomenon and the harness*, not the edge-network contribution.
- The reactive-layer idea isn't exercised here (PushT chunks are already dense at control rate);
  that's tested in the robosuite setting where the cognitive layer emits sparse EE targets.
