#!/bin/bash
# E4: does the AUTOCORRELATION of a real link's delay matter, at equal mean delay?
#
# Measured on a real 5 GHz link (scripts/wifi_trace.py, 2026-09-25, three 10-minute sessions with
# every sender at 100% of cadence and the probe cross-check clean):
#
#     one-way delay  p50 15-37 ms, p95 167-427 ms, min RTT 16-19 ms
#     loss           0.65-1.5%, in episodes of ~300-580 ms
#     lag-1 autocorrelation of delay   0.75-0.79
#     403 adjacent above-p95 samples where independent draws predict 30
#
# That last pair is the point. Every jitter model in the relay before this drew per message, so E2's
# equal-mean comparison between gaussian, uniform and lognormal found nothing -- it could not,
# because the property that distinguishes a real link from a synthetic one was not in the sweep.
# `jitter_model:=burst` reproduces it (lag-1 0.73-0.85 at these settings).
#
# The two conditions have the SAME mean one-way delay (34.8 vs 33.9 ms) and differ in structure:
#
#     burst      latency 16, jitter 300, bad_frac 0.06, burst 200 ms -> p50 16, p99 540, ac 0.79
#     lognormal  latency 16, jitter 18                              -> p50 27, p99 126, ac 0.00
#
# Loss is set to the measured 1% in 400 ms bursts in both, so only the delay structure varies.
# Reactive layer off: E3 measured it to do nothing. 16 cells, ~3 h.
#
#   docker run -d --name evh_paper1_realistic --gpus all --network host -e ROS_DOMAIN_ID=120 \
#     -v ~/edge-vla-hil:/ws edge-vla-hil:host bash /ws/scripts/paper1_realistic.sh
cd /ws || exit 1
source /opt/ros/humble/setup.bash
source /ws/ros2_ws/install/setup.bash

OUT=/ws/outputs/paper1
mkdir -p "$OUT"
STRATS=synchronous,naive_async,temporal_ensemble,rtc
COMMON="--backend dp --weights /ws/checkpoints/dp_square_ph_image_cnn.ckpt --absolute true \
  --env NutAssemblySquare --max_episode_s 20 --denoise_steps 4 \
  --trials ${TRIALS:-30} --duration 1800 --resume --log_dir $OUT/logs --image_quality 0 \
  --placement both --strategies $STRATS --reactive_modes off \
  --drop_prob 0.01 --loss_model gilbert --burst_ms 400"

for ex in policy robot; do
  # autocorrelated, as measured
  ros2 run evh_bringup benchmark $COMMON --executor $ex \
    --sweep latency --values 16 --jitter_ms 300 --jitter_model burst \
    --jitter_bad_frac 0.06 --jitter_burst_ms 200 --out "$OUT/e4_burst.csv"
  # independent draws at the same mean delay
  ros2 run evh_bringup benchmark $COMMON --executor $ex \
    --sweep latency --values 16 --jitter_ms 18 --jitter_model lognormal \
    --out "$OUT/e4_iid.csv"
done
echo "PAPER1_SWEEPS_DONE realistic"
