#!/bin/bash
# Does the compressed observation path cost task success?
#
# The graph can send JPEG frames instead of raw ones, because raw 84x84x3 frames saturate a real
# radio (scripts/wifi_trace.py). But JPEG is lossy: measured on 200 real Square frames, q90 is
# 2460 B against 21168 raw, at a mean absolute error of 1.81 grey levels. That is a change to what
# the POLICY consumes, not just to what the link carries, so it needs its own paired check before
# any measured cell uses it. Zero delay, buffered executor, reactive layer off (E3: it does
# nothing), same scenes on both sides.
#
#   docker run -d --name evh_jpeg --gpus all --network host -e ROS_DOMAIN_ID=120 \
#     -v ~/edge-vla-hil:/ws edge-vla-hil:host bash /ws/scripts/paper1_jpeg_check.sh
cd /ws || exit 1
source /opt/ros/humble/setup.bash
source /ws/ros2_ws/install/setup.bash

OUT=/ws/outputs/paper1
mkdir -p "$OUT"
COMMON="--backend dp --weights /ws/checkpoints/dp_square_ph_image_cnn.ckpt --absolute true \
  --env NutAssemblySquare --max_episode_s 20 --denoise_steps 4 \
  --trials ${TRIALS:-30} --duration 1800 --resume --log_dir $OUT/logs \
  --sweep latency --values 0 --placement act --executor robot \
  --strategies synchronous,rtc --reactive_modes off"

for q in 0 90; do
  ros2 run evh_bringup benchmark $COMMON --image_quality $q --out "$OUT/jpeg_q$q.csv"
done
echo "PAPER1_SWEEPS_DONE jpeg"
