#!/bin/bash
# E5: is the design rule really a RATIO?
#
# E1 found that buffered execution survives while the chunk outlasts the round trip, and that
# overlap-based methods (temporal ensembling, RTC) need roughly twice that. But every E1 cell used
# the checkpoint's full 15-action chunk, so 750 ms of horizon and "800 ms of delay" are the same
# number in that data: the ratio is inferred from the delay axis alone.
#
# This varies the DENOMINATOR at fixed delay. `max_chunk_actions` truncates every chunk to k
# actions, so the horizon is 50*k ms at 20 Hz:
#
#     k = 15 -> 750 ms   (the checkpoint default, E1's implicit horizon)
#     k = 10 -> 500 ms
#     k =  6 -> 300 ms
#     k =  4 -> 200 ms
#
# At a fixed 400 ms of action-path delay the round trip is ~10 control steps (500 ms), so the rule
# predicts k=15 works, k=10 is marginal, and k=6 and k=4 fail -- while plain "delay" is unchanged.
# If instead success tracks delay and ignores k, the ratio framing is wrong and E1's rule should be
# restated in absolute terms.
#
# Buffered executor only (the streamed one has no buffer to size), reactive layer off (E3), raw
# images to match the frozen cells. 16 cells, ~3 h.
#
#   docker run -d --name evh_paper1_horizon --gpus all --network host -e ROS_DOMAIN_ID=120 \
#     -v ~/edge-vla-hil:/ws edge-vla-hil:host bash /ws/scripts/paper1_horizon.sh
cd /ws || exit 1
source /opt/ros/humble/setup.bash
source /ws/ros2_ws/install/setup.bash

OUT=/ws/outputs/paper1
mkdir -p "$OUT"
COMMON="--backend dp --weights /ws/checkpoints/dp_square_ph_image_cnn.ckpt --absolute true \
  --env NutAssemblySquare --max_episode_s 20 --denoise_steps 4 \
  --trials ${TRIALS:-30} --duration 1800 --resume --log_dir $OUT/logs --image_quality 0 \
  --placement act --executor robot --reactive_modes off \
  --strategies synchronous,naive_async,temporal_ensemble,rtc"

for k in 15 10 6 4; do
  ros2 run evh_bringup benchmark $COMMON --max_chunk_actions $k \
    --sweep latency --values 400 --out "$OUT/e5_horizon_k$k.csv"
done
echo "PAPER1_SWEEPS_DONE horizon"
