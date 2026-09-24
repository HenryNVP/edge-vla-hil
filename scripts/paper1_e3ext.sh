#!/bin/bash
# E3-extension: the reactive layer OFF at the action-path delays E3 did not reach.
#
# Why this is needed. E3 measured reactive on/off at 0 and 200 ms of action-path delay and found
# no effect anywhere EXCEPT one cell, where it was large and in the wrong direction: streamed
# synchronous execution at 200 ms scored 0.73 with the reactive layer off against 0.30 with it on
# (13 of 30 matched scenes, p = 0.0002). The layer adds lag of its own, and past some delay that
# lag stops being free.
#
# That reading matters beyond RQ3, because every E1 cell ran with the layer ON. At 200 ms the
# streamed-vs-buffered gap is 0.57 with the layer on and only 0.20 with it off, so part of what E1
# attributes to placement is attributable to the layer. E1 never took the layer off above 200 ms,
# so the size of that correction at 400 and 800 ms is currently unknown — and those are the cells
# the headline rests on.
#
# 16 cells, ~3 h. Same settings as paper1_sweeps.sh so the numbers are comparable; the
# reactive-ON counterparts are the matching E1 `placement act` cells.
#
#   docker run -d --name evh_paper1_e3ext --gpus all --network host -e ROS_DOMAIN_ID=120 \
#     -v ~/edge-vla-hil:/ws edge-vla-hil:host bash /ws/scripts/paper1_e3ext.sh
cd /ws || exit 1
source /opt/ros/humble/setup.bash
source /ws/ros2_ws/install/setup.bash

OUT=/ws/outputs/paper1
mkdir -p "$OUT"
STRATS=synchronous,naive_async,temporal_ensemble,rtc
TRIALS=${TRIALS:-30}
COMMON="--backend dp --weights /ws/checkpoints/dp_square_ph_image_cnn.ckpt --absolute true \
  --env NutAssemblySquare --max_episode_s 20 --denoise_steps 4 \
  --trials $TRIALS --duration 1800 --resume --log_dir $OUT/logs \
  --image_quality 0"

for ex in policy robot; do
  ros2 run evh_bringup benchmark $COMMON \
    --sweep latency --values 400,800 --placement act --executor $ex \
    --strategies $STRATS --reactive_modes off --out $OUT/e3ext.csv
done
echo "PAPER1_SWEEPS_DONE e3ext"
