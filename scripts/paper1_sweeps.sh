#!/bin/bash
# Paper 1 main sweeps: E1 (delay placement x executor placement), E2 (channel shape), E3 (the
# reactive layer). Cells were frozen on 2026-09-22 from the week-3 pilots, before any main run:
#
#   * Square in the loop at zero delay: synchronous 25/30, RTC 26/30, temporal ensembling 16/20
#     (co-located 85/100), after the timing fixes in CLAUDE.md invariants 9-11.
#   * Streamed action-path delay collapses success between 0 and 200 ms, so E1 resolves 50 and
#     100 ms; robot-side execution and observation-path delay hold to 800 ms, so E1 reaches 1600.
#   * 15% and even 50% loss in 2 s outages cost little success at zero delay (the arm holds its
#     last absolute target and the 20 s horizon absorbs the lost time): E2 keeps 15% and 50%
#     and reads time-to-success from the per-episode log, not only the success rate.
#
# Every cell: Square, DP at 4 DDIM steps, reactive layer on unless E3 says off, and TRIALS
# episodes (default 30). 30 is enough for the large effects BECAUSE comparisons are paired:
# scene k is the same scene in every cell. Cells whose interval still straddles a conclusion get
# topped up afterwards with a second pass into <name>_topup.csv (same labels, pooled in analysis;
# --resume would otherwise skip them as done).
# Scene k is identical in every cell (same plant seed), so cells compare episode by episode
# via <out>.episodes.csv. --resume makes the whole script restartable after an interruption.
#
# Run inside the host image, one run at a time on the GPU (parallel runs change inference time):
#   docker run -d --name evh_paper1 --gpus all --network host -e ROS_DOMAIN_ID=120 \
#     -v ~/edge-vla-hil:/ws edge-vla-hil:host bash /ws/scripts/paper1_sweeps.sh [e1|e2|e3|all]
cd /ws || exit 1
source /opt/ros/humble/setup.bash
source /ws/ros2_ws/install/setup.bash

WHICH=${1:-all}
OUT=/ws/outputs/paper1
mkdir -p "$OUT"
STRATS=synchronous,naive_async,temporal_ensemble,rtc
TRIALS=${TRIALS:-30}
COMMON="--backend dp --weights /ws/checkpoints/dp_square_ph_image_cnn.ckpt --absolute true \
  --env NutAssemblySquare --max_episode_s 20 --denoise_steps 4 \
  --trials $TRIALS --duration 1800 --resume --log_dir $OUT/logs"

sweep() { ros2 run evh_bringup benchmark $COMMON "$@"; }

# ---------------------------------------------------------------- E1 (RQ1): 104 cells
# 1600 ms only for the buffered executor: streamed execution is already at 0.0 by 800 ms, so the
# extra level would buy nothing but 8 cells of timeouts
if [ "$WHICH" = e1 ] || [ "$WHICH" = all ]; then
  for pl in act obs; do
    sweep --sweep latency --values 0,50,100,200,400,800 --placement $pl --executor policy \
      --strategies $STRATS --reactive_modes on --out $OUT/e1.csv
    sweep --sweep latency --values 0,50,100,200,400,800,1600 --placement $pl --executor robot \
      --strategies $STRATS --reactive_modes on --out $OUT/e1.csv
  done
fi

# ---------------------------------------------------------------- E2 (RQ2): 64 cells
if [ "$WHICH" = e2 ] || [ "$WHICH" = all ]; then
  for ex in policy robot; do
    # equal mean one-way delay (100 ms, both paths), different tails. network_aware is NOT here:
    # its quantile forecast is paper 2's method, and only a diagnostic for this one
    sweep --sweep latency --values 100 --placement both --executor $ex \
      --strategies $STRATS --reactive_modes on --out $OUT/e2_jitter.csv
    sweep --sweep latency --values 100 --jitter_ms 30 --jitter_model gaussian --placement both \
      --executor $ex --strategies $STRATS --reactive_modes on --out $OUT/e2_jitter.csv
    sweep --sweep latency --values 70 --jitter_ms 30 --jitter_model lognormal --placement both \
      --executor $ex --strategies $STRATS --reactive_modes on --out $OUT/e2_jitter.csv
    sweep --sweep latency --values 50 --jitter_ms 50 --jitter_model lognormal --placement both \
      --executor $ex --strategies $STRATS --reactive_modes on --out $OUT/e2_jitter.csv
    # equal average loss, different burstiness. 50% only: 15% and 50% both left success intact at
    # zero delay in the pilot, so the cheaper half of the axis had nothing to separate
    sweep --sweep drop --values 0.5 --loss_model iid --placement both --executor $ex \
      --strategies $STRATS --reactive_modes on --out $OUT/e2_loss.csv
    for burst in 100 500 2000; do
      sweep --sweep drop --values 0.5 --loss_model gilbert --burst_ms $burst --placement both \
        --executor $ex --strategies $STRATS --reactive_modes on --out $OUT/e2_loss.csv
    done
  done
fi

# ---------------------------------------------------------------- E3 (RQ3): 40 cells
# reactive layer OFF at five channels; the reactive-on counterparts are the matching E1/E2 cells
if [ "$WHICH" = e3 ] || [ "$WHICH" = all ]; then
  for ex in policy robot; do
    sweep --sweep latency --values 0,200 --placement act --executor $ex --strategies $STRATS \
      --reactive_modes off --out $OUT/e3.csv
    sweep --sweep latency --values 800 --placement obs --executor $ex --strategies $STRATS \
      --reactive_modes off --out $OUT/e3.csv
    sweep --sweep latency --values 50 --jitter_ms 50 --jitter_model lognormal --placement both \
      --executor $ex --strategies $STRATS --reactive_modes off --out $OUT/e3.csv
    sweep --sweep drop --values 0.5 --loss_model gilbert --burst_ms 500 --placement both \
      --executor $ex --strategies $STRATS --reactive_modes off --out $OUT/e3.csv
  done
fi
echo "PAPER1_SWEEPS_DONE $WHICH"
