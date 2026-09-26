#!/bin/bash
# E6: is the collapse at long delay the DEADLINE, or a second failure mode?
#
# The duty-cycle law says blocking execution has no failure mode in this range -- it converts delay
# into time, at `completion = T0 / duty` with T0 = 7.49 +/- 0.25 s measured across six cells on two
# axes. Success rate is then just P(completion <= horizon), so what looks like a reliability cliff is
# the 20 s episode deadline cutting a continuous, predictable slowdown.
#
# E5 made that claim testable rather than rhetorical, by disagreeing with it once:
#
#     duty 0.306, reached by LENGTHENING the delay (d = 34, k = 15)  ->  success 0.00
#     duty 0.286, reached by SHORTENING the chunk  (d = 10, k =  4)  ->  success 0.21
#
# Two explanations fit. Either the deadline is doing the cutting, in which case the same cell with a
# longer deadline recovers; or long delay fails for a second reason that a short chunk does not share
# -- staleness -- in which case no deadline helps, and staleness has a hard limit independent of how
# much runway the buffer holds. The law predicts the first. Its residuals hint at the second (the
# model is pessimistic by 0.2-0.3 below duty 0.45, always on the horizon axis).
#
# So: hold the channel at 1600 ms (d ~ 34 steps, duty 0.306) and move ONLY the deadline.
#
#     max_episode_s = 20  ->  the matched control. E1's cell was recorded with the reactive layer
#                             ON and in a different sweep, so it is re-run here rather than compared
#                             across runs. Predicted: 0.00, reproducing E1.
#     max_episode_s = 40  ->  predicted median 7.49 / 0.306 = 24.5 s, p90 ~ 30 s, so the law says
#                             ~0.8. If it lands there, the cliff was the deadline.
#
# Reading the outcome:
#
#   blocking recovers            -> the cliff is the deadline. The sizing rule becomes arithmetic:
#                                   k > d * T0 / (horizon - T0), stated per task deadline.
#   blocking stays near zero     -> staleness has a hard limit the duty cycle does not capture, and
#                                   that is the more interesting result. It also means the law is a
#                                   rate/completion law only, and success needs the second term.
#   naive_async is the contrast  -> it pays in staleness (scales with d) rather than in time (1/duty),
#                                   so a longer deadline should NOT rescue it. If both recover the
#                                   two cost models are not separable; if only blocking does, the
#                                   E5 reversal's mechanism is confirmed from the other direction.
#   TE/RTC are a control         -> overlap is (15-34)/34 < 0, structurally zero. A deadline cannot
#                                   manufacture overlap, so these must stay 0.00 at both horizons.
#                                   If they recover, the overlap story is wrong.
#
# `max_episode_s` is a recorded column as of this sweep -- it is the independent variable here, and
# two cells differing only in it would otherwise be indistinguishable in the CSV.
#
# Buffered executor, reactive off, raw images: matches E5 and E3ext so the cells are comparable.
# 8 cells. The 20 s cells all time out (30 x 20 s ~ 10 min each); the 40 s cells run up to 20 min.
# Budget ~2.5 h.
#
#   docker run -d --name evh_paper1_deadline --gpus all --network host -e ROS_DOMAIN_ID=120 \
#     -v ~/edge-vla-hil:/ws edge-vla-hil:host bash /ws/scripts/paper1_deadline.sh
cd /ws || exit 1
source /opt/ros/humble/setup.bash
source /ws/ros2_ws/install/setup.bash

OUT=/ws/outputs/paper1
mkdir -p "$OUT"

# duration is the per-cell wall cap. 30 episodes x 40 s is 1200 s of episode time before resets,
# warmup and inference, so 1800 (the usual) would truncate the 40 s cells silently-ish; `truncated`
# is recorded, but a truncated cell is not the experiment.
COMMON="--backend dp --weights /ws/checkpoints/dp_square_ph_image_cnn.ckpt --absolute true \
  --env NutAssemblySquare --denoise_steps 4 \
  --trials ${TRIALS:-30} --duration 3000 --resume --log_dir $OUT/logs --image_quality 0 \
  --placement act --executor robot --reactive_modes off \
  --strategies synchronous,naive_async,temporal_ensemble,rtc"

for horizon in 20 40; do
  ros2 run evh_bringup benchmark $COMMON --max_episode_s "$horizon" \
    --sweep latency --values 1600 --out "$OUT/e6_deadline_h$horizon.csv"
done
echo "PAPER1_SWEEPS_DONE deadline"
