"""Phase-align periodic timers to the wall clock, so two nodes' ticks keep a fixed offset.

The plant publishes observations at control_hz and the controller samples them at control_hz, on
two independent ROS timers. Their relative phase was whatever it happened to be at startup, so
the age of the observation the policy used at zero injected delay was anywhere from 0 to a full
50 ms period, differently in every launch; near the top of that range timer jitter also made the
two-frame observation history alternately repeat and skip a frame. Measured on Square: pilot cells
whose median observation age was over 40 ms averaged ~17% success, under 16 ms ~50%, with nothing
else different. That is a testbed artefact, and it varied from cell to cell like a hidden factor.

The fix is a shared grid: every periodic timer that matters starts on a boundary of the wall
clock (`t mod period == offset`). The plant publishes at offset 0; the controller (and the
robot-side executor) tick a few milliseconds later, so the newest observation is fresh and the
history spacing is exact. ROS timers keep their period from creation, so aligning the start is
enough on one host; across machines the grids agree only as well as the clocks do (chrony).

Duplicated in evh_plant and evh_controller (separate deployables); test_mode_crosscheck.py pins
that the two copies agree.
"""
from __future__ import annotations


def seconds_to_phase(now_s: float, period_s: float, offset_s: float = 0.0) -> float:
    """Seconds in [0, period_s) from `now_s` to the next instant t with (t - offset_s) % period_s
    == 0. Already on the grid (to within a microsecond) means start now."""
    if period_s <= 0.0:
        raise ValueError(f'period must be positive, got {period_s}')
    wait = (offset_s - now_s) % period_s
    return 0.0 if wait > period_s - 1e-6 else wait
