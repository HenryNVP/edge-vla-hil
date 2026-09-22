"""Tests for timer phase alignment — pure Python.

Two 20 Hz timers at a random relative phase made the observation the policy used anywhere from
0 to 50 ms old at zero injected delay, differently per launch; this is what pins it.
"""
import pytest

from evh_controller.phase import seconds_to_phase as controller_phase
from evh_plant.phase import seconds_to_phase as plant_phase


@pytest.mark.parametrize('now,period,offset,expected', [
    (100.000, 0.05, 0.0, 0.0),        # exactly on a boundary: start now
    (100.001, 0.05, 0.0, 0.049),
    (100.049, 0.05, 0.0, 0.001),
    (100.000, 0.05, 0.01, 0.01),
    (100.020, 0.05, 0.01, 0.04),
])
def test_waits_until_the_next_grid_instant(now, period, offset, expected):
    assert plant_phase(now, period, offset) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize('now', [0.0, 12.3456, 1790000000.123, 5.049999])
def test_the_wait_lands_on_the_grid_and_never_exceeds_a_period(now):
    period, offset = 0.05, 0.01
    wait = plant_phase(now, period, offset)
    assert 0.0 <= wait < period
    landed = (now + wait - offset) / period
    assert abs(landed - round(landed)) < 1e-6


def test_two_nodes_started_apart_end_up_a_fixed_offset_apart():
    """The point of it: plant at phase 0, controller at +10 ms, whatever their start times."""
    period = 0.05
    plant_start = 1000.0123
    controller_start = 1000.0371
    p = plant_start + plant_phase(plant_start, period, 0.0)
    c = controller_start + controller_phase(controller_start, period, 0.010)
    assert ((c - p) % period) == pytest.approx(0.010, abs=1e-9)


def test_a_non_positive_period_is_refused():
    with pytest.raises(ValueError):
        plant_phase(1.0, 0.0)
