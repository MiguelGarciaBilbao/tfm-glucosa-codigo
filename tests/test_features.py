"""Tests for insulin/carbohydrate reconstruction and decay curves."""

import numpy as np
import pandas as pd
import pytest

from tfm.data import features as f


def test_insulin_curve_endpoints():
    assert f.insulin_remaining_fraction(np.array([0.0]))[0] == pytest.approx(1.0)
    assert f.insulin_remaining_fraction(
        np.array([f.INSULIN_DURATION_MIN])
    )[0] == pytest.approx(0.0, abs=1e-6)


def test_insulin_curve_is_monotonically_decreasing():
    t = np.arange(0, f.INSULIN_DURATION_MIN + 1, 5.0)
    iob = f.insulin_remaining_fraction(t)
    assert np.all(np.diff(iob) <= 1e-9)


def test_insulin_is_inactive_before_delivery_and_after_duration():
    assert f.insulin_remaining_fraction(np.array([-10.0]))[0] == 0.0
    assert f.insulin_remaining_fraction(np.array([400.0]))[0] == 0.0


def test_half_the_bolus_remains_around_the_expected_time():
    t = np.arange(0, 301, 1.0)
    iob = f.insulin_remaining_fraction(t)
    half_life = t[np.argmin(np.abs(iob - 0.5))]
    assert 90 <= half_life <= 150


def test_carb_absorption_is_linear_and_bounded():
    assert f.carbs_remaining_fraction(np.array([0.0]))[0] == pytest.approx(1.0)
    assert f.carbs_remaining_fraction(
        np.array([f.CARB_ABSORPTION_MIN / 2])
    )[0] == pytest.approx(0.5)
    assert f.carbs_remaining_fraction(np.array([1000.0]))[0] == 0.0


def test_rate_of_change_uses_a_fifteen_minute_window():
    s = pd.Series([100.0, 105, 110, 115, 120])
    roc = f.rate_of_change(s, minutes=15)
    # 115 - 100 over 15 minutes = 1 mg/dL per minute.
    assert roc.iloc[3] == pytest.approx(1.0)
    assert roc.iloc[:3].isna().sum() == 3


def test_decay_on_grid_places_a_dose_and_decays_it():
    grid = pd.date_range("2021-01-01", periods=12, freq="5min")
    events = pd.DataFrame({"ts": [grid[2]], "dose": [4.0]})
    out = f._decay_on_grid(grid, events, "ts", "dose", f.insulin_remaining_fraction, 300)
    assert out[:2].sum() == 0.0        # nothing before the bolus
    assert out[2] == pytest.approx(4.0)  # full dose on board at delivery
    assert out[3] < out[2]               # and decaying afterwards


def test_two_boluses_accumulate():
    grid = pd.date_range("2021-01-01", periods=24, freq="5min")
    events = pd.DataFrame({"ts": [grid[0], grid[1]], "dose": [2.0, 3.0]})
    out = f._decay_on_grid(grid, events, "ts", "dose", f.insulin_remaining_fraction, 300)
    assert out[1] > 4.5  # both doses on board, minus a little decay of the first
