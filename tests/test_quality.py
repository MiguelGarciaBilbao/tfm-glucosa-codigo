"""Tests for gap handling and event detection.
"""

import numpy as np
import pandas as pd
import pytest

from tfm.data import quality as q


def series(values, start="2021-01-01 00:00", freq_min=5):
    ts = pd.date_range(start, periods=len(values), freq=f"{freq_min}min")
    return pd.DataFrame({"ts": ts, "value": np.asarray(values, dtype=float)})


def test_regular_grid_exposes_a_missing_block_as_nan():
    df = pd.concat([series([100] * 6), series([100] * 6, start="2021-01-01 01:00")])
    grid = q.to_regular_grid(df)
    assert len(grid) == 18
    assert grid["value"].isna().sum() == 6


def test_short_gaps_are_interpolated_and_flagged():
    df = pd.concat([series([100, 100]), series([160, 160], start="2021-01-01 00:20")])
    out = q.add_segments(df)
    assert out["value"].notna().all()
    assert out["is_imputed"].sum() == 2
    assert out["segment"].nunique() == 1


def test_long_gaps_split_the_series_instead_of_being_filled():
    df = pd.concat([series([100] * 3), series([120] * 3, start="2021-01-01 01:05")])
    out = q.add_segments(df)
    assert out["value"].isna().sum() == 10
    assert out.dropna(subset=["value"])["segment"].nunique() == 2


def test_gap_table_reports_duration_in_minutes():
    df = pd.concat([series([100] * 3), series([100] * 3, start="2021-01-01 01:05")])
    gaps = q.gap_table(df)
    assert len(gaps) == 1
    assert gaps.loc[0, "minutes"] == 50


def test_short_dips_are_not_counted_as_events():
    df = series([100, 65, 65, 100, 100])
    assert q.glycemic_events(df).empty


def test_event_is_detected_with_nadir_and_duration():
    df = series([100, 65, 60, 55, 100])
    ev = q.glycemic_events(df)
    assert len(ev) == 1
    assert ev.loc[0, "minutes"] == 15
    assert ev.loc[0, "nadir"] == 55


def test_consecutive_samples_form_one_event_not_many():
    """The whole point of event-level counting."""
    df = series([100] + [60] * 12 + [100])
    ev = q.glycemic_events(df)
    assert len(ev) == 1
    assert ev.loc[0, "n_samples"] == 12


def test_separate_episodes_are_counted_separately():
    df = series([100] + [60] * 4 + [100] * 6 + [60] * 4 + [100])
    assert len(q.glycemic_events(df)) == 2


def test_time_in_range_percentages():
    df = series([50, 100, 100, 200])
    tir = q.time_in_range(df)
    assert tir["pct_hypo"] == pytest.approx(25.0)
    assert tir["pct_severe_hypo"] == pytest.approx(25.0)
    assert tir["pct_in_range"] == pytest.approx(50.0)
    assert tir["pct_hyper"] == pytest.approx(25.0)


def test_circadian_profile_uses_hour_of_day():
    df = series([100] * 24, freq_min=60)
    prof = q.circadian_profile(df)
    assert len(prof) == 24
    assert set(prof["hour"]) == set(range(24))


def test_oversized_gap_is_not_partially_filled():
    df = pd.concat([series([100] * 3), series([120] * 3, start="2021-01-01 01:05")])
    out = q.add_segments(df)
    assert not out["is_imputed"].any()
    assert out["value"].isna().sum() == 10


def test_a_gap_exactly_at_the_limit_is_still_filled():
    df = pd.concat([series([100] * 2), series([160] * 2, start="2021-01-01 00:40")])
    out = q.add_segments(df)
    assert out["is_imputed"].sum() == 6
    assert out["value"].notna().all()


def test_implausible_jumps_are_flagged():
    df = q.to_regular_grid(series([100, 105, 210, 215]))
    flags = q.flag_implausible_jumps(df)
    assert list(flags) == [False, False, True, False]


def test_normal_physiological_changes_are_not_flagged():
    df = q.to_regular_grid(series([100, 120, 140, 160]))
    assert not q.flag_implausible_jumps(df).any()
