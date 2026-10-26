"""Tests for windowing, scaling and the temporal split.
"""

import numpy as np
import pandas as pd
import pytest

from tfm.data import windows as w


def frame(n_segments=1, per_segment=40, start="2021-01-01"):
    rows = []
    ts = pd.Timestamp(start)
    for seg in range(n_segments):
        idx = pd.date_range(ts, periods=per_segment, freq="5min")
        rows.append(pd.DataFrame({
            "ts": idx,
            "value": np.arange(per_segment, dtype=float) + 100 * seg,
            "feat": np.ones(per_segment),
            "segment": seg,
        }))
        ts = idx[-1] + pd.Timedelta(hours=6)  # a long gap between segments
    return pd.concat(rows, ignore_index=True)


def test_window_count_accounts_for_history_and_horizon():
    df = frame(per_segment=40)
    ws = w.make_windows(df, ["value", "feat"], "999", "train", history=12, horizons=(6, 12))
    # 40 samples, 12 of history, 12 of horizon -> 40 - 12 - 12 + 1 windows.
    assert len(ws) == 17
    assert ws.X.shape == (17, 12, 2)
    assert ws.y.shape == (17, 2)


def test_targets_are_taken_at_the_right_horizons():
    df = frame(per_segment=40)
    ws = w.make_windows(df, ["value"], "999", "train", history=12, horizons=(6, 12))
    # First window: history covers samples 0-11, so the last observed value is 11.
    assert ws.X[0, -1, 0] == pytest.approx(11.0)
    assert ws.y[0, 0] == pytest.approx(17.0)  # +6 steps
    assert ws.y[0, 1] == pytest.approx(23.0)  # +12 steps


def test_horizon_minutes_are_reported_in_minutes():
    ws = w.make_windows(frame(), ["value"], "999", "train", horizons=(6, 12))
    assert ws.horizon_minutes == (30, 60)


def test_no_window_spans_a_segment_boundary():
    """The invariant that matters most: a window must never cross a dropout."""
    df = frame(n_segments=2, per_segment=40)
    ws = w.make_windows(df, ["value"], "999", "train", history=12, horizons=(12,))
    assert len(ws) == 2 * 17
    # Values are offset by 100 per segment, so a window mixing segments would
    # show a jump of ~100 inside its own history.
    jumps = np.abs(np.diff(ws.X[:, :, 0], axis=1)).max()
    assert jumps < 50


def test_short_segments_produce_no_windows():
    df = frame(per_segment=10)  # shorter than history + horizon
    assert len(w.make_windows(df, ["value"], "999", "train")) == 0


def test_temporal_split_is_chronological_not_random():
    ws = w.make_windows(frame(per_segment=100), ["value"], "999", "train")
    tr, val = w.temporal_split(ws, val_fraction=0.2)
    assert len(tr) + len(val) == len(ws)
    # Every training window must precede every validation window.
    assert tr.ts.max() < val.ts.min()


def test_no_timestamp_appears_in_both_split_sides():
    ws = w.make_windows(frame(per_segment=100), ["value"], "999", "train")
    tr, val = w.temporal_split(ws)
    assert not set(tr.ts.tolist()) & set(val.ts.tolist())


def test_scaler_standardises_and_is_fitted_on_train_only():
    ws = w.make_windows(frame(per_segment=100), ["value", "feat"], "999", "train")
    tr, val = w.temporal_split(ws)
    scaler = w.WindowScaler().fit(tr)
    scaled_tr = scaler.transform(tr)
    flat = scaled_tr.X.reshape(-1, scaled_tr.X.shape[-1])
    assert flat[:, 0].mean() == pytest.approx(0.0, abs=1e-5)
    assert flat[:, 0].std() == pytest.approx(1.0, abs=1e-5)
    # Validation is scaled with the training statistics, so it need not be
    # centred; if it were, the scaler would have seen it.
    scaled_val = scaler.transform(val)
    assert abs(scaled_val.X[:, :, 0].mean()) > 0.1


def test_constant_feature_does_not_divide_by_zero():
    ws = w.make_windows(frame(), ["value", "feat"], "999", "train")
    scaled = w.WindowScaler().fit_transform(ws)
    assert np.isfinite(scaled.X).all()


def test_scaler_rejects_a_different_feature_order():
    ws = w.make_windows(frame(), ["value", "feat"], "999", "train")
    scaler = w.WindowScaler().fit(ws)
    other = w.make_windows(frame(), ["feat", "value"], "999", "test")
    with pytest.raises(ValueError, match="Feature order"):
        scaler.transform(other)


def test_concat_tracks_which_patient_each_window_came_from():
    a = w.make_windows(frame(per_segment=40), ["value"], "559", "train")
    b = w.make_windows(frame(per_segment=40), ["value"], "563", "train")
    pooled = w.concat_windows([a, b])
    assert len(pooled) == len(a) + len(b)
    idx = w.window_patient_index(pooled)
    assert (idx[: len(a)] == 0).all()
    assert (idx[len(a) :] == 1).all()


def test_concat_rejects_incompatible_feature_sets():
    a = w.make_windows(frame(), ["value"], "559", "train")
    b = w.make_windows(frame(), ["value", "feat"], "563", "train")
    with pytest.raises(ValueError, match="Incompatible"):
        w.concat_windows([a, b])
