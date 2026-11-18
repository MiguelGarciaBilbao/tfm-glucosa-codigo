"""Tests for the HUPA external validation runner (phase 8A.3)."""

import numpy as np
import pandas as pd

from tfm.data import hupa
from tfm.eval.metrics import Predictions
from tfm.training import external


def _summary():
    rows = [
        ("HUPA0001P", "CSII", "completo", 13.9),
        ("HUPA0017P", "MDI", "completo", 12.4),
        ("HUPA0011P", "CSII", "parcial", 13.3),
        ("HUPA0015P", "MDI", "sin_registro", 13.2),
        ("HUPA0027P", "MDI", "completo", 574.0),
    ]
    return pd.DataFrame(rows, columns=["patient", "treatment", "record_tier", "days"])


def test_long_follow_ups_leave_the_headline_group():
    g = external.patient_groups(_summary())
    assert g["principal"] == ["HUPA0001P", "HUPA0017P"]
    assert g["principal_CSII"] == ["HUPA0001P"] and g["principal_MDI"] == ["HUPA0017P"]
    assert g["largos"] == ["HUPA0027P"]
    assert g["parcial"] == ["HUPA0011P"] and g["sin_registro"] == ["HUPA0015P"]


def _libre_frame(values, start="2018-06-13 08:02"):
    ts = pd.date_range(start, periods=len(values), freq="15min")
    cgm = pd.DataFrame({"ts": ts, "value": np.asarray(values, dtype=float)})
    return hupa.feature_frame("HUPA0001P", {"glucose_level": cgm})


def test_events_are_split_by_onset():
    values = [100] * 10 + [60, 60] + [100] * 70 + [60, 60, 60] + [100] * 17
    ev = external.events_by_scope({"HUPA0001P": _libre_frame(values)})
    assert len(ev["all"]["HUPA0001P"]) == 2
    assert len(ev["adapt"]["HUPA0001P"]) == 1 and len(ev["test"]["HUPA0001P"]) == 1


def test_an_onset_no_window_reaches_is_not_scorable():
    ts = pd.date_range("2021-01-01 01:00", periods=4, freq="15min")
    preds = [Predictions(np.ones(4) * 100, np.ones(4) * 100, ts, "p", h) for h in (30, 60)]
    # Targets at 30 min cover 01:30-02:15; at 60 min, 02:00-02:45.
    events = pd.DataFrame({"start": pd.to_datetime(["2021-01-01 02:00",
                                                    "2021-01-01 01:30"])})
    kept = external.scorable(events, preds)
    assert kept["start"].tolist() == [pd.Timestamp("2021-01-01 02:00")]


def test_persistence_holds_the_last_reading_and_declares_the_cadence():
    values = 120 + 30 * np.sin(np.arange(40) / 4)
    ws = hupa.patient_windows(_libre_frame(values))
    by_h = external.persistence(ws)
    last = ws.X[:, -1, ws.feature_names.index("value")]
    for h, pred in by_h.items():
        np.testing.assert_allclose(pred.y_pred, last)
        assert pred.meta["episode_gap_min"] == hupa.ALARM_EPISODE_GAP_MIN
    assert sorted(by_h) == [30, 60]


def test_window_sets_keep_the_test_part_free_of_earlier_history():
    values = 120 + 30 * np.sin(np.arange(120) / 4)
    frame = _libre_frame(values)
    sets = external.window_sets({"HUPA0001P": frame})
    cut = hupa.split_cut(frame)
    test = sets["test"]["HUPA0001P"]
    history_start = pd.to_datetime(test.ts) - pd.Timedelta(minutes=5 * (test.X.shape[1] - 1))
    assert (history_start >= cut).all()
    assert len(sets["all"]["HUPA0001P"]) >= len(sets["adapt"]["HUPA0001P"]) + len(test)
