"""Tests for the complementary analyses of phase 7."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm.eval.analysis import (
    AlarmTables,
    alarm_tables,
    bglp_metrics,
    coverage,
    is_night,
    lead_time_summary,
    master_row,
    period_breakdown,
    pick_examples,
)
from tfm.eval.metrics import Predictions
from tfm.viz.plots import plot_examples, plot_lead_times, plot_period_breakdown

START = np.datetime64("2027-03-01T00:00", "ns")


def _pred(patient, mu, sigma=5.0, n=288, y=None):
    ts = START + np.arange(n) * np.timedelta64(5, "m")
    y = np.full(n, 120.0) if y is None else y
    return Predictions(y, np.asarray(mu, dtype=float), ts, patient, 30,
                       sigma=np.full(n, sigma))


def _events(*starts, nadir=60.0):
    return pd.DataFrame({"start": pd.to_datetime(list(starts)),
                         "nadir": [nadir] * len(starts)})


def test_alarm_tables_list_every_event_and_every_false_alarm():
    mu = np.full(288, 120.0)
    mu[100:104] = 40.0                     # alarm whose targets are 08:50-09:05
    mu[200:202] = 40.0                     # alarm far from any event: false
    p = _pred("a", mu)
    ev = _events(START + np.timedelta64(9 * 60, "m"))   # onset at 09:00
    t = alarm_tables([p], {"a": ev}, alpha=0.5)
    assert list(t.events.columns) == ["patient", "start", "nadir", "detected", "lead_time_min"]
    assert t.events["detected"].tolist() == [True]
    assert t.events["lead_time_min"].iloc[0] == pytest.approx(9 * 60 - 100 * 5)
    assert len(t.false_alarms) == 1 and t.false_alarms["minutes"].iloc[0] == pytest.approx(10.0)
    assert t.span_days == pytest.approx(p.span_days)


def test_alarm_tables_survive_a_patient_without_events_or_alarms():
    t = alarm_tables([_pred("a", np.full(288, 120.0))], {"a": _events()}, alpha=0.5)
    assert t.events.empty and t.false_alarms.empty


def test_lead_time_summary_counts_late_alarms():
    events = pd.DataFrame({"detected": [True, True, True, False],
                           "lead_time_min": [30.0, 10.0, -5.0, np.nan]})
    s = lead_time_summary(events)
    assert s["n_detected"] == 3 and s["median"] == pytest.approx(10.0)
    assert s["n_negative"] == 1 and s["min"] == pytest.approx(-5.0)


def test_six_in_the_morning_is_already_day():
    ts = pd.to_datetime(["2027-03-01 05:55", "2027-03-01 06:00", "2027-03-01 00:00"])
    assert is_night(ts).tolist() == [True, False, True]


def test_period_breakdown_splits_events_and_adds_false_alarms_up_to_the_total():
    events = pd.DataFrame({
        "patient": ["a"] * 3,
        "start": pd.to_datetime(["2027-03-01 02:00", "2027-03-01 03:00", "2027-03-01 14:00"]),
        "nadir": [60.0] * 3, "detected": [True, False, True],
        "lead_time_min": [20.0, np.nan, 25.0]})
    false = pd.DataFrame({"patient": ["a", "a"],
                          "start": pd.to_datetime(["2027-03-01 01:00", "2027-03-01 13:00"]),
                          "end": pd.to_datetime(["2027-03-01 01:10", "2027-03-01 13:05"]),
                          "minutes": [15.0, 10.0]})
    y = np.full(288, 120.0)
    mu = y.copy()
    mu[:66] += 10.0                         # targets 00:30-05:55 are off by 10 (night)
    p = _pred("a", mu, y=y)
    table = period_breakdown(AlarmTables(events, false, span_days=2.0), [p])
    night, day = table.loc["noche"], table.loc["día"]
    assert night["n_events"] == 2 and night["n_detected"] == 1
    assert day["n_events"] == 1 and day["recall"] == pytest.approx(1.0)
    assert night["false_alarms_per_day"] + day["false_alarms_per_day"] == pytest.approx(1.0)
    assert night["rmse"] > day["rmse"]


def _frame(rows):
    return pd.DataFrame(rows, columns=["seed", "patient", "horizon_min", "ts", "y_true", "y_pred"])


def test_bglp_metrics_average_the_patients_then_the_seeds():
    t = pd.Timestamp("2027-03-01")
    rows = []
    for seed, err in ((0, 3.0), (1, 5.0)):
        for patient, scale in (("540", 1.0), ("544", 2.0)):
            for h in (30, 60):
                rows += [(seed, patient, h, t, 100.0, 100.0 + scale * err),
                         (seed, patient, h, t, 100.0, 100.0 - scale * err)]
    rows.append((0, "559", 30, t, 100.0, 900.0))       # 2018 patient: ignored
    out = bglp_metrics(_frame(rows))
    # seed 0: per-patient RMSE 3 and 6 -> 4.5; seed 1: 5 and 10 -> 7.5; mean 6.0
    assert out["rmse_30"] == pytest.approx(6.0) and out["mae_60"] == pytest.approx(6.0)
    assert out["sum"] == pytest.approx(24.0) and out["n_seeds"] == 2
    assert out["std"]["rmse_30"] == pytest.approx(np.std([4.5, 7.5], ddof=1))


def test_bglp_metrics_accept_frames_without_seed():
    t = pd.Timestamp("2027-03-01")
    frame = _frame([(0, "540", h, t, 100.0, 104.0) for h in (30, 60)]).drop(columns="seed")
    out = bglp_metrics(frame)
    assert out["rmse_30"] == pytest.approx(4.0) and out["n_seeds"] == 1


def test_coverage_is_the_share_of_official_instants_predicted():
    official = {"540": pd.date_range("2027-03-01 00:30", periods=4, freq="5min").to_numpy()}
    ts = pd.date_range("2027-03-01 00:00", periods=3, freq="5min")   # targets 00:30-00:40
    frame = pd.DataFrame({"patient": "540", "horizon_min": 30, "ts": ts,
                          "y_true": 100.0, "y_pred": 100.0})
    assert coverage(frame, official, 30) == pytest.approx(0.75)


def test_examples_follow_the_rule_and_break_ties_by_time():
    t = pd.to_datetime(["2027-03-01 01:00", "2027-03-01 02:00", "2027-03-01 03:00",
                        "2027-03-01 04:00", "2027-03-01 05:00"])
    events = pd.DataFrame({"patient": "a", "start": t,
                           "nadir": [60.0, 55.0, 50.0, 45.0, 45.0],
                           "detected": [True, True, True, False, False],
                           "lead_time_min": [20.0, 20.0, 35.0, np.nan, np.nan]})
    false = pd.DataFrame({"patient": ["a", "a"], "start": t[:2], "end": t[:2],
                          "minutes": [30.0, 30.0]})
    picks = pick_examples(AlarmTables(events, false, 1.0))
    assert picks["median_lead"]["start"] == t[0]      # leads 20, 20, 35: lower median, earliest
    assert picks["max_lead"]["start"] == t[2]
    assert picks["worst_miss"]["start"] == t[3]       # nadir tie at 45: earliest
    assert picks["longest_false_alarm"]["start"] == t[0]


def test_examples_are_none_when_there_is_nothing_to_show():
    empty = AlarmTables(pd.DataFrame(columns=["patient", "start", "nadir", "detected",
                                              "lead_time_min"]),
                        pd.DataFrame(columns=["patient", "start", "end", "minutes"]), 1.0)
    assert all(v is None for v in pick_examples(empty).values())


def _result_single_run():
    per_patient = [{"patient": "a", "mae": 10.0, "grmse": 20.0},
                   {"patient": "b", "mae": 14.0, "grmse": 30.0}]
    summary = {"rmse_pooled": 18.0, "clarke_AB_pooled": 99.0, "parkes_DE_pooled": 0.1,
               "pooled_event_recall": {"point": 0.2, "lo": 0.1, "hi": 0.3, "n": 61}}
    return {"by_horizon": {"30": {"summary": summary, "per_patient": per_patient}}}


def test_master_row_pairs_error_and_clinical_metrics():
    alarm = {"recall": {"point": 0.6, "lo": 0.5, "hi": 0.7, "n": 61},
             "false_alarms_per_day": 0.8, "median_lead_time_min": 25.0}
    row = master_row("Modelo", _result_single_run(), 30, alarm)
    assert row["rmse"] == pytest.approx(18.0)
    assert row["mae_media_pacientes"] == pytest.approx(12.0)
    assert row["recall_mu70"] == pytest.approx(0.2)
    assert row["recall_alarma"] == pytest.approx(0.6) and row["n_eventos"] == 61
    assert np.isnan(master_row("Sin alarma", _result_single_run(), 30)["recall_alarma"])


def _tables():
    events = pd.DataFrame({"patient": ["a", "a"],
                           "start": pd.to_datetime(["2027-03-01 02:00", "2027-03-01 14:00"]),
                           "nadir": [60.0, 55.0], "detected": [True, False],
                           "lead_time_min": [20.0, np.nan]})
    false = pd.DataFrame({"patient": ["a"], "start": pd.to_datetime(["2027-03-01 10:00"]),
                          "end": pd.to_datetime(["2027-03-01 10:10"]), "minutes": [15.0]})
    return AlarmTables(events, false, 1.0)


def test_lead_time_figure_has_one_panel_per_horizon():
    fig = plot_lead_times({30: _tables(), 60: _tables()})
    assert len(fig.axes) == 2


def test_period_figure_shows_recall_and_false_alarms():
    t = pd.DataFrame({"n_events": [2, 1], "n_detected": [1, 1], "recall": [0.5, 1.0],
                      "recall_lo": [0.0, 1.0], "recall_hi": [1.0, 1.0],
                      "false_alarms_per_day": [0.4, 0.6], "rmse": [20.0, 18.0]},
                     index=["noche", "día"])
    fig = plot_period_breakdown({30: t, 60: t})
    assert len(fig.axes) == 2


def test_example_figure_draws_one_panel_per_pick_and_skips_missing_ones():
    ts = pd.date_range("2027-03-01", periods=288, freq="5min")
    frame = pd.DataFrame({"patient": "a", "horizon_min": 30, "ts": ts,
                          "y_true": 110.0, "y_pred": 100.0, "sigma": 15.0})
    t = _tables()
    picks = {"median_lead": t.events.iloc[0], "max_lead": t.events.iloc[0],
             "worst_miss": t.events.iloc[1], "longest_false_alarm": None}
    fig = plot_examples(frame, picks, alpha=0.3, horizon_min=30)
    titled = [ax for ax in fig.axes if ax.get_title()]
    assert len(titled) == 3


def test_coverage_snaps_official_instants_to_the_grid_like_the_preprocessing():
    # raw sensor timestamps drift by seconds; the windows live on the 5 min grid
    official = {"540": pd.to_datetime(["2027-03-01 00:30:44", "2027-03-01 00:35:10",
                                       "2027-03-01 00:39:50", "2027-03-01 00:45:02"]).to_numpy()}
    ts = pd.date_range("2027-03-01 00:00", periods=3, freq="5min")   # targets 00:30-00:40
    frame = pd.DataFrame({"patient": "540", "horizon_min": 30, "ts": ts,
                          "y_true": 100.0, "y_pred": 100.0})
    assert coverage(frame, official, 30) == pytest.approx(0.75)


def test_example_figure_legend_names_the_probability_and_the_threshold():
    ts = pd.date_range("2027-03-01", periods=288, freq="5min")
    frame = pd.DataFrame({"patient": "a", "horizon_min": 30, "ts": ts,
                          "y_true": 110.0, "y_pred": 100.0, "sigma": 15.0})
    t = _tables()
    picks = {"median_lead": t.events.iloc[0], "max_lead": None, "worst_miss": None,
             "longest_false_alarm": None}
    fig = plot_examples(frame, picks, alpha=0.3, horizon_min=30)
    labels = {tx.get_text() for leg in fig.legends for tx in leg.get_texts()}
    assert {"CGM", "P(y < 70)", "α = 0,3"} <= labels


def test_false_alarm_marker_sits_at_the_predicted_instant_not_when_the_alarm_sounds():
    ts = pd.date_range("2027-03-01", periods=288, freq="5min")
    frame = pd.DataFrame({"patient": "a", "horizon_min": 30, "ts": ts,
                          "y_true": 110.0, "y_pred": 100.0, "sigma": 15.0})
    t = _tables()                                   # false alarm sounds at 10:00
    picks = {"median_lead": None, "max_lead": None, "worst_miss": None,
             "longest_false_alarm": t.false_alarms.iloc[0]}
    fig = plot_examples(frame, picks, alpha=0.3, horizon_min=30)
    ax = next(a for a in fig.axes if a.get_title())
    marks = [pd.Timestamp(ln.get_xdata()[0]) for ln in ax.lines
             if ln.get_linestyle() == ":" and not isinstance(ln.get_xdata()[0], (int, float))]
    assert marks == [pd.Timestamp("2027-03-01 10:30")]


def test_operating_curve_figure_marks_the_alpha_chosen_on_validation():
    from tfm.viz.plots import plot_operating_curve

    curve = pd.DataFrame({"alpha": [0.2, 0.3, 0.4], "recall": [0.9, 0.8, 0.4],
                          "recall_lo": [0.8, 0.7, 0.3], "recall_hi": [1.0, 0.9, 0.5],
                          "false_alarms_per_day": [1.6, 0.9, 0.6]})
    fig = plot_operating_curve({30: curve, 60: curve}, {30: 0.3, 60: 0.4})
    assert len(fig.axes) == 2
    marks = [(ln.get_xdata()[0], ln.get_ydata()[0]) for ln in fig.axes[1].lines
             if ln.get_marker() == "*"]
    assert marks == [(0.6, 0.4)]
