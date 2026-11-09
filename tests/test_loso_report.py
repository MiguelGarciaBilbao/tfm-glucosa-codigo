"""Tests for the phase 5 report: Holm, the per-patient tables, the tests and the figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from tfm.eval.compare import (
    adaptation_curve_frame,
    holm,
    loso_patient_table,
    personalization_tests,
    reference_lines,
)
from tfm.viz.plots import REGIME_LABELS, plot_adaptation_curves

PATIENTS = ["p1", "p2", "p3", "p4", "p5"]


def fake_result(regime, days, rmse_shift=0.0, detect=1):
    by_days = {}
    for n in days:
        by_seed, alarm, aggregate = {}, {}, {}
        for h in ("30", "60"):
            per_seed = {}
            for s in ("0", "1"):
                by_seed.setdefault(s, {})[h] = {
                    "summary": {"rmse_pooled": 20.0 + rmse_shift},
                    "per_patient": [{"patient": p, "rmse": 20.0 + i + rmse_shift - n}
                                    for i, p in enumerate(PATIENTS)]}
                per_seed[s] = [{"patient": p, "alpha": 0.3, "n_events": 2 if p != "p5" else 0,
                                "n_detected": min(detect, 2) if p != "p5" else 0,
                                "false_alarm_episodes": 1, "span_days": 10.0} for p in PATIENTS]
            alarm[h] = {"recall": {"point": detect / 2, "lo": 0.0, "hi": 1.0, "n": 8},
                        "recall_by_seed": [detect / 2] * 2,
                        "false_alarms_per_day_by_seed": [0.5, 0.7],
                        "per_patient_by_seed": per_seed}
            aggregate[h] = {"rmse_pooled": {"mean": 20.0 + rmse_shift - n, "std": 0.1}}
        by_days[str(n)] = {"by_seed": by_seed, "alarm": alarm, "aggregate": aggregate}
    return {"regime": regime, "n_days": list(days), "by_days": by_days}


RESULTS = {"global": fake_result("global", [0]),
           "individual": fake_result("individual", [1, 3, 7], rmse_shift=5.0),
           "ft": fake_result("ft", [1, 3, 7], detect=2),
           "ft_weighted": fake_result("ft_weighted", [1, 3, 7], rmse_shift=1.0, detect=2)}


def test_holm_steps_down_and_keeps_nans():
    adj = holm([0.01, 0.04, 0.03, np.nan])
    np.testing.assert_allclose(adj[:3], [0.03, 0.06, 0.06])
    assert np.isnan(adj[3])


def test_patient_table_averages_seeds_and_marks_patients_without_events():
    table = loso_patient_table(RESULTS["ft"], 1, 30).set_index("patient")
    assert table.loc["p1", "rmse"] == pytest.approx(19.0)
    assert table.loc["p1", "alarm_recall"] == pytest.approx(1.0)
    assert np.isnan(table.loc["p5", "alarm_recall"]) and table.loc["p5", "n_events"] == 0


def test_personalization_tests_cover_every_family_with_holm():
    df = personalization_tests(RESULTS)
    assert len(df) == 3 * 3 * 2 * 2
    assert (df["p_holm"].dropna() >= df.loc[df["p_holm"].notna(), "p_value"] - 1e-12).all()
    ft_vs_global = df[(df["comparison"] == "ft vs global") & (df["metric"] == "alarm_recall")]
    assert (ft_vs_global["n"] == 4).all()  # p5 has no events and drops out


def test_curve_frame_has_day_zero_for_global_only():
    frame = adaptation_curve_frame(RESULTS)
    assert set(frame.loc[frame["regime"] == "global", "n_days"]) == {0}
    assert set(frame.loc[frame["regime"] == "ft", "n_days"]) == {1, 3, 7}
    assert {"recall", "recall_lo", "recall_hi", "rmse_mean"} <= set(frame.columns)


def test_reference_lines_keep_the_lstm_out_of_the_recall_panel():
    alpha_val = {"aggregate": {h: {"1.0": {"recall": {"point": 0.6}}} for h in ("30", "60")}}
    glob = {"aggregate": {h: {"rmse_pooled": {"mean": 19.0}} for h in ("30", "60")}}
    lstm = {"aggregate": {h: {"rmse_pooled": {"mean": 22.0}} for h in ("30", "60")}}
    refs = reference_lines(alpha_val, glob, lstm)
    lstm_ref = [v for k, v in refs.items() if "LSTM" in k][0]
    assert "recall" not in lstm_ref and lstm_ref["rmse"][30] == 22.0


def test_plot_draws_one_line_per_regime_in_each_panel():
    fig = plot_adaptation_curves(adaptation_curve_frame(RESULTS))
    axes = fig.axes
    assert len(axes) == 4
    labels = {t.get_text() for t in axes[0].get_legend().get_texts()}
    assert set(REGIME_LABELS.values()) <= labels


def test_plot_takes_its_own_labels_for_other_regime_sets():
    results = {"global": RESULTS["global"], "ft_rmse": RESULTS["ft"]}
    labels = {"global": "Sin adaptar", "ft_rmse": "Ajuste fino con restricción de RMSE"}
    fig = plot_adaptation_curves(adaptation_curve_frame(results), labels=labels)
    drawn = {t.get_text() for ax in fig.axes if ax.get_legend() for t in ax.get_legend().get_texts()}
    drawn |= {h.get_label() for ax in fig.axes for h in ax.get_legend_handles_labels()[0]}
    assert set(labels.values()) <= drawn


def test_legend_names_references_drawn_only_in_the_rmse_panel():
    refs = {"Global-12": {"recall": {30: 0.6, 60: 0.1}, "rmse": {30: 19.0, 60: 31.0}},
            "LSTM por paciente": {"rmse": {30: 21.9, 60: 33.4}}}
    fig = plot_adaptation_curves(adaptation_curve_frame(RESULTS), refs)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "LSTM por paciente" in labels and labels.count("Global-12") == 1
