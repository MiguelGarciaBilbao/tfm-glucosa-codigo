"""Tests for the evaluation metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm.eval.metrics import (
    bootstrap_ci,
    evaluate,
    evaluate_cohort,
    glucose_penalty,
    gmse,
    Predictions,
    alarm_episodes,
    clarke_zones,
    event_metrics,
    parkes_zones,
    reference_events,
    regression_metrics,
    sample_hypo_metrics,
    zone_fractions,
)

# --------------------------------------------------------------------------
# Clarke Error Grid
# --------------------------------------------------------------------------

# (reference, prediction, expected zone). One clinical reading each:
CLARKE_POINTS = [
    (100, 105, "A"),   # accurate
    (60, 65, "A"),     # both hypoglycaemic: clinically identical action
    (250, 220, "A"),   # within 20 % at high glucose
    (100, 140, "B"),   # off by 40 %, but no wrong treatment
    (200, 140, "B"),   # under-estimate, still benign
    (100, 215, "C"),   # would trigger unnecessary correction insulin
    (160, 40, "C"),    # would trigger unnecessary glucose intake
    (250, 150, "D"),   # dangerous: real hyperglycaemia read as in-range
    (50, 120, "D"),    # dangerous: real hypoglycaemia read as in-range
    (50, 200, "E"),    # opposite treatment: hypo predicted as hyper
    (250, 50, "E"),    # opposite treatment: hyper predicted as hypo
]


@pytest.mark.parametrize("ref,pred,expected", CLARKE_POINTS)
def test_clarke_known_points(ref, pred, expected):
    assert clarke_zones([ref], [pred])[0] == expected


def test_clarke_perfect_prediction_is_all_a():
    ref = np.array([40, 70, 100, 180, 300, 400], dtype=float)
    assert set(clarke_zones(ref, ref)) == {"A"}


def test_clarke_is_vectorised():
    refs = [p[0] for p in CLARKE_POINTS]
    preds = [p[1] for p in CLARKE_POINTS]
    expected = [p[2] for p in CLARKE_POINTS]
    assert list(clarke_zones(refs, preds)) == expected


def test_clarke_rejects_non_positive_reference():
    with pytest.raises(ValueError):
        clarke_zones([0], [100])


# --------------------------------------------------------------------------
# Parkes (consensus) Error Grid, type 1
# --------------------------------------------------------------------------

PARKES_POINTS = [
    (100, 105, "A"),   # accurate
    (40, 45, "A"),     # accurate in the hypoglycaemic range
    (100, 150, "B"),   # over-estimate, above the A/B border (126 mg/dL)
    (200, 120, "B"),   # under-estimate, above the B/C border (87 mg/dL)
    (100, 300, "C"),   # over-estimate past the B/C border (180 mg/dL)
    (300, 100, "C"),   # under-estimate past the B/C border (147 mg/dL)
    (100, 500, "D"),   # over-estimate past the C/D border (364 mg/dL)
    (300, 50, "D"),    # under-estimate past the C/D border (58 mg/dL)
    (30, 400, "E"),    # hypoglycaemia predicted as severe hyperglycaemia
    # These two sit between the wrong vertices this module originally shipped
    # and the published ones, so they fail loudly if the table is edited back.
    (30, 180, "E"),    # D/E border at ref=30 is 154 mg/dL, not 200
    (300, 170, "B"),   # B/C border at ref=300 is 147 mg/dL, not 200
]


@pytest.mark.parametrize("ref,pred,expected", PARKES_POINTS)
def test_parkes_known_points(ref, pred, expected):
    assert parkes_zones([ref], [pred])[0] == expected


def test_parkes_perfect_prediction_is_all_a():
    ref = np.array([40, 70, 100, 180, 300, 400], dtype=float)
    assert set(parkes_zones(ref, ref)) == {"A"}


def test_parkes_type_1_has_no_zone_e_below_the_diagonal():
    """Type 1 defines no E region under the diagonal; D is as bad as it gets."""
    ref = np.array([300, 400, 500, 550], dtype=float)
    pred = np.array([5, 10, 10, 10], dtype=float)
    assert set(parkes_zones(ref, pred)) == {"D"}


def test_parkes_zones_worsen_monotonically_away_from_the_diagonal():
    """At a fixed reference, over-estimating more can never improve the zone."""
    order = {z: i for i, z in enumerate("ABCDE")}
    ref = np.full(60, 100.0)
    pred = np.linspace(100, 550, 60)
    levels = [order[z] for z in parkes_zones(ref, pred)]
    assert levels == sorted(levels)


def test_parkes_only_type_1_is_implemented():
    with pytest.raises(NotImplementedError):
        parkes_zones([100], [100], diabetes_type=2)


def test_zone_fractions_sum_to_one_hundred():
    zones = np.array(list("AAABBCDE"))
    frac = zone_fractions(zones)
    assert sum(frac[z] for z in "ABCDE") == pytest.approx(100.0)
    assert frac["AB"] == pytest.approx(62.5)
    assert frac["DE"] == pytest.approx(25.0)


# --------------------------------------------------------------------------
# Regression and per-sample alarms
# --------------------------------------------------------------------------

def _series(values, start="2027-01-01 00:00", freq_min=5):
    ts = pd.date_range(start, periods=len(values), freq=f"{freq_min}min")
    return np.asarray(values, dtype=float), ts


def test_regression_metrics_on_a_hand_computed_case():
    out = regression_metrics([100, 100, 100], [90, 110, 130])
    assert out["mae"] == pytest.approx((10 + 10 + 30) / 3)
    assert out["rmse"] == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert out["bias"] == pytest.approx(10.0)
    assert out["n_samples"] == 3


def test_sample_hypo_metrics_counts_the_confusion_matrix():
    y_true = [60, 65, 100, 120]
    y_pred = [65, 100, 60, 130]  # one hit, one miss, one false alarm
    out = sample_hypo_metrics(y_true, y_pred)
    assert (out["tp"], out["fn"], out["fp"], out["tn"]) == (1, 1, 1, 1)
    assert out["recall"] == pytest.approx(0.5)
    assert out["precision"] == pytest.approx(0.5)
    assert out["f1"] == pytest.approx(0.5)


def test_sample_hypo_metrics_recall_is_nan_without_positives():
    out = sample_hypo_metrics([100, 120], [100, 120])
    assert np.isnan(out["recall"])
    assert out["n_positive"] == 0


def test_alarm_episodes_groups_consecutive_samples():
    _, ts = _series(range(10))
    alarm = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
    assert alarm_episodes(ts, alarm) == [(1, 3), (6, 6)]


def test_alarm_episodes_splits_on_a_time_gap():
    """Two alarms adjacent in the array but an hour apart are two episodes."""
    ts = pd.to_datetime(
        ["2027-01-01 00:00", "2027-01-01 00:05", "2027-01-01 01:05"]
    ).to_numpy()
    assert alarm_episodes(ts, np.array([True, True, True])) == [(0, 1), (2, 2)]


def test_alarm_episodes_empty_when_no_alarm():
    _, ts = _series(range(5))
    assert alarm_episodes(ts, np.zeros(5, dtype=bool)) == []


# --------------------------------------------------------------------------
# Event-level detection
# --------------------------------------------------------------------------

def _predictions(y_true, y_pred, horizon_min=30):
    y_true, ts = _series(y_true)
    return Predictions(y_true, np.asarray(y_pred, dtype=float), ts, "999", horizon_min)


def test_predictions_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        Predictions([1, 2], [1], pd.date_range("2027-01-01", periods=2), "999", 30)


def test_predictions_target_ts_shifts_by_the_horizon():
    p = _predictions([100] * 3, [100] * 3, horizon_min=60)
    assert (p.target_ts - p.ts == np.timedelta64(60, "m")).all()


def test_reference_events_needs_fifteen_minutes_below_threshold():
    """Two samples under 70 are a 10-minute dip, not an event."""
    short = _predictions([100, 65, 65, 100, 100], [100] * 5)
    assert len(reference_events(short)) == 0
    long = _predictions([100, 65, 65, 65, 100], [100] * 5)
    assert len(reference_events(long)) == 1


def test_event_metrics_detects_an_event_within_tolerance():
    # Truth dips below 70 from index 4; the model predicts the dip one step
    # earlier, well inside a 30-minute tolerance.
    y_true = [120, 110, 100, 85, 65, 60, 62, 90, 110, 120]
    y_pred = [120, 110, 100, 65, 62, 60, 62, 90, 110, 120]
    out = event_metrics(_predictions(y_true, y_pred))
    assert out["n_events"] == 1
    assert out["n_detected"] == 1
    assert out["recall"] == pytest.approx(1.0)
    assert out["false_alarm_episodes"] == 0
    assert out["detected"].tolist() == [True]


def test_event_metrics_misses_an_event_the_model_never_flags():
    y_true = [120, 110, 100, 85, 65, 60, 62, 90, 110, 120]
    y_pred = [120] * 10
    out = event_metrics(_predictions(y_true, y_pred))
    assert (out["n_events"], out["n_detected"]) == (1, 0)
    assert out["recall"] == pytest.approx(0.0)


def test_event_metrics_counts_a_false_alarm_far_from_any_event():
    y_true = [120] * 20
    y_pred = [120] * 9 + [60, 60] + [120] * 9
    out = event_metrics(_predictions(y_true, y_pred))
    assert out["n_events"] == 0
    assert out["false_alarm_episodes"] == 1
    assert out["false_alarms_per_day"] > 0


def test_event_metrics_does_not_charge_alarms_on_real_short_dips():
    """A real 10-minute dip is too short to score, but flagging it is not wrong."""
    y_true = [120] * 9 + [65, 65] + [120] * 9
    y_pred = [120] * 9 + [65, 65] + [120] * 9
    out = event_metrics(_predictions(y_true, y_pred))
    assert out["n_events"] == 0
    assert out["false_alarm_episodes"] == 0
    assert out["alarms_on_short_dips"] == 1


def test_event_metrics_lead_time_is_measured_to_the_real_crossing():
    # y_true is indexed by target instant, so the crossing sits at 00:20 + 30 =
    # 00:50. The first alarm is issued at 00:15, hence 35 minutes of warning.
    # Lead time is bounded above by the horizon plus the tolerance.
    y_true = [120, 110, 100, 85, 65, 60, 62, 90, 110, 120]
    y_pred = [120, 110, 100, 65, 62, 60, 62, 90, 110, 120]
    out = event_metrics(_predictions(y_true, y_pred, horizon_min=30))
    assert out["median_lead_time_min"] == pytest.approx(35.0)
    assert out["median_lead_time_min"] <= 30 + out["tolerance_min"]


def test_event_metrics_is_empty_but_well_formed_without_events():
    out = event_metrics(_predictions([120] * 10, [120] * 10))
    assert out["n_events"] == 0
    assert np.isnan(out["recall"])
    assert out["events"].empty


# --------------------------------------------------------------------------
# gMSE
# --------------------------------------------------------------------------

def test_penalty_is_one_in_the_safe_band():
    """Between the thresholds, gMSE must reduce to plain MSE."""
    assert glucose_penalty([120], [140])[0] == pytest.approx(1.0)
    assert glucose_penalty([120], [100])[0] == pytest.approx(1.0)


def test_penalty_punishes_over_estimating_hypoglycaemia():
    """Reading a real 60 as 100 would withhold treatment: the dangerous error."""
    assert glucose_penalty([60], [100])[0] > 2.0


def test_penalty_ignores_under_estimating_hypoglycaemia():
    """Reading a real 60 as 40 is wrong but leads to the right action."""
    assert glucose_penalty([60], [40])[0] == pytest.approx(1.0)


def test_penalty_punishes_under_estimating_hyperglycaemia():
    assert glucose_penalty([300], [200])[0] > 1.5


def test_penalty_stays_within_its_published_range():
    rng = np.random.default_rng(0)
    ref = rng.uniform(20, 500, 5000)
    pred = rng.uniform(20, 500, 5000)
    pen = glucose_penalty(ref, pred)
    assert pen.min() >= 1.0
    assert pen.max() <= 1.0 + 1.5 + 1.0


def test_smooth_steps_are_continuous_at_every_knot():
    """A typo in the ramp shows up as a jump at one of the four knots."""
    from tfm.eval.metrics import _smooth_step, _smooth_step_down

    a, eps = 85.0, 30.0
    for fn in (_smooth_step, _smooth_step_down):
        x = np.linspace(a - 2 * eps, a + 2 * eps, 4001)
        y = fn(x, a, eps)
        assert np.abs(np.diff(y)).max() < 0.01
        assert y.min() == pytest.approx(0.0, abs=1e-9)
        assert y.max() == pytest.approx(1.0, abs=1e-9)


def test_gmse_equals_mse_when_no_penalty_applies():
    ref = np.array([120.0, 130.0, 140.0])
    pred = np.array([125.0, 125.0, 135.0])
    assert gmse(ref, pred)["gmse"] == pytest.approx(np.mean((pred - ref) ** 2))


def test_gmse_exceeds_mse_on_a_missed_hypoglycaemia():
    ref, pred = [60.0], [100.0]
    assert gmse(ref, pred)["gmse"] > np.mean((np.array(pred) - np.array(ref)) ** 2)


def test_gmse_is_asymmetric_for_equal_absolute_errors():
    """Same 40 mg/dL error either side of a real 60: only one is dangerous."""
    over = gmse([60.0], [100.0])["gmse"]
    under = gmse([60.0], [20.0])["gmse"]
    assert over > under


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def test_bootstrap_ci_brackets_the_point_estimate():
    successes = np.array([True] * 48 + [False] * 21)  # 69 events, recall 0.70
    ci = bootstrap_ci(successes, seed=0)
    assert ci["n"] == 69
    assert ci["point"] == pytest.approx(48 / 69)
    assert ci["lo"] < ci["point"] < ci["hi"]


def test_bootstrap_ci_width_matches_the_documented_event_budget():
    """With 69 events the interval is about +-0.11, the figure the project documentation quotes."""
    ci = bootstrap_ci(np.array([True] * 48 + [False] * 21), seed=0)
    half_width = (ci["hi"] - ci["lo"]) / 2
    assert 0.08 < half_width < 0.14


def test_bootstrap_ci_narrows_as_events_accumulate():
    rng = np.random.default_rng(0)
    widths = []
    for n in (50, 500, 5000):
        x = rng.random(n) < 0.7
        ci = bootstrap_ci(x, seed=1)
        widths.append(ci["hi"] - ci["lo"])
    assert widths[0] > widths[1] > widths[2]


def test_bootstrap_ci_is_reproducible_under_a_fixed_seed():
    x = np.array([True] * 30 + [False] * 20)
    assert bootstrap_ci(x, seed=7) == bootstrap_ci(x, seed=7)


def test_bootstrap_ci_handles_no_events():
    ci = bootstrap_ci(np.array([], dtype=bool))
    assert ci["n"] == 0 and np.isnan(ci["point"])


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def _cohort(n_patients=3, horizon_min=30, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_patients):
        n = 400
        y_true = 120 + 40 * np.sin(np.arange(n) / 20) + rng.normal(0, 5, n)
        y_pred = y_true + rng.normal(0, 12, n)
        ts = pd.date_range("2027-03-01", periods=n, freq="5min")
        out.append(Predictions(y_true, y_pred, ts, f"P{i}", horizon_min))
    return out


def test_evaluate_returns_regression_and_clinical_together():
    p = _cohort(1)[0]
    res = evaluate(p)
    assert {"regression", "gmse", "sample_hypo", "events", "clarke", "parkes"} <= res.keys()
    assert res["regression"]["rmse"] > 0
    assert res["clarke"]["AB"] <= 100.0


def test_evaluate_cohort_table_carries_absolute_event_counts():
    """Never a recall without its denominator: the table must expose n_events."""
    table, _ = evaluate_cohort(_cohort(3))
    assert {"n_events", "recall"} <= set(table.columns)
    assert len(table) == 3


def test_evaluate_cohort_pools_events_across_patients():
    preds = _cohort(3)
    table, summary = evaluate_cohort(preds)
    assert summary["pooled_event_recall"]["n"] == int(table["n_events"].sum())
    assert summary["n_patients"] == 3


def test_evaluate_cohort_rejects_mixed_horizons():
    preds = _cohort(2, horizon_min=30) + _cohort(1, horizon_min=60)
    with pytest.raises(ValueError, match="mixed horizons"):
        evaluate_cohort(preds)


def test_evaluate_cohort_rejects_an_empty_cohort():
    with pytest.raises(ValueError):
        evaluate_cohort([])


def test_event_metrics_rejects_a_tolerance_wider_than_the_horizon():
    """The guard that stops the recall metric from becoming meaningless.

    With tolerance >= horizon the detection window reaches the onset itself,
    where the truth is already below 70, so persistence scores a perfect recall.
    Measured on the official test set, persistence jumps from 0.095 to 1.000
    when the tolerance goes from 15 to 30 minutes at a 30-minute horizon.
    """
    p = _predictions([120] * 10, [120] * 10, horizon_min=30)
    with pytest.raises(ValueError, match="strictly below the horizon"):
        event_metrics(p, tolerance_min=30)


def test_persistence_does_not_get_a_free_perfect_recall():
    """End-to-end guard, on a synthetic drop, with the default tolerance."""
    n = 40
    y_true = np.concatenate([np.full(20, 120.0), np.full(20, 60.0)])
    ts = pd.date_range("2027-01-01", periods=n, freq="5min")
    # Persistence at a 30 min horizon: the prediction is the truth shifted back
    # six steps, which is what made the wide tolerance degenerate.
    y_pred = np.concatenate([np.full(6, 120.0), y_true[:-6]])
    out = event_metrics(Predictions(y_true, y_pred, ts, "999", 30), tolerance_min=15)
    assert out["recall"] < 1.0


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def test_persistence_reproduces_the_recorded_rmse():
    """Pins the baseline the whole TFM is measured against.

    If preprocessing ever shifts targets by one step this test fails loudly,
    which is the cheapest possible guard against a silent alignment bug.
    """
    pytest.importorskip("pandas")
    from tfm.eval.baselines import persistence_cohort
    from tfm.eval.metrics import regression_metrics

    for horizon_index, expected in ((0, 23.50), (1, 38.55)):
        try:
            preds = persistence_cohort(horizon_index)
        except FileNotFoundError:
            pytest.skip("data/processed not built")
        y_true = np.concatenate([p.y_true for p in preds])
        y_pred = np.concatenate([p.y_pred for p in preds])
        assert regression_metrics(y_true, y_pred)["rmse"] == pytest.approx(expected, abs=0.01)


# --------------------------------------------------------------------------
# Evaluation protocol
# --------------------------------------------------------------------------

def test_covered_instants_intersects_across_horizons():
    from tfm.eval.protocol import covered_instants

    ts = pd.date_range("2027-01-01", periods=10, freq="5min")
    a = Predictions(np.zeros(10), np.zeros(10), ts, "P", 30)
    b = Predictions(np.zeros(10), np.zeros(10), ts, "P", 60)
    common = covered_instants([a, b])
    # Targets are shifted by 30 and 60 min, so only the overlap survives.
    assert len(common) < 10
    assert set(common) <= set(a.target_ts) and set(common) <= set(b.target_ts)


def test_event_metrics_uses_the_event_set_it_is_given():
    """The onset must come from the caller, not be re-derived from the series."""
    y_true = [120] * 10
    p = _predictions(y_true, [120] * 10, horizon_min=30)
    assert event_metrics(p)["n_events"] == 0
    injected = pd.DataFrame({"start": [pd.Timestamp("2027-01-01 00:20")]})
    out = event_metrics(p, events=injected)
    assert out["n_events"] == 1
    assert out["n_detected"] == 0  # the model never alarms, so it is a miss


def test_scorable_event_budget_is_sixty_one_and_equal_at_both_horizons():
    """Pins the denominator every reported recall is divided by."""
    try:
        from tfm.eval.baselines import persistence_cohort
        from tfm.eval.protocol import scorable_events_by_patient

        by_patient = scorable_events_by_patient(
            {30: persistence_cohort(0), 60: persistence_cohort(1)}
        )
    except FileNotFoundError:
        pytest.skip("data/processed or data/interim not built")
    assert sum(len(v) for v in by_patient.values()) == 61


def test_scorable_events_keep_the_real_onsets():
    """Patient 584's third event really starts at 07:45, not at 07:55."""
    try:
        from tfm.eval.baselines import persistence
        from tfm.eval.protocol import official_events, scorable_events
    except ImportError:  # pragma: no cover
        pytest.skip("modules unavailable")
    try:
        preds = [persistence("584", 0), persistence("584", 1)]
        scored = scorable_events("584", preds)
        official = official_events("584")
    except FileNotFoundError:
        pytest.skip("data/processed or data/interim not built")
    assert set(scored["start"]) <= set(official["start"])


def test_scorable_events_by_patient_rejects_an_incomplete_horizon():
    ts = pd.date_range("2027-01-01", periods=10, freq="5min")
    a = Predictions(np.zeros(10), np.zeros(10), ts, "A", 30)
    b = Predictions(np.zeros(10), np.zeros(10), ts, "B", 60)
    from tfm.eval.protocol import scorable_events_by_patient

    with pytest.raises(ValueError, match="missing at some horizon"):
        scorable_events_by_patient({30: [a], 60: [b]})


def test_event_metrics_says_where_each_false_alarm_starts_and_ends():
    start = np.datetime64("2027-03-01T00:00", "ns")
    ts = start + np.arange(100) * np.timedelta64(5, "m")
    y = np.full(100, 120.0)
    pred = Predictions(y, y, ts, "a", 30)
    alarm = np.zeros(100, dtype=bool)
    alarm[10:14] = True
    alarm[50:52] = True
    no_events = pd.DataFrame({"start": pd.Series([], dtype="datetime64[ns]")})
    r = event_metrics(pred, events=no_events, alarm=alarm)
    assert r["false_alarm_episodes"] == 2
    assert r["false_alarm_spans"] == [(ts[10], ts[13]), (ts[50], ts[51])]
    assert "false_alarm_spans" not in evaluate(pred, events=no_events)["events"]


def test_alarm_episodes_follow_the_declared_cadence():
    """Libre predictions 15 min apart: three consecutive alarms are one episode."""
    ts = pd.date_range("2021-01-01", periods=20, freq="15min")
    y_true = np.full(20, 120.0)
    y_pred = np.full(20, 120.0)
    y_pred[8:11] = 60.0
    default = event_metrics(Predictions(y_true, y_pred, ts, "p", 30))
    libre = event_metrics(Predictions(y_true, y_pred, ts, "p", 30,
                                      meta={"episode_gap_min": 20}))
    assert default["false_alarm_episodes"] == 3
    assert libre["false_alarm_episodes"] == 1
