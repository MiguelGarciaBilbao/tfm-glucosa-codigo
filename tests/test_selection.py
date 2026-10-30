"""Tests for the pooled alarm counts, the seed-averaged recall and the nested choice."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm.eval.alarm import (
    alarm_counts,
    operating_curve,
    pooled_from_counts,
    seed_averaged_recall,
)
from tfm.eval.metrics import Predictions, bootstrap_ci, reference_events
from tfm.eval.selection import best_alpha, choose, pooled_curve, reference_rmse


def _cohort(horizon: int = 30) -> list[Predictions]:
    rng = np.random.default_rng(0)
    preds = []
    for pid in ("a", "b"):
        y = 110 + 60 * np.sin(np.arange(600) / 40)
        mu = y + rng.normal(0, 15, 600)
        ts = pd.date_range("2027-01-01", periods=600, freq="5min")
        preds.append(Predictions(y, mu, ts, pid, horizon, sigma=np.full(600, 15.0)))
    return preds


def test_counts_pool_to_the_operating_curve():
    preds = _cohort()
    events = {p.patient: reference_events(p) for p in preds}
    alphas = (0.1, 0.3, 0.6)
    curve = operating_curve(preds, events, alphas=alphas)
    pooled = pooled_from_counts(alarm_counts(preds, events, alphas=alphas))
    np.testing.assert_allclose(pooled["recall"], curve["recall"])
    np.testing.assert_allclose(pooled["false_alarms_per_day"], curve["false_alarms_per_day"])


def test_counts_have_one_row_per_patient_and_alpha():
    preds = _cohort()
    events = {p.patient: reference_events(p) for p in preds}
    counts = alarm_counts(preds, events, alphas=(0.2, 0.5))
    assert len(counts) == 4
    assert set(counts["patient"]) == {"a", "b"}


def test_seed_averaged_recall_with_one_seed_is_the_plain_bootstrap():
    detected = np.array([True, False, True, True, False])
    assert seed_averaged_recall([detected]) == pytest.approx(
        {**bootstrap_ci(detected), "n_seeds": 1})


def test_seed_averaged_recall_averages_each_event_over_seeds():
    res = seed_averaged_recall([np.array([1, 0, 1], bool), np.array([1, 1, 0], bool)])
    assert res["point"] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert res["n"] == 3 and res["n_seeds"] == 2


def test_seed_averaged_recall_rejects_different_event_sets():
    with pytest.raises(ValueError, match="same events"):
        seed_averaged_recall([np.ones(3, bool), np.ones(4, bool)])


def _card(folds=("a", "b", "c")) -> pd.DataFrame:
    """r1 detects 8/10 at alpha 0.2 in every fold, r2 only 5/10; alpha 0.5 is stricter."""
    rows = []
    for fold in folds:
        for recipe, good in (("r1", True), ("r2", False)):
            for h in (30, 60):
                for alpha in (0.2, 0.5):
                    detected = (8 if good else 5) if alpha == 0.2 else (6 if good else 3)
                    rows.append({"fold": fold, "seed": 0, "n_days": 1, "recipe": recipe,
                                 "horizon_min": h, "alpha": alpha, "n_events": 10,
                                 "n_detected": detected,
                                 "false_alarm_episodes": 5 if alpha == 0.2 else 1,
                                 "span_days": 10.0, "sse": 1000.0, "n_pred": 10})
    return pd.DataFrame(rows)


def _boost(card, fold, recipe):
    card = card.copy()
    card.loc[(card["fold"] == fold) & (card["recipe"] == recipe), "n_detected"] = 10
    return card


def test_choice_for_a_fold_ignores_that_fold():
    card = _card()
    assert choose(card, "a", 0, 1)["recipe"] == "r1"
    assert choose(_boost(card, "a", "r2"), "a", 0, 1)["recipe"] == "r1"


def test_choice_follows_the_other_folds():
    card = _boost(_boost(_card(), "b", "r2"), "c", "r2")
    assert choose(card, "a", 0, 1)["recipe"] == "r2"


def test_choice_picks_alpha_per_horizon_within_budget():
    choice = choose(_card(), "a", 0, 1)
    assert choice["alpha"] == {30: 0.2, 60: 0.2}
    assert choice["within_budget"] == {30: True, 60: True}
    assert choice["dev_recall"][30] == pytest.approx(0.8)


def test_pooled_curve_sums_counts_before_dividing():
    card = _card(folds=("b", "c"))
    card.loc[card["fold"] == "c", "n_events"] = 30  # recall 8/10 and 8/30
    curve = pooled_curve(card)
    row = curve[(curve["recipe"] == "r1") & (curve["horizon_min"] == 30) & (curve["alpha"] == 0.2)]
    assert row["recall"].item() == pytest.approx(16 / 40)
    assert row["rmse"].item() == pytest.approx(10.0)


def test_best_alpha_prefers_the_larger_alpha_on_ties():
    curve = pd.DataFrame({"alpha": [0.2, 0.5, 0.7], "recall": [0.8, 0.8, 0.4],
                          "false_alarms_per_day": [0.9, 0.5, 0.1]})
    assert best_alpha(curve, 1.0)["alpha"] == 0.5


def test_best_alpha_flags_when_no_alpha_meets_the_budget():
    curve = pd.DataFrame({"alpha": [0.2, 0.5, 0.7], "recall": [0.9, 0.8, 0.4],
                          "false_alarms_per_day": [3.0, 2.0, 1.5]})
    choice = best_alpha(curve, 1.0)
    assert choice["alpha"] == 0.7 and choice["within_budget"] is False


def test_rmse_breaks_ties_between_recipes():
    card = _card()
    card.loc[card["recipe"] == "r2", "n_detected"] = card.loc[card["recipe"] == "r1", "n_detected"].to_numpy()
    card.loc[card["recipe"] == "r2", "sse"] = 10.0
    assert choose(card, "a", 0, 1)["recipe"] == "r2"


def test_choice_survives_folds_without_events():
    card = _card()
    card.loc[card["fold"] != "a", ["n_events", "n_detected"]] = 0
    choice = choose(card, "a", 0, 1)
    assert choice["recipe"] in {"r1", "r2"}
    assert np.isnan(choice["dev_recall"][30])


def test_choice_without_other_folds_is_an_error():
    with pytest.raises(ValueError, match="no development rows"):
        choose(_card(folds=("a",)), "a", 0, 1)


def _with_rmse(card, recipe, sse):
    card = card.copy()
    card.loc[card["recipe"] == recipe, "sse"] = sse
    return card


def test_rmse_guard_skips_a_better_detector_that_worsens_the_error():
    card = _with_rmse(_card(), "r1", 4000.0)  # r1: recall 0.8, rmse 20; r2: recall 0.5, rmse 10
    assert choose(card, "a", 0, 1)["recipe"] == "r1"
    choice = choose(card, "a", 0, 1, rmse_ceiling={30: 15.0, 60: 15.0})
    assert choice["recipe"] == "r2"
    assert choice["rmse_guard"] == {"ceiling": {30: 15.0, 60: 15.0}, "met": True}


def test_rmse_guard_needs_every_horizon_under_its_ceiling():
    card = _with_rmse(_card(), "r1", 4000.0)
    choice = choose(card, "a", 0, 1, rmse_ceiling={30: 25.0, 60: 15.0})
    assert choice["recipe"] == "r2"


def test_rmse_guard_falls_back_to_the_lowest_error_and_says_so():
    card = _with_rmse(_with_rmse(_card(), "r1", 4000.0), "r2", 2250.0)  # rmse 20 and 15
    choice = choose(card, "a", 0, 1, rmse_ceiling={30: 12.0, 60: 12.0})
    assert choice["recipe"] == "r2"
    assert choice["rmse_guard"]["met"] is False


def test_reference_rmse_pools_the_other_folds_of_the_same_seed():
    ref = _card(folds=("a", "b", "c"))
    ref = ref[ref["recipe"] == "r1"].assign(recipe="none", n_days=0)
    ref.loc[ref["fold"] == "b", "sse"] = 4000.0   # rmse 20 in b, 10 in c
    ref.loc[ref["fold"] == "a", "sse"] = 1.0e9    # the fold being scored is ignored
    other_seed = ref.assign(seed=1, sse=1.0e9)
    rmse = reference_rmse(pd.concat([ref, other_seed]), "a", 0)
    assert rmse == pytest.approx({30: np.sqrt(5000 / 20), 60: np.sqrt(5000 / 20)})


def test_reference_rmse_rejects_a_table_with_several_recipes():
    with pytest.raises(ValueError, match="one recipe"):
        reference_rmse(_card(), "a", 0)
