"""Tests for the HUPA adaptation runner (phase 8A.4)."""

import numpy as np
import pandas as pd
import pytest

from tfm.eval.selection import choose
from tfm.training import external_adapt


class _Stub:

    def __init__(self, ts):
        self.ts = ts

    def __len__(self):
        return len(self.ts)


def _windows(days: float) -> _Stub:
    ts = pd.date_range("2018-06-13 08:00", periods=int(days * 96) + 1, freq="15min")
    return _Stub(ts.to_numpy())


def test_a_patient_needs_the_whole_budget_of_days():
    ws = _windows(2.5)
    assert external_adapt.eligible(ws, 1)
    assert not external_adapt.eligible(ws, 3)


def test_the_unadapted_regime_takes_every_patient_but_an_empty_one():
    assert external_adapt.eligible(_windows(0.5), 0)
    assert not external_adapt.eligible(_Stub(np.array([], dtype="datetime64[ns]")), 0)


def _card(folds=("a", "b", "c")) -> pd.DataFrame:
    rows = []
    for fold in folds:
        for recipe, detected in (("r1", 8), ("r2", 5)):
            for h in (30, 60):
                for alpha in (0.2, 0.5):
                    rows.append({"fold": fold, "seed": 0, "n_days": 1, "recipe": recipe,
                                 "horizon_min": h, "alpha": alpha, "n_events": 10,
                                 "n_detected": detected if alpha == 0.2 else detected - 2,
                                 "false_alarm_episodes": 5 if alpha == 0.2 else 1,
                                 "span_days": 10.0, "sse": 1000.0, "n_pred": 10})
    return pd.DataFrame(rows)


def _boost(card, folds, recipe):
    card = card.copy()
    card.loc[card["fold"].isin(folds) & (card["recipe"] == recipe), "n_detected"] = 10
    return card


def test_every_ohio_fold_takes_part_in_the_choice():
    # r2 wins only when both boosted folds count: (10 + 10 + 5) / 30 > 0.8.
    card = _boost(_card(), ["a", "b"], "r2")
    assert choose(card, "a", 0, 1)["recipe"] == "r1"
    choices = external_adapt.ohio_choices(card, seeds=[0], n_days=[1], budget=1.0)
    assert list(choices) == [(0, 1)]
    assert choices[(0, 1)]["recipe"] == "r2"


def test_the_rmse_guard_ceiling_pools_every_ohio_fold():
    card = _boost(_card(), ["a", "b"], "r2")
    card.loc[card["recipe"] == "r1", "sse"] = 640.0  # RMSE 8 against r2's 10
    ref = _card().query("recipe == 'r1'").assign(recipe="none", sse=810.0)  # ceiling 9
    choice = external_adapt.ohio_choices(card, [0], [1], 1.0, ref_card=ref)[(0, 1)]
    assert choice["recipe"] == "r1"
    assert choice["rmse_guard"]["met"]
    assert choice["rmse_guard"]["ceiling"] == {30: pytest.approx(9.0), 60: pytest.approx(9.0)}


def test_an_unknown_selection_rule_is_rejected_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="selection_rule"):
        external_adapt.run_external_adapt_experiment(
            {"exp_id": "x", "selection_rule": "best_test"}, tmp_path)

