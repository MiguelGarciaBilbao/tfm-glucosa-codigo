"""Nested choice of the adaptation recipe and the alarm threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tfm.eval.alarm import COUNT_COLUMNS

SCORECARD_COLUMNS = ("fold", "seed", "n_days", "recipe", "horizon_min", "alpha",
                     *COUNT_COLUMNS, "sse", "n_pred")


def pooled_curve(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["recipe", "horizon_min", "alpha"]
    g = rows.groupby(keys, as_index=False)[[*COUNT_COLUMNS, "sse", "n_pred"]].sum()
    g["recall"] = g["n_detected"] / g["n_events"].where(g["n_events"] > 0)
    g["false_alarms_per_day"] = g["false_alarm_episodes"] / g["span_days"].where(g["span_days"] > 0)
    g["rmse"] = np.sqrt(g["sse"] / g["n_pred"].where(g["n_pred"] > 0))
    return g


def best_alpha(curve: pd.DataFrame, budget: float) -> dict:
    c = curve.sort_values("alpha", ascending=False).reset_index(drop=True)
    feasible = c[c["false_alarms_per_day"] <= budget]
    if len(feasible):
        row, ok = feasible.loc[feasible["recall"].fillna(-1.0).idxmax()], True
    else:
        row, ok = c.loc[c["false_alarms_per_day"].idxmin()], False
    return {"alpha": float(row["alpha"]), "recall": float(row["recall"]),
            "false_alarms_per_day": float(row["false_alarms_per_day"]),
            "rmse": float(row.get("rmse", np.nan)), "within_budget": ok}


def reference_rmse(scorecard: pd.DataFrame, fold: str, seed: int) -> dict[int, float]:
    if scorecard["recipe"].nunique() != 1:
        raise ValueError("the reference table must hold one recipe (the unadapted backbone)")
    rows = scorecard[(scorecard["seed"] == seed) & (scorecard["fold"] != fold)]
    if rows.empty:
        raise ValueError(f"no reference rows outside fold {fold} (seed {seed})")
    curve = pooled_curve(rows)
    return {int(h): float(c["rmse"].iloc[0]) for h, c in curve.groupby("horizon_min")}


def choose(scorecard: pd.DataFrame, fold: str, seed: int, n_days: int,
           budget: float = 1.0, rmse_ceiling: dict[int, float] | None = None) -> dict:
    rows = scorecard[(scorecard["seed"] == seed) & (scorecard["n_days"] == n_days)
                     & (scorecard["fold"] != fold)]
    if rows.empty:
        raise ValueError(f"no development rows outside fold {fold} "
                         f"(seed {seed}, n_days {n_days})")
    best_key, best = None, None
    for recipe, rc in pooled_curve(rows).groupby("recipe"):
        per_h = {int(h): best_alpha(hc, budget) for h, hc in rc.groupby("horizon_min")}
        recall = np.nan_to_num(np.mean([v["recall"] for v in per_h.values()]), nan=-1.0)
        rmse = np.mean([v["rmse"] for v in per_h.values()])
        key = (-recall, rmse, recipe)
        if rmse_ceiling is not None:
            met = all(v["rmse"] <= rmse_ceiling[h] for h, v in per_h.items())
            key = (not met, rmse if not met else 0.0, *key)
        if best_key is None or key < best_key:
            best_key, best = key, (recipe, per_h)
    recipe, per_h = best
    guard = {} if rmse_ceiling is None else {"rmse_guard": {
        "ceiling": {int(h): float(v) for h, v in rmse_ceiling.items()},
        "met": all(v["rmse"] <= rmse_ceiling[h] for h, v in per_h.items())}}
    return {
        "recipe": recipe,
        "alpha": {h: v["alpha"] for h, v in per_h.items()},
        "within_budget": {h: v["within_budget"] for h, v in per_h.items()},
        "dev_recall": {h: v["recall"] for h, v in per_h.items()},
        "dev_false_alarms_per_day": {h: v["false_alarms_per_day"] for h, v in per_h.items()},
        "dev_rmse": {h: v["rmse"] for h, v in per_h.items()},
        **guard,
    }
