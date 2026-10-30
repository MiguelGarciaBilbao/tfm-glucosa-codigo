"""Comparing experiments.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tfm.paths import RESULTS_DIR


def load_result(exp_id: str) -> dict:
    path = RESULTS_DIR / exp_id / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the experiment first")
    return json.loads(path.read_text())


def _horizon_block(result: dict, horizon_min: int) -> tuple[dict, pd.DataFrame]:
    h = str(horizon_min)
    if "by_horizon" in result:
        block = result["by_horizon"][h]
        return block["summary"], pd.DataFrame(block["per_patient"])
    seeds = [str(s) for s in result["seeds"]]
    tables = [pd.DataFrame(result["by_seed"][s][h]["per_patient"]) for s in seeds]
    merged = pd.concat(tables).groupby("patient", as_index=False).mean(numeric_only=True)
    return result["by_seed"][seeds[0]][h]["summary"], merged


def comparison_table(exp_ids: list[str], horizon_min: int) -> pd.DataFrame:
    rows = []
    for exp_id in exp_ids:
        result = load_result(exp_id)
        summary, table = _horizon_block(result, horizon_min)
        recall = summary["pooled_event_recall"]
        agg = result.get("aggregate", {}).get(str(horizon_min), {})
        rows.append({
            "exp_id": exp_id,
            "rmse": agg.get("rmse_pooled", {}).get("mean", summary["rmse_pooled"]),
            "rmse_sd_seeds": agg.get("rmse_pooled", {}).get("std", 0.0),
            "recall": agg.get("event_recall", {}).get("mean", recall["point"]),
            "recall_ci_lo": recall["lo"],
            "recall_ci_hi": recall["hi"],
            "n_events": recall["n"],
            "false_alarms_per_day": summary["false_alarms_per_day_median"],
            "clarke_AB": summary["clarke_AB_pooled"],
            "parkes_AB": summary["parkes_AB_pooled"],
            "parkes_DE": summary["parkes_DE_pooled"],
        })
    return pd.DataFrame(rows)


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.stats import wilcoxon

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or np.allclose(a, b):
        return {"n": int(len(a)),
                "median_diff": float(np.median(a - b)) if len(a) else float("nan"),
                "p_value": float("nan")}
    stat, p = wilcoxon(a, b)
    return {"n": int(len(a)), "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "median_diff": float(np.median(a - b)), "statistic": float(stat), "p_value": float(p)}


def paired_wilcoxon(
    exp_a: str, exp_b: str, horizon_min: int, metric: str = "rmse"
) -> dict:
    _, table_a = _horizon_block(load_result(exp_a), horizon_min)
    _, table_b = _horizon_block(load_result(exp_b), horizon_min)
    merged = table_a.merge(table_b, on="patient", suffixes=("_a", "_b"))
    return {"metric": metric,
            **_paired(merged[f"{metric}_a"].to_numpy(), merged[f"{metric}_b"].to_numpy())}


PERSONALIZATION_COMPARISONS = (("ft", "global"), ("ft_weighted", "ft"), ("ft", "individual"))


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    adj = np.full_like(p, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    running = 0.0
    for rank, i in enumerate(idx[np.argsort(p[idx], kind="stable")]):
        running = max(running, min(1.0, (len(idx) - rank) * p[i]))
        adj[i] = running
    return adj


def loso_patient_table(result: dict, n_days: int, horizon_min: int) -> pd.DataFrame:
    block, h = result["by_days"][str(n_days)], str(horizon_min)
    rmse = pd.concat([pd.DataFrame(block["by_seed"][s][h]["per_patient"])[["patient", "rmse"]]
                      for s in block["by_seed"]]).groupby("patient", as_index=False).mean()
    alarm = pd.concat([pd.DataFrame(rows)
                       for rows in block["alarm"][h]["per_patient_by_seed"].values()])
    alarm["alarm_recall"] = alarm["n_detected"] / alarm["n_events"].where(alarm["n_events"] > 0)
    alarm = alarm.groupby("patient", as_index=False).agg(
        alarm_recall=("alarm_recall", "mean"), n_events=("n_events", "first"))
    return rmse.merge(alarm, on="patient")


def _days(regime: str, n: int) -> int:
    return 0 if regime == "global" else n


def personalization_tests(results: dict, n_days=(1, 3, 7), horizons=(30, 60),
                          metrics=("rmse", "alarm_recall"),
                          comparisons=PERSONALIZATION_COMPARISONS) -> pd.DataFrame:
    rows = []
    for a, b in comparisons:
        for n in n_days:
            for h in horizons:
                merged = loso_patient_table(results[a], _days(a, n), h).merge(
                    loso_patient_table(results[b], _days(b, n), h),
                    on="patient", suffixes=("_a", "_b"))
                for metric in metrics:
                    rows.append({"comparison": f"{a} vs {b}", "n_days": n, "horizon_min": h,
                                 "metric": metric,
                                 **_paired(merged[f"{metric}_a"].to_numpy(),
                                           merged[f"{metric}_b"].to_numpy())})
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for _, idx in df.groupby(["metric", "horizon_min"]).groups.items():
        df.loc[idx, "p_holm"] = holm(df.loc[idx, "p_value"].to_numpy())
    return df


def adaptation_curve_frame(results: dict, horizons=(30, 60)) -> pd.DataFrame:
    rows = []
    for regime, result in results.items():
        for n, block in result["by_days"].items():
            for h in horizons:
                alarm, agg = block["alarm"][str(h)], block["aggregate"][str(h)]
                rows.append({"regime": regime, "n_days": int(n), "horizon_min": h,
                             "recall": alarm["recall"]["point"],
                             "recall_lo": alarm["recall"]["lo"],
                             "recall_hi": alarm["recall"]["hi"],
                             "rmse_mean": agg["rmse_pooled"]["mean"],
                             "rmse_std": agg["rmse_pooled"]["std"],
                             "false_alarms_per_day": float(
                                 np.mean(alarm["false_alarms_per_day_by_seed"]))})
    return pd.DataFrame(rows)


def reference_lines(alpha_val: dict, global_12: dict, lstm_patient: dict,
                    budget: str = "1.0", horizons=(30, 60)) -> dict:
    return {
        "Global-12 (Fase 4, α en validación)": {
            "recall": {h: alpha_val["aggregate"][str(h)][budget]["recall"]["point"]
                       for h in horizons},
            "rmse": {h: global_12["aggregate"][str(h)]["rmse_pooled"]["mean"] for h in horizons},
        },
        "LSTM por paciente (todo su train)": {
            "rmse": {h: lstm_patient["aggregate"][str(h)]["rmse_pooled"]["mean"]
                     for h in horizons},
        },
    }
