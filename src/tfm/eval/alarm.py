"""The probabilistic alarm: ``P(y < 70) > alpha``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tfm.eval.metrics import N_BOOTSTRAP, Predictions, bootstrap_ci, event_metrics
from tfm.eval.protocol import EVENT_TOLERANCE_MIN
from tfm.data.quality import HYPO

#: Alpha values swept when drawing the operating curve.
DEFAULT_ALPHAS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)

#: Alpha grid of the matched-budget reading. Recall is steep in alpha near one
MATCHED_ALPHAS = tuple(round(0.02 * i, 2) for i in range(1, 36))


def probabilistic_alarm(pred: Predictions, alpha: float, threshold: float = HYPO):
    return pred.hypo_probability(threshold) > alpha


def operating_curve(
    predictions: list[Predictions],
    events_by_patient: dict[str, pd.DataFrame],
    alphas=DEFAULT_ALPHAS,
    threshold: float = HYPO,
    tolerance_min: int = EVENT_TOLERANCE_MIN,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for alpha in alphas:
        detected, false_alarms, days = [], 0, 0.0
        for pred in predictions:
            result = event_metrics(
                pred,
                threshold=threshold,
                tolerance_min=tolerance_min,
                events=events_by_patient[pred.patient],
                alarm=probabilistic_alarm(pred, alpha, threshold),
            )
            detected.append(result["detected"])
            false_alarms += result["false_alarm_episodes"]
            days += result["span_days"]
        pooled = np.concatenate(detected) if detected else np.array([], dtype=bool)
        ci = bootstrap_ci(pooled, seed=seed)
        rows.append({
            "alpha": alpha,
            "recall": ci["point"],
            "recall_lo": ci["lo"],
            "recall_hi": ci["hi"],
            "n_events": ci["n"],
            "false_alarms_per_day": false_alarms / days if days else float("nan"),
            "false_alarm_episodes": false_alarms,
        })
    return pd.DataFrame(rows)


def pick_alpha(curve: pd.DataFrame, max_false_alarms_per_day: float = 1.0) -> dict:
    feasible = curve[curve["false_alarms_per_day"] <= max_false_alarms_per_day]
    if feasible.empty:
        return {"alpha": None, "reason": "no alpha meets the false-alarm budget"}
    best = feasible.loc[feasible["recall"].idxmax()]
    return {
        "alpha": float(best["alpha"]),
        "recall": float(best["recall"]),
        "recall_lo": float(best["recall_lo"]),
        "recall_hi": float(best["recall_hi"]),
        "false_alarms_per_day": float(best["false_alarms_per_day"]),
        "budget": max_false_alarms_per_day,
    }


COUNT_COLUMNS = ("n_events", "n_detected", "false_alarm_episodes", "span_days")


def alarm_counts(
    predictions: list[Predictions],
    events_by_patient: dict[str, pd.DataFrame],
    alphas=DEFAULT_ALPHAS,
    threshold: float = HYPO,
    tolerance_min: int = EVENT_TOLERANCE_MIN,
) -> pd.DataFrame:
    rows = []
    for pred in predictions:
        prob = pred.hypo_probability(threshold)
        for alpha in alphas:
            r = event_metrics(pred, threshold=threshold, tolerance_min=tolerance_min,
                              events=events_by_patient[pred.patient], alarm=prob > alpha)
            rows.append({"patient": pred.patient, "horizon_min": pred.horizon_min,
                         "alpha": float(alpha),
                         **{c: r[c] for c in COUNT_COLUMNS}})
    return pd.DataFrame(rows)


def pooled_from_counts(counts: pd.DataFrame) -> pd.DataFrame:
    g = counts.groupby("alpha", as_index=False)[list(COUNT_COLUMNS)].sum()
    g["recall"] = g["n_detected"] / g["n_events"].where(g["n_events"] > 0)
    g["false_alarms_per_day"] = g["false_alarm_episodes"] / g["span_days"].where(g["span_days"] > 0)
    return g


def seed_averaged_recall(
    detected_by_seed: list[np.ndarray],
    n_resamples: int = N_BOOTSTRAP,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    arrays = [np.asarray(d, dtype=bool).ravel() for d in detected_by_seed]
    if len({len(a) for a in arrays}) > 1:
        raise ValueError("every seed must score the same events in the same order")
    rate = np.vstack(arrays).astype(float).mean(axis=0)
    n = len(rate)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "n_seeds": len(arrays), "confidence": confidence}
    rng = np.random.default_rng(seed)
    props = rate[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    half = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(props, [half, 1.0 - half])
    return {"point": float(rate.mean()), "lo": float(lo), "hi": float(hi), "n": int(n),
            "n_seeds": len(arrays), "confidence": confidence}


def predictions_from_experiment(exp_id: str, seed: int | None = None,
                                results_dir=None) -> dict:
    from pathlib import Path

    from tfm.paths import RESULTS_DIR

    path = Path(results_dir or RESULTS_DIR) / exp_id / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the experiment first")
    frame = pd.read_parquet(path)
    if "seed" in frame.columns:
        frame = frame[frame["seed"] == (frame["seed"].min() if seed is None else seed)]
    has_sigma = "sigma" in frame.columns
    out: dict[int, list[Predictions]] = {}
    for horizon, group in frame.groupby("horizon_min"):
        out[int(horizon)] = [
            Predictions(
                d["y_true"].to_numpy(), d["y_pred"].to_numpy(), d["ts"].to_numpy(),
                str(patient), int(horizon),
                sigma=d["sigma"].to_numpy() if has_sigma else None,
            )
            for patient, d in group.groupby("patient")
        ]
    return out


def alarm_analysis(exp_id: str, seed: int | None = None, budgets=(0.5, 1.0, 2.0)) -> dict:
    from tfm.eval.calibration import cohort_calibration
    from tfm.eval.protocol import scorable_events_by_patient

    by_horizon = predictions_from_experiment(exp_id, seed)
    if not by_horizon[next(iter(by_horizon))][0].is_probabilistic:
        raise ValueError(f"{exp_id} has no sigma; it is not a probabilistic model")
    events = scorable_events_by_patient(by_horizon)
    out = {}
    for horizon, preds in sorted(by_horizon.items()):
        curve = operating_curve(preds, events)
        out[str(horizon)] = {
            "calibration": cohort_calibration(preds),
            "operating_curve": curve.to_dict(orient="records"),
            "chosen": {str(b): pick_alpha(curve, b) for b in budgets},
        }
    return out


def regime_predictions(exp_id: str, episode_gap_min: int | None = None) -> dict:
    from tfm.paths import RESULTS_DIR

    df = pd.read_parquet(RESULTS_DIR / exp_id / "predictions.parquet")
    out = {}
    for (seed, n, p, h), g in df.groupby(["seed", "n_days", "patient", "horizon_min"]):
        pred = Predictions(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), g["ts"].to_numpy(),
                           str(p), int(h), sigma=g["sigma"].to_numpy())
        if episode_gap_min is not None:
            pred.meta["episode_gap_min"] = episode_gap_min
        out.setdefault((int(seed), int(n)), {}).setdefault(str(p), {})[int(h)] = pred
    return out


def recall_at_budget(test_preds: dict, events_fn, budget: float = 1.0,
                     alphas=MATCHED_ALPHAS, tolerance_min: int = EVENT_TOLERANCE_MIN,
                     patients=None) -> pd.DataFrame:
    from tfm.eval.selection import best_alpha

    rows = []
    for n in sorted({n for _, n in test_preds}):
        seeds = sorted(s for s, m in test_preds if m == n)
        pts = sorted(set.intersection(*(set(test_preds[(s, n)]) for s in seeds)))
        if patients is not None:
            pts = [p for p in pts if p in set(patients)]
        events = {s: {p: events_fn(p, list(test_preds[(s, n)][p].values())) for p in pts}
                  for s in seeds}
        for h in sorted(test_preds[(seeds[0], n)][pts[0]]):
            counts = pd.concat([alarm_counts([test_preds[(s, n)][p][h] for p in pts], events[s],
                                             alphas, tolerance_min=tolerance_min)
                                for s in seeds])
            pick = best_alpha(pooled_from_counts(counts), budget)
            detected, fa = [], []
            for s in seeds:
                res = [event_metrics(test_preds[(s, n)][p][h], tolerance_min=tolerance_min,
                                     events=events[s][p],
                                     alarm=test_preds[(s, n)][p][h].hypo_probability()
                                     > pick["alpha"]) for p in pts]
                detected.append(np.concatenate([r["detected"] for r in res]))
                fa.append(sum(r["false_alarm_episodes"] for r in res)
                          / sum(r["span_days"] for r in res))
            recall = seed_averaged_recall(detected)
            rows.append({"n_days": n, "horizon_min": h, "alpha": pick["alpha"],
                         "within_budget": pick["within_budget"], "recall": recall["point"],
                         "recall_lo": recall["lo"], "recall_hi": recall["hi"],
                         "n_events": recall["n"], "false_alarms_per_day": float(np.mean(fa)),
                         "n_patients": len(pts)})
    return pd.DataFrame(rows)
