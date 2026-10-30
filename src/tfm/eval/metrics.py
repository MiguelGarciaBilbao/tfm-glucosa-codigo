"""Regression, clinical and error-grid metrics for CGM prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tfm.data.parser import SAMPLING_MINUTES
from tfm.data.quality import HYPO, MIN_EVENT_MINUTES, SEVERE_HYPO  # noqa: F401

#: Tolerance around an event onset for counting it as detected, in minutes.
DEFAULT_TOLERANCE_MIN = 15

#: Bootstrap resamples for confidence intervals.
N_BOOTSTRAP = 2000


@dataclass
class Predictions:
    y_true: np.ndarray
    y_pred: np.ndarray
    ts: np.ndarray
    patient: str
    horizon_min: int
    meta: dict = field(default_factory=dict)
    sigma: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.y_true = np.asarray(self.y_true, dtype=float).ravel()
        self.y_pred = np.asarray(self.y_pred, dtype=float).ravel()
        self.ts = pd.to_datetime(pd.Series(self.ts)).to_numpy()
        if not (len(self.y_true) == len(self.y_pred) == len(self.ts)):
            raise ValueError(
                f"length mismatch: y_true={len(self.y_true)}, "
                f"y_pred={len(self.y_pred)}, ts={len(self.ts)}"
            )
        if self.sigma is not None:
            self.sigma = np.asarray(self.sigma, dtype=float).ravel()
            if len(self.sigma) != len(self.y_true):
                raise ValueError(
                    f"sigma has {len(self.sigma)} values for {len(self.y_true)} predictions"
                )
            if np.any(self.sigma <= 0):
                raise ValueError("sigma must be strictly positive")

    @property
    def is_probabilistic(self) -> bool:
        return self.sigma is not None

    def hypo_probability(self, threshold: float = HYPO) -> np.ndarray:
        if self.sigma is None:
            raise RuntimeError("this prediction has no sigma; the model is deterministic")
        from math import sqrt

        from scipy.special import erf

        z = (threshold - self.y_pred) / (self.sigma * sqrt(2.0))
        return 0.5 * (1.0 + erf(z))

    def __len__(self) -> int:
        return len(self.y_true)

    @property
    def target_ts(self) -> np.ndarray:
        return self.ts + np.timedelta64(self.horizon_min, "m")

    @property
    def span_days(self) -> float:
        if len(self) == 0:
            return 0.0
        t = np.sort(self.target_ts)
        delta = (t[-1] - t[0]) / np.timedelta64(1, "D")
        return float(delta) + SAMPLING_MINUTES / (60 * 24)

    def sorted(self) -> "Predictions":
        order = np.argsort(self.ts)
        return Predictions(
            self.y_true[order], self.y_pred[order], self.ts[order],
            self.patient, self.horizon_min, dict(self.meta),
            None if self.sigma is None else self.sigma[order],
        )


def ensemble_predictions(preds: list[Predictions]) -> Predictions:
    if not preds:
        raise ValueError("no predictions to combine")
    first = preds[0]
    for p in preds:
        if p.sigma is None:
            raise ValueError("every ensemble member needs a predictive sigma")
        if (p.patient != first.patient or p.horizon_min != first.horizon_min
                or len(p) != len(first) or not np.array_equal(p.ts, first.ts)
                or not np.allclose(p.y_true, first.y_true)):
            raise ValueError("ensemble members must predict the same windows "
                             f"(patient {first.patient}, horizon {first.horizon_min} min)")
    mus = np.stack([p.y_pred for p in preds])
    variances = np.stack([p.sigma ** 2 for p in preds])
    mu = mus.mean(axis=0)
    var = variances.mean(axis=0) + ((mus - mu) ** 2).mean(axis=0)
    return Predictions(first.y_true.copy(), mu, first.ts.copy(), first.patient,
                       first.horizon_min, dict(first.meta), sigma=np.sqrt(var))


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and signed bias in mg/dL."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "n_samples": int(len(err)),
    }


# --------------------------------------------------------------------------
# Hypoglycaemia as a binary alarm
# --------------------------------------------------------------------------

def sample_hypo_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = HYPO,
) -> dict[str, float]:
    true_pos = np.asarray(y_true, dtype=float).ravel() < threshold
    pred_pos = np.asarray(y_pred, dtype=float).ravel() < threshold
    tp = int(np.sum(true_pos & pred_pos))
    fp = int(np.sum(~true_pos & pred_pos))
    fn = int(np.sum(true_pos & ~pred_pos))
    tn = int(np.sum(~true_pos & ~pred_pos))
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if tp and not np.isnan(precision) and not np.isnan(recall)
        else 0.0 if tp + fp + fn else float("nan")
    )
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": float(recall), "precision": float(precision), "f1": float(f1),
        "n_positive": tp + fn,
    }


def alarm_episodes(
    ts: np.ndarray,
    alarm: np.ndarray,
    freq_min: int = SAMPLING_MINUTES,
) -> list[tuple[int, int]]:
    alarm = np.asarray(alarm, dtype=bool).ravel()
    idx = np.flatnonzero(alarm)
    if len(idx) == 0:
        return []
    t = pd.to_datetime(pd.Series(ts)).to_numpy()
    gap_ok = (t[idx[1:]] - t[idx[:-1]]) <= np.timedelta64(freq_min, "m")
    breaks = np.flatnonzero(~gap_ok | (np.diff(idx) > 1))
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def reference_events(
    pred: Predictions,
    threshold: float = HYPO,
    min_event_minutes: int = MIN_EVENT_MINUTES,
) -> pd.DataFrame:
    from tfm.data.quality import glycemic_events

    if len(pred) == 0:
        return pd.DataFrame(
            columns=["start", "end", "n_samples", "minutes", "nadir", "hour"]
        )
    p = pred.sorted()
    frame = pd.DataFrame({"ts": p.target_ts, "value": p.y_true})
    return glycemic_events(
        frame, threshold=threshold, below=True, min_minutes=min_event_minutes
    )


def event_metrics(
    pred: Predictions,
    threshold: float = HYPO,
    tolerance_min: int = DEFAULT_TOLERANCE_MIN,
    min_event_minutes: int = MIN_EVENT_MINUTES,
    events: pd.DataFrame | None = None,
    alarm: np.ndarray | None = None,
) -> dict:
    if tolerance_min >= pred.horizon_min:
        raise ValueError(
            f"tolerance ({tolerance_min} min) must be strictly below the horizon "
            f"({pred.horizon_min} min): the detection window would reach the event "
            "onset itself, where the true value is already below the threshold, and "
            "every model would score a recall of 1.0"
        )
    if alarm is not None:
        alarm = np.asarray(alarm, dtype=bool).ravel()
        if len(alarm) != len(pred):
            raise ValueError(
                f"alarm has {len(alarm)} values for {len(pred)} predictions"
            )
        order = np.argsort(pred.ts)
        alarm = alarm[order]
    p = pred.sorted()
    if events is None:
        events = reference_events(p, threshold, min_event_minutes)
    else:
        events = events.reset_index(drop=True)
    tol = np.timedelta64(int(tolerance_min), "m")
    if alarm is None:
        alarm = p.y_pred < threshold
    target = p.target_ts

    detected = np.zeros(len(events), dtype=bool)
    lead_times: list[float] = []
    for i, start in enumerate(events["start"].to_numpy()):
        in_window = alarm & (target >= start - tol) & (target <= start + tol)
        if in_window.any():
            detected[i] = True
            first = p.ts[in_window].min()
            lead_times.append(float((start - first) / np.timedelta64(1, "m")))
        else:
            lead_times.append(float("nan"))

    # An alarm episode is false only if it neither sits near an event onset nor
    onsets = events["start"].to_numpy()
    false_eps, short_dip_eps, false_spans = 0, 0, []
    gap = int(p.meta.get("episode_gap_min", SAMPLING_MINUTES))
    for a, b in alarm_episodes(p.ts, alarm, freq_min=gap):
        tgt = target[a : b + 1]
        near_event = (
            any(
                ((tgt >= o - tol) & (tgt <= o + tol)).any() for o in onsets
            )
            if len(onsets)
            else False
        )
        if near_event:
            continue
        if (p.y_true[a : b + 1] < threshold).any():
            short_dip_eps += 1
        else:
            false_eps += 1
            false_spans.append((p.ts[a], p.ts[b]))

    n_events = int(len(events))
    n_detected = int(detected.sum())
    recall = n_detected / n_events if n_events else float("nan")
    days = p.span_days
    precision_ep = (
        n_detected / (n_detected + false_eps) if n_detected + false_eps else float("nan")
    )
    f1 = (
        2 * precision_ep * recall / (precision_ep + recall)
        if n_detected
        else 0.0 if n_events or false_eps else float("nan")
    )
    events = events.assign(detected=detected, lead_time_min=lead_times)
    return {
        "patient": p.patient,
        "horizon_min": p.horizon_min,
        "tolerance_min": int(tolerance_min),
        "n_events": n_events,
        "n_detected": n_detected,
        "recall": float(recall),
        "precision_events": float(precision_ep),
        "f1_events": float(f1),
        "false_alarm_episodes": int(false_eps),
        "false_alarms_per_day": float(false_eps / days) if days else float("nan"),
        "alarms_on_short_dips": int(short_dip_eps),
        "median_lead_time_min": float(np.nanmedian(lead_times)) if n_detected else float("nan"),
        "span_days": float(days),
        "false_alarm_spans": false_spans,
        "detected": detected,
        "events": events,
    }


# --------------------------------------------------------------------------
# Error grids
# --------------------------------------------------------------------------

ZONES = ("A", "B", "C", "D", "E")


def clarke_zones(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float).ravel()
    pre = np.asarray(prediction, dtype=float).ravel()
    if ref.shape != pre.shape:
        raise ValueError("reference and prediction must have the same shape")
    if np.any(ref <= 0):
        raise ValueError("reference glucose must be strictly positive")

    zone = np.full(ref.shape, "B", dtype="<U1")

    in_a = ((pre <= 70) & (ref <= 70)) | (np.abs(pre - ref) <= 0.2 * ref)
    in_e = ((ref >= 180) & (pre <= 70)) | ((ref <= 70) & (pre >= 180))
    in_c = (((ref >= 70) & (ref <= 290)) & (pre >= ref + 110)) | (
        ((ref >= 130) & (ref <= 180)) & (pre <= (7 / 5) * ref - 182)
    )
    # Zone D is the clinically dangerous miss: the reference is out of range but
    # the prediction sits in the 70-180 band, so no action would be taken.
    in_d = (
        ((ref >= 240) & (pre >= 70) & (pre <= 180))
        | ((ref <= 175 / 3) & (pre <= 180) & (pre >= 70))
        | (((ref >= 175 / 3) & (ref <= 70)) & (pre >= (6 / 5) * ref))
    )

    # Order matters: A wins over everything, then E, then C, then D.
    zone[in_d] = "D"
    zone[in_c] = "C"
    zone[in_e] = "E"
    zone[in_a] = "A"
    return zone


# Parkes (consensus) Error Grid for type 1 diabetes, as the polylines that
# separate consecutive zones. Vertices from Pfuetzner et al. (2013),
_PARKES_T1_UPPER: dict[str, list[tuple[float, float]]] = {
    "A": [(0, 50), (30, 50), (140, 170), (280, 380), (430, 550)],
    "B": [(0, 60), (30, 60), (50, 80), (70, 110), (260, 550)],
    "C": [(0, 100), (25, 100), (50, 125), (80, 215), (125, 550)],
    "D": [(0, 150), (35, 155), (50, 550)],
}
_PARKES_T1_LOWER: dict[str, list[tuple[float, float]]] = {
    "A": [(50, 0), (50, 30), (170, 145), (385, 300), (550, 450)],
    "B": [(120, 0), (120, 30), (260, 130), (550, 250)],
    "C": [(250, 0), (250, 40), (550, 150)],
}


def _upper_bound(poly: list[tuple[float, float]], ref: np.ndarray) -> np.ndarray:
    """Boundary height at each ``ref``; +inf where the polyline is not defined."""
    xs = np.array([p[0] for p in poly], dtype=float)
    ys = np.array([p[1] for p in poly], dtype=float)
    out = np.interp(ref, xs, ys, left=ys[0], right=ys[-1])
    return np.where(ref > xs[-1], np.inf, out)


def _lower_bound(poly: list[tuple[float, float]], ref: np.ndarray) -> np.ndarray:
    xs = np.array([p[0] for p in poly], dtype=float)
    ys = np.array([p[1] for p in poly], dtype=float)
    keep = np.concatenate([np.diff(xs) > 0, [True]])
    xs_i, ys_i = xs[keep], ys[keep]
    out = np.interp(ref, xs_i, ys_i, left=ys_i[0], right=ys_i[-1])
    return np.where(ref < xs[0], -np.inf, out)


def parkes_zones(
    reference: np.ndarray,
    prediction: np.ndarray,
    diabetes_type: int = 1,
) -> np.ndarray:
    if diabetes_type != 1:
        raise NotImplementedError("only the type 1 Parkes grid is implemented")
    ref = np.asarray(reference, dtype=float).ravel()
    pre = np.asarray(prediction, dtype=float).ravel()
    if ref.shape != pre.shape:
        raise ValueError("reference and prediction must have the same shape")

    level = np.zeros(ref.shape, dtype=int)  # 0 = A .. 4 = E
    for i, name in enumerate(("A", "B", "C", "D")):
        above = pre > _upper_bound(_PARKES_T1_UPPER[name], ref)
        lower = _PARKES_T1_LOWER.get(name)
        below = (
            pre < _lower_bound(lower, ref)
            if lower is not None
            else np.zeros(ref.shape, dtype=bool)
        )
        level = np.maximum(level, np.where(above | below, i + 1, 0))
    return np.array(ZONES, dtype="<U1")[level]


def zone_fractions(zones: np.ndarray) -> dict[str, float]:
    zones = np.asarray(zones)
    n = len(zones)
    out = {z: float(100.0 * np.sum(zones == z) / n) if n else float("nan") for z in ZONES}
    out["AB"] = out["A"] + out["B"]
    out["DE"] = out["D"] + out["E"]
    return out


# --------------------------------------------------------------------------
# Glucose-specific MSE
# --------------------------------------------------------------------------

# Penalty constants from Del Favero, Facchinetti and Cobelli (2012).
GMSE_ALPHA_L, GMSE_ALPHA_H = 1.5, 1.0
GMSE_BETA_L, GMSE_BETA_H = 30.0, 100.0
GMSE_GAMMA_L, GMSE_GAMMA_H = 10.0, 20.0
GMSE_T_L, GMSE_T_H = 85.0, 155.0


def _smooth_step(x: np.ndarray, a: float | np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xi = (2.0 / eps) * (x - a - eps / 2.0)
    return np.select(
        [x <= a, x <= a + eps / 2.0, x <= a + eps],
        [0.0, -0.5 * xi**4 - xi**3 + xi + 0.5, 0.5 * xi**4 - xi**3 + xi + 0.5],
        default=1.0,
    )


def _smooth_step_down(x: np.ndarray, a: float | np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xi = -(2.0 / eps) * (x - a + eps / 2.0)
    return np.select(
        [x <= a - eps, x <= a - eps / 2.0, x <= a],
        [1.0, 0.5 * xi**4 - xi**3 + xi + 0.5, -0.5 * xi**4 - xi**3 + xi + 0.5],
        default=0.0,
    )


def glucose_penalty(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float).ravel()
    pre = np.asarray(prediction, dtype=float).ravel()
    low = _smooth_step_down(ref, GMSE_T_L, GMSE_BETA_L) * _smooth_step(pre, ref, GMSE_GAMMA_L)
    high = _smooth_step(ref, GMSE_T_H, GMSE_BETA_H) * _smooth_step_down(pre, ref, GMSE_GAMMA_H)
    return 1.0 + GMSE_ALPHA_L * low + GMSE_ALPHA_H * high


def gmse(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=float).ravel()
    pre = np.asarray(prediction, dtype=float).ravel()
    value = float(np.mean(glucose_penalty(ref, pre) * (pre - ref) ** 2))
    return {"gmse": value, "grmse": float(np.sqrt(value))}


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def bootstrap_ci(
    successes: np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    x = np.asarray(successes, dtype=bool).ravel()
    n = len(x)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_resamples, n))
    props = x[draws].mean(axis=1)
    half = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(props, [half, 1.0 - half])
    return {
        "point": float(x.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(n),
        "confidence": confidence,
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def evaluate(
    pred: Predictions,
    threshold: float = HYPO,
    tolerance_min: int = DEFAULT_TOLERANCE_MIN,
    seed: int = 0,
    events: pd.DataFrame | None = None,
) -> dict:
    p = pred.sorted()
    ev = event_metrics(
        p, threshold=threshold, tolerance_min=tolerance_min, events=events
    )
    clarke = zone_fractions(clarke_zones(p.y_true, p.y_pred))
    parkes = zone_fractions(parkes_zones(p.y_true, p.y_pred))
    return {
        "patient": p.patient,
        "horizon_min": p.horizon_min,
        "regression": regression_metrics(p.y_true, p.y_pred),
        "gmse": gmse(p.y_true, p.y_pred),
        "sample_hypo": sample_hypo_metrics(p.y_true, p.y_pred, threshold),
        "events": {k: v for k, v in ev.items()
                   if k not in ("detected", "events", "false_alarm_spans")},
        "event_recall_ci": bootstrap_ci(ev["detected"], seed=seed),
        "clarke": clarke,
        "parkes": parkes,
        "_detected": ev["detected"],
        "_event_table": ev["events"],
    }


def evaluate_cohort(
    predictions: list[Predictions],
    threshold: float = HYPO,
    tolerance_min: int = DEFAULT_TOLERANCE_MIN,
    seed: int = 0,
    events_by_patient: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict]:
    if not predictions:
        raise ValueError("no predictions given")
    horizons = {p.horizon_min for p in predictions}
    if len(horizons) != 1:
        raise ValueError(f"mixed horizons in one cohort: {sorted(horizons)}")

    rows, detected_all = [], []
    for p in predictions:
        events = None if events_by_patient is None else events_by_patient[p.patient]
        res = evaluate(p, threshold, tolerance_min, seed, events=events)
        detected_all.append(res["_detected"])
        rows.append(
            {
                "patient": p.patient,
                "n_events": res["events"]["n_events"],
                "n_detected": res["events"]["n_detected"],
                "recall": res["events"]["recall"],
                "false_alarms_per_day": res["events"]["false_alarms_per_day"],
                "median_lead_time_min": res["events"]["median_lead_time_min"],
                "rmse": res["regression"]["rmse"],
                "mae": res["regression"]["mae"],
                "grmse": res["gmse"]["grmse"],
                "clarke_AB": res["clarke"]["AB"],
                "clarke_DE": res["clarke"]["DE"],
                "parkes_AB": res["parkes"]["AB"],
                "parkes_DE": res["parkes"]["DE"],
            }
        )
    table = pd.DataFrame(rows).sort_values("patient").reset_index(drop=True)
    pooled = np.concatenate(detected_all) if detected_all else np.array([], dtype=bool)

    y_true = np.concatenate([p.y_true for p in predictions])
    y_pred = np.concatenate([p.y_pred for p in predictions])
    summary = {
        "horizon_min": horizons.pop(),
        "n_patients": len(predictions),
        "tolerance_min": int(tolerance_min),
        "pooled_event_recall": bootstrap_ci(pooled, seed=seed),
        "rmse_pooled": regression_metrics(y_true, y_pred)["rmse"],
        "rmse_mean_over_patients": float(table["rmse"].mean()),
        "rmse_std_over_patients": float(table["rmse"].std(ddof=1)),
        "false_alarms_per_day_median": float(table["false_alarms_per_day"].median()),
        "clarke_AB_pooled": zone_fractions(clarke_zones(y_true, y_pred))["AB"],
        "parkes_AB_pooled": zone_fractions(parkes_zones(y_true, y_pred))["AB"],
        "parkes_DE_pooled": zone_fractions(parkes_zones(y_true, y_pred))["DE"],
    }
    return table, summary
