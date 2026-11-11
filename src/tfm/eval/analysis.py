"""Complementary analyses of phase 7, computed from stored predictions.

Everything here works on :class:`~tfm.eval.metrics.Predictions` or on the
frames saved as ``predictions.parquet``; nothing reads or writes files, so the
notebook decides what to load and every function is testable on synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tfm.data.parser import PATIENTS_2020
from tfm.eval.alarm import probabilistic_alarm
from tfm.eval.compare import _horizon_block
from tfm.eval.metrics import Predictions, bootstrap_ci, event_metrics
from tfm.eval.protocol import EVENT_TOLERANCE_MIN

#: Night is [00:00, NIGHT_END_HOUR:00); the hour of day survived anonymisation.
NIGHT_END_HOUR = 6

EVENT_COLUMNS = ["patient", "start", "nadir", "detected", "lead_time_min"]
FALSE_ALARM_COLUMNS = ["patient", "start", "end", "minutes"]
SAMPLE_MINUTES = 5


@dataclass
class AlarmTables:
    """Scored events and false-alarm episodes of one alarm at one horizon."""

    events: pd.DataFrame
    false_alarms: pd.DataFrame
    span_days: float


def alarm_tables(preds: list[Predictions], events_by_patient: dict[str, pd.DataFrame],
                 alpha: float, tolerance_min: int = EVENT_TOLERANCE_MIN) -> AlarmTables:
    """One row per scored event and per false-alarm episode of ``P(y<70) > alpha``."""
    ev_frames, fa_rows, span = [], [], 0.0
    for p in preds:
        r = event_metrics(p, tolerance_min=tolerance_min, events=events_by_patient[p.patient],
                          alarm=probabilistic_alarm(p, alpha))
        e = r["events"]
        if len(e):
            ev_frames.append(e.assign(patient=p.patient)[EVENT_COLUMNS])
        for a, b in r["false_alarm_spans"]:
            fa_rows.append({"patient": p.patient, "start": a, "end": b,
                            "minutes": float((b - a) / np.timedelta64(1, "m")) + SAMPLE_MINUTES})
        span += r["span_days"]
    events = (pd.concat(ev_frames, ignore_index=True) if ev_frames
              else pd.DataFrame(columns=EVENT_COLUMNS))
    return AlarmTables(events, pd.DataFrame(fa_rows, columns=FALSE_ALARM_COLUMNS), span)


def lead_time_summary(events: pd.DataFrame) -> dict:
    """Minutes from the first alarm to the onset, over the detected events.

    Negative values are alarms raised after the onset but inside the tolerance.
    """
    lead = events.loc[events["detected"].astype(bool), "lead_time_min"].astype(float).to_numpy()
    if len(lead) == 0:
        return {"n_detected": 0, "median": np.nan, "q1": np.nan, "q3": np.nan,
                "min": np.nan, "max": np.nan, "n_negative": 0}
    q1, med, q3 = np.percentile(lead, [25, 50, 75])
    return {"n_detected": int(len(lead)), "median": float(med), "q1": float(q1),
            "q3": float(q3), "min": float(lead.min()), "max": float(lead.max()),
            "n_negative": int((lead < 0).sum())}


def is_night(ts) -> np.ndarray:
    """True for instants in [00:00, 06:00)."""
    return np.asarray(pd.DatetimeIndex(ts).hour < NIGHT_END_HOUR)


def period_breakdown(tables: AlarmTables, preds: list[Predictions]) -> pd.DataFrame:
    """Recall, false alarms per day and RMSE at night and during the day.

    False alarms are divided by the days covered by the whole test, so the two
    periods add up to the overall rate. The RMSE uses the windows whose target
    instant falls in the period.
    """
    target = np.concatenate([p.target_ts for p in preds])
    err = np.concatenate([p.y_pred - p.y_true for p in preds])
    night_target = is_night(target)
    ev_night = is_night(tables.events["start"]) if len(tables.events) else np.array([], bool)
    fa_night = (is_night(tables.false_alarms["start"]) if len(tables.false_alarms)
                else np.array([], bool))
    rows = {}
    for name, flag in (("noche", True), ("día", False)):
        detected = tables.events["detected"].to_numpy(bool)[ev_night == flag]
        ci = bootstrap_ci(detected)
        mask = night_target == flag
        rows[name] = {
            "n_events": int(len(detected)), "n_detected": int(detected.sum()),
            "recall": ci["point"], "recall_lo": ci["lo"], "recall_hi": ci["hi"],
            "false_alarms_per_day": float((fa_night == flag).sum() / tables.span_days)
            if tables.span_days else np.nan,
            "rmse": float(np.sqrt(np.mean(err[mask] ** 2))) if mask.any() else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


BGLP_KEYS = ("rmse_30", "mae_30", "rmse_60", "mae_60")


def bglp_metrics(frame: pd.DataFrame, patients=PATIENTS_2020) -> dict:
    """RMSE and MAE as the BGLP 2020 ranks them: mean over the six 2020 patients.

    Hameed and Kleinberg (2020) report per-patient RMSEs whose mean is exactly
    the official 19.21, which fixes the aggregation. Several seeds give mean and
    standard deviation of that figure.
    """
    f = frame[frame["patient"].astype(str).isin(patients)]
    groups = f.groupby("seed") if "seed" in f.columns else [(None, f)]
    per_seed = []
    for _, g in groups:
        err = g["y_pred"] - g["y_true"]
        per_patient = (g.assign(se=err ** 2, ae=err.abs())
                       .groupby(["horizon_min", "patient"])
                       .agg(mse=("se", "mean"), mae=("ae", "mean")))
        per_patient["rmse"] = np.sqrt(per_patient["mse"])
        m = per_patient.groupby("horizon_min")[["rmse", "mae"]].mean()
        per_seed.append({f"{k}_{int(h)}": float(m.loc[h, k]) for h in m.index
                         for k in ("rmse", "mae")})
    table = pd.DataFrame(per_seed)
    out = {k: float(table[k].mean()) for k in BGLP_KEYS}
    out["sum"] = float(sum(out[k] for k in BGLP_KEYS))
    out["std"] = {k: float(table[k].std(ddof=1)) if len(table) > 1 else 0.0 for k in BGLP_KEYS}
    out["n_seeds"] = int(len(table))
    return out


def coverage(frame: pd.DataFrame, official_ts: dict[str, np.ndarray], horizon_min: int) -> float:
    """Share of the official test instants that the predictions reach at one horizon.

    BGLP participants predicted every official point; our windows never cross a
    gap longer than 30 minutes, so part of the test is not predicted. Raw sensor
    timestamps drift by seconds, so the official instants are snapped to the
    5 minute grid first, exactly as :func:`tfm.data.quality.to_regular_grid` does
    when the windows are built.
    """
    f = frame[frame["horizon_min"] == horizon_min]
    if "seed" in f.columns:
        f = f[f["seed"] == f["seed"].min()]
    hit = total = 0
    for patient, official in official_ts.items():
        target = (pd.DatetimeIndex(f.loc[f["patient"].astype(str) == patient, "ts"])
                  + pd.Timedelta(minutes=horizon_min))
        official = pd.DatetimeIndex(official).round(f"{SAMPLE_MINUTES}min").unique()
        hit += int(official.isin(target).sum())
        total += len(official)
    return hit / total if total else float("nan")


def pick_examples(tables: AlarmTables) -> dict:
    """Four episodes chosen by a fixed rule, never by hand; ties go to the earliest."""
    ev, fa = tables.events, tables.false_alarms
    det = ev[ev["detected"].astype(bool)].sort_values(["lead_time_min", "start"])
    miss = ev[~ev["detected"].astype(bool)].sort_values(["nadir", "start"])
    longest = fa.sort_values(["minutes", "start"], ascending=[False, True])
    by_max = det.sort_values(["lead_time_min", "start"], ascending=[False, True])
    median = None
    if len(det):
        # lower median of the lead times, then the earliest event with that value
        value = det["lead_time_min"].iloc[(len(det) - 1) // 2]
        median = det[det["lead_time_min"] == value].sort_values("start").iloc[0]
    return {
        "median_lead": median,
        "max_lead": by_max.iloc[0] if len(det) else None,
        "worst_miss": miss.iloc[0] if len(miss) else None,
        "longest_false_alarm": longest.iloc[0] if len(longest) else None,
    }


def master_row(label: str, result: dict, horizon_min: int, alarm: dict | None = None) -> dict:
    """One row of the master table: regression error next to the clinical metrics.

    RMSE, Clarke and Parkes are pooled over patients (mean over seeds when there
    are several); MAE and gRMSE are means over patients, as the column names say.
    ``recall_mu70`` is the ``mu < 70`` rule; the alarm columns are filled only
    for models with a probabilistic alarm chosen on validation.
    """
    summary, table = _horizon_block(result, horizon_min)
    agg = result.get("aggregate", {}).get(str(horizon_min), {})

    def pooled(key):
        return agg.get(key, {}).get("mean", summary[key])

    row = {
        "modelo": label, "PH": horizon_min, "rmse": pooled("rmse_pooled"),
        "mae_media_pacientes": float(table["mae"].mean()),
        "grmse_media_pacientes": float(table["grmse"].mean()),
        "clarke_AB": pooled("clarke_AB_pooled"), "parkes_DE": pooled("parkes_DE_pooled"),
        "recall_mu70": agg.get("event_recall", {}).get("mean",
                                                       summary["pooled_event_recall"]["point"]),
        "recall_alarma": np.nan, "ic_lo": np.nan, "ic_hi": np.nan, "n_eventos": np.nan,
        "fa_dia_alarma": np.nan, "anticipacion_mediana_min": np.nan,
    }
    if alarm is not None:
        r = alarm["recall"]
        row.update(recall_alarma=r["point"], ic_lo=r["lo"], ic_hi=r["hi"], n_eventos=r["n"],
                   fa_dia_alarma=alarm["false_alarms_per_day"],
                   anticipacion_mediana_min=alarm.get("median_lead_time_min", np.nan))
    return row
