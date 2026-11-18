"""Phase 8A.3: the OhioT1DM models on HUPA-UCM, with no retraining.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tfm.data import hupa
from tfm.data.loaders import load_processed
from tfm.data.windows import WindowScaler, concat_windows, temporal_split
from tfm.eval.alarm import DEFAULT_ALPHAS, alarm_counts, pooled_from_counts
from tfm.eval.analysis import alarm_tables, lead_time_summary
from tfm.eval.baselines import PATIENTS as OHIO_PATIENTS
from tfm.eval.calibration import cohort_calibration
from tfm.eval.metrics import (
    Predictions,
    bootstrap_ci,
    ensemble_predictions,
    evaluate_cohort,
    event_metrics,
)
from tfm.eval.protocol import EVENT_TOLERANCE_MIN, covered_instants
from tfm.eval.selection import best_alpha
from tfm.models.linear import RidgeForecaster
from tfm.paths import RESULTS_DIR
from tfm.training.alpha_val import DEFAULT_BUDGETS
from tfm.training.baselines import VAL_FRACTION, TargetScaler
from tfm.training.global_model import load_checkpoint, predict_windows
from tfm.training.loop import pick_device

#: A patient followed for longer than this is reported apart: the three of them
#: hold 699 of the 986 events and would otherwise decide every pooled number.
LONG_FOLLOW_UP_DAYS = 30

SCOPES = ("all", "adapt", "test")


def patient_groups(summary: pd.DataFrame) -> dict[str, list[str]]:
    long = summary["days"] > LONG_FOLLOW_UP_DAYS
    tier = summary["record_tier"]
    principal = summary[(tier == "completo") & ~long]
    pick = lambda m: summary.loc[m, "patient"].tolist()  # noqa: E731
    return {
        "principal": principal["patient"].tolist(),
        "principal_CSII": principal.loc[principal["treatment"] == "CSII", "patient"].tolist(),
        "principal_MDI": principal.loc[principal["treatment"] == "MDI", "patient"].tolist(),
        "largos": pick(long),
        "parcial": pick((tier == "parcial") & ~long),
        "sin_registro": pick((tier == "sin_registro") & ~long),
    }


def window_sets(frames: dict[str, pd.DataFrame], imputed_flag: str = "dropouts") -> dict:
    out = {scope: {} for scope in SCOPES}
    for patient, frame in frames.items():
        parts = hupa.split_frame(frame)
        out["all"][patient] = hupa.patient_windows(frame, imputed_flag=imputed_flag, split="all")
        for scope in ("adapt", "test"):
            out[scope][patient] = hupa.patient_windows(parts[scope], imputed_flag=imputed_flag,
                                                       split=scope)
    return out


def events_by_scope(frames: dict[str, pd.DataFrame]) -> dict:
    out = {scope: {} for scope in SCOPES}
    for patient, frame in frames.items():
        events = hupa.libre_events(frame)
        cut = hupa.split_cut(frame)
        out["all"][patient] = events
        out["adapt"][patient] = events[events["start"] < cut].reset_index(drop=True)
        out["test"][patient] = events[events["start"] >= cut].reset_index(drop=True)
    return out


def scorable(events: pd.DataFrame, predictions: list[Predictions]) -> pd.DataFrame:
    if events.empty:
        return events
    covered = pd.DatetimeIndex(covered_instants(predictions))
    return events[pd.DatetimeIndex(events["start"]).isin(covered)].reset_index(drop=True)


def _tag(pred: Predictions) -> Predictions:
    pred.meta["episode_gap_min"] = hupa.ALARM_EPISODE_GAP_MIN
    return pred


def _wrap(ws, y_pred: np.ndarray) -> dict[int, Predictions]:
    return {hmin: _tag(Predictions(ws.y[:, i].astype(float), y_pred[:, i].astype(float),
                                   ws.ts, ws.patient, hmin))
            for i, hmin in enumerate(ws.horizon_minutes)}


def persistence(ws) -> dict[int, Predictions]:
    value = ws.X[:, -1, ws.feature_names.index("value")]
    return _wrap(ws, np.repeat(value[:, None], len(ws.horizons), axis=1))


def fit_ohio_ridge(alpha: float = 1.0):
    train = concat_windows([temporal_split(load_processed(p, "train"), VAL_FRACTION)[0]
                            for p in OHIO_PATIENTS])
    scaler = WindowScaler().fit(train)
    target = TargetScaler().fit(train.y)
    model = RidgeForecaster(alpha).fit(scaler.transform(train).X, target.transform(train.y))
    return model, scaler, target


def _alarm_score(preds: list[Predictions], events: dict, alpha: float, tol: int) -> dict:
    detected, false_alarms, days = [], 0, 0.0
    for p in preds:
        r = event_metrics(p, tolerance_min=tol, events=events[p.patient],
                          alarm=p.hypo_probability() > alpha)
        detected.append(r["detected"])
        false_alarms += r["false_alarm_episodes"]
        days += r["span_days"]
    det = np.concatenate(detected) if detected else np.array([], dtype=bool)
    tables = alarm_tables(preds, events, alpha, tol)
    per_patient = (tables.events.groupby("patient")["detected"]
                   .agg(n_events="size", n_detected="sum").reset_index()
                   if len(tables.events) else
                   pd.DataFrame(columns=["patient", "n_events", "n_detected"]))
    return {"alpha": float(alpha), "recall": bootstrap_ci(det),
            "false_alarms_per_day": false_alarms / days if days else float("nan"),
            "false_alarm_episodes": int(false_alarms), "span_days": float(days),
            "lead_time": lead_time_summary(tables.events),
            "per_patient": per_patient.to_dict(orient="records")}


def _subset(preds: list[Predictions], patients: list[str]) -> list[Predictions]:
    keep = set(patients)
    return [p for p in preds if p.patient in keep]


def run_external_experiment(cfg: dict, out_dir: Path, progress=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(cfg.get("source_dir") or RESULTS_DIR / cfg["source"])
    seeds = [int(s) for s in cfg.get("seeds", [0])]
    budgets = tuple(float(b) for b in cfg.get("budgets", DEFAULT_BUDGETS))
    alphas = tuple(float(a) for a in cfg.get("alphas", DEFAULT_ALPHAS))
    tol = int(cfg.get("evaluation", {}).get("event_tolerance_min", EVENT_TOLERANCE_MIN))
    imputed_flag = cfg.get("imputed_flag", "dropouts")
    scaler_mode = cfg.get("scaler", "ohio")
    if scaler_mode not in ("ohio", "hupa_adapt"):
        raise ValueError(f"scaler must be 'ohio' or 'hupa_adapt', got {scaler_mode!r}")
    device = pick_device(cfg.get("device"))

    frames = hupa.load_frames()
    patients = list(frames)
    groups = patient_groups(hupa.load_summary())
    windows = window_sets(frames, imputed_flag)
    raw_events = events_by_scope(frames)
    scopes = SCOPES if scaler_mode == "ohio" else ("adapt", "test")

    hupa_scaler = None
    if scaler_mode == "hupa_adapt":
        hupa_scaler = WindowScaler().fit(concat_windows(list(windows["adapt"].values()),
                                                        split="adapt"))

    preds: dict = {}

    def add(model_name, scope, by_h):
        for h, p in by_h.items():
            preds.setdefault(model_name, {}).setdefault(scope, {}).setdefault(h, []).append(p)

    ridge, ridge_scaler, ridge_target = fit_ohio_ridge(float(cfg.get("ridge_alpha", 1.0)))
    for scope in scopes:
        for p in patients:
            ws = windows[scope][p]
            add("persistencia", scope, persistence(ws))
            sc = hupa_scaler or ridge_scaler
            add("ridge_global_ohio", scope,
                _wrap(ws, ridge_target.inverse(ridge.predict(sc.transform(ws).X))))
    progress("  líneas base predichas")

    for seed in seeds:
        model, scaler, target, _ = load_checkpoint(source_dir / f"model_seed{seed}.pt")
        for scope in scopes:
            for p in patients:
                by_h = predict_windows(model, hupa_scaler or scaler, target,
                                       windows[scope][p], device)
                add(f"bigru_seed{seed}", scope, {h: _tag(v) for h, v in by_h.items()})
        progress(f"  semilla {seed} predicha")
    horizons = sorted(preds[f"bigru_seed{seeds[0]}"][scopes[0]])
    for scope in scopes:
        for h in horizons:
            members = [preds[f"bigru_seed{s}"][scope][h] for s in seeds]
            preds.setdefault("ensemble", {}).setdefault(scope, {})[h] = [
                _tag(ensemble_predictions([m[i] for m in members])) for i in range(len(patients))]

    events = {}
    for scope in scopes:
        events[scope] = {}
        for i, p in enumerate(patients):
            both = [preds["ensemble"][scope][h][i] for h in horizons]
            events[scope][p] = scorable(raw_events[scope][p], both)

    regression: dict = {}
    for model_name, by_scope in preds.items():
        regression[model_name] = {}
        for scope in scopes:
            regression[model_name][scope] = {}
            for group, members in groups.items():
                if not members:
                    continue
                regression[model_name][scope][group] = {}
                for h in horizons:
                    sub = _subset(by_scope[scope][h], members)
                    table, summary = evaluate_cohort(
                        sub, tolerance_min=tol,
                        events_by_patient={p: events[scope][p] for p in members})
                    regression[model_name][scope][group][str(h)] = {
                        "summary": summary, "per_patient": table.to_dict(orient="records")}
        progress(f"  {model_name} puntuado")

    ohio = cfg.get("ohio_alpha_source")
    ohio_alpha = {}
    if ohio:
        import json

        alarm = json.loads((RESULTS_DIR / ohio / "metrics.json").read_text())["alarm"]
        ohio_alpha = {int(h): {float(b): float(v["alpha"]) for b, v in per_b.items()}
                      for h, per_b in alarm.items()}

    alarm_block: dict = {}
    for model_name in ["ensemble"] + [f"bigru_seed{s}" for s in seeds]:
        alarm_block[model_name] = {}
        for h in horizons:
            alarm_block[model_name][str(h)] = {}
            for budget in budgets:
                entry: dict = {}
                for group, members in groups.items():
                    if not members:
                        continue
                    g = {}
                    if h in ohio_alpha and budget in ohio_alpha[h]:
                        a = ohio_alpha[h][budget]
                        g["ohio_alpha"] = {scope: _alarm_score(
                            _subset(preds[model_name][scope][h], members), events[scope], a, tol)
                            for scope in scopes if scope != "adapt"}
                    # Recalibrated on the adaptation part of the same group, scored on its test.
                    curve = pooled_from_counts(alarm_counts(
                        _subset(preds[model_name]["adapt"][h], members), events["adapt"],
                        alphas, tolerance_min=tol))
                    choice = best_alpha(curve, budget)
                    g["recalibrated"] = {
                        "alpha": choice["alpha"], "within_budget": choice["within_budget"],
                        "adapt_recall": choice["recall"],
                        "adapt_false_alarms_per_day": choice["false_alarms_per_day"],
                        "test": _alarm_score(_subset(preds[model_name]["test"][h], members),
                                             events["test"], choice["alpha"], tol)}
                    entry[group] = g
                alarm_block[model_name][str(h)][str(budget)] = entry
        progress(f"  alarma de {model_name} puntuada")

    curves = {}
    for scope in scopes:
        curves[scope] = {}
        for h in horizons:
            sub = _subset(preds["ensemble"][scope][h], groups["principal"])
            c = pooled_from_counts(alarm_counts(sub, events[scope], alphas, tolerance_min=tol))
            curves[scope][str(h)] = c.to_dict(orient="records")

    calibration = {scope: {str(h): cohort_calibration(
        _subset(preds["ensemble"][scope][h], groups["principal"])) for h in horizons}
        for scope in scopes}

    frames_out = []
    for model_name in ("ensemble", "persistencia", "ridge_global_ohio"):
        for scope in scopes:
            for h in horizons:
                for p in preds[model_name][scope][h]:
                    frames_out.append(pd.DataFrame({
                        "model": model_name, "scope": scope, "patient": p.patient,
                        "horizon_min": h, "ts": p.ts, "y_true": p.y_true, "y_pred": p.y_pred,
                        "sigma": p.sigma if p.sigma is not None else np.nan}))
    pd.concat(frames_out).to_parquet(out_dir / "predictions.parquet", index=False)

    n_events = {scope: {p: int(len(events[scope][p])) for p in patients} for scope in scopes}
    n_raw = {scope: {p: int(len(raw_events[scope][p])) for p in patients} for scope in scopes}
    return {"exp_id": cfg["exp_id"], "kind": "external_hupa", "config": cfg,
            "source": cfg.get("source"), "seeds": seeds, "patients": patients,
            "groups": groups, "scopes": list(scopes), "imputed_flag": imputed_flag,
            "scaler": scaler_mode, "ohio_alpha": {str(h): {str(b): a for b, a in v.items()}
                                                  for h, v in ohio_alpha.items()},
            "n_events_scorable": n_events, "n_events_found": n_raw,
            "regression": regression, "alarm": alarm_block,
            "operating_curve": curves, "calibration": calibration}


# --------------------------------------------------------------------------
# Reading the results back (notebook 08)
# --------------------------------------------------------------------------

def per_patient(result: dict, model: str, scope: str, group: str, horizon_min: int) -> pd.DataFrame:
    return pd.DataFrame(result["regression"][model][scope][group][str(horizon_min)]["per_patient"])


def paired_test(result: dict, model_a: str, model_b: str, scope: str = "all",
                group: str = "principal", horizon_min: int = 30, metric: str = "rmse") -> dict:
    from tfm.eval.compare import _paired

    a = per_patient(result, model_a, scope, group, horizon_min).set_index("patient")[metric]
    b = per_patient(result, model_b, scope, group, horizon_min).set_index("patient")[metric]
    common = a.index.intersection(b.index)
    return {"model_a": model_a, "model_b": model_b, "metric": metric,
            **_paired(a[common].to_numpy(), b[common].to_numpy())}


def load_predictions(exp_id: str, model: str = "ensemble", scope: str = "all") -> dict:
    df = pd.read_parquet(RESULTS_DIR / exp_id / "predictions.parquet",
                         filters=[("model", "==", model), ("scope", "==", scope)])
    out: dict = {}
    for (h, patient), g in df.groupby(["horizon_min", "patient"], sort=True):
        sigma = g["sigma"].to_numpy() if g["sigma"].notna().all() else None
        out.setdefault(int(h), []).append(_tag(Predictions(
            g["y_true"].to_numpy(), g["y_pred"].to_numpy(), g["ts"].to_numpy(),
            str(patient), int(h), sigma=sigma)))
    return out


def scorable_events(frames: dict[str, pd.DataFrame], by_h: dict, scope: str = "all") -> dict:
    raw = events_by_scope(frames)[scope]
    by_patient: dict = {}
    for h in sorted(by_h):
        for p in by_h[h]:
            by_patient.setdefault(p.patient, []).append(p)
    return {pid: scorable(raw[pid], preds) for pid, preds in by_patient.items()}


def ablate_inputs(patient: str, zero: tuple[str, ...], source: str = "20260916-global-bigru",
                  seeds=(0, 1, 2), device=None) -> dict[int, float]:
    from tfm.data.windows import WindowSet
    from tfm.eval.metrics import regression_metrics

    ws = hupa.patient_windows(hupa.load_frame(patient))
    X = ws.X.copy()
    for name in zero:
        X[:, :, ws.feature_names.index(name)] = 0.0
    ws = WindowSet(X, ws.y, ws.ts, ws.patient, ws.split, ws.feature_names, ws.horizons)
    members = []
    for seed in seeds:
        model, scaler, target, _ = load_checkpoint(RESULTS_DIR / source / f"model_seed{seed}.pt")
        members.append(predict_windows(model, scaler, target, ws, pick_device(device)))
    out = {}
    for h in sorted(members[0]):
        ens = ensemble_predictions([m[h] for m in members])
        out[h] = regression_metrics(ens.y_true, ens.y_pred)["rmse"]
    return out
