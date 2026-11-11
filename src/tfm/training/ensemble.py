"""Phase 7: the seed ensemble of the global model, with alpha chosen on validation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tfm.data.loaders import load_processed
from tfm.data.windows import temporal_split
from tfm.eval.alarm import DEFAULT_ALPHAS
from tfm.eval.analysis import alarm_tables, lead_time_summary
from tfm.eval.baselines import PATIENTS
from tfm.eval.calibration import cohort_calibration
from tfm.eval.metrics import bootstrap_ci, ensemble_predictions, evaluate_cohort
from tfm.eval.protocol import EVENT_TOLERANCE_MIN, scorable_events_by_patient
from tfm.paths import RESULTS_DIR
from tfm.training.alpha_val import DEFAULT_BUDGETS, alpha_from_validation
from tfm.training.baselines import VAL_FRACTION
from tfm.training.global_model import load_checkpoint, predict_windows
from tfm.training.loop import pick_device


def _predict(source_dir, seed, patients, processed_dir, device) -> dict:
    model, scaler, target, _ = load_checkpoint(Path(source_dir) / f"model_seed{seed}.pt")
    out = {"val": {}, "test": {}}
    for p in patients:
        _, val = temporal_split(load_processed(p, "train", processed_dir=processed_dir),
                                VAL_FRACTION)
        test = load_processed(p, "test", processed_dir=processed_dir)
        for split, ws in (("val", val), ("test", test)):
            for h, pred in predict_windows(model, scaler, target, ws, device).items():
                out[split].setdefault(h, []).append(pred)
    return out


def _alarm_block(by_split, val_events, test_events, budgets, alphas, tol) -> dict:
    chosen = alpha_from_validation(by_split["val"], by_split["test"], val_events, test_events,
                                   budgets, alphas, tol)
    out = {}
    for h, per_budget in chosen.items():
        out[str(h)] = {}
        for budget, c in per_budget.items():
            tables = alarm_tables(by_split["test"][h], test_events, c["alpha"], tol)
            per_patient = (tables.events.groupby("patient")["detected"]
                           .agg(n_events="size", n_detected="sum").reset_index()
                           if len(tables.events) else
                           pd.DataFrame(columns=["patient", "n_events", "n_detected"]))
            out[str(h)][str(budget)] = {
                "alpha": c["alpha"], "within_budget": c["within_budget"],
                "val_recall": c["val_recall"],
                "val_false_alarms_per_day": c["val_false_alarms_per_day"],
                "recall": bootstrap_ci(c["test_detected"]),
                "false_alarms_per_day": c["test_false_alarms_per_day"],
                "lead_time": lead_time_summary(tables.events),
                "per_patient": per_patient.to_dict(orient="records"),
            }
    return out


def run_ensemble_experiment(cfg: dict, out_dir: Path, events_fn=None, progress=print) -> dict:
    events_fn = events_fn or scorable_events_by_patient
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(cfg.get("source_dir") or RESULTS_DIR / cfg["source"])
    seeds = [int(s) for s in cfg.get("seeds", [0])]
    patients = [str(p) for p in cfg.get("patients", PATIENTS)]
    processed_dir = cfg.get("data", {}).get("processed_dir")
    budgets = tuple(float(b) for b in cfg.get("budgets", DEFAULT_BUDGETS))
    alphas = tuple(float(a) for a in cfg.get("alphas", DEFAULT_ALPHAS))
    tol = int(cfg.get("evaluation", {}).get("event_tolerance_min", EVENT_TOLERANCE_MIN))
    device = pick_device(cfg.get("device"))

    members = {}
    for seed in seeds:
        members[seed] = _predict(source_dir, seed, patients, processed_dir, device)
        progress(f"  semilla {seed} predicha")
    horizons = sorted(members[seeds[0]]["test"])
    ens = {split: {h: [ensemble_predictions([members[s][split][h][i] for s in seeds])
                       for i in range(len(patients))] for h in horizons}
           for split in ("val", "test")}
    val_events = events_fn(ens["val"], split="train")
    test_events = events_fn(ens["test"])

    by_horizon, calibration, frames = {}, {}, []
    for h in horizons:
        table, summary = evaluate_cohort(ens["test"][h], tolerance_min=tol,
                                         events_by_patient=test_events)
        by_horizon[str(h)] = {"summary": summary, "per_patient": table.to_dict(orient="records")}
        calibration[str(h)] = cohort_calibration(ens["test"][h])
        for p in ens["test"][h]:
            frames.append(pd.DataFrame({"patient": p.patient, "horizon_min": h, "ts": p.ts,
                                        "y_true": p.y_true, "y_pred": p.y_pred,
                                        "sigma": p.sigma}))
    pd.concat(frames).to_parquet(out_dir / "predictions.parquet", index=False)

    member_blocks = {}
    for seed in seeds:
        alarm = _alarm_block(members[seed], val_events, test_events, budgets, alphas, tol)
        member_blocks[str(seed)] = {}
        for h in horizons:
            table, _ = evaluate_cohort(members[seed]["test"][h], tolerance_min=tol,
                                       events_by_patient=test_events)
            member_blocks[str(seed)][str(h)] = {
                "per_patient": table.to_dict(orient="records"),
                "alarm": {b: {"alpha": a["alpha"], "recall": a["recall"],
                              "per_patient": a["per_patient"]}
                          for b, a in alarm[str(h)].items()}}
        progress(f"  semilla {seed} puntuada")

    return {"exp_id": cfg["exp_id"], "kind": "ensemble", "source": cfg.get("source"),
            "config": cfg, "seeds": seeds, "patients": patients,
            "by_horizon": by_horizon,
            "alarm": _alarm_block(ens, val_events, test_events, budgets, alphas, tol),
            "calibration": calibration, "members": member_blocks}
