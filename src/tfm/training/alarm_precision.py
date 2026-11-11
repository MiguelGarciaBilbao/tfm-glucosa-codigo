"""Phase 7 revisited: precision and F1 of the final model's alarm, on record.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tfm.eval.alarm import predictions_from_experiment
from tfm.eval.metrics import event_metrics
from tfm.eval.protocol import EVENT_TOLERANCE_MIN, scorable_events_by_patient
from tfm.paths import RESULTS_DIR

COUNTS = ("n_events", "n_detected", "false_alarm_episodes", "alarms_on_short_dips", "span_days")


def pooled_precision_f1(rows) -> dict:
    """Recall, precision and F1 from counts summed over patients, not averaged ratios."""
    events = sum(r["n_events"] for r in rows)
    hits = sum(r["n_detected"] for r in rows)
    false = sum(r["false_alarm_episodes"] for r in rows)
    recall = hits / events if events else float("nan")
    precision = hits / (hits + false) if hits + false else float("nan")
    if hits:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0 if events or false else float("nan")
    return {"recall": recall, "precision": precision, "f1": f1}


def _num(value) -> float:
    return float("nan") if value is None else float(value)


def run_alarm_precision_experiment(cfg: dict, out_dir: Path, events_fn=None) -> dict:
    events_fn = events_fn or scorable_events_by_patient
    source_dir = Path(cfg.get("source_dir") or RESULTS_DIR / cfg["source"])
    source = json.loads((source_dir / "metrics.json").read_text())
    tol = int(cfg.get("evaluation", {}).get("event_tolerance_min", EVENT_TOLERANCE_MIN))
    by_horizon = predictions_from_experiment(source_dir.name, results_dir=source_dir.parent)
    events = events_fn(by_horizon)

    out = {}
    for h, preds in sorted(by_horizon.items()):
        out[str(h)] = {}
        for budget, chosen in source["alarm"][str(h)].items():
            alpha = chosen["alpha"]
            per_patient = []
            for pred in preds:
                r = event_metrics(pred, tolerance_min=tol, events=events[pred.patient],
                                  alarm=pred.hypo_probability() > alpha)
                per_patient.append({"patient": pred.patient, **{c: r[c] for c in COUNTS}})
            totals = {c: sum(p[c] for p in per_patient) for c in COUNTS}
            pooled = pooled_precision_f1(per_patient)
            fa_day = (totals["false_alarm_episodes"] / totals["span_days"]
                      if totals["span_days"] else float("nan"))
            same = (np.isclose(pooled["recall"], _num(chosen["recall"]["point"]), equal_nan=True)
                    and np.isclose(fa_day, _num(chosen["false_alarms_per_day"]), equal_nan=True))
            if not same:
                raise RuntimeError(
                    f"PH{h}, budget {budget}: rescoring does not reproduce {source_dir.name} "
                    f"(recall {pooled['recall']:.4f}, {fa_day:.4f} false alarms/day)")
            out[str(h)][budget] = {
                "alpha": alpha, **totals,
                "n_missed": totals["n_events"] - totals["n_detected"],
                **pooled, "false_alarms_per_day": fa_day, "per_patient": per_patient,
            }
    return {"exp_id": cfg["exp_id"], "kind": "alarm_precision", "source": cfg.get("source"),
            "config": cfg, "tolerance_min": tol, "by_horizon": out}
