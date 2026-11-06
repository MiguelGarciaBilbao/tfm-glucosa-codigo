"""Phase 4 revisited: the alarm threshold chosen on validation, not on test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tfm.data.loaders import load_processed
from tfm.data.windows import temporal_split
from tfm.eval.alarm import (
    DEFAULT_ALPHAS,
    alarm_counts,
    pooled_from_counts,
    predictions_from_experiment,
    seed_averaged_recall,
)
from tfm.eval.baselines import PATIENTS
from tfm.eval.metrics import event_metrics
from tfm.eval.protocol import EVENT_TOLERANCE_MIN, scorable_events_by_patient
from tfm.eval.selection import best_alpha
from tfm.paths import RESULTS_DIR
from tfm.training.baselines import VAL_FRACTION
from tfm.training.global_model import load_checkpoint, predict_windows
from tfm.training.loop import pick_device

DEFAULT_BUDGETS = (0.5, 1.0, 2.0)


def alpha_from_validation(val_by_h, test_by_h, val_events, test_events, budgets, alphas,
                          tolerance_min) -> dict:
    out = {}
    for h in sorted(val_by_h):
        curve = pooled_from_counts(alarm_counts(val_by_h[h], val_events, alphas,
                                                tolerance_min=tolerance_min))
        out[h] = {}
        for budget in budgets:
            choice = best_alpha(curve, budget)
            detected, false_alarms, days = [], 0, 0.0
            for pred in test_by_h[h]:
                r = event_metrics(pred, tolerance_min=tolerance_min,
                                  events=test_events[pred.patient],
                                  alarm=pred.hypo_probability() > choice["alpha"])
                detected.append(r["detected"])
                false_alarms += r["false_alarm_episodes"]
                days += r["span_days"]
            det = np.concatenate(detected) if detected else np.array([], dtype=bool)
            out[h][budget] = {
                "alpha": choice["alpha"], "within_budget": choice["within_budget"],
                "val_recall": choice["recall"],
                "val_false_alarms_per_day": choice["false_alarms_per_day"],
                "test_recall": float(det.mean()) if len(det) else float("nan"),
                "test_false_alarms_per_day": false_alarms / days if days else float("nan"),
                "test_detected": det,
            }
    return out


def run_alpha_val_experiment(cfg: dict, out_dir: Path, progress=print) -> dict:
    source = cfg["source"]
    seeds = [int(s) for s in cfg.get("seeds", [0])]
    budgets = tuple(float(b) for b in cfg.get("budgets", DEFAULT_BUDGETS))
    alphas = tuple(float(a) for a in cfg.get("alphas", DEFAULT_ALPHAS))
    tol = int(cfg.get("evaluation", {}).get("event_tolerance_min", EVENT_TOLERANCE_MIN))
    device = pick_device(cfg.get("device"))
    per_seed = {}
    for seed in seeds:
        model, scaler, target, _ = load_checkpoint(RESULTS_DIR / source / f"model_seed{seed}.pt")
        val_by_h: dict[int, list] = {}
        for p in PATIENTS:
            _, val = temporal_split(load_processed(p, "train"), VAL_FRACTION)
            for h, pred in predict_windows(model, scaler, target, val, device).items():
                val_by_h.setdefault(h, []).append(pred)
        test_by_h = predictions_from_experiment(source, seed)
        per_seed[seed] = alpha_from_validation(
            val_by_h, test_by_h, scorable_events_by_patient(val_by_h, split="train"),
            scorable_events_by_patient(test_by_h), budgets, alphas, tol)
        progress(f"  semilla {seed} lista")

    first = per_seed[seeds[0]]
    aggregate = {}
    for h in first:
        aggregate[str(h)] = {}
        for b in budgets:
            runs = [per_seed[s][h][b] for s in seeds]
            aggregate[str(h)][str(b)] = {
                "recall": seed_averaged_recall([r["test_detected"] for r in runs]),
                "alpha_by_seed": [r["alpha"] for r in runs],
                "recall_by_seed": [r["test_recall"] for r in runs],
                "false_alarms_per_day_by_seed": [r["test_false_alarms_per_day"] for r in runs],
                "within_budget_by_seed": [r["within_budget"] for r in runs],
            }
    by_seed = {str(s): {str(h): {str(b): {k: v for k, v in r.items() if k != "test_detected"}
                                 for b, r in hb.items()}
                        for h, hb in per_seed[s].items()}
               for s in seeds}
    return {"exp_id": cfg["exp_id"], "kind": "alpha_val", "source": source, "config": cfg,
            "seeds": seeds, "by_seed": by_seed, "aggregate": aggregate}
