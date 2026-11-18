"""Phase 8A.4 (E3): adapting the OhioT1DM model to HUPA-UCM patients with N days.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tfm.data import hupa
from tfm.eval.selection import choose, reference_rmse
from tfm.paths import RESULTS_DIR
from tfm.training import external, loso
from tfm.training.global_model import load_checkpoint
from tfm.training.loop import pick_device

#: Fold label given to ``choose``: it excludes one fold, and no Ohio fold is
#: called this, so every one of the twelve takes part in the selection.
NO_FOLD = "__hupa__"

SELECTION_RULES = ("spec", "rmse_guard")



def eligible(ws, n_days: int) -> bool:
    if len(ws) == 0:
        return False
    span = (ws.ts.max() - ws.ts.min()) / np.timedelta64(1, "D")
    return bool(span >= n_days)


def ohio_choices(card: pd.DataFrame, seeds, n_days, budget: float,
                 ref_card: pd.DataFrame | None = None) -> dict:
    """``{(seed, n_days): choice}`` from all twelve Ohio folds' development."""
    out = {}
    for seed in seeds:
        ceiling = None if ref_card is None else reference_rmse(ref_card, NO_FOLD, seed)
        for n in n_days:
            out[(seed, n)] = choose(card, NO_FOLD, seed, n, budget, rmse_ceiling=ceiling)
    return out


def _tag_all(preds: dict) -> dict:
    return {h: external._tag(p) for h, p in preds.items()}


def run_external_adapt_experiment(cfg: dict, out_dir: Path, progress=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rule = cfg.get("selection_rule", "spec")
    if rule not in SELECTION_RULES:
        raise ValueError(f"selection_rule must be one of {SELECTION_RULES}, got {rule!r}")

    source_dir = RESULTS_DIR / cfg["source"]
    source = json.loads((source_dir / "metrics.json").read_text())
    rc = loso.regime_config(source["config"])
    card = pd.read_parquet(source_dir / "dev_scorecard.parquet")
    ref_card = None
    if rule == "rmse_guard":
        if rc.regime == "global":
            raise ValueError("the RMSE guard compares against the global regime itself")
        ref_card = pd.read_parquet(RESULTS_DIR / cfg["reference"] / "dev_scorecard.parquet")
    seeds = [int(s) for s in cfg.get("seeds", source.get("seeds", [0]))]
    n_days = [int(n) for n in cfg.get("n_days", rc.n_days)]
    device = pick_device(rc.adapt.device)

    frames = hupa.load_frames()
    groups = external.patient_groups(hupa.load_summary())
    patients = [p for p in groups[cfg.get("group", "principal")] if p in frames]
    frames = {p: frames[p] for p in patients}
    windows = external.window_sets(frames, cfg.get("imputed_flag", "dropouts"))
    raw_test_events = external.events_by_scope(frames)["test"]

    def events_fn(patient, preds, split="test"):
        return external.scorable(raw_test_events[patient], preds)

    per_n_patients = {n: [p for p in patients if eligible(windows["adapt"][p], n)]
                      for n in n_days}
    choices_by_seed = ohio_choices(card, seeds, n_days, rc.budget, ref_card)

    test_preds, frames_out = {}, []
    for seed in seeds:
        backbone, scaler, target, meta = load_checkpoint(
            RESULTS_DIR / cfg["backbone"] / f"model_seed{seed}.pt")
        for p in patients:
            fold = loso.Fold(p, seed, backbone, scaler, target, meta,
                             windows["adapt"][p], windows["test"][p])
            for n in n_days:
                if p not in per_n_patients[n]:
                    continue
                key = choices_by_seed[(seed, n)]["recipe"]
                preds = _tag_all(loso._test_predictions(fold, rc, n, key, device))
                test_preds.setdefault((seed, n), {})[p] = preds
                for h, pr in preds.items():
                    frames_out.append(pd.DataFrame({
                        "seed": seed, "n_days": n, "patient": p, "horizon_min": h,
                        "ts": pr.ts, "y_true": pr.y_true, "y_pred": pr.y_pred,
                        "sigma": pr.sigma}))
        progress(f"  semilla {seed}: {len(patients)} pacientes adaptados")
    pd.concat(frames_out).to_parquet(out_dir / "predictions.parquet", index=False)

    by_days = {}
    for n in n_days:
        pts = per_n_patients[n]
        choices = {(s, p, n): choices_by_seed[(s, n)] for s in seeds for p in pts}
        by_days[str(n)] = loso._score_days(test_preds, choices, rc, pts, seeds, n, events_fn)
        by_days[str(n)]["patients"] = pts
    return {"exp_id": cfg["exp_id"], "kind": "external_hupa_adapt", "regime": rc.regime,
            "selection_rule": rule, "source": cfg["source"], "backbone": cfg["backbone"],
            "reference": cfg.get("reference"), "config": cfg, "seeds": seeds,
            "patients": patients, "n_days": n_days,
            "excluded_by_days": {str(n): [p for p in patients if p not in per_n_patients[n]]
                                 for n in n_days},
            "ohio_choices": {f"{s}|{n}": c for (s, n), c in choices_by_seed.items()},
            "by_days": by_days}

