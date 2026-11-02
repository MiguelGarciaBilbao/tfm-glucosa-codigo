"""Config-driven experiment runner.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tfm.eval.metrics import evaluate_cohort
from tfm.eval.protocol import (
    EVENT_TOLERANCE_MIN,
    SENSITIVITY_TOLERANCES_MIN,
    scorable_events_by_patient,
)
from tfm.paths import RESULTS_DIR
from tfm.training.baselines import (
    BaselineRun,
    run_lstm_global,
    run_lstm_per_patient,
    run_ridge,
)
from tfm.training.global_model import run_global
from tfm.training.loop import TrainConfig

RUNNERS = {
    "ridge": run_ridge,
    "lstm_per_patient": run_lstm_per_patient,
    "lstm_global": run_lstm_global,
}


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _run_one(cfg: dict, seed: int):
    if cfg.get("kind") == "global":
        data = cfg.get("data", {})
        ckpt = RESULTS_DIR / cfg["exp_id"] / f"model_seed{seed}.pt"
        return run_global(
            architecture=cfg["architecture"],
            seed=seed,
            cfg=TrainConfig(**{**cfg.get("training", {}), "seed": seed}),
            model_kwargs=cfg.get("model", {}),
            loss=cfg.get("loss", {}).get("name", "gaussian_nll"),
            loss_kwargs={k: v for k, v in cfg.get("loss", {}).items() if k != "name"},
            features=data.get("features"),
            processed_dir=data.get("processed_dir"),
            checkpoint_path=ckpt,
        )
    kind = cfg["baseline"]
    if kind == "ridge":
        return run_ridge(alpha=float(cfg.get("model", {}).get("alpha", 1.0)))
    train_cfg = TrainConfig(**{**cfg.get("training", {}), "seed": seed})
    return RUNNERS[kind](seed=seed, cfg=train_cfg, model_kwargs=cfg.get("model", {}))


def _score(run: BaselineRun, tolerance: int) -> dict:
    events = scorable_events_by_patient(run.by_horizon)
    out = {}
    for hmin, preds in sorted(run.by_horizon.items()):
        table, summary = evaluate_cohort(
            preds, tolerance_min=tolerance, events_by_patient=events
        )
        out[str(hmin)] = {"summary": summary, "per_patient": table.to_dict(orient="records")}
    return out


PHASE5_KINDS = ("loso_backbones", "loso", "loso_reselect", "alpha_val", "ensemble",
                "external_hupa", "external_hupa_adapt", "global_sweep", "alarm_precision")


def smoke_config(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["exp_id"] = f"{cfg['exp_id']}-smoke"
    cfg["seeds"] = list(cfg.get("seeds", [0]))[:1]
    kind = cfg.get("kind")
    if kind == "loso_backbones":
        cfg["folds"] = [str(p) for p in cfg["data"]["patients"]][:2]
        cfg["training"] = {**cfg.get("training", {}), "max_epochs": 2, "warmup_epochs": 1}
    elif kind == "loso":
        cfg["backbones"] = f"{cfg['backbones']}-smoke"
        cfg["patients"] = [str(p) for p in cfg["patients"]][:2]
        cfg["n_days"] = list(cfg.get("n_days", [1]))[:2]
        if "adaptation" in cfg:
            a = dict(cfg["adaptation"])
            a["lr"] = list(a["lr"])[:1]
            a["epochs"] = sorted(a["epochs"])[:2]
            a["layers"] = list(a.get("layers", ["all"]))[:1]
            if "lam" in a:
                a["lam"] = list(a["lam"])[:1]
            cfg["adaptation"] = a
    elif kind == "global_sweep":
        chosen = cfg.get("chosen", {})
        cfg["grid"] = {k: [chosen[k]] if k in chosen else list(v)[:1] for k, v in cfg["grid"].items()}
        cfg["training"] = {**cfg.get("training", {}), "max_epochs": 2, "warmup_epochs": 1}
    return cfg


def _main_phase5(cfg: dict, smoke: bool, resume: bool) -> dict:
    from tfm.training import alpha_val, loso

    if smoke:
        cfg = smoke_config(cfg)
    out_dir = RESULTS_DIR / cfg["exp_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = cfg["kind"]
    if kind == "loso_backbones":
        results = loso.run_backbones_experiment(cfg, out_dir, resume=resume)
    elif kind == "loso":
        results = loso.run_regime_experiment(cfg, out_dir)
    elif kind == "loso_reselect":
        results = loso.run_reselect_experiment(cfg, out_dir)
    elif kind == "ensemble":
        from tfm.training import ensemble

        results = ensemble.run_ensemble_experiment(cfg, out_dir)
    elif kind == "external_hupa":
        from tfm.training import external

        results = external.run_external_experiment(cfg, out_dir)
    elif kind == "external_hupa_adapt":
        from tfm.training import external_adapt

        results = external_adapt.run_external_adapt_experiment(cfg, out_dir)
    elif kind == "global_sweep":
        from tfm.training import sweep

        results = sweep.run_sweep_experiment(cfg, out_dir)
    elif kind == "alarm_precision":
        from tfm.training import alarm_precision

        results = alarm_precision.run_alarm_precision_experiment(cfg, out_dir)
    else:
        results = alpha_val.run_alpha_val_experiment(cfg, out_dir)
    (out_dir / "metrics.json").write_text(
        json.dumps(_json_safe(results), indent=2, ensure_ascii=False))
    print(f"\n{cfg['exp_id']}: resultados en {out_dir.name}/metrics.json")
    return results


def _main_persistence(cfg: dict) -> dict:
    from tfm.eval.baselines import persistence_cohort

    exp_id = cfg["exp_id"]
    out_dir = RESULTS_DIR / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluation = cfg.get("evaluation", {})
    tolerance = int(evaluation.get("event_tolerance_min", EVENT_TOLERANCE_MIN))
    split = cfg.get("data", {}).get("split", "test")

    run = BaselineRun(name="persistence", seed=0)
    for horizon_index in range(len(cfg["data"]["horizons_min"])):
        preds = persistence_cohort(horizon_index, split)
        run.by_horizon[preds[0].horizon_min] = preds
    events = scorable_events_by_patient(run.by_horizon)

    by_tolerance = {tol: _score(run, tol)
                    for tol in evaluation.get("sensitivity_tolerances_min", SENSITIVITY_TOLERANCES_MIN)}
    results = {
        "exp_id": exp_id, "model": "persistence", "split": "test oficial (BGLP)",
        "protocol": {"event_tolerance_min": tolerance,
                     "n_scorable_events": sum(len(e) for e in events.values()),
                     "event_onsets": "CGM oficial, interseccion de cobertura PH30/PH60"},
        "by_horizon": _score(run, tolerance),
        "sensitivity": {str(h): {str(tol): scored[str(h)]["summary"]["pooled_event_recall"]
                                 for tol, scored in by_tolerance.items()}
                        for h in sorted(run.by_horizon)},
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(_json_safe(results), indent=2, ensure_ascii=False))
    pd.concat([
        pd.DataFrame({"patient": p.patient, "horizon_min": hmin, "ts": p.ts,
                      "y_true": p.y_true, "y_pred": p.y_pred})
        for hmin, preds in sorted(run.by_horizon.items()) for p in preds
    ]).to_parquet(out_dir / "predictions.parquet", index=False)

    print(f"\n{exp_id}")
    for hmin, block in results["by_horizon"].items():
        s = block["summary"]
        print(f"  PH {hmin:>2}: RMSE {s['rmse_pooled']:.2f}"
              f" | recall {s['pooled_event_recall']['point']:.3f} (n={s['pooled_event_recall']['n']})"
              f" | Parkes A+B {s['parkes_AB_pooled']:.2f} %")
    return results


def main(config_path: str, smoke: bool = False, resume: bool = False) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    if cfg.get("kind") in PHASE5_KINDS:
        return _main_phase5(cfg, smoke, resume)
    if cfg.get("model") == "persistence":
        return _main_persistence(cfg)
    exp_id = cfg["exp_id"]
    out_dir = RESULTS_DIR / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tolerance = int(cfg.get("evaluation", {}).get("event_tolerance_min", EVENT_TOLERANCE_MIN))
    seeds = cfg.get("seeds", [0])

    results = {"exp_id": exp_id,
               "baseline": cfg.get("baseline", cfg.get("architecture")),
               "config": cfg, "seeds": seeds, "by_seed": {}}
    frames, logs = [], []

    first_run = None
    for seed in seeds:
        run = _run_one(cfg, seed)
        if first_run is None:
            first_run = run
        results["by_seed"][str(seed)] = _score(run, tolerance)
        if getattr(run, "n_parameters", 0):
            results["n_parameters"] = run.n_parameters
        for hmin, preds in run.by_horizon.items():
            for p in preds:
                frame = pd.DataFrame({
                    "seed": seed, "patient": p.patient, "horizon_min": hmin,
                    "ts": p.ts, "y_true": p.y_true, "y_pred": p.y_pred})
                if p.sigma is not None:
                    frame["sigma"] = p.sigma
                frames.append(frame)
        for who, log in run.logs.items():
            logs.append(pd.DataFrame({
                "seed": seed, "unit": who, "epoch": log["epoch"],
                "train_loss": log["train_loss"], "val_loss": log["val_loss"]}))
        print(f"  semilla {seed} lista")

    agg = {}
    for hmin in results["by_seed"][str(seeds[0])]:
        rows = [results["by_seed"][str(s)][hmin]["summary"] for s in seeds]
        agg[hmin] = {
            key: {"mean": float(np.mean([r[key] for r in rows])),
                  "std": float(np.std([r[key] for r in rows], ddof=1)) if len(seeds) > 1 else 0.0}
            for key in ("rmse_pooled", "false_alarms_per_day_median",
                        "clarke_AB_pooled", "parkes_AB_pooled", "parkes_DE_pooled")
        }
        recalls = [r["pooled_event_recall"]["point"] for r in rows]
        agg[hmin]["event_recall"] = {
            "mean": float(np.mean(recalls)),
            "std": float(np.std(recalls, ddof=1)) if len(seeds) > 1 else 0.0,
            "n_events": rows[0]["pooled_event_recall"]["n"],
            "ci_first_seed": rows[0]["pooled_event_recall"],
        }
    results["aggregate"] = agg

    sensitivity = {}
    if cfg.get("evaluation", {}).get("sensitivity", True) and first_run is not None:
        for tol in SENSITIVITY_TOLERANCES_MIN:
            scored = _score(first_run, tol)
            sensitivity[str(tol)] = {
                h: v["summary"]["pooled_event_recall"] for h, v in scored.items()
            }
    results["sensitivity"] = sensitivity

    if first_run is not None and frames and "sigma" in frames[0].columns:
        pd.concat(frames).to_parquet(out_dir / "predictions.parquet", index=False)
        from tfm.eval.alarm import alarm_analysis

        results["alarm"] = alarm_analysis(exp_id, seed=seeds[0])

    (out_dir / "metrics.json").write_text(
        json.dumps(_json_safe(results), indent=2, ensure_ascii=False)
    )
    if frames:
        pd.concat(frames).to_parquet(out_dir / "predictions.parquet", index=False)
    if logs:
        pd.concat(logs).to_csv(out_dir / "train_log.csv", index=False)

    print(f"\n{exp_id}")
    for hmin, a in agg.items():
        print(f"  PH {hmin:>2}: RMSE {a['rmse_pooled']['mean']:.2f} +- {a['rmse_pooled']['std']:.2f}"
              f" | recall {a['event_recall']['mean']:.3f} +- {a['event_recall']['std']:.3f}"
              f" (n={a['event_recall']['n_events']})"
              f" | Parkes A+B {a['parkes_AB_pooled']['mean']:.2f} %")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="phase 5 only: 2 folds, 1 seed, reduced grid, into <exp_id>-smoke")
    ap.add_argument("--resume", action="store_true",
                    help="loso_backbones only: reuse checkpoints trained with the same config")
    args = ap.parse_args()
    main(args.config, smoke=args.smoke, resume=args.resume)
