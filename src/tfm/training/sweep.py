"""Phase 4 revisited: the hyperparameter sweep of the global model, on record.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tfm.eval.baselines import PATIENTS
from tfm.eval.calibration import crps_gaussian, gaussian_nll
from tfm.eval.metrics import regression_metrics
from tfm.training.global_model import ARCHITECTURES, predict_windows, prepare_cohort
from tfm.training.loop import TrainConfig, train_model
from tfm.training.losses import build_loss

SWEEPABLE = ("model", "training")


def _point_id(overrides: dict) -> str:
    return "|".join(f"{key.split('.', 1)[1]}={value}" for key, value in overrides.items())


def grid_points(cfg: dict) -> list[dict]:
    grid = cfg.get("grid", {})
    bad = [k for k in grid if "." not in k or k.split(".", 1)[0] not in SWEEPABLE]
    if bad:
        raise ValueError(f"grid keys must start with 'model.' or 'training.': {bad}")
    points = []
    for values in itertools.product(*grid.values()):
        overrides = dict(zip(grid, values))
        point = {block: dict(cfg.get(block, {})) for block in SWEEPABLE}
        for key, value in overrides.items():
            block, name = key.split(".", 1)
            point[block][name] = value
        points.append({"point_id": _point_id(overrides), "overrides": overrides, **point})
    return points


def _chosen_id(cfg: dict, points: list[dict]) -> str:
    chosen = cfg.get("chosen", {})
    matches = [p["point_id"] for p in points
               if all(p["overrides"].get(k) == v for k, v in chosen.items())]
    if len(matches) != 1:
        raise ValueError(f"chosen {chosen} must match exactly one grid point, "
                         f"it matches {len(matches)}")
    return matches[0]


def _mean_sd(values) -> dict:
    values = [float(v) for v in values]
    return {"mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def _summarise(points: list[dict], runs: list[dict], chosen_id: str) -> list[dict]:
    ranking = []
    for point in points:
        rs = [r for r in runs if r["point_id"] == point["point_id"]]
        horizons = list(rs[0]["val_rmse"])
        ranking.append({
            "point_id": point["point_id"], "overrides": point["overrides"],
            "chosen": point["point_id"] == chosen_id,
            "n_parameters": rs[0]["n_parameters"],
            "val_nll": _mean_sd(r["val_nll"] for r in rs),
            "val_nll_by_seed": {str(r["seed"]): r["val_nll"] for r in rs},
            "val_rmse": {h: _mean_sd(r["val_rmse"][h] for r in rs) for h in horizons},
            "val_crps": {h: _mean_sd(r["val_crps"][h] for r in rs) for h in horizons},
            "epochs_run": _mean_sd(r["epochs_run"] for r in rs),
        })
    ranking.sort(key=lambda r: r["val_nll"]["mean"])
    for k, r in enumerate(ranking, 1):
        r["rank"] = k
    for h in ranking[0]["val_rmse"]:
        by_rmse = sorted(ranking, key=lambda r: r["val_rmse"][h]["mean"])
        for k, r in enumerate(by_rmse, 1):
            r.setdefault("rank_rmse", {})[h] = k
    return ranking


def run_sweep_experiment(cfg: dict, out_dir: Path, progress=print) -> dict:
    points = grid_points(cfg)
    chosen_id = _chosen_id(cfg, points)
    seeds = [int(s) for s in cfg.get("seeds", [0])]
    architecture = cfg.get("architecture", "bigru")
    data = cfg.get("data", {})
    train, val, _, scaler, target = prepare_cohort(
        patients=[str(p) for p in data.get("patients", PATIENTS)],
        features=data.get("features"), processed_dir=data.get("processed_dir"),
        include_test=False,
    )
    X_train, y_train = scaler.transform(train).X, target.transform(train.y)
    X_val, y_val = scaler.transform(val).X, target.transform(val.y)
    loss_cfg = dict(cfg.get("loss", {"name": "gaussian_nll"}))
    loss_fn = build_loss(loss_cfg.pop("name"), **loss_cfg)

    runs, logs = [], []
    for i, point in enumerate(points, 1):
        for seed in seeds:
            start = time.perf_counter()
            run_cfg = TrainConfig(**{**point["training"], "seed": seed})
            model = ARCHITECTURES[architecture](
                n_features=X_train.shape[-1], n_horizons=y_train.shape[-1], **point["model"])
            model, log = train_model(
                model, X_train, y_train, X_val, y_val, run_cfg, loss_fn=loss_fn,
                warmup_loss_fn=build_loss("mse") if run_cfg.warmup_epochs else None)
            preds = predict_windows(model, scaler, target, val, run_cfg.device)
            run = {
                "point_id": point["point_id"], "seed": seed,
                "val_nll": float(np.mean([gaussian_nll(p) for p in preds.values()])),
                "val_nll_by_horizon": {str(h): gaussian_nll(p) for h, p in preds.items()},
                "val_rmse": {str(h): regression_metrics(p.y_true, p.y_pred)["rmse"]
                             for h, p in preds.items()},
                "val_crps": {str(h): crps_gaussian(p) for h, p in preds.items()},
                "val_beta_nll": log["best_val_loss"], "best_epoch": log["best_epoch"],
                "epochs_run": log["epochs_run"], "device": log["device"],
                "n_parameters": sum(p.numel() for p in model.parameters()),
                "seconds": time.perf_counter() - start,
            }
            runs.append(run)
            logs.append(pd.DataFrame({
                "point_id": point["point_id"], "seed": seed, "epoch": log["epoch"],
                "train_loss": log["train_loss"], "val_loss": log["val_loss"]}))
            rmse = " / ".join(f"{v:.2f}" for v in run["val_rmse"].values())
            progress(f"  [{i}/{len(points)}] {point['point_id']} semilla {seed}: "
                     f"NLL val {run['val_nll']:.4f}, RMSE val {rmse} ({run['seconds']:.0f} s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(logs).to_csv(out_dir / "train_log.csv", index=False)
    ranking = _summarise(points, runs, chosen_id)
    best, chosen = ranking[0], next(r for r in ranking if r["chosen"])
    return {
        "exp_id": cfg["exp_id"], "kind": "global_sweep", "config": cfg, "seeds": seeds,
        "selection": "NLL gaussiana en validación (mg/dL), media de los horizontes",
        "n_windows": {"train": len(train), "val": len(val)},
        "chosen": {"point_id": chosen["point_id"], "rank": chosen["rank"],
                   "rank_rmse": chosen["rank_rmse"], "n_points": len(ranking),
                   "best_point_id": best["point_id"],
                   "val_nll_minus_best": chosen["val_nll"]["mean"] - best["val_nll"]["mean"]},
        "ranking": ranking, "runs": runs,
    }
