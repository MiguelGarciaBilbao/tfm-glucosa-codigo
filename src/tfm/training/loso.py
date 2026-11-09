"""Leave-one-subject-out: the phase 5 experiment.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch import nn

from tfm.data.loaders import load_processed
from tfm.data.windows import WindowScaler, WindowSet
from tfm.eval.alarm import DEFAULT_ALPHAS, alarm_counts, seed_averaged_recall
from tfm.eval.baselines import PATIENTS
from tfm.eval.metrics import evaluate_cohort, event_metrics
from tfm.eval.protocol import EVENT_TOLERANCE_MIN, scorable_events
from tfm.eval.selection import SCORECARD_COLUMNS, choose, reference_rmse
from tfm.paths import RESULTS_DIR
from tfm.training.baselines import TargetScaler
from tfm.training.global_model import (
    ARCHITECTURES,
    load_checkpoint,
    predict_windows,
    prepare_cohort,
    save_checkpoint,
)
from tfm.training.loop import TrainConfig, pick_device, set_seed, train_model
from tfm.training.losses import build_loss, hypo_weights
from tfm.training.personalize import (
    DEV_START_DAYS,
    AdaptConfig,
    Recipe,
    adaptation_split,
    fit_with_snapshots,
)


def fold_patients(held_out: str, patients=PATIENTS) -> list[str]:
    patients = [str(p) for p in patients]
    if held_out not in patients:
        raise ValueError(f"{held_out!r} is not one of {patients}")
    return [p for p in patients if p != held_out]


def backbone_path(exp_dir: Path, held_out: str, seed: int) -> Path:
    return Path(exp_dir) / f"model_fold{held_out}_seed{seed}.pt"


def config_fingerprint(cfg: dict) -> str:
    keys = ("architecture", "model", "loss", "training", "data")
    blob = json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def train_backbone(held_out: str, seed: int, cfg: TrainConfig, architecture: str = "bigru",
                   model_kwargs: dict | None = None, loss_kwargs: dict | None = None,
                   patients=PATIENTS, processed_dir=None):
    others = fold_patients(held_out, patients)
    train, val, _, scaler, target = prepare_cohort(patients=others, processed_dir=processed_dir)
    set_seed(seed)  # the initial weights depend on the seed too
    model = ARCHITECTURES[architecture](
        n_features=train.X.shape[-1], n_horizons=train.y.shape[-1], **(model_kwargs or {}))
    run_cfg = TrainConfig(**{**cfg.__dict__, "seed": seed, "history": {}})
    model, log = train_model(
        model,
        scaler.transform(train).X, target.transform(train.y),
        scaler.transform(val).X, target.transform(val.y),
        run_cfg, loss_fn=build_loss("gaussian_nll", **(loss_kwargs or {})),
        warmup_loss_fn=build_loss("mse") if run_cfg.warmup_epochs else None,
    )
    return model.cpu(), scaler, target, log


def run_backbones_experiment(cfg: dict, out_dir: Path, resume: bool = False,
                             progress=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = cfg.get("data", {})
    patients = [str(p) for p in data.get("patients", PATIENTS)]
    held_out = [str(p) for p in cfg.get("folds", patients)]
    architecture = cfg.get("architecture", "bigru")
    model_kwargs = cfg.get("model", {})
    loss_kwargs = {k: v for k, v in cfg.get("loss", {}).items() if k != "name"}
    train_cfg = TrainConfig(**cfg.get("training", {}))
    fingerprint = config_fingerprint(cfg)

    metrics_path, log_path = out_dir / "metrics.json", out_dir / "train_log.csv"
    folds = {}
    logs = []
    if resume and metrics_path.exists():
        folds = json.loads(metrics_path.read_text()).get("folds", {})
    if resume and log_path.exists():
        logs.append(pd.read_csv(log_path, dtype={"unit": str}))

    for seed in cfg.get("seeds", [0]):
        for p in held_out:
            path = backbone_path(out_dir, p, seed)
            if resume and path.exists():
                _, _, _, meta = load_checkpoint(path)
                if meta.get("fingerprint") != fingerprint:
                    raise ValueError(f"{path} was trained with a different configuration; "
                                     "delete it or run without --resume")
                progress(f"  pliegue {p} semilla {seed}: reutilizado")
                continue
            model, scaler, target, log = train_backbone(
                p, seed, train_cfg, architecture, model_kwargs, loss_kwargs,
                patients, data.get("processed_dir"))
            save_checkpoint(path, model, scaler, target, architecture, model_kwargs, seed,
                            extra={"fingerprint": fingerprint, "held_out": p,
                                   "trained_on": fold_patients(p, patients)})
            folds[f"{p}/{seed}"] = {"best_val_loss": log["best_val_loss"],
                                    "best_epoch": log["best_epoch"],
                                    "epochs_run": log["epochs_run"]}
            logs.append(pd.DataFrame({"seed": seed, "unit": p, "epoch": log["epoch"],
                                      "train_loss": log["train_loss"],
                                      "val_loss": log["val_loss"]}))
            progress(f"  pliegue {p} semilla {seed}: val {log['best_val_loss']:.4f} "
                     f"({log['epochs_run']} épocas)")
    if logs:
        pd.concat(logs).drop_duplicates(["seed", "unit", "epoch"], keep="last").to_csv(
            log_path, index=False)
    result = {"exp_id": cfg["exp_id"], "kind": "loso_backbones", "config": cfg,
              "fingerprint": fingerprint, "folds": folds}
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


REGIMES = ("global", "individual", "ft", "ft_weighted")

#: Metrics averaged over seeds in each regime's ``aggregate`` block.
AGGREGATE_KEYS = ("rmse_pooled", "clarke_AB_pooled", "parkes_AB_pooled", "parkes_DE_pooled")


@dataclass(frozen=True)
class RegimeConfig:
    regime: str
    n_days: tuple[int, ...]
    recipes: tuple[Recipe, ...]
    epochs: tuple[int, ...]
    adapt: AdaptConfig
    alphas: tuple[float, ...] = tuple(DEFAULT_ALPHAS)
    budget: float = 1.0
    tolerance_min: int = EVENT_TOLERANCE_MIN
    dev_start_days: int = DEV_START_DAYS

    @property
    def grid(self) -> dict[str, tuple[Recipe | None, int]]:
        if self.regime == "global":
            return {"none": (None, 0)}
        return {r.key(e): (r, e) for r in self.recipes for e in self.epochs}


def regime_config(cfg: dict) -> RegimeConfig:
    regime = cfg.get("regime")
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    sel, ev = cfg.get("selection", {}), cfg.get("evaluation", {})
    common = {
        "alphas": tuple(float(a) for a in sel.get("alphas", DEFAULT_ALPHAS)),
        "budget": float(sel.get("budget", 1.0)),
        "dev_start_days": int(sel.get("dev_start_days", DEV_START_DAYS)),
        "tolerance_min": int(ev.get("event_tolerance_min", EVENT_TOLERANCE_MIN)),
    }
    if regime == "global":
        return RegimeConfig(regime, (0,), (), (), AdaptConfig(), **common)
    a = cfg.get("adaptation", {})
    layers = tuple(a.get("layers", ["all"]))
    lams = tuple(float(x) for x in a.get("lam", [0.0]))
    if regime == "individual" and layers != ("all",):
        raise ValueError("the individual regime trains every layer: layers must be [all]")
    if regime == "ft_weighted" and any(lam <= 0 for lam in lams):
        raise ValueError("ft_weighted needs every lam > 0")
    if regime in ("ft", "individual") and any(lam != 0 for lam in lams):
        raise ValueError(f"{regime} uses the plain loss: lam must be [0]")
    recipes = tuple(Recipe(float(lr), layer, lam)
                    for layer in layers for lr in a["lr"] for lam in lams)
    adapt = AdaptConfig(
        batch_size=int(a.get("batch_size", 64)),
        weight_decay=float(a.get("weight_decay", 0.0)),
        grad_clip=float(a.get("grad_clip", 1.0)),
        warmup_epochs=int(a.get("warmup_epochs", 0)),
        beta=float(cfg.get("loss", {}).get("beta", 0.5)),
        device=a.get("device"),
    )
    return RegimeConfig(regime, tuple(int(n) for n in cfg["n_days"]), recipes,
                        tuple(sorted(int(e) for e in a["epochs"])), adapt, **common)


@dataclass
class Fold:
    patient: str
    seed: int
    backbone: nn.Module
    scaler: WindowScaler
    target: TargetScaler
    meta: dict
    train: WindowSet
    test: WindowSet


def load_folds(backbones_dir, patients, seeds, processed_dir=None) -> list[Fold]:
    folds = []
    for seed in seeds:
        for p in patients:
            model, scaler, target, meta = load_checkpoint(backbone_path(backbones_dir, p, seed))
            folds.append(Fold(p, int(seed), model, scaler, target, meta,
                              load_processed(p, "train", processed_dir=processed_dir),
                              load_processed(p, "test", processed_dir=processed_dir)))
    return folds


def _adapt(fold: Fold, rc: RegimeConfig, n_days: int, recipe: Recipe, epochs) -> dict:
    adapt, _ = adaptation_split(fold.train, n_days, rc.dev_start_days)
    if len(adapt) == 0:
        raise ValueError(f"patient {fold.patient} has no windows in its first {n_days} days")
    if rc.regime == "individual":
        scaler, target = WindowScaler().fit(adapt), TargetScaler().fit(adapt.y)
        set_seed(fold.seed)
        model = ARCHITECTURES[fold.meta["architecture"]](
            n_features=adapt.X.shape[-1], n_horizons=adapt.y.shape[-1],
            **fold.meta["model_kwargs"])
    else:
        scaler, target = fold.scaler, fold.target
        model = copy.deepcopy(fold.backbone)
    snapshots = fit_with_snapshots(
        model, scaler.transform(adapt).X, target.transform(adapt.y),
        hypo_weights(adapt.y, recipe.lam), recipe, tuple(epochs), rc.adapt, fold.seed)
    out = {}
    for ep, state in snapshots.items():
        snap = copy.deepcopy(model)
        snap.load_state_dict(state)
        out[ep] = (snap, scaler, target)
    return out


def _candidates(fold: Fold, rc: RegimeConfig, n_days: int):
    if rc.regime == "global":
        yield "none", (fold.backbone, fold.scaler, fold.target)
        return
    for recipe in rc.recipes:
        for ep, trio in _adapt(fold, rc, n_days, recipe, rc.epochs).items():
            yield recipe.key(ep), trio


def dev_scorecard(folds, rc: RegimeConfig, events_fn=scorable_events,
                  progress=print) -> pd.DataFrame:
    device = pick_device(rc.adapt.device)
    frames = []
    for fold in folds:
        _, dev = adaptation_split(fold.train, 0, rc.dev_start_days)
        dev_events = None
        for n_days in rc.n_days:
            for key, (model, scaler, target) in _candidates(fold, rc, n_days):
                preds = predict_windows(model, scaler, target, dev, device)
                if dev_events is None:
                    dev_events = events_fn(fold.patient, list(preds.values()), "train")
                counts = alarm_counts(list(preds.values()), {fold.patient: dev_events},
                                      rc.alphas, tolerance_min=rc.tolerance_min)
                sse = {h: float(np.sum((p.y_true - p.y_pred) ** 2)) for h, p in preds.items()}
                size = {h: len(p) for h, p in preds.items()}
                frames.append(counts.drop(columns="patient").assign(
                    fold=fold.patient, seed=fold.seed, n_days=n_days, recipe=key,
                    sse=counts["horizon_min"].map(sse), n_pred=counts["horizon_min"].map(size)))
        progress(f"  desarrollo: pliegue {fold.patient} semilla {fold.seed}")
    return pd.concat(frames, ignore_index=True)[list(SCORECARD_COLUMNS)]


def _test_predictions(fold: Fold, rc: RegimeConfig, n_days: int, key: str, device) -> dict:
    recipe, epochs = rc.grid[key]
    if recipe is None:
        model, scaler, target = fold.backbone, fold.scaler, fold.target
    else:
        model, scaler, target = _adapt(fold, rc, n_days, recipe, (epochs,))[epochs]
    return predict_windows(model, scaler, target, fold.test, device)


def _mean_std(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def _score_days(test_preds, choices, rc, patients, seeds, n_days, events_fn) -> dict:
    block = {"by_seed": {}, "selection": {}, "alarm": {}, "aggregate": {}}
    detected, alarm_rows, summaries = {}, {}, {}
    for seed in seeds:
        per_p = test_preds[(seed, n_days)]
        events = {p: events_fn(p, list(per_p[p].values()), "test") for p in patients}
        horizons = sorted(per_p[patients[0]])
        block["by_seed"][str(seed)] = {}
        block["selection"][str(seed)] = {p: choices[(seed, p, n_days)] for p in patients}
        for h in horizons:
            table, summary = evaluate_cohort([per_p[p][h] for p in patients],
                                             tolerance_min=rc.tolerance_min,
                                             events_by_patient=events)
            block["by_seed"][str(seed)][str(h)] = {
                "summary": summary, "per_patient": table.to_dict(orient="records")}
            summaries.setdefault(h, []).append(summary)
            rows, det = [], []
            for p in patients:
                pred, alpha = per_p[p][h], choices[(seed, p, n_days)]["alpha"][h]
                r = event_metrics(pred, tolerance_min=rc.tolerance_min, events=events[p],
                                  alarm=pred.hypo_probability() > alpha)
                det.append(r["detected"])
                rows.append({"patient": p, "alpha": alpha, "n_events": r["n_events"],
                             "n_detected": r["n_detected"],
                             "false_alarm_episodes": r["false_alarm_episodes"],
                             "span_days": r["span_days"]})
            detected.setdefault(h, []).append(np.concatenate(det))
            alarm_rows.setdefault(h, {})[str(seed)] = rows
    for h in detected:
        by_seed = alarm_rows[h]
        block["alarm"][str(h)] = {
            "recall": seed_averaged_recall(detected[h]),
            "recall_by_seed": [float(d.mean()) if len(d) else float("nan") for d in detected[h]],
            "false_alarms_per_day_by_seed": [
                sum(r["false_alarm_episodes"] for r in rows) / sum(r["span_days"] for r in rows)
                for rows in by_seed.values()],
            "per_patient_by_seed": by_seed,
        }
        block["aggregate"][str(h)] = {k: _mean_std([s[k] for s in summaries[h]])
                                      for k in AGGREGATE_KEYS}
    return block


def run_regime_experiment(cfg: dict, out_dir: Path, events_fn=None,
                          progress=print) -> dict:
    events_fn = events_fn or scorable_events
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = regime_config(cfg)
    patients = [str(p) for p in cfg["patients"]]
    seeds = [int(s) for s in cfg.get("seeds", [0])]
    backbones_dir = Path(cfg["backbones_dir"]) if "backbones_dir" in cfg \
        else RESULTS_DIR / cfg["backbones"]
    folds = load_folds(backbones_dir, patients, seeds, cfg.get("data", {}).get("processed_dir"))

    card = dev_scorecard(folds, rc, events_fn, progress)
    card.to_parquet(out_dir / "dev_scorecard.parquet", index=False)
    choices = {(f.seed, f.patient, n): choose(card, f.patient, f.seed, n, rc.budget)
               for f in folds for n in rc.n_days}
    by_days = _test_pass(folds, rc, choices, patients, seeds, out_dir, events_fn, progress)
    return {"exp_id": cfg["exp_id"], "kind": "loso", "regime": rc.regime, "config": cfg,
            "seeds": seeds, "patients": patients, "n_days": list(rc.n_days),
            "by_days": by_days}


def _test_pass(folds, rc, choices, patients, seeds, out_dir, events_fn, progress) -> dict:
    device = pick_device(rc.adapt.device)
    test_preds, frames = {}, []
    for fold in folds:
        for n in rc.n_days:
            preds = _test_predictions(fold, rc, n, choices[(fold.seed, fold.patient, n)]["recipe"],
                                      device)
            test_preds.setdefault((fold.seed, n), {})[fold.patient] = preds
            for h, p in preds.items():
                frames.append(pd.DataFrame({
                    "seed": fold.seed, "n_days": n, "patient": fold.patient, "horizon_min": h,
                    "ts": p.ts, "y_true": p.y_true, "y_pred": p.y_pred, "sigma": p.sigma}))
        progress(f"  test: pliegue {fold.patient} semilla {fold.seed}")
    pd.concat(frames).to_parquet(out_dir / "predictions.parquet", index=False)
    return {str(n): _score_days(test_preds, choices, rc, patients, seeds, n, events_fn)
            for n in rc.n_days}


def run_reselect_experiment(cfg: dict, out_dir: Path, events_fn=None,
                            progress=print) -> dict:
    events_fn = events_fn or scorable_events
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(cfg.get("source_dir") or RESULTS_DIR / cfg["source"])
    reference_dir = Path(cfg.get("reference_dir") or RESULTS_DIR / cfg["reference"])
    source = json.loads((source_dir / "metrics.json").read_text())
    scfg = source["config"]
    rc = regime_config(scfg)
    card = pd.read_parquet(source_dir / "dev_scorecard.parquet")
    ref_card = pd.read_parquet(reference_dir / "dev_scorecard.parquet")
    patients = [str(p) for p in scfg["patients"]]
    seeds = [int(s) for s in scfg.get("seeds", [0])]
    backbones_dir = Path(scfg["backbones_dir"]) if "backbones_dir" in scfg \
        else RESULTS_DIR / scfg["backbones"]
    folds = load_folds(backbones_dir, patients, seeds, scfg.get("data", {}).get("processed_dir"))

    choices = {}
    for f in folds:
        ceiling = reference_rmse(ref_card, f.patient, f.seed)
        for n in rc.n_days:
            choices[(f.seed, f.patient, n)] = choose(card, f.patient, f.seed, n, rc.budget,
                                                     rmse_ceiling=ceiling)
    by_days = _test_pass(folds, rc, choices, patients, seeds, out_dir, events_fn, progress)
    return {"exp_id": cfg["exp_id"], "kind": "loso", "regime": rc.regime,
            "selection_rule": "rmse_guard", "source": source["exp_id"],
            "reference": cfg.get("reference", str(reference_dir)), "config": cfg,
            "source_config": scfg, "seeds": seeds, "patients": patients,
            "n_days": list(rc.n_days), "by_days": by_days}
