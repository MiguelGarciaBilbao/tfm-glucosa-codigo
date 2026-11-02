from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tfm.data.loaders import load_processed
from tfm.data.windows import WindowSet, WindowScaler, concat_windows, temporal_split
from tfm.eval.baselines import PATIENTS
from tfm.eval.metrics import Predictions
from tfm.models.rnn import BiGRUForecaster, LSTMForecaster
from tfm.models.transformer import TransformerForecaster
from tfm.paths import RESULTS_DIR
from tfm.training.baselines import VAL_FRACTION, TargetScaler
from tfm.training.losses import build_loss, split_output
from tfm.training.loop import TrainConfig, pick_device, train_model

ARCHITECTURES = {
    "bigru": BiGRUForecaster,
    "transformer": TransformerForecaster,
    "lstm": LSTMForecaster,
}


@dataclass
class GlobalRun:
    architecture: str
    seed: int
    by_horizon: dict[int, list[Predictions]] = field(default_factory=dict)
    logs: dict = field(default_factory=dict)
    n_parameters: int = 0


def _select_features(ws: WindowSet, features: list[str] | None) -> WindowSet:
    if not features:
        return ws
    missing = [f for f in features if f not in ws.feature_names]
    if missing:
        raise ValueError(f"features not in the dataset: {missing}")
    idx = [ws.feature_names.index(f) for f in features]
    return WindowSet(ws.X[:, :, idx], ws.y, ws.ts, ws.patient, ws.split,
                     list(features), ws.horizons, dict(ws.meta))


def prepare_cohort(
    patients=PATIENTS,
    features: list[str] | None = None,
    processed_dir: Path | None = None,
    include_test: bool = True,
):
    trains, vals, tests = [], [], {}
    for patient in patients:
        full = load_processed(patient, "train", processed_dir=processed_dir)
        train, val = temporal_split(_select_features(full, features), VAL_FRACTION)
        trains.append(train)
        vals.append(val)
        if include_test:
            tests[patient] = _select_features(
                load_processed(patient, "test", processed_dir=processed_dir), features
            )
    pooled_train = concat_windows(trains, split="train")
    pooled_val = concat_windows(vals, split="val")
    scaler = WindowScaler().fit(pooled_train)
    target = TargetScaler().fit(pooled_train.y)
    return pooled_train, pooled_val, tests, scaler, target


@torch.no_grad()
def predict_distribution(model, X: np.ndarray, device, batch_size: int = 512):
    model.eval()
    mus, sigmas = [], []
    for i in range(0, len(X), batch_size):
        chunk = torch.from_numpy(np.ascontiguousarray(X[i : i + batch_size], dtype=np.float32))
        out = model(chunk.to(device))
        if out.ndim == 3:
            mu, log_var = split_output(out)
            mus.append(mu.cpu().numpy())
            sigmas.append(torch.exp(0.5 * log_var).cpu().numpy())
        else:
            mus.append(out.cpu().numpy())
    mu = np.concatenate(mus) if mus else np.empty((0, 0), dtype=np.float32)
    sigma = np.concatenate(sigmas) if sigmas else None
    return mu, sigma


def predict_windows(model, scaler: WindowScaler, target: TargetScaler, ws: WindowSet,
                    device=None) -> dict[int, Predictions]:
    device = device if isinstance(device, torch.device) else pick_device(device)
    model.to(device)
    mu, sigma = predict_distribution(model, scaler.transform(ws).X, device)
    mu = target.inverse(mu)
    sigma_mgdl = None if sigma is None else sigma * target.std_
    return {
        hmin: Predictions(
            ws.y[:, i].astype(float), mu[:, i].astype(float), ws.ts, ws.patient, hmin,
            sigma=None if sigma_mgdl is None else sigma_mgdl[:, i].astype(float),
        )
        for i, hmin in enumerate(ws.horizon_minutes)
    }


def save_checkpoint(path: Path, model, scaler: WindowScaler, target: TargetScaler,
                    architecture: str, model_kwargs: dict, seed: int,
                    extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
         "architecture": architecture, "model_kwargs": dict(model_kwargs), "seed": seed,
         "feature_names": list(scaler.feature_names),
         "scaler_mean": scaler.mean_, "scaler_std": scaler.std_,
         "target_mean": target.mean_, "target_std": target.std_, **(extra or {})},
        path,
    )


def load_checkpoint(path) -> tuple:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; train or restore the model first")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    names = [str(f) for f in ck["feature_names"]]
    n_horizons = len(np.atleast_1d(ck["target_mean"]))
    model = ARCHITECTURES[ck["architecture"]](
        n_features=len(names), n_horizons=n_horizons, **ck["model_kwargs"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    scaler = WindowScaler()
    scaler.mean_, scaler.std_ = np.asarray(ck["scaler_mean"]), np.asarray(ck["scaler_std"])
    scaler.feature_names = names
    target = TargetScaler(np.asarray(ck["target_mean"]), np.asarray(ck["target_std"]))
    skip = {"state_dict", "scaler_mean", "scaler_std", "target_mean", "target_std"}
    meta = {k: v for k, v in ck.items() if k not in skip}
    return model, scaler, target, meta


def run_global(
    architecture: str = "bigru",
    seed: int = 0,
    cfg: TrainConfig | None = None,
    model_kwargs: dict | None = None,
    loss: str = "gaussian_nll",
    loss_kwargs: dict | None = None,
    features: list[str] | None = None,
    processed_dir: Path | None = None,
    checkpoint_path: Path | None = None,
) -> GlobalRun:
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"unknown architecture {architecture!r}; available: {sorted(ARCHITECTURES)}"
        )
    cfg = cfg or TrainConfig()
    model_kwargs = dict(model_kwargs or {})
    train, val, tests, scaler, target = prepare_cohort(
        features=features, processed_dir=processed_dir
    )

    model = ARCHITECTURES[architecture](
        n_features=train.X.shape[-1], n_horizons=train.y.shape[-1], **model_kwargs
    )
    run_cfg = TrainConfig(**{**cfg.__dict__, "seed": seed, "history": {}})
    loss_fn = build_loss(loss, **(loss_kwargs or {}))
    warmup_fn = build_loss("mse") if run_cfg.warmup_epochs else None

    model, log = train_model(
        model,
        scaler.transform(train).X, target.transform(train.y),
        scaler.transform(val).X, target.transform(val.y),
        run_cfg, loss_fn=loss_fn, warmup_loss_fn=warmup_fn,
    )
    run = GlobalRun(architecture=architecture, seed=seed,
                    n_parameters=sum(p.numel() for p in model.parameters()))
    run.logs["global"] = log

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, scaler, target, architecture, model_kwargs, seed)

    device = pick_device(run_cfg.device)
    for patient, test in tests.items():
        for hmin, pred in predict_windows(model, scaler, target, test, device).items():
            run.by_horizon.setdefault(hmin, []).append(pred)
    return run
