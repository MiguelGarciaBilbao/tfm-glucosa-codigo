"""Baselines 1 to 3, fitted and evaluated under the TFM's protocol.

* **1 - ridge** on the flattened history window, one model per patient.
* **2 - LSTM per patient**, fitted only on that patient's training split.
* **3 - global LSTM**, fitted on all twelve patients pooled and applied to every
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from tfm.data.loaders import load_processed
from tfm.data.windows import WindowScaler, concat_windows, temporal_split
from tfm.eval.baselines import PATIENTS
from tfm.eval.metrics import Predictions
from tfm.models import LSTMForecaster, RidgeForecaster
from tfm.training.loop import TrainConfig, predict, train_model

VAL_FRACTION = 0.2


@dataclass
class TargetScaler:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "TargetScaler":
        self.mean_ = y.mean(axis=0)
        std = y.std(axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mean_) / self.std_).astype(np.float32)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return (y * self.std_ + self.mean_).astype(np.float32)


@dataclass
class BaselineRun:
    name: str
    seed: int
    by_horizon: dict[int, list[Predictions]] = field(default_factory=dict)
    logs: dict = field(default_factory=dict)


def _splits(patient: str):
    train_full = load_processed(patient, "train")
    test = load_processed(patient, "test")
    train, val = temporal_split(train_full, VAL_FRACTION)
    scaler = WindowScaler().fit(train)
    target = TargetScaler().fit(train.y)
    return train, val, test, scaler, target


def _to_predictions(patient: str, test, y_pred: np.ndarray) -> dict[int, Predictions]:
    return {
        hmin: Predictions(test.y[:, i].astype(float), y_pred[:, i].astype(float),
                          test.ts, patient, hmin)
        for i, hmin in enumerate(test.horizon_minutes)
    }


def run_ridge(alpha: float = 1.0) -> BaselineRun:
    run = BaselineRun(name="ridge_per_patient", seed=0)
    for patient in PATIENTS:
        train, _, test, scaler, target = _splits(patient)
        model = RidgeForecaster(alpha).fit(
            scaler.transform(train).X, target.transform(train.y)
        )
        y_pred = target.inverse(model.predict(scaler.transform(test).X))
        for hmin, pred in _to_predictions(patient, test, y_pred).items():
            run.by_horizon.setdefault(hmin, []).append(pred)
    return run


def run_lstm_per_patient(
    seed: int = 0,
    cfg: TrainConfig | None = None,
    model_kwargs: dict | None = None,
) -> BaselineRun:
    cfg = cfg or TrainConfig()
    model_kwargs = model_kwargs or {}
    run = BaselineRun(name="lstm_per_patient", seed=seed)
    for patient in PATIENTS:
        train, val, test, scaler, target = _splits(patient)
        model = LSTMForecaster(
            n_features=train.X.shape[-1], n_horizons=train.y.shape[-1], **model_kwargs
        )
        run_cfg = TrainConfig(**{**cfg.__dict__, "seed": seed, "history": {}})
        model, log = train_model(
            model,
            scaler.transform(train).X, target.transform(train.y),
            scaler.transform(val).X, target.transform(val.y),
            run_cfg,
        )
        device = next(model.parameters()).device
        y_pred = target.inverse(predict(model, scaler.transform(test).X, device))
        run.logs[patient] = log
        for hmin, pred in _to_predictions(patient, test, y_pred).items():
            run.by_horizon.setdefault(hmin, []).append(pred)
    return run


def run_lstm_global(
    seed: int = 0,
    cfg: TrainConfig | None = None,
    model_kwargs: dict | None = None,
) -> BaselineRun:
    cfg = cfg or TrainConfig()
    model_kwargs = model_kwargs or {}
    run = BaselineRun(name="lstm_global", seed=seed)

    trains, vals, per_patient = [], [], {}
    for patient in PATIENTS:
        train_full = load_processed(patient, "train")
        train, val = temporal_split(train_full, VAL_FRACTION)
        trains.append(train)
        vals.append(val)
        per_patient[patient] = load_processed(patient, "test")

    pooled_train = concat_windows(trains, split="train")
    pooled_val = concat_windows(vals, split="val")
    scaler = WindowScaler().fit(pooled_train)
    target = TargetScaler().fit(pooled_train.y)

    model = LSTMForecaster(
        n_features=pooled_train.X.shape[-1],
        n_horizons=pooled_train.y.shape[-1],
        **model_kwargs,
    )
    run_cfg = TrainConfig(**{**cfg.__dict__, "seed": seed, "history": {}})
    model, log = train_model(
        model,
        scaler.transform(pooled_train).X, target.transform(pooled_train.y),
        scaler.transform(pooled_val).X, target.transform(pooled_val.y),
        run_cfg,
    )
    run.logs["global"] = log

    device = next(model.parameters()).device
    for patient, test in per_patient.items():
        y_pred = target.inverse(predict(model, scaler.transform(test).X, device))
        for hmin, pred in _to_predictions(patient, test, y_pred).items():
            run.by_horizon.setdefault(hmin, []).append(pred)
    return run
