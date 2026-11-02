"""Tests for models, the training loop and the baseline runners.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tfm.models import LSTMForecaster, RidgeForecaster
from tfm.training.baselines import TargetScaler
from tfm.training.loop import TrainConfig, predict, set_seed, train_model


def _toy(n=300, history=12, features=4, horizons=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, history, features)).astype(np.float32)
    # A target the models can actually learn: a linear read-out of the window.
    w = rng.normal(size=(history * features, horizons))
    y = (X.reshape(n, -1) @ w).astype(np.float32)
    return X, y


# --------------------------------------------------------------------------
# Ridge
# --------------------------------------------------------------------------

def test_ridge_recovers_a_linear_target():
    X, y = _toy()
    model = RidgeForecaster(alpha=1e-6).fit(X, y)
    assert np.allclose(model.predict(X), y, atol=1e-2)


def test_ridge_is_deterministic():
    X, y = _toy()
    a = RidgeForecaster(1.0).fit(X, y).predict(X)
    b = RidgeForecaster(1.0).fit(X, y).predict(X)
    assert np.array_equal(a, b)


def test_ridge_shrinks_coefficients_as_alpha_grows():
    X, y = _toy()
    small = np.abs(RidgeForecaster(1e-3).fit(X, y).coef_).sum()
    large = np.abs(RidgeForecaster(1e4).fit(X, y).coef_).sum()
    assert large < small


def test_ridge_refuses_to_predict_before_fitting():
    with pytest.raises(RuntimeError):
        RidgeForecaster().predict(_toy()[0])


# --------------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------------

def test_lstm_output_shape_matches_the_horizons():
    X, _ = _toy(n=8, horizons=2)
    out = LSTMForecaster(n_features=X.shape[-1], n_horizons=2)(torch.from_numpy(X))
    assert out.shape == (8, 2)


def test_lstm_only_reads_the_past():
    torch.manual_seed(0)
    model = LSTMForecaster(n_features=4, n_horizons=2).eval()
    X = torch.zeros(1, 12, 4)
    base = model(X)
    X2 = X.clone()
    X2[0, -1] = 5.0
    assert not torch.allclose(base, model(X2))


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def test_training_reduces_validation_loss():
    X, y = _toy(n=400)
    model = LSTMForecaster(n_features=X.shape[-1], n_horizons=y.shape[-1])
    _, log = train_model(model, X[:300], y[:300], X[300:], y[300:],
                         TrainConfig(max_epochs=15, seed=0))
    assert log["val_loss"][-1] < log["val_loss"][0]


def test_training_restores_the_best_epoch_not_the_last():
    X, y = _toy(n=200)
    model = LSTMForecaster(n_features=X.shape[-1], n_horizons=y.shape[-1])
    model, log = train_model(model, X[:150], y[:150], X[150:], y[150:],
                             TrainConfig(max_epochs=20, patience=3, seed=0))
    device = next(model.parameters()).device
    final = np.mean((predict(model, X[150:], device) - y[150:]) ** 2)
    assert final == pytest.approx(log["best_val_loss"], rel=0.05)


def test_training_stops_early_when_validation_stalls():
    X, y = _toy(n=200)
    model = LSTMForecaster(n_features=X.shape[-1], n_horizons=y.shape[-1])
    _, log = train_model(model, X[:150], y[:150], X[150:], y[150:],
                         TrainConfig(max_epochs=200, patience=2, seed=0))
    assert log["epochs_run"] < 200


def test_set_seed_makes_a_run_reproducible_on_cpu():
    X, y = _toy(n=120)
    outs = []
    for _ in range(2):
        set_seed(3)
        model = LSTMForecaster(n_features=X.shape[-1], n_horizons=y.shape[-1])
        model, _ = train_model(model, X[:90], y[:90], X[90:], y[90:],
                               TrainConfig(max_epochs=3, seed=3, device="cpu"))
        outs.append(predict(model, X[90:], torch.device("cpu")))
    assert np.allclose(outs[0], outs[1])


# --------------------------------------------------------------------------
# Target scaling
# --------------------------------------------------------------------------

def test_target_scaler_round_trips():
    y = np.array([[100.0, 120.0], [80.0, 90.0], [150.0, 200.0]], dtype=np.float32)
    sc = TargetScaler().fit(y)
    assert np.allclose(sc.inverse(sc.transform(y)), y, atol=1e-4)


def test_target_scaler_survives_a_constant_target():
    y = np.full((10, 2), 100.0, dtype=np.float32)
    sc = TargetScaler().fit(y)
    assert np.all(np.isfinite(sc.transform(y)))


def test_target_scaler_is_fitted_on_training_data_only():
    rng = np.random.default_rng(0)
    train = rng.normal(100, 10, size=(500, 2)).astype(np.float32)
    val = rng.normal(160, 10, size=(200, 2)).astype(np.float32)
    sc = TargetScaler().fit(train)
    assert np.abs(sc.transform(val).mean()) > 1.0


# --------------------------------------------------------------------------
# Comparison between experiments
# --------------------------------------------------------------------------

def test_comparison_table_reports_regression_and_clinical_together():
    from tfm.eval.compare import comparison_table

    try:
        table = comparison_table(
            ["20260915-baseline-persistencia", "20260916-baseline-ridge"], 30
        )
    except FileNotFoundError:
        pytest.skip("baseline experiments not run")
    assert {"rmse", "recall", "n_events", "parkes_DE"} <= set(table.columns)
    assert (table["n_events"] == 61).all()


def test_paired_wilcoxon_compares_twelve_patients():
    from tfm.eval.compare import paired_wilcoxon

    try:
        r = paired_wilcoxon(
            "20260916-baseline-ridge", "20260915-baseline-persistencia", 30, "rmse"
        )
    except FileNotFoundError:
        pytest.skip("baseline experiments not run")
    assert r["n"] == 12
    assert r["median_diff"] < 0  # ridge beats persistence on RMSE


def test_multi_seed_results_average_one_value_per_patient():
    from tfm.eval.compare import _horizon_block, load_result

    try:
        result = load_result("20260916-baseline-lstm-global")
    except FileNotFoundError:
        pytest.skip("baseline experiments not run")
    _, table = _horizon_block(result, 30)
    assert len(table) == 12
    assert table["patient"].is_unique


# --------------------------------------------------------------------------
# Config-driven runner
# --------------------------------------------------------------------------

def _same_json(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys {sorted(a.keys() ^ b.keys())}"]
        return [d for k in a for d in _same_json(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _same_json(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float):
        return [] if a == pytest.approx(b, rel=0, abs=1e-9) else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def test_persistence_config_runs_and_reproduces_its_versioned_metrics(tmp_path, monkeypatch):
    import json

    import pandas as pd

    from tfm.paths import EXPERIMENTS_DIR, PROCESSED_DIR
    from tfm.training import run

    exp_id = "20260915-baseline-persistencia"
    versioned = EXPERIMENTS_DIR / "results" / exp_id / "metrics.json"
    if not (PROCESSED_DIR / "559_test.npz").exists() or not versioned.exists():
        pytest.skip("data/processed not built")
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)

    run.main(str(EXPERIMENTS_DIR / "configs" / f"{exp_id}.yaml"))

    written = json.loads((tmp_path / exp_id / "metrics.json").read_text())
    assert _same_json(written, json.loads(versioned.read_text())) == []
    preds = pd.read_parquet(tmp_path / exp_id / "predictions.parquet")
    assert list(preds.columns) == ["patient", "horizon_min", "ts", "y_true", "y_pred"]
    assert sorted(preds["horizon_min"].unique()) == [30, 60]
