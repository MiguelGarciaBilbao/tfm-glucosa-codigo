"""Tests for the seed ensemble: combining predictions and the experiment runner."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from test_loso import write_fake_processed
from tfm.data.loaders import load_processed
from tfm.data.windows import WindowScaler, concat_windows
from tfm.eval.metrics import Predictions, ensemble_predictions, reference_events
from tfm.models.rnn import BiGRUForecaster
from tfm.training import run as run_module
from tfm.training.baselines import TargetScaler
from tfm.training.ensemble import run_ensemble_experiment
from tfm.training.global_model import save_checkpoint

START = np.datetime64("2027-03-01T00:00", "ns")


def _pred(mu, sigma, patient="a", n=4, ts_shift=0):
    ts = START + (np.arange(n) + ts_shift) * np.timedelta64(5, "m")
    return Predictions(np.full(n, 110.0), np.full(n, float(mu)), ts, patient, 30,
                       sigma=np.full(n, float(sigma)))


def test_ensemble_of_one_member_is_that_member():
    p = _pred(100, 10)
    e = ensemble_predictions([p])
    np.testing.assert_allclose(e.y_pred, p.y_pred)
    np.testing.assert_allclose(e.sigma, p.sigma)
    assert e.patient == "a" and e.horizon_min == 30


def test_ensemble_matches_the_moments_of_the_mixture():
    e = ensemble_predictions([_pred(100, 10), _pred(120, 20)])
    np.testing.assert_allclose(e.y_pred, 110.0)
    np.testing.assert_allclose(e.sigma, np.sqrt(350.0))


def test_ensemble_refuses_members_on_different_windows():
    with pytest.raises(ValueError, match="same windows"):
        ensemble_predictions([_pred(100, 10), _pred(100, 10, ts_shift=1)])
    with pytest.raises(ValueError, match="same windows"):
        ensemble_predictions([_pred(100, 10), _pred(100, 10, patient="b")])


def test_ensemble_needs_probabilistic_members():
    p = _pred(100, 10)
    det = Predictions(p.y_true, p.y_pred, p.ts, "a", 30)
    with pytest.raises(ValueError, match="sigma"):
        ensemble_predictions([p, det])
    with pytest.raises(ValueError, match="no predictions"):
        ensemble_predictions([])


PATIENTS3 = ["p1", "p2", "p3"]


def fake_events(by_horizon, split="test"):
    return {p.patient: reference_events(p) for p in by_horizon[30]}


@pytest.fixture(scope="module")
def fake_source(tmp_path_factory):
    root = write_fake_processed(tmp_path_factory.mktemp("processed"))
    src = tmp_path_factory.mktemp("source")
    train = concat_windows([load_processed(p, "train", processed_dir=root) for p in PATIENTS3])
    scaler, target = WindowScaler().fit(train), TargetScaler().fit(train.y)
    for seed in (0, 1):
        torch.manual_seed(seed)
        model = BiGRUForecaster(n_features=2, n_horizons=2, hidden_size=8)
        save_checkpoint(src / f"model_seed{seed}.pt", model, scaler, target, "bigru",
                        {"hidden_size": 8}, seed=seed)
    return root, src


def ensemble_cfg(root, src, seeds=(0, 1)):
    return {"exp_id": "test-ensemble", "kind": "ensemble", "source_dir": str(src),
            "seeds": list(seeds), "patients": PATIENTS3, "device": "cpu",
            "data": {"processed_dir": str(root)},
            "budgets": [1.0], "alphas": [0.2, 0.5], "evaluation": {"event_tolerance_min": 15}}


def test_ensemble_experiment_scores_the_ensemble_and_every_member(fake_source, tmp_path):
    root, src = fake_source
    result = run_ensemble_experiment(ensemble_cfg(root, src), tmp_path,
                                     events_fn=fake_events, progress=lambda *_: None)
    assert result["kind"] == "ensemble" and set(result["members"]) == {"0", "1"}
    for h in ("30", "60"):
        assert "rmse_pooled" in result["by_horizon"][h]["summary"]
        alarm = result["alarm"][h]["1.0"]
        assert alarm["recall"]["n"] == sum(r["n_events"] for r in alarm["per_patient"])
        assert alarm["alpha"] in (0.2, 0.5)
        assert "crps" in result["calibration"][h]
        assert "1.0" in result["members"]["0"][h]["alarm"]
    saved = pd.read_parquet(tmp_path / "predictions.parquet")
    assert {"sigma", "y_pred"} <= set(saved.columns) and "seed" not in saved.columns


def test_alpha_does_not_look_at_the_test(fake_source, tmp_path):
    root, src = fake_source
    other = write_fake_processed(tmp_path / "other", seed=7)
    swapped = tmp_path / "swapped"
    shutil.copytree(root, swapped)
    for p in PATIENTS3:
        shutil.copy(other / f"{p}_test.npz", swapped / f"{p}_test.npz")
    quiet = {"events_fn": fake_events, "progress": lambda *_: None}
    a = run_ensemble_experiment(ensemble_cfg(root, src), tmp_path / "a", **quiet)
    b = run_ensemble_experiment(ensemble_cfg(swapped, src), tmp_path / "b", **quiet)
    for h in ("30", "60"):
        assert a["alarm"][h]["1.0"]["alpha"] == b["alarm"][h]["1.0"]["alpha"]


def test_cli_runs_an_ensemble_of_one_seed(fake_source, tmp_path, monkeypatch):
    root, src = fake_source
    monkeypatch.setattr(run_module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr("tfm.training.ensemble.scorable_events_by_patient", fake_events)
    (tmp_path / "e.yaml").write_text(yaml.safe_dump(ensemble_cfg(root, src, seeds=(0,))))
    run_module.main(str(tmp_path / "e.yaml"))
    saved = json.loads((tmp_path / "test-ensemble" / "metrics.json").read_text())
    assert saved["kind"] == "ensemble" and saved["seeds"] == [0]
