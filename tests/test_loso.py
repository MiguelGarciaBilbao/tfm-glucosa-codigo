"""Tests for phase 5 LOSO: checkpoints, backbones, regimes, alpha on validation and CLI.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from tfm.data.windows import WindowScaler, WindowSet
from tfm.eval.alarm import alarm_counts, pooled_from_counts
from tfm.eval.metrics import Predictions, reference_events
from tfm.eval.selection import best_alpha
from tfm.models.rnn import BiGRUForecaster
from tfm.training import run as run_module
from tfm.training.alpha_val import alpha_from_validation, run_alpha_val_experiment
from tfm.training.baselines import TargetScaler
from tfm.training.global_model import (
    load_checkpoint,
    predict_windows,
    prepare_cohort,
    save_checkpoint,
)
from tfm.training.loop import TrainConfig
from tfm.training.loso import (
    backbone_path,
    fold_patients,
    regime_config,
    run_backbones_experiment,
    run_regime_experiment,
    run_reselect_experiment,
    train_backbone,
)

FEATURES = ["value", "hour_sin"]
START = np.datetime64("2027-03-01T00:00", "ns")


def _series(n: int, rng) -> np.ndarray:
    t = np.arange(n) * 5.0
    value = (125 + 55 * np.sin(2 * np.pi * t / 1440) + 20 * np.sin(2 * np.pi * t / 170)
             + rng.normal(0, 4, n))
    hour = np.sin(2 * np.pi * (t % 1440) / 1440)
    return np.stack([value, hour], axis=1).astype(np.float32)


def write_fake_processed(root, patients=("p1", "p2", "p3"), train_days=10, test_days=2, seed=0):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for patient in patients:
        offset = 0
        for split, days in (("train", train_days), ("test", test_days)):
            n = days * 288
            feats = _series(n, rng)
            times = START + (offset + np.arange(n)) * np.timedelta64(5, "m")
            offset += n
            idx = np.arange(n - 12 - 12 + 1)
            X = np.stack([feats[i:i + 12] for i in idx])
            y = np.stack([[feats[i + 11 + 6, 0], feats[i + 11 + 12, 0]] for i in idx])
            np.savez(root / f"{patient}_{split}.npz", X=X, y=y.astype(np.float32),
                     ts=times[idx + 11], feature_names=np.array(FEATURES),
                     horizons=np.array([6, 12]))
    return root


def _small_windows(n: int = 50) -> WindowSet:
    rng = np.random.default_rng(1)
    ts = START + np.arange(n) * np.timedelta64(5, "m")
    return WindowSet(rng.normal(size=(n, 12, 2)).astype(np.float32),
                     (120 + 30 * rng.normal(size=(n, 2))).astype(np.float32),
                     ts, "p1", "test", FEATURES, (6, 12))


def test_checkpoint_roundtrip_predicts_the_same(tmp_path):
    torch.manual_seed(0)
    model = BiGRUForecaster(n_features=2, n_horizons=2, hidden_size=8)
    ws = _small_windows()
    scaler, target = WindowScaler().fit(ws), TargetScaler().fit(ws.y)
    save_checkpoint(tmp_path / "m.pt", model, scaler, target, "bigru", {"hidden_size": 8},
                    seed=0, extra={"held_out": "p1"})
    model2, scaler2, target2, meta = load_checkpoint(tmp_path / "m.pt")
    a = predict_windows(model, scaler, target, ws, device="cpu")
    b = predict_windows(model2, scaler2, target2, ws, device="cpu")
    assert sorted(a) == [30, 60]
    for h in a:
        np.testing.assert_allclose(a[h].y_pred, b[h].y_pred, rtol=1e-6)
        np.testing.assert_allclose(a[h].sigma, b[h].sigma, rtol=1e-6)
    assert meta["held_out"] == "p1" and meta["architecture"] == "bigru"


def test_loading_a_missing_checkpoint_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="m.pt"):
        load_checkpoint(tmp_path / "m.pt")


PATIENTS3 = ["p1", "p2", "p3"]


def backbones_cfg(root, **training):
    return {"exp_id": "bb", "kind": "loso_backbones", "architecture": "bigru", "seeds": [0],
            "model": {"hidden_size": 8},
            "loss": {"name": "gaussian_nll", "beta": 0.5},
            "training": {"max_epochs": 1, "warmup_epochs": 0, "device": "cpu", **training},
            "data": {"patients": PATIENTS3, "processed_dir": str(root)}}


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    return write_fake_processed(tmp_path_factory.mktemp("processed"))


@pytest.fixture(scope="module")
def backbones(fake_data, tmp_path_factory):
    out = tmp_path_factory.mktemp("backbones")
    run_backbones_experiment(backbones_cfg(fake_data), out, progress=lambda *_: None)
    return out


def test_fold_patients_leave_the_patient_out():
    assert fold_patients("p2", PATIENTS3) == ["p1", "p3"]
    with pytest.raises(ValueError):
        fold_patients("p9", PATIENTS3)


def test_backbone_is_fitted_without_the_held_out_patient(fake_data):
    _, scaler, _, _ = train_backbone("p1", 0, TrainConfig(max_epochs=1, device="cpu"),
                                     model_kwargs={"hidden_size": 8},
                                     patients=PATIENTS3, processed_dir=fake_data)
    _, _, _, expected, _ = prepare_cohort(patients=["p2", "p3"], processed_dir=fake_data)
    np.testing.assert_allclose(scaler.mean_, expected.mean_)


def test_backbones_experiment_writes_one_checkpoint_per_fold_and_seed(backbones):
    for p in PATIENTS3:
        assert backbone_path(backbones, p, 0).exists()


def test_resume_reuses_checkpoints_trained_with_the_same_config(fake_data, backbones):
    path = backbone_path(backbones, "p1", 0)
    stamp = path.stat().st_mtime_ns
    result = run_backbones_experiment(backbones_cfg(fake_data), backbones, resume=True,
                                      progress=lambda *_: None)
    assert path.stat().st_mtime_ns == stamp
    assert set(result["folds"]) == {"p1/0", "p2/0", "p3/0"}


def test_resume_refuses_a_checkpoint_from_another_config(fake_data, backbones):
    with pytest.raises(ValueError, match="different configuration"):
        run_backbones_experiment(backbones_cfg(fake_data, lr=0.5), backbones, resume=True,
                                 progress=lambda *_: None)


def fake_events(patient, predictions, split):
    return reference_events(predictions[0])


def regime_cfg(regime, backbones, root, **adaptation):
    cfg = {"exp_id": f"test-{regime}", "kind": "loso", "regime": regime,
           "backbones_dir": str(backbones), "patients": PATIENTS3, "seeds": [0],
           "n_days": [1, 3], "data": {"processed_dir": str(root)},
           "selection": {"budget": 1.0, "alphas": [0.2, 0.5], "dev_start_days": 7},
           "evaluation": {"event_tolerance_min": 15}}
    if regime != "global":
        cfg["adaptation"] = {"layers": ["all"], "lr": [1.0e-3], "epochs": [1, 2],
                             "lam": [0.0], "batch_size": 64, "device": "cpu", **adaptation}
    return cfg


def test_ft_grid_is_layers_by_lr_by_epochs():
    cfg = {"regime": "ft", "n_days": [1, 3, 7],
           "adaptation": {"layers": ["head", "all"], "lr": [1.0e-4, 3.0e-4, 1.0e-3],
                          "epochs": [5, 10, 20, 40]}}
    assert len(regime_config(cfg).grid) == 24


def test_global_regime_has_a_single_recipe_and_no_days():
    rc = regime_config({"regime": "global"})
    assert rc.grid == {"none": (None, 0)} and rc.n_days == (0,)


@pytest.mark.parametrize("regime,adaptation", [
    ("ft_weighted", {"lam": [0.0, 2.0]}),
    ("ft", {"lam": [5.0]}),
    ("individual", {"layers": ["head"]}),
    ("nonsense", {}),
])
def test_incoherent_regime_configs_fail_when_read(regime, adaptation):
    cfg = {"regime": regime, "n_days": [1],
           "adaptation": {"layers": ["all"], "lr": [1.0e-3], "epochs": [1], **adaptation}}
    with pytest.raises(ValueError):
        regime_config(cfg)


def test_ft_regime_runs_end_to_end(fake_data, backbones, tmp_path):
    cfg = regime_cfg("ft", backbones, fake_data)
    result = run_regime_experiment(cfg, tmp_path, events_fn=fake_events, progress=lambda *_: None)
    assert set(result["by_days"]) == {"1", "3"}
    grid = regime_config(cfg).grid
    for n in ("1", "3"):
        block = result["by_days"][n]
        assert set(block["selection"]["0"]) == set(PATIENTS3)
        assert all(c["recipe"] in grid for c in block["selection"]["0"].values())
        for h in ("30", "60"):
            rows = block["alarm"][h]["per_patient_by_seed"]["0"]
            assert block["alarm"][h]["recall"]["n"] == sum(r["n_events"] for r in rows)
            assert "rmse_pooled" in block["aggregate"][h]
    card = pd.read_parquet(tmp_path / "dev_scorecard.parquet")
    assert set(card["fold"]) == set(PATIENTS3) and set(card["n_days"]) == {1, 3}
    assert (tmp_path / "predictions.parquet").exists()


def test_global_regime_runs_without_adaptation(fake_data, backbones, tmp_path):
    result = run_regime_experiment(regime_cfg("global", backbones, fake_data), tmp_path,
                                   events_fn=fake_events, progress=lambda *_: None)
    assert set(result["by_days"]) == {"0"}
    assert {c["recipe"] for c in result["by_days"]["0"]["selection"]["0"].values()} == {"none"}


def test_individual_regime_trains_from_scratch(fake_data, backbones, tmp_path):
    cfg = regime_cfg("individual", backbones, fake_data, warmup_epochs=1)
    result = run_regime_experiment(cfg, tmp_path, events_fn=fake_events, progress=lambda *_: None)
    assert set(result["by_days"]) == {"1", "3"}


def _probabilistic(patient, sigma, seed):
    rng = np.random.default_rng(seed)
    y = 110 + 60 * np.sin(np.arange(600) / 40)
    ts = START + np.arange(600) * np.timedelta64(5, "m")
    return Predictions(y, y + rng.normal(0, 15, 600), ts, patient, 30,
                       sigma=np.full(600, sigma))


def test_alpha_comes_from_validation_and_ignores_the_test():
    val = {30: [_probabilistic("a", 15.0, 0), _probabilistic("b", 15.0, 1)]}
    val_events = {p.patient: reference_events(p) for p in val[30]}
    expected = best_alpha(pooled_from_counts(alarm_counts(val[30], val_events, (0.1, 0.3, 0.6))), 1.0)
    for test_sigma in (5.0, 40.0):
        test = {30: [_probabilistic("a", test_sigma, 2), _probabilistic("b", test_sigma, 3)]}
        test_events = {p.patient: reference_events(p) for p in test[30]}
        out = alpha_from_validation(val, test, val_events, test_events, (1.0,), (0.1, 0.3, 0.6), 15)
        assert out[30][1.0]["alpha"] == expected["alpha"]
        assert len(out[30][1.0]["test_detected"]) == sum(len(e) for e in test_events.values())


def test_alpha_val_fails_clearly_without_the_phase4_checkpoints(tmp_path, monkeypatch):
    import tfm.training.alpha_val as av

    monkeypatch.setattr(av, "RESULTS_DIR", tmp_path)
    cfg = {"exp_id": "x", "kind": "alpha_val", "source": "missing", "seeds": [0]}
    with pytest.raises(FileNotFoundError, match="model_seed0.pt"):
        run_alpha_val_experiment(cfg, tmp_path / "out", progress=lambda *_: None)


def test_smoke_config_trims_a_regime_to_minutes():
    cfg = {"exp_id": "20260928-loso-ft", "kind": "loso", "regime": "ft",
           "backbones": "20260928-loso-backbones", "patients": [559, 563, 570],
           "seeds": [0, 1, 2], "n_days": [1, 3, 7],
           "adaptation": {"layers": ["head", "all"], "lr": [1.0e-4, 3.0e-4], "epochs": [5, 10, 20]}}
    smoke = run_module.smoke_config(cfg)
    assert smoke["exp_id"] == "20260928-loso-ft-smoke"
    assert smoke["backbones"] == "20260928-loso-backbones-smoke"
    assert smoke["patients"] == ["559", "563"] and smoke["seeds"] == [0]
    assert smoke["n_days"] == [1, 3]
    assert smoke["adaptation"]["epochs"] == [5, 10] and smoke["adaptation"]["lr"] == [1.0e-4]
    assert cfg["seeds"] == [0, 1, 2]  # the original is not modified


def test_cli_runs_backbones_then_a_regime(fake_data, tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "RESULTS_DIR", tmp_path)
    bb = backbones_cfg(fake_data)
    (tmp_path / "bb.yaml").write_text(yaml.safe_dump(bb))
    run_module.main(str(tmp_path / "bb.yaml"))
    assert (tmp_path / "bb" / "metrics.json").exists()

    reg = regime_cfg("global", tmp_path / "bb", fake_data)
    (tmp_path / "reg.yaml").write_text(yaml.safe_dump(reg))
    monkeypatch.setattr("tfm.training.loso.scorable_events", fake_events)
    result = run_module.main(str(tmp_path / "reg.yaml"))
    saved = json.loads((tmp_path / "test-global" / "metrics.json").read_text())
    assert saved["by_days"]["0"]["alarm"]["30"]["recall"]["n"] == \
        result["by_days"]["0"]["alarm"]["30"]["recall"]["n"]


def test_reselect_reuses_the_development_table_and_keeps_the_error_under_the_guard(
        fake_data, backbones, tmp_path):
    quiet = {"events_fn": fake_events, "progress": lambda *_: None}
    ft_dir, gl_dir, out = tmp_path / "ft", tmp_path / "gl", tmp_path / "out"
    ft = run_regime_experiment(regime_cfg("ft", backbones, fake_data, lr=[1.0e-4, 1.0e-2]),
                               ft_dir, **quiet)
    (ft_dir / "metrics.json").write_text(json.dumps(ft, default=str))
    run_regime_experiment(regime_cfg("global", backbones, fake_data), gl_dir, **quiet)
    card_before = (ft_dir / "dev_scorecard.parquet").stat().st_mtime_ns

    cfg = {"exp_id": "test-ft-rmse", "kind": "loso_reselect",
           "source_dir": str(ft_dir), "reference_dir": str(gl_dir)}
    result = run_reselect_experiment(cfg, out, **quiet)

    assert (ft_dir / "dev_scorecard.parquet").stat().st_mtime_ns == card_before
    assert not (out / "dev_scorecard.parquet").exists()
    assert (out / "predictions.parquet").exists()
    assert result["regime"] == "ft" and result["selection_rule"] == "rmse_guard"
    assert set(result["by_days"]) == set(ft["by_days"])
    for block in result["by_days"].values():
        for choice in block["selection"]["0"].values():
            guard = choice["rmse_guard"]
            if guard["met"]:
                assert all(choice["dev_rmse"][h] <= guard["ceiling"][h] for h in guard["ceiling"])
        rows = block["alarm"]["30"]["per_patient_by_seed"]["0"]
        assert block["alarm"]["30"]["recall"]["n"] == sum(r["n_events"] for r in rows)


def test_reselect_refuses_a_reference_that_is_not_the_global_regime(fake_data, backbones, tmp_path):
    quiet = {"events_fn": fake_events, "progress": lambda *_: None}
    ft = run_regime_experiment(regime_cfg("ft", backbones, fake_data), tmp_path / "ft", **quiet)
    (tmp_path / "ft" / "metrics.json").write_text(json.dumps(ft, default=str))
    cfg = {"exp_id": "x", "kind": "loso_reselect",
           "source_dir": str(tmp_path / "ft"), "reference_dir": str(tmp_path / "ft")}
    with pytest.raises(ValueError, match="one recipe"):
        run_reselect_experiment(cfg, tmp_path / "out", **quiet)


def test_cli_runs_a_reselection(fake_data, backbones, tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr("tfm.training.loso.scorable_events", fake_events)
    quiet = {"events_fn": fake_events, "progress": lambda *_: None}
    ft = run_regime_experiment(regime_cfg("ft", backbones, fake_data), tmp_path / "ft", **quiet)
    (tmp_path / "ft" / "metrics.json").write_text(json.dumps(ft, default=str))
    run_regime_experiment(regime_cfg("global", backbones, fake_data), tmp_path / "gl", **quiet)
    cfg = {"exp_id": "ft-rmse", "kind": "loso_reselect",
           "source_dir": str(tmp_path / "ft"), "reference_dir": str(tmp_path / "gl")}
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(cfg))
    run_module.main(str(tmp_path / "r.yaml"))
    saved = json.loads((tmp_path / "ft-rmse" / "metrics.json").read_text())
    assert saved["selection_rule"] == "rmse_guard"
