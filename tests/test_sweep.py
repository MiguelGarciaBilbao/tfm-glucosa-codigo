"""Tests for the hyperparameter sweep of the global model.
"""

from __future__ import annotations

import pytest

from test_loso import write_fake_processed
from tfm.training import global_model
from tfm.training import run as run_module
from tfm.training.sweep import grid_points, run_sweep_experiment


def sweep_cfg(root, **overrides):
    cfg = {"exp_id": "sw", "kind": "global_sweep", "architecture": "bigru", "seeds": [0, 1],
           "model": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0},
           "loss": {"name": "gaussian_nll", "beta": 0.5},
           "training": {"lr": 1e-3, "max_epochs": 2, "warmup_epochs": 1, "device": "cpu"},
           "data": {"patients": ["p1", "p2"], "processed_dir": str(root)},
           "grid": {"model.hidden_size": [4, 8], "training.lr": [1e-3]},
           "chosen": {"model.hidden_size": 8, "training.lr": 1e-3}}
    return {**cfg, **overrides}


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    return write_fake_processed(tmp_path_factory.mktemp("processed"), patients=("p1", "p2"))


def test_grid_points_override_the_base_model_and_training_blocks():
    cfg = {"model": {"hidden_size": 128, "num_layers": 1},
           "training": {"lr": 1e-3, "batch_size": 256},
           "grid": {"model.hidden_size": [64, 128], "training.lr": [3e-4, 1e-3]}}
    points = grid_points(cfg)
    assert {(p["model"]["hidden_size"], p["training"]["lr"]) for p in points} == {
        (64, 3e-4), (64, 1e-3), (128, 3e-4), (128, 1e-3)}
    assert all(p["model"]["num_layers"] == 1 for p in points)
    assert all(p["training"]["batch_size"] == 256 for p in points)
    assert len({p["point_id"] for p in points}) == 4


def test_grid_rejects_keys_outside_model_and_training():
    with pytest.raises(ValueError, match="loss.beta"):
        grid_points({"model": {}, "training": {}, "grid": {"loss.beta": [0.0, 0.5]}})


def test_sweep_never_reads_the_test_split(fake_data, tmp_path, monkeypatch):
    seen = []
    real = global_model.load_processed

    def spy(patient, split, **kwargs):
        seen.append(split)
        return real(patient, split, **kwargs)

    monkeypatch.setattr(global_model, "load_processed", spy)
    run_sweep_experiment(sweep_cfg(fake_data), tmp_path, progress=lambda *_: None)
    assert seen and set(seen) == {"train"}


def test_sweep_ranks_points_by_validation_nll_and_flags_the_chosen_one(fake_data, tmp_path):
    result = run_sweep_experiment(sweep_cfg(fake_data), tmp_path, progress=lambda *_: None)
    ranking = result["ranking"]
    assert [r["rank"] for r in ranking] == [1, 2]
    nll = [r["val_nll"]["mean"] for r in ranking]
    assert nll == sorted(nll)
    assert sum(r["chosen"] for r in ranking) == 1
    for r in ranking:
        assert set(r["val_nll_by_seed"]) == {"0", "1"}
        assert set(r["val_rmse"]) == {"30", "60"}
        assert set(r["val_crps"]) == {"30", "60"}
    assert result["chosen"]["point_id"] in {r["point_id"] for r in ranking}
    assert (tmp_path / "train_log.csv").exists()


def test_sweep_fails_clearly_when_the_chosen_point_is_not_in_the_grid(fake_data, tmp_path):
    cfg = sweep_cfg(fake_data, chosen={"model.hidden_size": 16, "training.lr": 1e-3})
    with pytest.raises(ValueError, match="chosen"):
        run_sweep_experiment(cfg, tmp_path, progress=lambda *_: None)


def test_runner_dispatches_the_sweep_kind():
    assert "global_sweep" in run_module.PHASE5_KINDS
