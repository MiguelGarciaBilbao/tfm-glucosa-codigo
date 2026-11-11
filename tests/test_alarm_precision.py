"""Tests for the precision and F1 of a stored probabilistic alarm.
"""

from __future__ import annotations

import json
import math
import shutil

import pytest

from test_ensemble import ensemble_cfg, fake_events, fake_source  # noqa: F401 (fixture)
from tfm.training import run as run_module
from tfm.training.alarm_precision import pooled_precision_f1, run_alarm_precision_experiment
from tfm.training.ensemble import run_ensemble_experiment


def precision_cfg(source_dir):
    return {"exp_id": "test-precision", "kind": "alarm_precision",
            "source_dir": str(source_dir), "evaluation": {"event_tolerance_min": 15}}


@pytest.fixture(scope="module")
def stored_source(fake_source, tmp_path_factory):
    root, src = fake_source
    out = tmp_path_factory.mktemp("test-ensemble")
    result = run_ensemble_experiment(ensemble_cfg(root, src), out, events_fn=fake_events,
                                     progress=lambda *_: None)
    (out / "metrics.json").write_text(json.dumps(run_module._json_safe(result)))
    return out


def test_precision_and_f1_pool_the_counts_not_the_ratios():
    rows = [{"n_events": 10, "n_detected": 9, "false_alarm_episodes": 1},
            {"n_events": 1, "n_detected": 0, "false_alarm_episodes": 9}]
    out = pooled_precision_f1(rows)
    precision, recall = 9 / 19, 9 / 11
    assert out["precision"] == pytest.approx(precision)
    assert out["recall"] == pytest.approx(recall)
    assert out["f1"] == pytest.approx(2 * precision * recall / (precision + recall))


def test_precision_is_undefined_without_alarms_and_f1_is_zero_without_hits():
    out = pooled_precision_f1([{"n_events": 3, "n_detected": 0, "false_alarm_episodes": 0}])
    assert math.isnan(out["precision"]) and out["f1"] == 0.0


def test_rescoring_reproduces_the_source_and_adds_precision(stored_source, tmp_path):
    result = run_alarm_precision_experiment(precision_cfg(stored_source), tmp_path,
                                            events_fn=fake_events)
    source = json.loads((stored_source / "metrics.json").read_text())
    for h in ("30", "60"):
        r, s = result["by_horizon"][h]["1.0"], source["alarm"][h]["1.0"]
        assert r["alpha"] == s["alpha"]
        assert r["recall"] == pytest.approx(s["recall"]["point"])
        assert r["false_alarms_per_day"] == pytest.approx(s["false_alarms_per_day"])
        assert r["n_detected"] + r["n_missed"] == r["n_events"] == s["recall"]["n"]
        assert sum(p["n_detected"] for p in r["per_patient"]) == r["n_detected"]
        alarms = r["n_detected"] + r["false_alarm_episodes"]
        if alarms:
            assert r["precision"] == pytest.approx(r["n_detected"] / alarms)
        else:
            assert math.isnan(r["precision"])


def test_rescoring_refuses_a_source_it_does_not_reproduce(stored_source, tmp_path):
    tampered = tmp_path / "tampered"
    shutil.copytree(stored_source, tampered)
    metrics = json.loads((tampered / "metrics.json").read_text())
    metrics["alarm"]["30"]["1.0"]["false_alarms_per_day"] += 0.5
    (tampered / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError, match="reproduce"):
        run_alarm_precision_experiment(precision_cfg(tampered), tmp_path / "out",
                                       events_fn=fake_events)


def test_runner_dispatches_the_precision_kind():
    assert "alarm_precision" in run_module.PHASE5_KINDS
