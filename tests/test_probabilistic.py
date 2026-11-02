"""Tests for the heteroscedastic models, losses and probabilistic predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tfm.eval.metrics import Predictions
from tfm.models.heads import ForecastHead
from tfm.models.rnn import BiGRUForecaster
from tfm.models.transformer import TransformerForecaster
from tfm.training.losses import (
    LOG_VAR_MAX,
    LOG_VAR_MIN,
    GaussianNLL,
    HeteroscedasticMSE,
    build_loss,
    split_output,
)
from tfm.training.loop import TrainConfig, train_model


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def test_gaussian_nll_matches_the_analytic_value():
    out = torch.zeros(1, 1, 2)
    assert GaussianNLL(beta=0.0)(out, torch.ones(1, 1)).item() == pytest.approx(0.5)


def test_gaussian_nll_prefers_the_right_variance():
    target = torch.full((1, 1), 2.0)
    losses = {}
    for log_var in (-2.0, np.log(4.0), 4.0):  # sigma^2 = 0.14, 4, 54
        out = torch.zeros(1, 1, 2)
        out[..., 1] = float(log_var)
        losses[float(log_var)] = GaussianNLL(beta=0.0)(out, target).item()
    assert losses[float(np.log(4.0))] == min(losses.values())


def _grad_wrt_mu(beta: float, log_var: float) -> float:
    out = torch.zeros(1, 1, 2, requires_grad=True)
    filled = out.clone()
    filled = torch.stack([out[..., 0], out[..., 1] + log_var], dim=-1)
    GaussianNLL(beta=beta)(filled, torch.ones(1, 1)).backward()
    return float(out.grad[..., 0].item())


def test_beta_one_makes_the_mean_gradient_independent_of_sigma():
    assert _grad_wrt_mu(1.0, -2.0) == pytest.approx(_grad_wrt_mu(1.0, 2.0), rel=1e-6)
    assert abs(_grad_wrt_mu(0.0, -2.0)) > 10 * abs(_grad_wrt_mu(0.0, 2.0))


def test_beta_half_sits_between_nll_and_mse():
    ratio_nll = abs(_grad_wrt_mu(0.0, -2.0) / _grad_wrt_mu(0.0, 2.0))
    ratio_half = abs(_grad_wrt_mu(0.5, -2.0) / _grad_wrt_mu(0.5, 2.0))
    assert 1.0 < ratio_half < ratio_nll


def test_log_variance_is_clamped():
    out = torch.zeros(2, 1, 2)
    out[..., 1] = torch.tensor([[100.0], [-100.0]])
    _, log_var = split_output(out)
    assert log_var.max().item() <= LOG_VAR_MAX
    assert log_var.min().item() >= LOG_VAR_MIN


def test_build_loss_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss("no_such_loss")


def test_heteroscedastic_mse_ignores_the_variance_head():
    target = torch.ones(3, 2)
    a, b = torch.zeros(3, 2, 2), torch.zeros(3, 2, 2)
    b[..., 1] = 5.0
    assert HeteroscedasticMSE()(a, target) == HeteroscedasticMSE()(b, target)


# --------------------------------------------------------------------------
# Heads and architectures
# --------------------------------------------------------------------------

def test_head_starts_at_unit_variance():
    head = ForecastHead(8, n_horizons=2, heteroscedastic=True)
    out = head(torch.zeros(1, 8))
    assert out[..., 1].abs().max().item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("cls", [BiGRUForecaster, TransformerForecaster])
def test_architectures_emit_mean_and_log_variance(cls):
    out = cls(n_features=6, n_horizons=2)(torch.randn(5, 12, 6))
    assert out.shape == (5, 2, 2)


@pytest.mark.parametrize("cls", [BiGRUForecaster, TransformerForecaster])
def test_architectures_can_be_deterministic(cls):
    out = cls(n_features=6, n_horizons=2, heteroscedastic=False)(torch.randn(5, 12, 6))
    assert out.shape == (5, 2)


def test_bigru_reads_both_ends_of_the_window():
    torch.manual_seed(0)
    model = BiGRUForecaster(n_features=3, n_horizons=1).eval()
    x = torch.zeros(1, 12, 3)
    changed_first = x.clone()
    changed_first[0, 0] = 4.0
    assert not torch.allclose(model(x), model(changed_first))


def test_variance_head_learns_input_dependent_uncertainty():
    rng = np.random.default_rng(0)
    n = 2000
    flag = rng.integers(0, 2, n)
    X = np.zeros((n, 6, 2), dtype=np.float32)
    X[:, :, 0] = flag[:, None]
    signal = rng.normal(size=n)
    X[:, -1, 1] = signal
    noise = np.where(flag == 1, rng.normal(0, 2.0, n), 0.0)
    y = (signal + noise).astype(np.float32)[:, None]

    model = BiGRUForecaster(n_features=2, n_horizons=1, hidden_size=32)
    model, _ = train_model(model, X[:1600], y[:1600], X[1600:], y[1600:],
                           TrainConfig(max_epochs=60, patience=10, warmup_epochs=5,
                                       seed=0, device="cpu"),
                           loss_fn=GaussianNLL(0.5), warmup_loss_fn=HeteroscedasticMSE())
    with torch.no_grad():
        _, log_var = split_output(model(torch.from_numpy(X[1600:])))
    sigma = torch.exp(0.5 * log_var).numpy().ravel()
    noisy = flag[1600:] == 1
    assert sigma[noisy].mean() > 2 * sigma[~noisy].mean()


# --------------------------------------------------------------------------
# Probabilistic predictions
# --------------------------------------------------------------------------

def _pred(mu, sigma, y_true=None):
    ts = pd.date_range("2027-01-01", periods=len(mu), freq="5min")
    y_true = np.full(len(mu), 100.0) if y_true is None else y_true
    return Predictions(y_true, mu, ts, "999", 30, sigma=sigma)


def test_predictions_detect_a_probabilistic_model():
    assert _pred([100.0], [10.0]).is_probabilistic
    assert not Predictions([100.0], [100.0],
                           pd.date_range("2027-01-01", periods=1), "9", 30).is_probabilistic


def test_hypo_probability_matches_the_normal_cdf():
    p = _pred([75.0], [10.0]).hypo_probability()
    assert p[0] == pytest.approx(0.3085, abs=1e-4)


def test_hypo_probability_rises_as_the_mean_falls():
    p = _pred([120.0, 90.0, 70.0, 50.0], [15.0] * 4).hypo_probability()
    assert list(p) == sorted(p)


def test_wider_sigma_pulls_probability_towards_one_half():
    narrow = _pred([100.0], [5.0]).hypo_probability()[0]
    wide = _pred([100.0], [50.0]).hypo_probability()[0]
    assert narrow < wide < 0.5


def test_probabilistic_alarm_can_fire_where_the_mean_alone_never_would():
    pred = _pred([85.0], [20.0])
    assert pred.y_pred[0] > 70
    assert pred.hypo_probability()[0] > 0.15


def test_predictions_reject_a_non_positive_sigma():
    with pytest.raises(ValueError, match="strictly positive"):
        _pred([100.0, 100.0], [10.0, 0.0])


def test_predictions_reject_a_mismatched_sigma_length():
    with pytest.raises(ValueError, match="sigma has"):
        _pred([100.0, 100.0], [10.0])


def test_deterministic_predictions_refuse_to_give_a_probability():
    p = Predictions([100.0], [100.0], pd.date_range("2027-01-01", periods=1), "9", 30)
    with pytest.raises(RuntimeError, match="no sigma"):
        p.hypo_probability()


def test_sigma_survives_sorting():
    ts = pd.to_datetime(["2027-01-01 00:10", "2027-01-01 00:00"]).to_numpy()
    p = Predictions([100.0, 100.0], [90.0, 80.0], ts, "9", 30, sigma=[5.0, 15.0]).sorted()
    assert list(p.y_pred) == [80.0, 90.0]
    assert list(p.sigma) == [15.0, 5.0]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def _calibrated(n=20000, sigma=15.0, seed=0):
    from tfm.eval.metrics import Predictions

    rng = np.random.default_rng(seed)
    mu = rng.normal(120, 40, n)
    y = mu + rng.normal(0, sigma, n)
    ts = pd.date_range("2027-01-01", periods=n, freq="5min")
    return Predictions(y, mu, ts, "X", 30, sigma=np.full(n, sigma))


@pytest.mark.parametrize("level", [0.5, 0.8, 0.9, 0.95])
def test_picp_matches_the_nominal_level_when_sigma_is_right(level):
    from tfm.eval.calibration import interval_metrics

    m = interval_metrics(_calibrated(), level)
    assert m["picp"] == pytest.approx(level, abs=0.02)


def test_picp_falls_when_the_model_is_overconfident():
    from tfm.eval.calibration import interval_metrics

    p = _calibrated()
    p.sigma = p.sigma / 3
    assert interval_metrics(p, 0.95)["picp"] < 0.7


def test_mpiw_grows_with_sigma():
    from tfm.eval.calibration import interval_metrics

    narrow = interval_metrics(_calibrated(sigma=5.0), 0.95)["mpiw"]
    wide = interval_metrics(_calibrated(sigma=20.0), 0.95)["mpiw"]
    assert wide > 3 * narrow


def test_crps_matches_its_closed_form_expectation():
    from tfm.eval.calibration import crps_gaussian

    sigma = 15.0
    assert crps_gaussian(_calibrated(sigma=sigma)) == pytest.approx(
        sigma / np.sqrt(np.pi), rel=0.02
    )


def test_crps_penalises_both_overconfidence_and_vagueness():
    from tfm.eval.calibration import crps_gaussian

    right = _calibrated(sigma=15.0)
    over, vague = _calibrated(sigma=15.0), _calibrated(sigma=15.0)
    over.sigma = over.sigma / 4
    vague.sigma = vague.sigma * 4
    base = crps_gaussian(right)
    assert crps_gaussian(over) > base
    assert crps_gaussian(vague) > base


def test_gaussian_nll_matches_its_closed_form_expectation():
    from tfm.eval.calibration import gaussian_nll

    sigma = 15.0
    expected = np.log(sigma) + 0.5 * (1 + np.log(2 * np.pi))
    assert gaussian_nll(_calibrated(sigma=sigma)) == pytest.approx(expected, abs=0.02)


def test_gaussian_nll_penalises_both_overconfidence_and_vagueness():
    from tfm.eval.calibration import gaussian_nll

    over, vague = _calibrated(sigma=15.0), _calibrated(sigma=15.0)
    over.sigma = over.sigma / 4
    vague.sigma = vague.sigma * 4
    base = gaussian_nll(_calibrated(sigma=15.0))
    assert gaussian_nll(over) > base
    assert gaussian_nll(vague) > base


def test_calibration_report_exposes_the_sigma_spread():
    from tfm.eval.calibration import calibration_report

    report = calibration_report(_calibrated())
    assert report["sigma_iqr"] == pytest.approx(0.0)
    assert set(report) >= {"crps", "sigma_mean", "sigma_iqr", "coverage"}


def test_calibration_refuses_a_deterministic_prediction():
    from tfm.eval.calibration import crps_gaussian, interval_metrics
    from tfm.eval.metrics import Predictions

    p = Predictions([100.0], [100.0], pd.date_range("2027-01-01", periods=1), "9", 30)
    with pytest.raises(RuntimeError):
        interval_metrics(p)
    with pytest.raises(RuntimeError):
        crps_gaussian(p)


# --------------------------------------------------------------------------
# The probabilistic alarm
# --------------------------------------------------------------------------

def _alarm_pred(mu, sigma, y_true, patient="999"):
    from tfm.eval.metrics import Predictions

    ts = pd.date_range("2027-01-01", periods=len(mu), freq="5min")
    return Predictions(np.asarray(y_true, float), np.asarray(mu, float), ts,
                       patient, 30, sigma=np.asarray(sigma, float))


def test_probabilistic_alarm_is_monotonic_in_alpha():
    from tfm.eval.alarm import probabilistic_alarm

    pred = _alarm_pred(np.linspace(50, 150, 40), np.full(40, 20.0), np.full(40, 100.0))
    counts = [probabilistic_alarm(pred, a).sum() for a in (0.1, 0.3, 0.5, 0.7)]
    assert counts == sorted(counts, reverse=True)


def test_probabilistic_alarm_fires_above_the_threshold_when_uncertain():
    from tfm.eval.alarm import probabilistic_alarm

    pred = _alarm_pred([85.0], [20.0], [60.0])
    assert not (pred.y_pred < 70).any()
    assert probabilistic_alarm(pred, 0.15).all()


def test_event_metrics_rejects_a_mismatched_alarm_length():
    from tfm.eval.metrics import event_metrics

    pred = _alarm_pred([100.0] * 5, [10.0] * 5, [100.0] * 5)
    with pytest.raises(ValueError, match="alarm has"):
        event_metrics(pred, alarm=np.ones(3, dtype=bool))


def test_explicit_alarm_is_reordered_with_the_predictions():
    from tfm.eval.metrics import Predictions, event_metrics

    ts = pd.to_datetime(
        ["2027-01-01 01:00", "2027-01-01 00:00", "2027-01-01 00:05"]
    ).to_numpy()
    pred = Predictions([100.0] * 3, [100.0] * 3, ts, "9", 30)
    events = pd.DataFrame({"start": [pd.Timestamp("2027-01-01 00:35")]})
    # Only the first element (01:00, target 01:30) is flagged: far from the onset.
    far = event_metrics(pred, events=events, alarm=np.array([True, False, False]))
    # The second element (00:00, target 00:30) sits right on the onset.
    near = event_metrics(pred, events=events, alarm=np.array([False, True, False]))
    assert far["n_detected"] == 0
    assert near["n_detected"] == 1


def test_operating_curve_trades_recall_against_false_alarms():
    from tfm.eval.alarm import operating_curve

    rng = np.random.default_rng(0)
    n = 600
    y = np.where(rng.random(n) < 0.2, 60.0, 130.0)
    mu = y + rng.normal(0, 15, n)
    preds = [_alarm_pred(mu, np.full(n, 25.0), y, patient="P")]
    events = {"P": pd.DataFrame({"start": [pd.Timestamp("2027-01-01 00:30")]})}
    curve = operating_curve(preds, events, alphas=(0.05, 0.3, 0.7))
    assert list(curve["false_alarms_per_day"]) == sorted(
        curve["false_alarms_per_day"], reverse=True
    )


def test_pick_alpha_respects_the_false_alarm_budget():
    from tfm.eval.alarm import pick_alpha

    curve = pd.DataFrame({
        "alpha": [0.1, 0.2, 0.3, 0.4],
        "recall": [0.95, 0.90, 0.70, 0.40],
        "recall_lo": [0.9, 0.85, 0.6, 0.3],
        "recall_hi": [1.0, 0.95, 0.8, 0.5],
        "false_alarms_per_day": [5.0, 2.0, 0.9, 0.3],
    })
    chosen = pick_alpha(curve, max_false_alarms_per_day=1.0)
    assert chosen["alpha"] == 0.3
    assert chosen["false_alarms_per_day"] <= 1.0


def test_pick_alpha_reports_when_no_operating_point_is_feasible():
    from tfm.eval.alarm import pick_alpha

    curve = pd.DataFrame({
        "alpha": [0.1], "recall": [0.9], "recall_lo": [0.8], "recall_hi": [1.0],
        "false_alarms_per_day": [9.0],
    })
    assert pick_alpha(curve, 1.0)["alpha"] is None


def test_alarm_analysis_refuses_a_deterministic_experiment():
    from tfm.eval.alarm import alarm_analysis

    try:
        with pytest.raises(ValueError, match="not a probabilistic model"):
            alarm_analysis("20260916-baseline-lstm-global")
    except FileNotFoundError:
        pytest.skip("baseline experiments not run")
