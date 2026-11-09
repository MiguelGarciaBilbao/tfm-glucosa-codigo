"""Tests for phase 5 adaptation: the weighted loss, the day splits and fine-tuning."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tfm.data.windows import WindowSet
from tfm.models.rnn import BiGRUForecaster
from tfm.training.losses import GaussianNLL, WeightedGaussianNLL, hypo_weights
from tfm.training.personalize import (
    AdaptConfig,
    Recipe,
    adaptation_split,
    fit_with_snapshots,
)


def test_weighted_nll_with_unit_weights_is_the_beta_nll():
    torch.manual_seed(0)
    out, y = torch.randn(8, 2, 2), torch.randn(8, 2)
    weighted = WeightedGaussianNLL(beta=0.5)(out, y, torch.ones(8, 2)).item()
    assert weighted == pytest.approx(GaussianNLL(beta=0.5)(out, y).item())


def test_weighted_nll_is_a_weighted_mean_of_the_terms():
    out, y = torch.zeros(2, 1, 2), torch.tensor([[0.0], [2.0]])
    loss = WeightedGaussianNLL(beta=0.0)
    assert loss(out, y, torch.ones(2, 1)).item() == pytest.approx(1.0)
    assert loss(out, y, torch.tensor([[1.0], [11.0]])).item() == pytest.approx(22 / 12)


def test_hypo_weights_mark_only_targets_below_70():
    w = hypo_weights(np.array([[69.9, 70.0], [150.0, 40.0]]), lam=5)
    np.testing.assert_array_equal(w, [[6.0, 1.0], [1.0, 6.0]])
    assert w.dtype == np.float32


def test_hypo_weights_are_all_one_without_hypoglycaemia():
    assert (hypo_weights(np.full((3, 2), 120.0), lam=10) == 1.0).all()


def _windows(days: int = 10, patient: str = "p") -> WindowSet:
    start = np.datetime64("2027-01-01T00:00", "ns")
    ts = start + np.arange(days * 288) * np.timedelta64(5, "m")
    n = len(ts)
    return WindowSet(np.zeros((n, 12, 1), np.float32), np.zeros((n, 2), np.float32), ts,
                     patient, "train", ["value"], (6, 12))


@pytest.mark.parametrize("n_days", [1, 3, 7])
def test_adaptation_and_development_never_share_a_target(n_days):
    adapt, dev = adaptation_split(_windows(), n_days)
    last_adapt_target = adapt.ts.max() + np.timedelta64(60, "m")
    first_dev_target = dev.ts.min() + np.timedelta64(30, "m")
    assert last_adapt_target < first_dev_target
    assert len(np.intersect1d(adapt.ts, dev.ts)) == 0


def test_development_is_the_same_for_every_budget():
    devs = [adaptation_split(_windows(), n)[1].ts for n in (1, 3, 7)]
    np.testing.assert_array_equal(devs[0], devs[1])
    np.testing.assert_array_equal(devs[0], devs[2])


def test_adaptation_counts_calendar_days_from_the_first_window():
    assert len(adaptation_split(_windows(), 1)[0]) == 288
    assert len(adaptation_split(_windows(), 0)[0]) == 0


def test_a_budget_beyond_the_development_start_is_rejected():
    with pytest.raises(ValueError, match="n_days"):
        adaptation_split(_windows(), 8)


def _tiny_model() -> BiGRUForecaster:
    torch.manual_seed(0)
    return BiGRUForecaster(n_features=3, n_horizons=2, hidden_size=8)


def _data(n: int = 40):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 12, 3)).astype(np.float32)
    y = rng.normal(size=(n, 2)).astype(np.float32)
    return X, y, np.ones_like(y)


def test_head_only_leaves_the_encoder_bit_for_bit():
    model = _tiny_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    snaps = fit_with_snapshots(model, *_data(), Recipe(lr=1e-2, layers="head"), (2,),
                               AdaptConfig(device="cpu"), seed=0)
    after = snaps[2]
    for k in before:
        if k.startswith("gru."):
            assert torch.equal(before[k], after[k]), k
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("head."))


def test_snapshots_are_taken_at_the_requested_epochs():
    snaps = fit_with_snapshots(_tiny_model(), *_data(), Recipe(lr=1e-2), (1, 3),
                               AdaptConfig(device="cpu"), seed=0)
    assert sorted(snaps) == [1, 3]
    assert any(not torch.equal(snaps[1][k], snaps[3][k]) for k in snaps[1])


def test_every_parameter_is_trainable_again_after_fitting():
    model = _tiny_model()
    fit_with_snapshots(model, *_data(), Recipe(lr=1e-2, layers="head"), (1,),
                       AdaptConfig(device="cpu"), seed=0)
    assert all(p.requires_grad for p in model.parameters())


def test_fitting_without_windows_fails_loudly():
    X, y, w = _data(0)
    with pytest.raises(ValueError, match="no adaptation windows"):
        fit_with_snapshots(_tiny_model(), X, y, w, Recipe(lr=1e-3), (1,),
                           AdaptConfig(device="cpu"), seed=0)


def test_recipe_rejects_unknown_layers():
    with pytest.raises(ValueError, match="layers"):
        Recipe(lr=1e-3, layers="gru")


def test_recipe_key_names_every_setting():
    assert Recipe(lr=3e-4, layers="head", lam=5).key(20) == "layers=head|lr=0.0003|lam=5|epochs=20"
