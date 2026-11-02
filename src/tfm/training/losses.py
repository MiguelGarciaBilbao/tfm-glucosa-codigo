"""Losses for the heteroscedastic models.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tfm.data.quality import HYPO

LOG_VAR_MIN, LOG_VAR_MAX = -6.0, 6.0


def split_output(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = out[..., 0]
    log_var = out[..., 1].clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    return mu, log_var


class GaussianNLL(nn.Module):
    def __init__(self, beta: float = 0.5):
        super().__init__()
        self.beta = float(beta)

    def forward(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu, log_var = split_output(out)
        var = log_var.exp()
        nll = 0.5 * (log_var + (target - mu) ** 2 / var)
        if self.beta:
            # Detached, so the weight rescales the gradient without adding a
            # path through which the model can game the variance.
            nll = nll * var.detach() ** self.beta
        return nll.mean()


class HeteroscedasticMSE(nn.Module):
    def forward(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu, _ = split_output(out)
        return ((target - mu) ** 2).mean()


class WeightedGaussianNLL(nn.Module):
    def __init__(self, beta: float = 0.5):
        super().__init__()
        self.beta = float(beta)

    def forward(self, out: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        mu, log_var = split_output(out)
        var = log_var.exp()
        nll = 0.5 * (log_var + (target - mu) ** 2 / var)
        if self.beta:
            nll = nll * var.detach() ** self.beta
        return (weight * nll).sum() / weight.sum()


def hypo_weights(y_mgdl: np.ndarray, lam: float, threshold: float = HYPO) -> np.ndarray:
    return (1.0 + lam * (np.asarray(y_mgdl) < threshold)).astype(np.float32)


def build_loss(name: str = "gaussian_nll", **kwargs) -> nn.Module:
    losses = {
        "gaussian_nll": GaussianNLL,
        "mse": HeteroscedasticMSE,
    }
    if name not in losses:
        raise ValueError(f"unknown loss {name!r}; available: {sorted(losses)}")
    return losses[name](**kwargs) if name == "gaussian_nll" else losses[name]()
