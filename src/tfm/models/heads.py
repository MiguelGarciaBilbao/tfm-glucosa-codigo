"""The shared forecasting head.
"""

from __future__ import annotations

import torch
from torch import nn


class ForecastHead(nn.Module):
    def __init__(self, in_features: int, n_horizons: int, heteroscedastic: bool = True):
        super().__init__()
        self.n_horizons = n_horizons
        self.heteroscedastic = heteroscedastic
        self.linear = nn.Linear(in_features, n_horizons * (2 if heteroscedastic else 1))
        if heteroscedastic:
            with torch.no_grad():
                self.linear.bias.view(n_horizons, 2)[:, 1].fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.heteroscedastic:
            return out.view(-1, self.n_horizons, 2)
        return out
