"""Recurrent baselines: an LSTM over the history window.
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_horizons: int = 2,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1]))


class BiGRUForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_horizons: int = 2,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        heteroscedastic: bool = True,
    ):
        super().__init__()
        from tfm.models.heads import ForecastHead

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = ForecastHead(2 * hidden_size, n_horizons, heteroscedastic)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        forward_last = out[:, -1, : out.shape[-1] // 2]
        backward_last = out[:, 0, out.shape[-1] // 2 :]
        state = torch.cat([forward_last, backward_last], dim=-1)
        return self.head(self.dropout(state))
