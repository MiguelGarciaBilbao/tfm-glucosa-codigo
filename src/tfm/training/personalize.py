"""Phase 5: adapting a model to one patient with their first days of data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tfm.data.windows import WindowSet
from tfm.training.loop import pick_device, set_seed
from tfm.training.losses import HeteroscedasticMSE, WeightedGaussianNLL


DEV_START_DAYS = 7

LAYER_CHOICES = ("head", "all")


def _subset(ws: WindowSet, mask: np.ndarray, split: str) -> WindowSet:
    return WindowSet(ws.X[mask], ws.y[mask], ws.ts[mask], ws.patient, split,
                     ws.feature_names, ws.horizons, dict(ws.meta))


def adaptation_split(
    ws: WindowSet, n_days: int, dev_start_days: int = DEV_START_DAYS
) -> tuple[WindowSet, WindowSet]:
    if not 0 <= n_days <= dev_start_days:
        raise ValueError(f"n_days must be in [0, {dev_start_days}], got {n_days}")
    if len(ws) == 0:
        raise ValueError(f"patient {ws.patient} has no training windows")
    t0 = ws.ts.min()
    adapt = ws.ts < t0 + np.timedelta64(n_days, "D")
    dev_start = (t0 + np.timedelta64(dev_start_days, "D")
                 + np.timedelta64(max(ws.horizon_minutes), "m"))
    return _subset(ws, adapt, "adapt"), _subset(ws, ws.ts >= dev_start, "dev")


@dataclass(frozen=True)
class Recipe:
    lr: float
    layers: str = "all"
    lam: float = 0.0

    def __post_init__(self) -> None:
        if self.layers not in LAYER_CHOICES:
            raise ValueError(f"layers must be one of {LAYER_CHOICES}, got {self.layers!r}")
        if self.lam < 0:
            raise ValueError(f"lam must be >= 0, got {self.lam}")

    def key(self, epochs: int) -> str:
        return f"layers={self.layers}|lr={self.lr:g}|lam={self.lam:g}|epochs={epochs}"


@dataclass(frozen=True)
class AdaptConfig:

    batch_size: int = 64
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup_epochs: int = 0
    beta: float = 0.5
    device: str | None = None


def _set_trainable(model: nn.Module, layers: str) -> list[nn.Parameter]:
    for name, param in model.named_parameters():
        param.requires_grad = layers == "all" or name.startswith("head.")
    return [p for p in model.parameters() if p.requires_grad]


def fit_with_snapshots(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    recipe: Recipe,
    epochs: tuple[int, ...],
    cfg: AdaptConfig,
    seed: int,
) -> dict[int, dict[str, torch.Tensor]]:
    if len(X) == 0:
        raise ValueError("no adaptation windows to fit on")
    if not epochs or min(epochs) < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")
    set_seed(seed)
    device = pick_device(cfg.device)
    model.to(device)
    params = _set_trainable(model, recipe.layers)
    opt = torch.optim.Adam(params, lr=recipe.lr, weight_decay=cfg.weight_decay)
    main_loss, warm_loss = WeightedGaussianNLL(cfg.beta), HeteroscedasticMSE()
    data = TensorDataset(*(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
                           for a in (X, y, weight)))
    loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=True)
    wanted, snapshots = set(epochs), {}
    try:
        for epoch in range(1, max(epochs) + 1):
            model.train()
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                opt.zero_grad()
                out = model(xb)
                if epoch <= cfg.warmup_epochs:
                    loss = warm_loss(out, yb)
                else:
                    loss = main_loss(out, yb, wb)
                loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
            if epoch in wanted:
                snapshots[epoch] = {k: v.detach().cpu().clone()
                                    for k, v in model.state_dict().items()}
    finally:
        for param in model.parameters():
            param.requires_grad = True
    return snapshots
