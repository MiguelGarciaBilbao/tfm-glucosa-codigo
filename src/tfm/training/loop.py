"""Training loop shared by every learned model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 0
    device: str | None = None
    warmup_epochs: int = 0
    history: dict = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        chunk = torch.from_numpy(np.ascontiguousarray(X[i : i + batch_size], dtype=np.float32))
        out.append(model(chunk.to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 0), dtype=np.float32)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    loss_fn: nn.Module | None = None,
    warmup_loss_fn: nn.Module | None = None,
) -> tuple[nn.Module, dict]:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    model = model.to(device)
    loss_fn = loss_fn or nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = _loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = _loader(X_val, y_val, cfg.batch_size, shuffle=False)

    best_loss, best_state, best_epoch, stale = float("inf"), None, -1, 0
    log = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(cfg.max_epochs):
        warming = epoch < cfg.warmup_epochs and warmup_loss_fn is not None
        active_loss = warmup_loss_fn if warming else loss_fn
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = active_loss(model(xb), yb)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item() * len(xb)
            n += len(xb)
        train_loss = running / max(n, 1)
        model.eval()
        running, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                running += loss_fn(model(xb), yb).item() * len(xb)
                n += len(xb)
        val_loss = running / max(n, 1)

        log["epoch"].append(epoch)
        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)
        if not warming and val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    log.update({"best_epoch": best_epoch, "best_val_loss": best_loss,
                "epochs_run": len(log["epoch"]), "device": str(device)})
    return model, log
