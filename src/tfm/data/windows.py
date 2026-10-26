"""Sliding-window construction and scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tfm.data.parser import SAMPLING_MINUTES

#: History length and prediction horizons, in samples (60 min / 30 and 60 min).
DEFAULT_HISTORY = 12
DEFAULT_HORIZONS = (6, 12)


@dataclass
class WindowSet:
    """Windows for one patient and split."""

    X: np.ndarray  # (n_windows, history, n_features)
    y: np.ndarray  # (n_windows, n_horizons)
    ts: np.ndarray  # timestamp of the last observed sample of each window
    patient: str
    split: str
    feature_names: list[str]
    horizons: tuple[int, ...]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.X)

    @property
    def horizon_minutes(self) -> tuple[int, ...]:
        return tuple(h * SAMPLING_MINUTES for h in self.horizons)


def make_windows(
    frame: pd.DataFrame,
    feature_names: list[str],
    patient: str,
    split: str,
    history: int = DEFAULT_HISTORY,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    target: str = "value",
) -> WindowSet:
    max_h = max(horizons)
    Xs, ys, tss = [], [], []

    for _, seg in frame.dropna(subset=["segment"]).groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        n = len(seg)
        if n < history + max_h:
            continue
        feats = seg[feature_names].to_numpy(dtype=np.float32)
        tgt = seg[target].to_numpy(dtype=np.float32)
        times = seg["ts"].to_numpy()
        for i in range(n - history - max_h + 1):
            end = i + history
            Xs.append(feats[i:end])
            ys.append([tgt[end - 1 + h] for h in horizons])
            tss.append(times[end - 1])

    if not Xs:
        return WindowSet(
            np.empty((0, history, len(feature_names)), dtype=np.float32),
            np.empty((0, len(horizons)), dtype=np.float32),
            np.array([], dtype="datetime64[ns]"),
            patient, split, list(feature_names), tuple(horizons),
        )

    return WindowSet(
        np.stack(Xs), np.array(ys, dtype=np.float32), np.array(tss),
        patient, split, list(feature_names), tuple(horizons),
    )


def temporal_split(ws: WindowSet, val_fraction: float = 0.2) -> tuple[WindowSet, WindowSet]:
    if len(ws) == 0:
        return ws, ws
    order = np.argsort(ws.ts)
    cut = int(len(order) * (1 - val_fraction))
    parts = []
    for idx, split_name in ((order[:cut], "train"), (order[cut:], "val")):
        parts.append(
            WindowSet(ws.X[idx], ws.y[idx], ws.ts[idx], ws.patient, split_name,
                      ws.feature_names, ws.horizons, dict(ws.meta))
        )
    return parts[0], parts[1]


class WindowScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.feature_names: list[str] | None = None

    def fit(self, ws: WindowSet) -> "WindowScaler":
        flat = ws.X.reshape(-1, ws.X.shape[-1])
        self.mean_ = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std)
        self.feature_names = list(ws.feature_names)
        return self

    def transform(self, ws: WindowSet) -> WindowSet:
        if self.mean_ is None:
            raise RuntimeError("WindowScaler must be fitted before transforming")
        if ws.feature_names != self.feature_names:
            raise ValueError(
                "Feature order differs from the fitted scaler:\n"
                f"  fitted:  {self.feature_names}\n  given:   {ws.feature_names}"
            )
        X = (ws.X - self.mean_) / self.std_
        return WindowSet(X.astype(np.float32), ws.y, ws.ts, ws.patient, ws.split,
                         ws.feature_names, ws.horizons, dict(ws.meta))

    def fit_transform(self, ws: WindowSet) -> WindowSet:
        return self.fit(ws).transform(ws)


def concat_windows(sets: list[WindowSet], split: str = "train") -> WindowSet:
    sets = [s for s in sets if len(s) > 0]
    if not sets:
        raise ValueError("No non-empty WindowSets to concatenate")
    names, horizons = sets[0].feature_names, sets[0].horizons
    for s in sets:
        if s.feature_names != names or s.horizons != horizons:
            raise ValueError(f"Incompatible WindowSet for patient {s.patient}")
    return WindowSet(
        np.concatenate([s.X for s in sets]),
        np.concatenate([s.y for s in sets]),
        np.concatenate([s.ts for s in sets]),
        patient="+".join(s.patient for s in sets),
        split=split,
        feature_names=names,
        horizons=horizons,
        meta={"patients": [s.patient for s in sets],
              "sizes": [len(s) for s in sets]},
    )


def window_patient_index(ws: WindowSet) -> np.ndarray:
    patients = ws.meta.get("patients")
    sizes = ws.meta.get("sizes")
    if not patients:
        return np.zeros(len(ws), dtype=int)
    return np.concatenate([np.full(n, i, dtype=int) for i, n in enumerate(sizes)])
