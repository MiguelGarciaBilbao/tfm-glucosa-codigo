"""PyTorch datasets"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tfm.data import quality as q
from tfm.data.features import FEATURE_COLUMNS, WEARABLE_COLUMNS, build_feature_frame
from tfm.data.loaders import load_cgm
from tfm.data.windows import (
    DEFAULT_HISTORY,
    DEFAULT_HORIZONS,
    WindowSet,
    make_windows,
)


class GlucoseWindows(Dataset):
    def __init__(self, ws: WindowSet):
        self.ws = ws
        self.X = torch.from_numpy(np.ascontiguousarray(ws.X))
        self.y = torch.from_numpy(np.ascontiguousarray(ws.y))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


def make_loader(ws: WindowSet, batch_size: int = 256, shuffle: bool = False, **kw) -> DataLoader:
    return DataLoader(GlucoseWindows(ws), batch_size=batch_size, shuffle=shuffle, **kw)


def patient_windows(
    patient: str,
    split: str,
    history: int = DEFAULT_HISTORY,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    wearable: bool = False,
    official: bool = True,
) -> WindowSet:
    cgm = load_cgm(patient, split, official=official)
    segments = q.add_segments(cgm)
    frame = build_feature_frame(patient, split, segments, wearable=wearable)
    names = list(FEATURE_COLUMNS) + (list(WEARABLE_COLUMNS) if wearable else [])
    ws = make_windows(frame, names, patient, split, history=history, horizons=horizons)
    ws.X[:, :, names.index("roc")] = np.nan_to_num(ws.X[:, :, names.index("roc")])
    return ws
