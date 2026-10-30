"""The evaluation protocol of the TFM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tfm.data.quality import HYPO, MIN_EVENT_MINUTES, glycemic_events
from tfm.eval.metrics import Predictions

#: Detection tolerance used everywhere in the TFM, in minutes.
EVENT_TOLERANCE_MIN = 15

#: Tolerances reported as a sensitivity analysis alongside the headline number.
SENSITIVITY_TOLERANCES_MIN = (5, 15)


def covered_instants(predictions: list[Predictions]) -> np.ndarray:
    if not predictions:
        return np.array([], dtype="datetime64[ns]")
    common = set(pd.to_datetime(predictions[0].target_ts))
    for p in predictions[1:]:
        common &= set(pd.to_datetime(p.target_ts))
    return np.array(sorted(common), dtype="datetime64[ns]")


def official_events(
    patient: str,
    split: str = "test",
    threshold: float = HYPO,
    min_event_minutes: int = MIN_EVENT_MINUTES,
) -> pd.DataFrame:
    from tfm.data.loaders import load_cgm

    return glycemic_events(
        load_cgm(patient, split, official=True),
        threshold=threshold,
        below=True,
        min_minutes=min_event_minutes,
    )


def scorable_events(
    patient: str,
    predictions: list[Predictions],
    split: str = "test",
    threshold: float = HYPO,
    min_event_minutes: int = MIN_EVENT_MINUTES,
) -> pd.DataFrame:
    events = official_events(patient, split, threshold, min_event_minutes)
    if events.empty:
        return events
    covered = pd.DatetimeIndex(covered_instants(predictions))
    keep = pd.DatetimeIndex(events["start"]).isin(covered)
    return events.loc[keep].reset_index(drop=True)


def scorable_events_by_patient(
    predictions_by_horizon: dict[int, list[Predictions]],
    split: str = "test",
) -> dict[str, pd.DataFrame]:
    horizons = sorted(predictions_by_horizon)
    if not horizons:
        raise ValueError("no horizons given")
    by_patient: dict[str, list[Predictions]] = {}
    for h in horizons:
        for p in predictions_by_horizon[h]:
            by_patient.setdefault(p.patient, []).append(p)
    n = len(horizons)
    missing = [pid for pid, preds in by_patient.items() if len(preds) != n]
    if missing:
        raise ValueError(f"patients missing at some horizon: {sorted(missing)}")
    return {
        pid: scorable_events(pid, preds, split) for pid, preds in by_patient.items()
    }
