"""Data quality and clinical characterisation of the CGM signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tfm.data.parser import SAMPLING_MINUTES

#: Clinical thresholds in mg/dL.
HYPO = 70.0
SEVERE_HYPO = 54.0
TARGET_LOW, TARGET_HIGH = 70.0, 180.0
HYPER = 180.0
SEVERE_HYPER = 250.0

#: Longest gap that is filled by interpolation; anything longer splits the series.
MAX_INTERP_GAP_MIN = 30

#: An episode counts as clinically significant from 15 min (3 samples) below
MIN_EVENT_MINUTES = 15

#: Fastest plausible glucose excursion, mg/dL per minute.
MAX_ROC_MG_DL_MIN = 5.0


def to_regular_grid(cgm: pd.DataFrame, freq_min: int = SAMPLING_MINUTES) -> pd.DataFrame:
    s = cgm.set_index("ts")["value"]
    s.index = s.index.round(f"{freq_min}min")
    s = s[~s.index.duplicated(keep="first")]
    grid = pd.date_range(s.index.min(), s.index.max(), freq=f"{freq_min}min")
    out = s.reindex(grid).to_frame("value")
    out.index.name = "ts"
    return out.reset_index()


def gap_table(cgm: pd.DataFrame, freq_min: int = SAMPLING_MINUTES) -> pd.DataFrame:
    grid = to_regular_grid(cgm, freq_min)
    missing = grid["value"].isna().to_numpy()
    if not missing.any():
        return pd.DataFrame(columns=["start", "end", "n_missing", "minutes"])
    idx = np.flatnonzero(missing)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return pd.DataFrame(
        {
            "start": grid["ts"].iloc[starts].to_numpy(),
            "end": grid["ts"].iloc[ends].to_numpy(),
            "n_missing": ends - starts + 1,
            "minutes": (ends - starts + 1) * freq_min,
        }
    )


def add_segments(
    cgm: pd.DataFrame,
    max_interp_gap_min: int = MAX_INTERP_GAP_MIN,
    freq_min: int = SAMPLING_MINUTES,
) -> pd.DataFrame:
    grid = to_regular_grid(cgm, freq_min)
    max_steps = max_interp_gap_min // freq_min
    was_na = grid["value"].isna()
    run_id = (was_na != was_na.shift()).cumsum()
    run_len = was_na.groupby(run_id).transform("size")
    fillable = was_na & (run_len <= max_steps)
    filled = grid["value"].interpolate(limit_area="inside")
    grid["value"] = grid["value"].where(~fillable, filled)
    grid["is_imputed"] = fillable & grid["value"].notna()
    breaks = grid["value"].isna()
    grid["segment"] = breaks.cumsum().where(~breaks)
    grid["segment"] = grid["segment"].astype("Int64")
    return grid


def segment_table(cgm: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Length in samples and minutes of each usable contiguous segment."""
    seg = add_segments(cgm, **kwargs).dropna(subset=["value"])
    g = seg.groupby("segment")
    return pd.DataFrame(
        {
            "start": g["ts"].min(),
            "end": g["ts"].max(),
            "n_samples": g.size(),
            "minutes": g.size() * SAMPLING_MINUTES,
            "pct_imputed": 100 * g["is_imputed"].mean(),
        }
    ).reset_index(drop=True)


def glycemic_events(
    cgm: pd.DataFrame,
    threshold: float = HYPO,
    below: bool = True,
    min_minutes: int = MIN_EVENT_MINUTES,
    freq_min: int = SAMPLING_MINUTES,
) -> pd.DataFrame:
    grid = to_regular_grid(cgm, freq_min)
    v = grid["value"]
    flag = (v < threshold) if below else (v > threshold)
    flag = flag.fillna(False).to_numpy()
    if not flag.any():
        return pd.DataFrame(columns=["start", "end", "n_samples", "minutes", "nadir", "hour"])
    idx = np.flatnonzero(flag)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    rows = []
    for a, b in zip(starts, ends):
        n = b - a + 1
        if n * freq_min < min_minutes:
            continue
        chunk = v.iloc[a : b + 1]
        rows.append(
            {
                "start": grid["ts"].iloc[a],
                "end": grid["ts"].iloc[b],
                "n_samples": n,
                "minutes": n * freq_min,
                "nadir": chunk.min() if below else chunk.max(),
                "hour": grid["ts"].iloc[a].hour,
            }
        )
    return pd.DataFrame(rows, columns=["start", "end", "n_samples", "minutes", "nadir", "hour"])


def time_in_range(cgm: pd.DataFrame) -> dict[str, float]:
    v = cgm["value"].dropna()
    return {
        "pct_severe_hypo": 100 * (v < SEVERE_HYPO).mean(),
        "pct_hypo": 100 * (v < HYPO).mean(),
        "pct_in_range": 100 * v.between(TARGET_LOW, TARGET_HIGH).mean(),
        "pct_hyper": 100 * (v > HYPER).mean(),
        "pct_severe_hyper": 100 * (v > SEVERE_HYPER).mean(),
        "mean": v.mean(),
        "std": v.std(),
        "cv": 100 * v.std() / v.mean(),
    }


def circadian_profile(cgm: pd.DataFrame) -> pd.DataFrame:
    df = cgm.dropna(subset=["value"]).copy()
    df["hour"] = pd.to_datetime(df["ts"]).dt.hour
    g = df.groupby("hour")["value"]
    return pd.DataFrame(
        {
            "mean": g.mean(),
            "std": g.std(),
            "p10": g.quantile(0.10),
            "p90": g.quantile(0.90),
            "pct_hypo": 100 * g.apply(lambda s: (s < HYPO).mean()),
            "n": g.size(),
        }
    ).reset_index()


def flag_implausible_jumps(
    gridded: pd.DataFrame,
    max_roc: float = MAX_ROC_MG_DL_MIN,
    freq_min: int = SAMPLING_MINUTES,
) -> pd.Series:
    step = gridded["value"].diff().abs()
    return (step > max_roc * freq_min).fillna(False)
