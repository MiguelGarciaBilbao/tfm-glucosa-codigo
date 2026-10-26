"""Feature engineering on the 5-minute grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tfm.data import quality as q
from tfm.data.loaders import load_channel
from tfm.data.parser import SAMPLING_MINUTES

INSULIN_PEAK_MIN = 75
INSULIN_DURATION_MIN = 300

CARB_ABSORPTION_MIN = 180


def insulin_remaining_fraction(
    minutes: np.ndarray,
    peak: float = INSULIN_PEAK_MIN,
    duration: float = INSULIN_DURATION_MIN,
) -> np.ndarray:
    td, tp = float(duration), float(peak)
    tau = tp * (1 - tp / td) / (1 - 2 * tp / td)
    a = 2 * tau / td
    s = 1 / (1 - a + (1 + a) * np.exp(-td / tau))

    t = np.asarray(minutes, dtype=float)
    with np.errstate(over="ignore"):
        iob = 1 - s * (1 - a) * (
            (t**2 / (tau * td * (1 - a)) - t / tau - 1) * np.exp(-t / tau) + 1
        )
    return np.clip(np.where(t < 0, 0.0, np.where(t > td, 0.0, iob)), 0.0, 1.0)


def carbs_remaining_fraction(
    minutes: np.ndarray, duration: float = CARB_ABSORPTION_MIN
) -> np.ndarray:
    t = np.asarray(minutes, dtype=float)
    return np.clip(1 - t / float(duration), 0.0, 1.0)


def _decay_on_grid(
    grid: pd.DatetimeIndex,
    events: pd.DataFrame,
    ts_col: str,
    amount_col: str,
    kernel,
    horizon_min: float,
) -> np.ndarray:
    out = np.zeros(len(grid), dtype=float)
    if events.empty:
        return out
    max_steps = int(horizon_min // SAMPLING_MINUTES) + 1
    offsets = np.arange(max_steps) * SAMPLING_MINUTES
    weights = kernel(offsets)
    positions = grid.searchsorted(events[ts_col].to_numpy(), side="left")
    for pos, amount in zip(positions, events[amount_col].to_numpy()):
        if pos >= len(grid) or not np.isfinite(amount):
            continue
        end = min(pos + max_steps, len(grid))
        out[pos:end] += amount * weights[: end - pos]
    return out


def _channel(patient: str, split: str, name: str, channels: dict | None) -> pd.DataFrame:
    if channels is None:
        return load_channel(patient, split, name)
    if name in channels:
        return channels[name]
    from tfm.data.parser import CHANNELS

    cols = CHANNELS[name]
    return pd.DataFrame(columns=[*cols.time_cols, *cols.numeric_cols, *cols.text_cols])


def basal_rate_on_grid(
    patient: str, split: str, grid: pd.DatetimeIndex, channels: dict | None = None
) -> pd.DataFrame:
    basal = _channel(patient, split, "basal", channels)
    rate = pd.Series(np.nan, index=grid, dtype=float)
    if not basal.empty:
        b = basal.dropna(subset=["ts"]).set_index("ts")["value"]
        b = b[~b.index.duplicated(keep="last")]
        rate = b.reindex(b.index.union(grid)).ffill().reindex(grid)
    rate = rate.ffill().bfill().fillna(0.0)

    temp = _channel(patient, split, "temp_basal", channels)
    suspended = pd.Series(0.0, index=grid, dtype=float)
    for _, row in temp.iterrows():
        if pd.isna(row["ts_begin"]) or pd.isna(row["ts_end"]):
            continue
        mask = (grid >= row["ts_begin"]) & (grid < row["ts_end"])
        if not mask.any():
            continue
        rate.loc[mask] = row["value"]
        if row["value"] == 0:
            suspended.loc[mask] = 1.0
    return pd.DataFrame({"basal_rate": rate.to_numpy(), "pump_suspended": suspended.to_numpy()},
                        index=grid)


def bolus_events(patient: str, split: str, channels: dict | None = None) -> pd.DataFrame:
    bolus = _channel(patient, split, "bolus", channels)
    if bolus.empty:
        return pd.DataFrame(columns=["ts", "dose"])
    rows = []
    for _, row in bolus.iterrows():
        begin, end, dose = row["ts_begin"], row["ts_end"], row["dose"]
        if pd.isna(begin) or not np.isfinite(dose):
            continue
        if pd.isna(end) or end <= begin:
            rows.append({"ts": begin, "dose": dose})
            continue
        steps = pd.date_range(begin, end, freq=f"{SAMPLING_MINUTES}min")
        if len(steps) <= 1:
            rows.append({"ts": begin, "dose": dose})
        else:
            share = dose / len(steps)
            rows.extend({"ts": s, "dose": share} for s in steps)
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def rate_of_change(values: pd.Series, minutes: int = 15) -> pd.Series:
    steps = minutes // SAMPLING_MINUTES
    return (values - values.shift(steps)) / minutes


def build_feature_frame(
    patient: str,
    split: str,
    cgm_segments: pd.DataFrame,
    wearable: bool = True,
    channels: dict | None = None,
) -> pd.DataFrame:
    df = cgm_segments.copy()
    grid = pd.DatetimeIndex(df["ts"])
    basal = basal_rate_on_grid(patient, split, grid, channels)
    df["basal_rate"] = basal["basal_rate"].to_numpy()
    df["pump_suspended"] = basal["pump_suspended"].to_numpy()

    boluses = bolus_events(patient, split, channels)
    df["bolus_dose"] = _decay_on_grid(
        grid, boluses, "ts", "dose", lambda t: (t == 0).astype(float), SAMPLING_MINUTES
    )
    df["iob"] = _decay_on_grid(
        grid, boluses, "ts", "dose", insulin_remaining_fraction, INSULIN_DURATION_MIN
    )

    meals = _channel(patient, split, "meal", channels)
    meals = meals.dropna(subset=["ts"]) if not meals.empty else meals
    df["carbs"] = _decay_on_grid(
        grid, meals, "ts", "carbs", lambda t: (t == 0).astype(float), SAMPLING_MINUTES
    )
    df["cob"] = _decay_on_grid(
        grid, meals, "ts", "carbs", carbs_remaining_fraction, CARB_ABSORPTION_MIN
    )

    df["roc"] = rate_of_change(df["value"])
    df["sensor_jump"] = q.flag_implausible_jumps(df).astype(float)

    hour = grid.hour + grid.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    if wearable:
        for channel, name in (("basis_gsr", "gsr"), ("basis_skin_temperature", "skin_temp")):
            raw = _channel(patient, split, channel, channels)
            if raw.empty:
                df[name] = np.nan
                continue
            s = raw.dropna(subset=["ts"]).set_index("ts")["value"]
            s = s.resample(f"{SAMPLING_MINUTES}min").mean()
            df[name] = s.reindex(grid).to_numpy()
        for name in ("gsr", "skin_temp"):
            df[f"{name}_missing"] = df[name].isna().astype(float)
            df[name] = df[name].ffill(limit=6).fillna(df[name].median())

    df["is_imputed"] = df["is_imputed"].astype(float)
    return df

FEATURE_COLUMNS = [
    "value",
    "roc",
    "iob",
    "bolus_dose",
    "basal_rate",
    "pump_suspended",
    "cob",
    "carbs",
    "hour_sin",
    "hour_cos",
    "is_imputed",
    "sensor_jump",
]

WEARABLE_COLUMNS = ["gsr", "skin_temp", "gsr_missing", "skin_temp_missing"]
