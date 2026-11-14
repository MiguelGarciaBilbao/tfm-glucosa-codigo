"""HUPA-UCM, the external validation set, in the OhioT1DM channel schema.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from tfm.data import quality as q
from tfm.data.features import FEATURE_COLUMNS, build_feature_frame
from tfm.data.parser import SAMPLING_MINUTES
from tfm.data.windows import DEFAULT_HISTORY, DEFAULT_HORIZONS, WindowSet, make_windows
from tfm.paths import HUPA_DIR, HUPA_RAW_DIR

PATIENTS = tuple(
    f"HUPA{n:04d}P"
    for n in (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21,
              22, 23, 24, 25, 26, 27, 28)
)

#: Treatment per patient, from Table 1 of Hidalgo et al. (2024): continuous
CSII = frozenset(f"HUPA{n:04d}P" for n in (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 16, 18, 22, 25))
TREATMENT = {p: ("CSII" if p in CSII else "MDI") for p in PATIENTS}

#: No raw Libre export: HUPA0009P's file is empty and HUPA0010P's holds no
EXCLUDED = {
    "HUPA0009P": "sin lecturas del Libre (fichero vacío)",
    "HUPA0010P": "sin lecturas históricas del Libre",
}

#: Patients whose ``carb_input`` is already in grams. The paper states that
CARBS_IN_GRAMS = frozenset({"HUPA0002P", "HUPA0004P", "HUPA0011P", "HUPA0022P"})

#: Patients whose boluses and meals are always exactly 1: they logged *that*
UNQUANTIFIED_DOSES = frozenset({"HUPA0019P", "HUPA0020P", "HUPA0021P"})

#: How much of the treatment each patient recorded, verified against the data:
RECORD_TIER = {
    **{p: "completo" for p in PATIENTS},
    "HUPA0011P": "parcial",       # pump user with no insulin in the export
    "HUPA0014P": "parcial",       # boluses and meals, no long-acting insulin
    "HUPA0019P": "parcial",       # flags only (UNQUANTIFIED_DOSES)
    "HUPA0020P": "parcial",
    "HUPA0021P": "parcial",
    "HUPA0015P": "sin_registro",
    "HUPA0018P": "sin_registro",
}

#: A pump basal of zero for at most this long is a suspension (the Paradigm Veo
MAX_SUSPEND_MIN = 120

#: Libre historic readings are 15 minutes apart. Snapped to the 5-minute grid
LIBRE_CADENCE_STEPS = 3

#: Two predictions this far apart still belong to one alarm episode: one Libre
ALARM_EPISODE_GAP_MIN = (LIBRE_CADENCE_STEPS + 1) * SAMPLING_MINUTES

#: Hypoglycaemia events on Libre data: consecutive readings below threshold,
MIN_EVENT_READINGS = 2

#: Share of each patient's readings, at the end of the series, held out as test.
TEST_FRACTION = 0.3

_TIME_HEADERS = ("Hora", "Sello de tiempo del dispositivo")
_GLUCOSE_HEADERS = ("Histórico glucosa", "Historial de glucosa")
_RECORD_HISTORIC = "0"
_TS_FORMATS = ("%Y/%m/%d %H:%M", "%d/%m/%y %H:%M", "%d-%m-%Y %H:%M", "%m-%d-%Y %H:%M")


def _parse_timestamps(raw: pd.Series) -> pd.Series:
    best, best_score = None, -1.0
    for fmt in _TS_FORMATS:
        ts = pd.to_datetime(raw.str.strip(), format=fmt, errors="coerce")
        if ts.isna().mean() > 0.01:
            continue
        step = ts.sort_values().diff().dt.total_seconds().div(60)
        score = step.between(13, 17).mean()
        if score > best_score:
            best, best_score = ts, score
    if best is None:
        raise ValueError(f"unrecognised Libre timestamp format, e.g. {raw.iloc[0]!r}")
    return best


def read_libre(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8-sig", newline=None) as fh:
        lines = fh.read().splitlines()
    header_at = next((i for i, line in enumerate(lines) if "Tipo de registro" in line), None)
    if header_at is None:
        return pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns]"),
                             "value": pd.Series(dtype=float)})
    header_line = lines[header_at]
    sep = max((";", "\t", ","), key=header_line.count)
    rows = list(csv.reader(lines[header_at:], delimiter=sep))
    header = [h.strip() for h in rows[0]]
    t_col = next(i for i, h in enumerate(header) if h in _TIME_HEADERS)
    r_col = header.index("Tipo de registro")
    g_col = next(i for i, h in enumerate(header) if h.startswith(_GLUCOSE_HEADERS))
    body = [
        (r[t_col], r[g_col]) for r in rows[1:]
        if len(r) > g_col and r[r_col].strip() == _RECORD_HISTORIC and r[g_col].strip()
    ]
    if not body:
        return pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns]"),
                             "value": pd.Series(dtype=float)})
    df = pd.DataFrame(body, columns=["ts", "value"])
    df["ts"] = _parse_timestamps(df["ts"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().sort_values("ts").reset_index(drop=True)


def read_preprocessed(patient: str, raw_dir: Path = HUPA_RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "Preprocessed" / f"{patient}.csv", sep=";")
    df["time"] = pd.to_datetime(df["time"])
    return df


def libre_glucose(
    patient: str, window: tuple[pd.Timestamp, pd.Timestamp], raw_dir: Path = HUPA_RAW_DIR
) -> pd.DataFrame:
    files = sorted((raw_dir / "Raw_Data" / patient / "free_style_sensor").glob("*.csv"))
    parts = [read_libre(f) for f in files]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns]"),
                             "value": pd.Series(dtype=float)})
    df = pd.concat(parts).drop_duplicates("ts").sort_values("ts")
    start, end = window
    return df[(df["ts"] >= start) & (df["ts"] <= end)].reset_index(drop=True)


def _runs(flag: np.ndarray) -> list[tuple[int, int]]:
    if not flag.any():
        return []
    idx = np.flatnonzero(flag)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def basal_events(patient: str, pre: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rate = pre["basal_rate"].to_numpy(dtype=float) * 12
    ts = pre["time"]
    change = np.r_[True, np.diff(rate) != 0]
    basal = pd.DataFrame({"ts": ts[change].to_numpy(), "value": rate[change]})

    rows = []
    if TREATMENT[patient] == "CSII" and RECORD_TIER[patient] != "sin_registro":
        for a, b in _runs(rate == 0):
            minutes = (b - a + 1) * SAMPLING_MINUTES
            if minutes <= MAX_SUSPEND_MIN:
                rows.append({
                    "ts_begin": ts.iloc[a],
                    "ts_end": ts.iloc[b] + pd.Timedelta(minutes=SAMPLING_MINUTES),
                    "value": 0.0,
                })
    temp = pd.DataFrame(rows, columns=["ts_begin", "ts_end", "value"])
    return basal, temp


def treatment_channels(patient: str, pre: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Insulin and carbohydrates in the parser schema, units repaired."""
    channels: dict[str, pd.DataFrame] = {}
    channels["basal"], channels["temp_basal"] = basal_events(patient, pre)
    if patient in UNQUANTIFIED_DOSES:
        return channels

    dose = pre["bolus_volume_delivered"].to_numpy(dtype=float)
    hit = dose > 0
    channels["bolus"] = pd.DataFrame({
        "ts_begin": pre["time"][hit].to_numpy(),
        "ts_end": pre["time"][hit].to_numpy(),
        "dose": dose[hit],
        "bwz_carb_input": np.nan,
        "type": "normal",
    })

    carbs = pre["carb_input"].to_numpy(dtype=float)
    grams = carbs if patient in CARBS_IN_GRAMS else carbs * 10
    hit = grams > 0
    channels["meal"] = pd.DataFrame({
        "ts": pre["time"][hit].to_numpy(),
        "carbs": grams[hit],
        "type": "meal",
    })
    return channels


def patient_channels(patient: str, raw_dir: Path = HUPA_RAW_DIR) -> dict[str, pd.DataFrame]:
    if patient in EXCLUDED:
        raise ValueError(f"{patient} is excluded: {EXCLUDED[patient]}")
    pre = read_preprocessed(patient, raw_dir)
    window = (pre["time"].min(), pre["time"].max())
    return {"glucose_level": libre_glucose(patient, window, raw_dir),
            **treatment_channels(patient, pre)}


def mark_libre_cadence(gridded: pd.DataFrame) -> pd.Series:
    was_na = gridded["value"].isna()
    run_id = (was_na != was_na.shift()).cumsum()
    run_len = was_na.groupby(run_id).transform("size")
    return was_na & (run_len <= LIBRE_CADENCE_STEPS)


def feature_frame(patient: str, channels: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    channels = patient_channels(patient) if channels is None else channels
    cgm = channels["glucose_level"]
    gridded = q.to_regular_grid(cgm)
    segments = q.add_segments(cgm)
    segments["is_reading"] = gridded["value"].notna().to_numpy()
    segments["libre_cadence"] = mark_libre_cadence(gridded).to_numpy() & segments["is_imputed"]
    frame = build_feature_frame(patient, "all", segments, wearable=False, channels=channels)
    frame["patient"] = patient
    return frame


def libre_events(
    frame: pd.DataFrame,
    threshold: float = q.HYPO,
    min_readings: int = MIN_EVENT_READINGS,
) -> pd.DataFrame:
    r = frame[frame["is_reading"]].reset_index(drop=True)
    cols = ["start", "end", "n_readings", "minutes", "nadir", "hour"]
    if r.empty:
        return pd.DataFrame(columns=cols)
    low = (r["value"] < threshold).to_numpy()
    same_segment = np.r_[False, r["segment"].to_numpy()[1:] == r["segment"].to_numpy()[:-1]]
    rows = []
    run_start = None
    for i in range(len(r) + 1):
        inside = i < len(r) and low[i] and (run_start is None or same_segment[i])
        if inside and run_start is None:
            run_start = i
            continue
        if inside:
            continue
        if run_start is not None:
            end = i - 1
            n = end - run_start + 1
            if n >= min_readings:
                chunk = r.iloc[run_start : end + 1]
                rows.append({
                    "start": chunk["ts"].iloc[0],
                    "end": chunk["ts"].iloc[-1],
                    "n_readings": n,
                    "minutes": (chunk["ts"].iloc[-1] - chunk["ts"].iloc[0]).total_seconds() / 60,
                    "nadir": chunk["value"].min(),
                    "hour": chunk["ts"].iloc[0].hour,
                })

            run_start = i if (i < len(r) and low[i]) else None
    return pd.DataFrame(rows, columns=cols)


def patient_windows(
    frame: pd.DataFrame,
    history: int = DEFAULT_HISTORY,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    imputed_flag: str = "dropouts",
    split: str = "all",
) -> WindowSet:
    if imputed_flag not in ("dropouts", "all"):
        raise ValueError(f"imputed_flag must be 'dropouts' or 'all', got {imputed_flag!r}")
    frame = frame.copy()
    if imputed_flag == "dropouts":
        frame["is_imputed"] = (frame["is_imputed"].astype(bool) & ~frame["libre_cadence"]).astype(float)
    frame["is_reading"] = frame["is_reading"].astype(float)
    patient = str(frame["patient"].iloc[0])
    names = list(FEATURE_COLUMNS)

    ws = make_windows(frame, names, patient, split, history=history, horizons=horizons)
    flags = make_windows(frame, ["is_reading"], patient, split,
                         history=history, horizons=horizons, target="is_reading")
    keep = (flags.X[:, -1, 0] == 1) & (flags.y == 1).all(axis=1)
    roc = names.index("roc")
    X = ws.X[keep]
    X[:, :, roc] = np.nan_to_num(X[:, :, roc])
    return WindowSet(X, ws.y[keep], ws.ts[keep], patient, split, names, tuple(horizons),
                     meta={"n_all_windows": len(ws), "imputed_flag": imputed_flag})


def split_cut(frame: pd.DataFrame, test_fraction: float = TEST_FRACTION) -> pd.Timestamp:
    ts = frame.loc[frame["is_reading"], "ts"].reset_index(drop=True)
    if ts.empty:
        raise ValueError("no readings to split")
    return ts.iloc[int(len(ts) * (1 - test_fraction))]


def split_frame(frame: pd.DataFrame, test_fraction: float = TEST_FRACTION) -> dict[str, pd.DataFrame]:
    cut = split_cut(frame, test_fraction)
    return {"adapt": frame[frame["ts"] < cut].reset_index(drop=True),
            "test": frame[frame["ts"] >= cut].reset_index(drop=True)}


def event_budget(frames: dict[str, pd.DataFrame], test_fraction: float = TEST_FRACTION) -> pd.DataFrame:
    rows = []
    for patient, frame in frames.items():
        events = libre_events(frame)
        cut = split_cut(frame, test_fraction)
        days = (frame["ts"].iloc[-1] - frame["ts"].iloc[0]).total_seconds() / 86400
        rows.append({
            "patient": patient,
            "treatment": TREATMENT[patient],
            "record_tier": RECORD_TIER[patient],
            "days": round(days, 1),
            "events_all": int(len(events)),
            "events_test": int((events["start"] >= cut).sum()) if len(events) else 0,
        })
    return pd.DataFrame(rows)


def preprocessed_agreement(patient: str, raw_dir: Path = HUPA_RAW_DIR) -> dict:
    pre = read_preprocessed(patient, raw_dir)
    window = (pre["time"].min(), pre["time"].max())
    raw = libre_glucose(patient, window, raw_dir)
    raw = raw.assign(ts=raw["ts"].dt.round(f"{SAMPLING_MINUTES}min")).drop_duplicates("ts")
    pre_g = pre.drop_duplicates("time").set_index("time")["glucose"]
    common = pd.DatetimeIndex(raw["ts"]).intersection(pre_g.index)
    raw_g = raw.set_index("ts")["value"]
    linear = (pre["glucose"].diff().diff().abs() < 1e-6).to_numpy()
    longest = max((b - a + 1 for a, b in _runs(linear)), default=0)

    # A cell is interpolated when it sits on a straight line with both neighbours.
    on_line = linear & np.r_[linear[1:], False]
    unverifiable = 0.0
    if len(raw):
        r = np.sort(raw["ts"].to_numpy())
        t = pre["time"].to_numpy()
        i = np.clip(np.searchsorted(r, t), 1, max(len(r) - 1, 1))
        nearest = np.minimum(np.abs(t - r[i - 1]), np.abs(t - r[np.minimum(i, len(r) - 1)]))
        unverifiable = float(((nearest > np.timedelta64(30, "m")) & ~on_line).sum())
    return {
        "patient": patient,
        "n_common": int(len(common)),
        "pct_equal": round(100 * float((raw_g[common] - pre_g[common]).abs().lt(0.5).mean()), 1)
        if len(common) else float("nan"),
        "longest_linear_h": round((longest + 1) * SAMPLING_MINUTES / 60, 1),
        "unverifiable_h": round(unverifiable * SAMPLING_MINUTES / 60, 1),
    }


def characterise(frame: pd.DataFrame) -> dict:
    """One row of the HUPA characterisation table."""
    patient = str(frame["patient"].iloc[0])
    readings = frame.loc[frame["is_reading"], "value"]
    days = (frame["ts"].iloc[-1] - frame["ts"].iloc[0]).total_seconds() / 86400
    seg = frame.dropna(subset=["segment"]).groupby("segment").size() * SAMPLING_MINUTES
    events = libre_events(frame)
    return {
        "patient": patient,
        "treatment": TREATMENT[patient],
        "record_tier": RECORD_TIER[patient],
        "days": round(days, 1),
        "n_readings": int(len(readings)),
        "n_segments": int(len(seg)),
        "median_segment_h": round(float(seg.median()) / 60, 1) if len(seg) else 0.0,
        "pct_cells_dropout": round(100 * float(
            (frame["is_imputed"].astype(bool) & ~frame["libre_cadence"]).mean()), 2),
        "pct_readings_below_70": round(100 * float((readings < q.HYPO).mean()), 2),
        "pct_readings_below_54": round(100 * float((readings < q.SEVERE_HYPO).mean()), 2),
        "pct_readings_70_180": round(100 * float(readings.between(q.TARGET_LOW, q.TARGET_HIGH).mean()), 2),
        "hypo_events": int(len(events)),
        "pct_pump_suspended": round(100 * float(frame["pump_suspended"].mean()), 2),
        "pct_sensor_jump": round(100 * float(frame["sensor_jump"].mean()), 2),
        "bolus_u_per_day": round(float(frame["bolus_dose"].sum()) / days, 1),
        "carbs_g_per_day": round(float(frame["carbs"].sum()) / days, 1),
        "basal_u_per_day": round(float(frame["basal_rate"].mean()) * 24, 1),
    }


def build(patients=PATIENTS, out_dir: Path = HUPA_DIR) -> pd.DataFrame:
    """Write ``<patient>.parquet`` feature frames and ``summary.parquet``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for patient in patients:
        if patient in EXCLUDED:
            continue
        frame = feature_frame(patient)
        frame.to_parquet(out_dir / f"{patient}.parquet", index=False)
        rows.append(characterise(frame))
    summary = pd.DataFrame(rows)
    summary.to_parquet(out_dir / "summary.parquet", index=False)
    return summary


def load_frame(patient: str, out_dir: Path = HUPA_DIR) -> pd.DataFrame:
    path = out_dir / f"{patient}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m tfm.data.hupa` first")
    return pd.read_parquet(path)


def load_summary(out_dir: Path = HUPA_DIR) -> pd.DataFrame:
    path = out_dir / "summary.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m tfm.data.hupa` first")
    return pd.read_parquet(path)


def load_frames(out_dir: Path = HUPA_DIR) -> dict[str, pd.DataFrame]:
    """Every built patient frame, in ``PATIENTS`` order."""
    return {p: load_frame(p, out_dir) for p in PATIENTS if p not in EXCLUDED}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    summary = build()
    pd.set_option("display.width", 250)
    print(summary.to_string(index=False))
    print(f"\nPacientes: {len(summary)} (excluidos: {', '.join(EXCLUDED)})")
    print(f"Lecturas: {summary.n_readings.sum():,}  Eventos de hipoglucemia: {summary.hypo_events.sum()}")
    print(f"Escrito en {HUPA_DIR}")


if __name__ == "__main__":
    main()
