"""Parse OhioT1DM XML files into tidy pandas DataFrames.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tfm.paths import RAW_DIR

TS_FORMAT = "%d-%m-%Y %H:%M:%S"

PATIENTS_2018 = ("559", "563", "570", "575", "588", "591")
PATIENTS_2020 = ("540", "544", "552", "567", "584", "596")
PATIENTS = PATIENTS_2018 + PATIENTS_2020

SAMPLING_MINUTES = 5

EVAL_SKIP_STEPS = {"2018": 0, "2020": 12}


@dataclass(frozen=True)
class Channel:
    name: str
    time_cols: tuple[str, ...]
    numeric_cols: tuple[str, ...] = ()
    text_cols: tuple[str, ...] = ()
    cohort: str | None = None


CHANNELS: dict[str, Channel] = {
    c.name: c
    for c in (
        Channel("glucose_level", ("ts",), ("value",)),
        Channel("finger_stick", ("ts",), ("value",)),
        Channel("basal", ("ts",), ("value",)),
        Channel("temp_basal", ("ts_begin", "ts_end"), ("value",)),
        Channel("bolus", ("ts_begin", "ts_end"), ("dose", "bwz_carb_input"), ("type",)),
        Channel("meal", ("ts",), ("carbs",), ("type",)),
        Channel("exercise", ("ts",), ("duration", "intensity", "competitive"), ("type",)),
        Channel("sleep", ("ts_begin", "ts_end"), ("quality",)),
        Channel("work", ("ts_begin", "ts_end"), ("intensity",)),
        Channel("stressors", ("ts",), (), ("type", "description")),
        Channel("illness", ("ts_begin", "ts_end"), (), ("type", "description")),
        Channel("hypo_event", ("ts",)),
        Channel("basis_heart_rate", ("ts",), ("value",), cohort="2018"),
        Channel("basis_steps", ("ts",), ("value",), cohort="2018"),
        Channel("basis_air_temperature", ("ts",), ("value",), cohort="2018"),
        Channel("basis_gsr", ("ts",), ("value",)),
        Channel("basis_skin_temperature", ("ts",), ("value",)),
        Channel("basis_sleep", ("tbegin", "tend"), ("quality",), ("type",)),
        Channel("acceleration", ("ts",), ("value",), cohort="2020"),
    )
}


def eval_skip_steps(patient: str) -> int:
    return EVAL_SKIP_STEPS[cohort_of(patient)]


def cohort_of(patient: str) -> str:
    patient = str(patient)
    if patient in PATIENTS_2018:
        return "2018"
    if patient in PATIENTS_2020:
        return "2020"
    raise ValueError(f"Unknown patient id: {patient!r}. Known ids: {PATIENTS}")


def patient_xml_path(patient: str, split: str, raw_dir: Path = RAW_DIR) -> Path:
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    suffix = "training" if split == "train" else "testing"
    path = raw_dir / cohort_of(patient) / split / f"{patient}-ws-{suffix}.xml"
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    return path


def _events_to_frame(channel: Channel, events: list[ET.Element]) -> pd.DataFrame:
    rows = [ev.attrib for ev in events]
    df = pd.DataFrame(rows)
    for col in channel.time_cols:
        series = df[col] if col in df.columns else pd.Series(dtype="object", index=df.index)
        df[col] = pd.to_datetime(series, format=TS_FORMAT, errors="coerce")
    for col in channel.numeric_cols:
        series = df[col] if col in df.columns else pd.Series(dtype="object", index=df.index)
        df[col] = pd.to_numeric(series.replace(r"^\s*$", None, regex=True), errors="coerce")
    for col in channel.text_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    ordered = [*channel.time_cols, *channel.numeric_cols, *channel.text_cols]
    df = df.reindex(columns=[c for c in ordered if c in df.columns or c in ordered])
    return df.sort_values(channel.time_cols[0], kind="stable").reset_index(drop=True)


def read_patient_xml(path: Path) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
    root = ET.parse(path).getroot()
    channels: dict[str, pd.DataFrame] = {}
    for node in root:
        channel = CHANNELS.get(node.tag)
        if channel is None:
            raise KeyError(f"Unknown channel {node.tag!r} in {path.name}; update CHANNELS")
        events = list(node.findall("event"))
        if events:
            channels[node.tag] = _events_to_frame(channel, events)
        else:
            cols = [*channel.time_cols, *channel.numeric_cols, *channel.text_cols]
            channels[node.tag] = pd.DataFrame(columns=cols)
    return dict(root.attrib), channels


def load_patient_channels(
    patient: str, split: str, raw_dir: Path = RAW_DIR
) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
    return read_patient_xml(patient_xml_path(patient, split, raw_dir))


def channel_availability(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    records = []
    for patient in PATIENTS:
        for split in ("train", "test"):
            _, channels = load_patient_channels(patient, split, raw_dir)
            for name, df in channels.items():
                records.append(
                    {
                        "patient": patient,
                        "cohort": cohort_of(patient),
                        "split": split,
                        "channel": name,
                        "n_events": len(df),
                    }
                )
    return pd.DataFrame.from_records(records)


def official_test_frame(patient: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    _, channels = load_patient_channels(patient, "test", raw_dir)
    return channels["glucose_level"].iloc[eval_skip_steps(patient) :].reset_index(drop=True)
