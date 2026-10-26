from __future__ import annotations

import pandas as pd

from tfm.data import parser
from tfm.paths import INTERIM_DIR


class InterimMissingError(FileNotFoundError):
    """Raised when the interim layer has not been built yet."""


def _require(path):
    if not path.exists():
        raise InterimMissingError(
            f"{path} not found. Build the interim layer first:\n"
            "    python -m tfm.data.build"
        )
    return path


def load_channel(patient: str, split: str, channel: str) -> pd.DataFrame:
    path = INTERIM_DIR / f"{patient}_{split}_{channel}.parquet"
    if not path.exists():
        cols = parser.CHANNELS[channel]
        return pd.DataFrame(columns=[*cols.time_cols, *cols.numeric_cols, *cols.text_cols])
    return pd.read_parquet(path)


def load_cgm(patient: str, split: str, official: bool = True) -> pd.DataFrame:
    df = load_channel(patient, split, "glucose_level")
    if official and split == "test":
        df = df.iloc[parser.eval_skip_steps(patient) :].reset_index(drop=True)
    return df


def load_summary() -> pd.DataFrame:
    return pd.read_parquet(_require(INTERIM_DIR / "summary.parquet"))


def load_availability() -> pd.DataFrame:
    return pd.read_parquet(_require(INTERIM_DIR / "availability.parquet"))


def availability_matrix(split: str = "train") -> pd.DataFrame:
    av = load_availability()
    av = av[av["split"] == split]
    mat = av.pivot_table(
        index="channel", columns="patient", values="n_events", fill_value=0
    ).astype(int)
    order = [p for p in parser.PATIENTS_2018 + parser.PATIENTS_2020 if p in mat.columns]
    return mat[order]


def load_processed(patient: str, split: str, processed_dir=None) -> "WindowSet":
    from pathlib import Path

    import numpy as np

    from tfm.data.windows import WindowSet
    from tfm.paths import PROCESSED_DIR

    base = PROCESSED_DIR if processed_dir is None else Path(processed_dir)
    path = base / f"{patient}_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python -m tfm.data.build_processed` first"
        )
    d = np.load(path, allow_pickle=True)
    return WindowSet(
        X=d["X"], y=d["y"], ts=pd.to_datetime(d["ts"]).to_numpy(),
        patient=patient, split=split,
        feature_names=[str(f) for f in d["feature_names"]],
        horizons=tuple(int(h) for h in d["horizons"]),
    )
