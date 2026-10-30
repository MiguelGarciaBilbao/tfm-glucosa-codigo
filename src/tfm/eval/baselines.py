"""Non-learned baselines and the loader that feeds them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tfm.eval.metrics import Predictions
from tfm.paths import PROCESSED_DIR

#: Cohort order used across the project: 2018 first, then 2020.
PATIENTS = ("559", "563", "570", "575", "588", "591",
            "540", "544", "552", "567", "584", "596")


def load_windows(patient: str, split: str = "test") -> dict:
    """Load one patient's processed windows."""
    path = PROCESSED_DIR / f"{patient}_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python -m tfm.data.build_processed` first"
        )
    d = np.load(path, allow_pickle=True)
    return {
        "X": d["X"], "y": d["y"], "ts": pd.to_datetime(d["ts"]),
        "feature_names": list(d["feature_names"]),
        "horizons": [int(h) for h in d["horizons"]],
    }


def persistence(patient: str, horizon_index: int, split: str = "test") -> Predictions:
    """Predict the last observed CGM value for every horizon."""
    w = load_windows(patient, split)
    value_idx = w["feature_names"].index("value")
    from tfm.data.parser import SAMPLING_MINUTES

    return Predictions(
        y_true=w["y"][:, horizon_index].astype(float),
        y_pred=w["X"][:, -1, value_idx].astype(float),
        ts=w["ts"],
        patient=patient,
        horizon_min=w["horizons"][horizon_index] * SAMPLING_MINUTES,
    )


def persistence_cohort(horizon_index: int, split: str = "test") -> list[Predictions]:
    """Persistence predictions for the whole cohort at one horizon."""
    return [persistence(p, horizon_index, split) for p in PATIENTS]
