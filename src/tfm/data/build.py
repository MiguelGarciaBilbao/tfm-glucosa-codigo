"""Dump the raw XML files to parquet, one file per patient and split.
"""

from __future__ import annotations

import argparse

import pandas as pd

from tfm.data import parser
from tfm.paths import INTERIM_DIR, ensure_dirs


def build_interim(patients: tuple[str, ...] = parser.PATIENTS) -> pd.DataFrame:
    ensure_dirs()
    records = []
    for patient in patients:
        for split in ("train", "test"):
            attrs, channels = parser.load_patient_channels(patient, split)
            for name, df in channels.items():
                if df.empty:
                    continue
                out = INTERIM_DIR / f"{patient}_{split}_{name}.parquet"
                df.to_parquet(out, index=False)
            cgm = channels["glucose_level"]
            records.append(
                {
                    "patient": patient,
                    "cohort": parser.cohort_of(patient),
                    "insulin_type": attrs.get("insulin_type"),
                    "split": split,
                    "n_cgm": len(cgm),
                    "first_ts": cgm["ts"].min(),
                    "last_ts": cgm["ts"].max(),
                    "days": (cgm["ts"].max() - cgm["ts"].min()).total_seconds() / 86400,
                    "pct_hypo": 100 * (cgm["value"] < 70).mean(),
                    "pct_severe_hypo": 100 * (cgm["value"] < 54).mean(),
                    "pct_in_range": 100 * cgm["value"].between(70, 180).mean(),
                    "pct_hyper": 100 * (cgm["value"] > 180).mean(),
                    "mean_cgm": cgm["value"].mean(),
                    "std_cgm": cgm["value"].std(),
                    "nonempty_channels": sum(not d.empty for d in channels.values()),
                }
            )
    summary = pd.DataFrame.from_records(records)
    summary.to_parquet(INTERIM_DIR / "summary.parquet", index=False)
    parser.channel_availability().to_parquet(INTERIM_DIR / "availability.parquet", index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patients", nargs="*", default=list(parser.PATIENTS))
    args = ap.parse_args()
    summary = build_interim(tuple(args.patients))
    cols = ["patient", "cohort", "split", "n_cgm", "days", "pct_hypo", "pct_in_range"]
    print(summary[cols].to_string(index=False, float_format="%.1f"))
    print(f"\nWritten to {INTERIM_DIR}")


if __name__ == "__main__":
    main()
