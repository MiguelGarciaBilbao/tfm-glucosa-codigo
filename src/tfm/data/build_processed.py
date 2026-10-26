from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tfm.data import quality as q
from tfm.data.datasets import patient_windows
from tfm.data.parser import PATIENTS, cohort_of
from tfm.data.windows import DEFAULT_HISTORY, DEFAULT_HORIZONS
from tfm.paths import PROCESSED_DIR, ensure_dirs


def build(
    patients=PATIENTS,
    history: int = DEFAULT_HISTORY,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    wearable: bool = False,
    out_dir=None,
) -> pd.DataFrame:
    ensure_dirs()
    out_dir = PROCESSED_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for patient in patients:
        for split in ("train", "test"):
            ws = patient_windows(patient, split, history=history,
                                 horizons=horizons, wearable=wearable)
            np.savez_compressed(
                out_dir / f"{patient}_{split}.npz",
                X=ws.X, y=ws.y, ts=ws.ts.astype("datetime64[ns]").astype(np.int64),
                feature_names=np.array(ws.feature_names), horizons=np.array(ws.horizons),
            )
            hypo = (ws.y < q.HYPO).mean(axis=0) * 100 if len(ws) else np.zeros(len(horizons))
            rows.append({
                "patient": patient,
                "cohort": cohort_of(patient),
                "split": split,
                "n_windows": len(ws),
                "n_features": ws.X.shape[-1],
                **{f"pct_hypo_h{h * 5}": round(float(v), 2)
                   for h, v in zip(horizons, hypo)},
            })
    summary = pd.DataFrame(rows)
    summary.to_parquet(out_dir / "windows_summary.parquet", index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    ap.add_argument("--wearable", action="store_true")
    ap.add_argument("--out-dir", default=None,
                    help="write elsewhere than data/processed (use it for ablations)")
    args = ap.parse_args()
    summary = build(history=args.history, wearable=args.wearable, out_dir=args.out_dir)
    print(summary.to_string(index=False))
    print(f"\nVentanas totales: train={summary.query('split==\"train\"').n_windows.sum():,} "
          f"test={summary.query('split==\"test\"').n_windows.sum():,}")
    print(f"Escrito en {args.out_dir or PROCESSED_DIR}")


if __name__ == "__main__":
    main()
