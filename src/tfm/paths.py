"""Project paths. Every module resolves files through here, never with relative paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "datasets" / "OhioT1DM"
#: External validation datasets: raw, read-only and outside git like OhioT1DM.
EXTERNAL_RAW_DIR = ROOT / "datasets" / "external"
HUPA_RAW_DIR = EXTERNAL_RAW_DIR / "HUPA-UCM"
DATA_DIR = ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
HUPA_DIR = DATA_DIR / "external" / "hupa"
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
FIGURES_DIR = ROOT / "memoria" / "figuras"


def ensure_dirs() -> None:
    for d in (INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
