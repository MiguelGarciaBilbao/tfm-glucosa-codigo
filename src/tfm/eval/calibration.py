"""Is the predicted sigma trustworthy?
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf, erfinv

from tfm.eval.metrics import Predictions

#: Nominal coverage levels reported in the memoria.
COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)


def _z(level: float) -> float:
    return float(np.sqrt(2.0) * erfinv(level))


def interval_metrics(pred: Predictions, level: float = 0.95) -> dict[str, float]:
    if pred.sigma is None:
        raise RuntimeError("this prediction has no sigma; the model is deterministic")
    half = _z(level) * pred.sigma
    inside = np.abs(pred.y_true - pred.y_pred) <= half
    return {
        "level": float(level),
        "picp": float(inside.mean()),
        "mpiw": float((2 * half).mean()),
        "calibration_error": float(inside.mean() - level),
    }


def coverage_curve(pred: Predictions, levels=COVERAGE_LEVELS) -> list[dict[str, float]]:
    return [interval_metrics(pred, level) for level in levels]


def crps_gaussian(pred: Predictions) -> float:
    if pred.sigma is None:
        raise RuntimeError("this prediction has no sigma; the model is deterministic")
    z = (pred.y_true - pred.y_pred) / pred.sigma
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    return float(np.mean(pred.sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))))


def gaussian_nll(pred: Predictions) -> float:
    if pred.sigma is None:
        raise RuntimeError("this prediction has no sigma; the model is deterministic")
    z = (pred.y_true - pred.y_pred) / pred.sigma
    return float(np.mean(np.log(pred.sigma) + 0.5 * np.log(2 * np.pi) + 0.5 * z**2))


def calibration_report(pred: Predictions) -> dict:
    return {
        "patient": pred.patient,
        "horizon_min": pred.horizon_min,
        "crps": crps_gaussian(pred),
        "sigma_mean": float(pred.sigma.mean()),
        "sigma_min": float(pred.sigma.min()),
        "sigma_max": float(pred.sigma.max()),
        "sigma_iqr": float(np.subtract(*np.percentile(pred.sigma, [75, 25]))),
        "coverage": coverage_curve(pred),
    }


def cohort_calibration(predictions: list[Predictions]) -> dict:
    pooled = Predictions(
        np.concatenate([p.y_true for p in predictions]),
        np.concatenate([p.y_pred for p in predictions]),
        np.concatenate([p.ts for p in predictions]),
        "cohorte",
        predictions[0].horizon_min,
        sigma=np.concatenate([p.sigma for p in predictions]),
    )
    return calibration_report(pooled)
