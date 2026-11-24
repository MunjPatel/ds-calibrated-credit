"""Probability calibration (Platt scaling + isotonic regression).

Grounded in Niculescu-Mizil & Caruana (2005), ICML.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


@dataclass
class CalibrationResult:
    name: str
    probs: np.ndarray
    brier: float
    ece: float
    fraction_of_positives: np.ndarray
    mean_predicted_value: np.ndarray


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if not np.any(mask):
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def fit_platt(y_cal: np.ndarray, p_cal: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(max_iter=2000)
    lr.fit(p_cal.reshape(-1, 1), y_cal)
    return lr


def fit_isotonic(y_cal: np.ndarray, p_cal: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    return iso


def apply_platt(model: LogisticRegression, p: np.ndarray) -> np.ndarray:
    return model.predict_proba(p.reshape(-1, 1))[:, 1]


def apply_isotonic(model: IsotonicRegression, p: np.ndarray) -> np.ndarray:
    return model.predict(p)


def evaluate_probs(name: str, y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> CalibrationResult:
    frac, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="quantile")
    return CalibrationResult(
        name=name,
        probs=probs,
        brier=float(brier_score_loss(y_true, probs)),
        ece=expected_calibration_error(y_true, probs, n_bins=n_bins),
        fraction_of_positives=frac,
        mean_predicted_value=mean_pred,
    )


def sklearn_calibrated_clone(estimator, X_cal, y_cal, method: str = "sigmoid"):
    """Optional sklearn CalibratedClassifierCV helper (prefit)."""
    cal = CalibratedClassifierCV(estimator, method=method, cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal
