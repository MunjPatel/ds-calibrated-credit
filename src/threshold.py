"""Expected-loss / expected-profit threshold selection for underwriting decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Economics:
    """Assumed unit economics — always state these; never claim bank P&L."""

    lgd: float = 0.45  # loss given default
    ead_fraction: float = 1.0  # exposure = limit_bal * ead_fraction (simplified)
    margin: float = 0.08  # expected net interest / fee margin on performing loans
    review_cost: float = 0.0


def expected_loss(pd: np.ndarray, exposure: np.ndarray, econ: Economics) -> np.ndarray:
    return pd * econ.lgd * exposure * econ.ead_fraction


def expected_profit(pd: np.ndarray, exposure: np.ndarray, econ: Economics) -> np.ndarray:
    """Approve everyone at their PD: E[profit] ≈ (1-PD)*margin*EAD - PD*LGD*EAD."""
    ead = exposure * econ.ead_fraction
    return (1.0 - pd) * econ.margin * ead - pd * econ.lgd * ead


def threshold_sweep(
    y_true: np.ndarray,
    pd_hat: np.ndarray,
    exposure: np.ndarray,
    econ: Economics | None = None,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """For each PD cutoff, approve if pd_hat < threshold; compute portfolio metrics."""
    econ = econ or Economics()
    thresholds = thresholds if thresholds is not None else np.linspace(0.02, 0.50, 25)
    rows = []
    for t in thresholds:
        approve = pd_hat < t
        n = max(int(approve.sum()), 1)
        # Realized loss proxy on approved book using actual labels
        approved_exp = exposure[approve]
        approved_y = y_true[approve]
        realized_loss = float((approved_y * econ.lgd * approved_exp * econ.ead_fraction).sum())
        realized_revenue = float(((1 - approved_y) * econ.margin * approved_exp * econ.ead_fraction).sum())
        realized_profit = realized_revenue - realized_loss
        rows.append(
            {
                "threshold": float(t),
                "approval_rate": float(approve.mean()),
                "n_approved": int(approve.sum()),
                "default_rate_approved": float(approved_y.mean()) if n else np.nan,
                "realized_profit": realized_profit,
                "realized_loss": realized_loss,
                "avg_pd_approved": float(pd_hat[approve].mean()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def best_threshold(sweep: pd.DataFrame) -> dict:
    idx = int(sweep["realized_profit"].idxmax())
    row = sweep.loc[idx]
    return {k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v) for k, v in row.items()}
