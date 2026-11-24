"""Expert-style feature engineering for revolving credit default risk."""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_FEATURES = [
    "limit_bal",
    "sex",
    "education",
    "marriage",
    "age",
    "pay_1",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
    "pay_amt1",
    "pay_amt2",
    "pay_amt3",
    "pay_amt4",
    "pay_amt5",
    "pay_amt6",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add utilization, repayment ratios, and delinquency aggregates (no label leakage)."""
    out = df.copy()
    bill_cols = [f"bill_amt{i}" for i in range(1, 7)]
    pay_cols = [f"pay_amt{i}" for i in range(1, 7)]
    status_cols = [f"pay_{i}" for i in range(1, 7)]

    out["avg_bill"] = out[bill_cols].mean(axis=1)
    out["avg_pay"] = out[pay_cols].mean(axis=1)
    out["utilization"] = out["bill_amt1"] / out["limit_bal"].replace(0, np.nan)
    out["utilization"] = out["utilization"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 5)
    out["pay_to_bill"] = out["pay_amt1"] / out["bill_amt1"].abs().replace(0, np.nan)
    out["pay_to_bill"] = out["pay_to_bill"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 5)
    out["max_delinquency"] = out[status_cols].max(axis=1)
    out["n_late"] = (out[status_cols] > 0).sum(axis=1)
    out["bill_trend"] = out["bill_amt1"] - out["bill_amt6"]
    out["pay_trend"] = out["pay_amt1"] - out["pay_amt6"]
    out["credit_age_proxy"] = out["age"]  # demographic only; documented as proxy
    return out


def baseline_feature_list() -> list[str]:
    return [
        "limit_bal",
        "age",
        "education",
        "marriage",
        "pay_1",
        "bill_amt1",
        "pay_amt1",
        "utilization",
        "n_late",
    ]


def full_feature_list(df: pd.DataFrame) -> list[str]:
    exclude = {"default"}
    return [c for c in df.columns if c not in exclude]
