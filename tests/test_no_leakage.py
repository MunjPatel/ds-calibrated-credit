"""Basic hygiene tests for the calibrated credit pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import engineer  # noqa: E402
from train import split_four_way  # noqa: E402


def test_split_disjoint():
    df = pd.DataFrame(
        {
            "row_id": np.arange(400),
            "limit_bal": np.random.randint(10000, 200000, 400),
            "age": np.random.randint(21, 70, 400),
            "default": np.r_[np.zeros(320, dtype=int), np.ones(80, dtype=int)],
        }
    )
    parts = split_four_way(df)
    ids = {k: set(v["row_id"].tolist()) for k, v in parts.items()}
    keys = list(ids)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert ids[keys[i]].isdisjoint(ids[keys[j]])
    assert sum(len(v) for v in ids.values()) == 400


def test_engineer_no_target_in_features_math():
    df = pd.DataFrame(
        {
            "limit_bal": [10000, 20000],
            "sex": [1, 2],
            "education": [2, 1],
            "marriage": [1, 2],
            "age": [30, 40],
            "pay_1": [0, 2],
            "pay_2": [0, 1],
            "pay_3": [0, 0],
            "pay_4": [0, 0],
            "pay_5": [0, 0],
            "pay_6": [0, 0],
            "bill_amt1": [5000, 15000],
            "bill_amt2": [4000, 14000],
            "bill_amt3": [3000, 13000],
            "bill_amt4": [2000, 12000],
            "bill_amt5": [1000, 11000],
            "bill_amt6": [500, 10000],
            "pay_amt1": [1000, 500],
            "pay_amt2": [1000, 500],
            "pay_amt3": [1000, 500],
            "pay_amt4": [1000, 500],
            "pay_amt5": [1000, 500],
            "pay_amt6": [1000, 500],
            "default": [0, 1],
        }
    )
    out = engineer(df)
    assert "utilization" in out.columns
    # Engineering must not create features from the label
    assert out["utilization"].iloc[0] == 5000 / 10000
