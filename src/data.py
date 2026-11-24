"""Load the UCI Default of Credit Card Clients dataset (public, no Kaggle auth).

Primary source: Yeh & Lien (2009), UCI ML Repository dataset 350.
We use this instead of Home Credit when Kaggle credentials are unavailable;
methodology (calibration + expected-loss thresholds) is identical.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

RENAME = {
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_1",
    "PAY_2": "pay_2",
    "PAY_3": "pay_3",
    "PAY_4": "pay_4",
    "PAY_5": "pay_5",
    "PAY_6": "pay_6",
    "BILL_AMT1": "bill_amt1",
    "BILL_AMT2": "bill_amt2",
    "BILL_AMT3": "bill_amt3",
    "BILL_AMT4": "bill_amt4",
    "BILL_AMT5": "bill_amt5",
    "BILL_AMT6": "bill_amt6",
    "PAY_AMT1": "pay_amt1",
    "PAY_AMT2": "pay_amt2",
    "PAY_AMT3": "pay_amt3",
    "PAY_AMT4": "pay_amt4",
    "PAY_AMT5": "pay_amt5",
    "PAY_AMT6": "pay_amt6",
    "default payment next month": "default",
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    for id_col in ("ID", "id", "Unnamed: 0"):
        if id_col in df.columns:
            df = df.drop(columns=[id_col])
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})
    if "default" not in df.columns:
        # last column is the label in the UCI workbook
        df = df.rename(columns={df.columns[-1]: "default"})
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["default"]).reset_index(drop=True)
    df["default"] = df["default"].astype(int)
    if "education" in df.columns:
        df.loc[~df["education"].isin([1, 2, 3, 4]), "education"] = 4
    if "marriage" in df.columns:
        df.loc[~df["marriage"].isin([1, 2, 3]), "marriage"] = 3
    return df


def load_credit_default(force: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / "uci_credit_default.csv"
    if cached.exists() and not force:
        return pd.read_csv(cached)

    zip_path = CACHE / "uci_default.zip"
    if force or not zip_path.exists():
        r = requests.get(UCI_ZIP_URL, timeout=120)
        r.raise_for_status()
        zip_path.write_bytes(r.content)

    with zipfile.ZipFile(zip_path) as zf:
        # Prefer xls inside the archive
        members = zf.namelist()
        xls_name = next((m for m in members if m.lower().endswith((".xls", ".xlsx"))), None)
        if xls_name is None:
            raise RuntimeError(f"No Excel file in UCI zip. Members={members}")
        raw_bytes = zf.read(xls_name)

    # header=1: UCI sheet row0 is X1..Y labels, row1 is real names
    df = pd.read_excel(io.BytesIO(raw_bytes), header=1)
    df = _normalize(df)
    df.to_csv(cached, index=False)
    (CACHE / "SOURCE.txt").write_text(
        "UCI Default of Credit Card Clients (Yeh & Lien, 2009).\n"
        f"Downloaded from: {UCI_ZIP_URL}\n"
        "License: public UCI ML Repository dataset.\n",
        encoding="utf-8",
    )
    return df
