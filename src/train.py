"""Train baseline logistic and LightGBM default models."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelBundle:
    name: str
    model: object
    features: list[str]
    raw_probs: np.ndarray
    auc: float
    ap: float


def split_four_way(
    df: pd.DataFrame,
    target: str = "default",
    seed: int = 42,
    ratios=(0.55, 0.15, 0.15, 0.15),
):
    """Stratified split into train / valid / calibrate / test."""
    assert abs(sum(ratios) - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    y = df[target].to_numpy()
    parts = {"train": [], "valid": [], "calibrate": [], "test": []}
    for label in (0, 1):
        lab_idx = idx[y == label]
        rng.shuffle(lab_idx)
        n = len(lab_idx)
        n_tr = int(ratios[0] * n)
        n_va = int(ratios[1] * n)
        n_cal = int(ratios[2] * n)
        parts["train"].extend(lab_idx[:n_tr])
        parts["valid"].extend(lab_idx[n_tr : n_tr + n_va])
        parts["calibrate"].extend(lab_idx[n_tr + n_va : n_tr + n_va + n_cal])
        parts["test"].extend(lab_idx[n_tr + n_va + n_cal :])
    return {k: df.iloc[sorted(v)].reset_index(drop=True) for k, v in parts.items()}


def train_logistic(X_tr, y_tr, X_te, y_te, features: list[str]) -> ModelBundle:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    pipe.fit(X_tr[features], y_tr)
    probs = pipe.predict_proba(X_te[features])[:, 1]
    return ModelBundle(
        name="logistic_baseline",
        model=pipe,
        features=features,
        raw_probs=probs,
        auc=float(roc_auc_score(y_te, probs)),
        ap=float(average_precision_score(y_te, probs)),
    )


def train_lightgbm(X_tr, y_tr, X_va, y_va, X_te, y_te, features: list[str]) -> ModelBundle:
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_tr[features],
        y_tr,
        eval_set=[(X_va[features], y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    probs = model.predict_proba(X_te[features])[:, 1]
    return ModelBundle(
        name="lightgbm",
        model=model,
        features=features,
        raw_probs=probs,
        auc=float(roc_auc_score(y_te, probs)),
        ap=float(average_precision_score(y_te, probs)),
    )


def predict_proba(bundle: ModelBundle, X: pd.DataFrame) -> np.ndarray:
    return bundle.model.predict_proba(X[bundle.features])[:, 1]


def feature_importance_table(bundle: ModelBundle, top_n: int = 15) -> pd.DataFrame:
    if bundle.name == "logistic_baseline":
        coefs = bundle.model.named_steps["clf"].coef_[0]
        s = pd.Series(coefs, index=bundle.features).abs().sort_values(ascending=False).head(top_n)
        return s.rename("abs_coefficient").reset_index().rename(columns={"index": "feature"})
    imp = pd.Series(bundle.model.feature_importances_, index=bundle.features)
    s = imp.sort_values(ascending=False).head(top_n)
    return s.rename("importance").reset_index().rename(columns={"index": "feature"})
