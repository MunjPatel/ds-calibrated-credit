from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "src"))

from calibrate import (  # noqa: E402
    apply_isotonic,
    apply_platt,
    evaluate_probs,
    fit_isotonic,
    fit_platt,
)
from data import load_credit_default  # noqa: E402
from features import baseline_feature_list, engineer, full_feature_list  # noqa: E402
from threshold import Economics, best_threshold, threshold_sweep  # noqa: E402
from train import (  # noqa: E402
    feature_importance_table,
    predict_proba,
    split_four_way,
    train_lightgbm,
    train_logistic,
)


def main():
    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    raw = load_credit_default()
    df = engineer(raw)
    parts = split_four_way(df)

    base_feats = [c for c in baseline_feature_list() if c in df.columns]
    full_feats = [c for c in full_feature_list(df) if c != "default"]

    y_test = parts["test"]["default"].to_numpy()
    exposure = parts["test"]["limit_bal"].to_numpy().astype(float)

    # --- Baseline logistic on curated features ---
    logit = train_logistic(
        parts["train"],
        parts["train"]["default"],
        parts["test"],
        parts["test"]["default"],
        base_feats,
    )

    # --- LightGBM on full engineered set ---
    lgbm = train_lightgbm(
        parts["train"],
        parts["train"]["default"],
        parts["valid"],
        parts["valid"]["default"],
        parts["test"],
        parts["test"]["default"],
        full_feats,
    )

    # Calibration on held-out calibrate split using LightGBM raw scores
    p_cal_raw = predict_proba(lgbm, parts["calibrate"])
    y_cal = parts["calibrate"]["default"].to_numpy()
    platt = fit_platt(y_cal, p_cal_raw)
    iso = fit_isotonic(y_cal, p_cal_raw)

    p_raw = lgbm.raw_probs
    p_platt = apply_platt(platt, p_raw)
    p_iso = apply_isotonic(iso, p_raw)

    cal_rows = []
    for name, probs in (
        ("logistic_raw", logit.raw_probs),
        ("lgbm_raw", p_raw),
        ("lgbm_platt", p_platt),
        ("lgbm_isotonic", p_iso),
    ):
        # For logistic, evaluate on same test labels
        y_eval = y_test
        res = evaluate_probs(name, y_eval, probs)
        cal_rows.append(
            {
                "model": name,
                "brier": res.brier,
                "ece": res.ece,
                "auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y_eval, probs)),
                "pr_auc": float(
                    __import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(
                        y_eval, probs
                    )
                ),
            }
        )
        pd.DataFrame(
            {
                "mean_predicted": res.mean_predicted_value,
                "fraction_positive": res.fraction_of_positives,
            }
        ).to_csv(out / f"reliability_{name}.csv", index=False)

    metrics = pd.DataFrame(cal_rows).sort_values("brier")
    metrics.to_csv(out / "metrics.csv", index=False)

    # Choose calibrated probs by lowest Brier among lgbm variants
    lgbm_metrics = metrics[metrics["model"].str.startswith("lgbm")]
    best_cal_name = lgbm_metrics.iloc[0]["model"]
    best_pd = {"lgbm_raw": p_raw, "lgbm_platt": p_platt, "lgbm_isotonic": p_iso}[best_cal_name]

    econ = Economics(lgd=0.45, ead_fraction=1.0, margin=0.08)
    sweep = threshold_sweep(y_test, best_pd, exposure, econ=econ)
    sweep.to_csv(out / "threshold_sweep.csv", index=False)
    chosen = best_threshold(sweep)

    fi = feature_importance_table(lgbm)
    fi.to_csv(out / "feature_importance.csv", index=False)

    # Scorecard-style sample decisions for dashboard
    sample = parts["test"].head(200).copy()
    sample_pd = predict_proba(lgbm, sample)
    if best_cal_name == "lgbm_platt":
        sample_pd = apply_platt(platt, sample_pd)
    elif best_cal_name == "lgbm_isotonic":
        sample_pd = apply_isotonic(iso, sample_pd)
    sample["calibrated_pd"] = sample_pd
    sample["expected_loss"] = sample["calibrated_pd"] * econ.lgd * sample["limit_bal"]
    sample["decision"] = np.where(
        sample["calibrated_pd"] < chosen["threshold"],
        "APPROVE",
        np.where(sample["calibrated_pd"] < chosen["threshold"] * 1.4, "REVIEW", "DENY"),
    )
    sample[
        ["limit_bal", "age", "utilization", "n_late", "pay_1", "calibrated_pd", "expected_loss", "decision", "default"]
    ].to_csv(out / "sample_decisions.csv", index=False)

    model_card = {
        "name": "Calibrated revolving-credit default model",
        "intended_use": "Portfolio demonstration of PD calibration + expected-loss thresholding; not for production underwriting.",
        "data": "UCI Default of Credit Card Clients (Yeh & Lien, 2009), n=30000",
        "default_rate": float(df["default"].mean()),
        "features": {
            "baseline_count": len(base_feats),
            "full_count": len(full_feats),
            "leakage_controls": "No post-default outcomes; splits before calibration; calibrate set held out from training.",
        },
        "metrics_test": metrics.to_dict(orient="records"),
        "selected_calibration": best_cal_name,
        "economics_assumptions": {
            "lgd": econ.lgd,
            "margin": econ.margin,
            "ead_fraction": econ.ead_fraction,
            "note": "Assumptions for demonstration only — state these in interviews.",
        },
        "selected_threshold": chosen,
        "sources": [
            "Niculescu-Mizil & Caruana (2005), Predicting Good Probabilities with Supervised Learning, ICML",
            "Ke et al. (2017), LightGBM, NeurIPS",
            "Stripe Radar engineering blog — FP/FN cost framing",
            "Yeh & Lien (2009), UCI credit card default dataset",
        ],
    }
    (out / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")

    meta = {
        "n_rows": int(len(df)),
        "default_rate": float(df["default"].mean()),
        "logistic_auc": logit.auc,
        "lgbm_auc": lgbm.auc,
        "best_calibration": best_cal_name,
        "best_brier": float(lgbm_metrics.iloc[0]["brier"]),
        "best_threshold": chosen["threshold"],
        "approval_rate_at_best": chosen["approval_rate"],
        "realized_profit_at_best": chosen["realized_profit"],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Persist calibration curves comparison for the three lgbm variants
    rel_compare = []
    for name, probs in (("lgbm_raw", p_raw), ("lgbm_platt", p_platt), ("lgbm_isotonic", p_iso)):
        res = evaluate_probs(name, y_test, probs)
        for mp, fp in zip(res.mean_predicted_value, res.fraction_of_positives):
            rel_compare.append({"model": name, "mean_predicted": float(mp), "fraction_positive": float(fp)})
    pd.DataFrame(rel_compare).to_csv(out / "reliability_compare.csv", index=False)

    print("P5 calibrated credit MVP complete ->", out)
    print(pd.Series(meta))


if __name__ == "__main__":
    main()
