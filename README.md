# Calibrated Credit Default Model with Expected-Loss Thresholding

**Role focus:** Data Scientist  
**Decision:** Approve / review / deny revolving credit using **calibrated PD**, not raw classifier scores.

## Problem

I built this around a practical credit decision loop: calibrated probabilities that mean something in dollars, and approval thresholds that trade off false approvals against false declines — not AUC alone.

## What I built

1. UCI Credit Card Default dataset load + expert features (utilization, delinquency counts, pay-to-bill)
2. **Baseline:** balanced logistic regression  
3. **Improved:** LightGBM with early stopping  
4. **Calibration:** Platt scaling + isotonic regression on a held-out calibrate split (Niculescu-Mizil & Caruana 2005)
5. **Thresholding:** expected-loss / expected-profit sweep under explicit LGD & margin assumptions
6. Streamlit underwriter console + model card

## Sources (verified)

- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning.* ICML.
- Ke, G. et al. (2017). *LightGBM.* NeurIPS.
- Stripe Engineering. *How we built it: Stripe Radar* (FP/FN cost framing).
- Yeh, I-C. & Lien, C-H. (2009). UCI Default of Credit Card Clients dataset.

> Home Credit (Kaggle) is the blueprint’s preferred corpus; this MVP uses UCI so the portfolio runs without Kaggle credentials. The methodology is unchanged.

## Reproduce

```bash
cd portfolio/projects
.venv\Scripts\activate          # Windows
pip install -r requirements.txt # includes lightgbm
python 05_ds_calibrated_credit/scripts/run_mvp.py
streamlit run 05_ds_calibrated_credit/app/dashboard.py
```

## Key outputs

| File | Meaning |
|------|---------|
| `outputs/metrics.csv` | AUC / PR-AUC / Brier / ECE by model |
| `outputs/reliability_compare.csv` | Calibration curves |
| `outputs/threshold_sweep.csv` | Profit vs PD cutoff |
| `outputs/model_card.json` | Intended use, assumptions, limitations |
| `DECISION_MEMO.md` | 1-page stakeholder memo |

## Limitations

- Demo LGD/margin are **assumptions**, not bank economics.
- UCI features are classic revolving-credit attributes, not a modern bureau stack.
- Not a production underwriting system; no reject inference, no fair-lending legal review.



1. Push this repo to GitHub (already done if you cloned from MunjPatel/ds-calibrated-credit).
2. Go to https://share.streamlit.io/ → **New app**
3. Select repo `MunjPatel/ds-calibrated-credit`, branch `main`
4. Main file path: `app/dashboard.py`
5. Python version 3.11+ recommended → Deploy

The app reads artifacts under `outputs/`. Re-run `scripts/run_mvp.py` locally if you need to refresh them before pushing.

## Live dashboard

[https://ds-calibrated-credit.streamlit.app/](https://ds-calibrated-credit.streamlit.app/)

