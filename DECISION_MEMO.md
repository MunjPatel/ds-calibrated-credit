# Decision Memo — Calibrated PD Underwriting Demo

**To:** Hiring manager / risk partner (portfolio artifact)  
**From:** Munj Patel  
**Re:** Ship a calibrated default model with an explicit approval threshold  

## Recommendation

Use the **calibrated LightGBM PD** (selected by lowest Brier among raw / Platt / isotonic) and set the initial approval cutoff at the **profit-maximizing threshold** on the held-out test sweep under stated unit economics (LGD = 45%, performing-loan margin = 8%). Route a band above the cutoff to human **REVIEW** rather than hard deny.

## Why this over “max AUC”

- Raw ranking quality (AUC) does not guarantee usable probabilities.
- Underwriting needs `EL ≈ PD × LGD × EAD` and a threshold policy.
- Calibration is measured with reliability diagrams, Brier score, and ECE (Niculescu-Mizil & Caruana, 2005).

## Evidence (see `outputs/`)

- Baseline logistic vs LightGBM comparison in `metrics.csv`
- Reliability curves in `reliability_compare.csv`
- Threshold vs realized profit in `threshold_sweep.csv`
- Assumptions locked in `model_card.json`

## Risks / do-not-use

- Economics are **illustrative assumptions**, not a priced loan book.
- Public UCI data ≠ institution’s applicant population (population shift risk).
- No reject-inference; approved-book metrics are conditional on historical issuance.

## Next step

Wire prediction logging + PSI drift monitors (companion monitoring project) before any live pilot narrative.
