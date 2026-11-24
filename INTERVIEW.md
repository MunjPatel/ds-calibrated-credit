# Interview narrative — Calibrated Credit Decisioning

**30-second version**  
“I built a revolving-credit default model where the deliverable is a **decision**, not an AUC. LightGBM ranks risk; Platt/isotonic calibration makes probabilities honest; then I pick the approve threshold by expected profit under explicit LGD and margin assumptions — the same FP/FN economics Stripe describes for Radar.”

## Structure (2–3 minutes)

1. **Business decision:** Approve vs deny changes dollars. AUC alone doesn’t.
2. **Replication source:** Niculescu-Mizil & Caruana (2005) on calibration; LightGBM (Ke et al. 2017) for tabular PD.
3. **What I built:** Four-way split (train/valid/calibrate/test), baseline logistic → LightGBM, reliability diagrams, threshold sweep.
4. **What broke / surprised me:** Raw tree scores can look sharp on ROC but be poorly calibrated; isotonic can overfit small calibrate sets — I compare Brier/ECE before picking.
5. **Limitations:** Assumed LGD/margin; UCI ≠ bank book; no reject inference.
6. **Next:** Time-based vintages + PSI monitoring (Project 2), cost matrix by segment.

## Likely follow-ups

| Question | Answer direction |
|----------|------------------|
| Why not just maximize F1? | F1 ignores dollar asymmetry of FN vs FP. |
| Why hold out a calibrate set? | Fitting isotonic on test leaks; Niculescu-Mizil warns about data needs. |
| How would you monitor in prod? | Log PD, join outcomes, watch Brier + PSI (see monitoring project). |
| Fairness? | I report slice gaps; I don’t “fix” them without policy context. |
