# Project notes — Calibrated Credit Decisioning

I trained a LightGBM default model on public revolving-credit data, calibrated probabilities (Platt / isotonic), and selected approval thresholds from expected-loss / profit economics rather than AUC alone.

**What I found:** calibration cut ECE sharply versus raw scores, and the threshold sweep makes the approval policy an economic choice, not an ROC-max rule.
