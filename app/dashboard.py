from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # local shared/ for standalone + Streamlit Cloud

from shared.plot_theme import COLORS, apply_plotly_theme  # noqa: E402

st.set_page_config(page_title="Calibrated Credit Decisioning", layout="wide")
st.markdown(
    f"<style>.stApp {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}</style>",
    unsafe_allow_html=True,
)
st.title("Calibrated Credit Default Model")
st.caption("PD calibration · expected-loss thresholds · underwriter-style decisions")

out = ROOT / "outputs"
needed = ["meta.json", "metrics.csv", "threshold_sweep.csv", "reliability_compare.csv"]
if any(not (out / f).exists() for f in needed):
    st.warning("Outputs missing. Run `python 05_ds_calibrated_credit/scripts/run_mvp.py` first.")
    st.stop()

meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
metrics = pd.read_csv(out / "metrics.csv")
sweep = pd.read_csv(out / "threshold_sweep.csv")
rel = pd.read_csv(out / "reliability_compare.csv")
fi = pd.read_csv(out / "feature_importance.csv") if (out / "feature_importance.csv").exists() else pd.DataFrame()
sample = pd.read_csv(out / "sample_decisions.csv") if (out / "sample_decisions.csv").exists() else pd.DataFrame()
card = json.loads((out / "model_card.json").read_text(encoding="utf-8"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test default rate", f"{meta['default_rate']:.1%}")
c2.metric("LightGBM AUC", f"{meta['lgbm_auc']:.3f}")
c3.metric("Best calibration", meta["best_calibration"])
c4.metric("Chosen PD threshold", f"{meta['best_threshold']:.2f}")

tab1, tab2, tab3, tab4 = st.tabs(["Calibration", "Threshold economics", "Features", "Underwriter console"])

with tab1:
    st.subheader("Reliability curves (quantile bins)")
    fig = px.line(
        rel,
        x="mean_predicted",
        y="fraction_positive",
        color="model",
        markers=True,
        title="Predicted PD vs observed default rate",
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="perfect",
            line=dict(dash="dash", color=COLORS["muted"]),
        )
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(metrics, use_container_width=True)
    st.info(
        "Calibration follows Niculescu-Mizil & Caruana (2005): compare raw scores vs "
        "Platt scaling and isotonic regression using Brier score / ECE — not AUC alone."
    )

with tab2:
    st.subheader("Approval threshold vs realized portfolio profit (demo economics)")
    fig = px.line(
        sweep,
        x="threshold",
        y="realized_profit",
        markers=True,
        title="Realized profit on test approvals vs PD cutoff",
    )
    fig.add_vline(x=meta["best_threshold"], line_dash="dot", line_color=COLORS["accent"])
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.line(sweep, x="threshold", y="approval_rate", title="Approval rate vs threshold")
    apply_plotly_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)
    st.json(card["economics_assumptions"])

with tab3:
    if not fi.empty:
        fig = px.bar(fi.head(12), x="importance", y="feature", orientation="h", title="LightGBM feature importance")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Leakage controls:** " + card["features"]["leakage_controls"])

with tab4:
    st.subheader("Sample decisions (test slice)")
    if not sample.empty:
        st.dataframe(sample, use_container_width=True)
        fig = px.histogram(
            sample,
            x="calibrated_pd",
            color="decision",
            nbins=30,
            title="Calibrated PD by decision bucket",
            barmode="overlay",
            opacity=0.7,
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("APPROVE / REVIEW / DENY bands are demo policy around the profit-maximizing threshold.")
