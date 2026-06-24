"""Microvest — model viewer. Run: streamlit run streamlit_app/Home.py"""

import streamlit as st

import engine
import ui

st.set_page_config(page_title="Microvest Model Viewer", page_icon="📈", layout="wide")

cfg = ui.controls()
view = cfg["view"]

st.title("📈 MICROVEST // MODEL VIEWER")
ui.banner("LSTM RANKING SIGNAL · live console")
st.write("A window into what the LSTM signal actually does. Use the pages on the "
         "left: **Classification** for a single-ticker view + manual backtest, "
         "**Ranking** for the cross-sectional signal that actually matters.")

# Headline signal metrics over everything visible (the honest scorecard).
rep = engine.signal_report(view)
if rep:
    c = st.columns(5)
    c[0].metric("Rank IC", f"{rep.get('ic_spearman', float('nan')):+.3f}",
                help="Spearman corr of signal vs forward return. >0.02 is a real edge.")
    c[1].metric("ICIR (ann.)", f"{rep.get('icir', float('nan')):+.2f}")
    c[2].metric("Decile spread", f"{rep.get('decile_spread', float('nan')):+.4f}",
                help="Top-decile minus bottom-decile mean forward return.")
    c[3].metric("Long-short Sharpe", f"{rep.get('ls_sharpe', float('nan')):+.2f}")
    c[4].metric("Hit rate", f"{rep.get('hit_hit_rate', float('nan')):.1%}")

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Universe")
    st.write(", ".join(cfg["universe"]))
    st.subheader("Class legend")
    for cls, name in enumerate(engine.CLASS_NAMES):
        st.markdown(f"<span style='color:{engine.CLASS_COLOR[cls]};font-weight:600'>"
                    f"● {name}</span>", unsafe_allow_html=True)
with right:
    st.subheader("Predicted-direction mix (visible window)")
    counts = view["pred_class"].value_counts().sort_index()
    labels = []
    for i in counts.index:
        labels.append(engine.CLASS_NAMES[i])
    counts.index = labels
    st.bar_chart(counts)

st.divider()
ui.caveats()
