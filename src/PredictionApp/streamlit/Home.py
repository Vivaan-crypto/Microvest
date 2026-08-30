"""Microvest — model viewer. Run: streamlit run streamlit_app/Home.py"""

import streamlit as st

import engine
import ui

st.set_page_config(page_title="Microvest Model Viewer", page_icon="📈", layout="wide")

cfg = ui.controls()
view = cfg["view"]

ui.page_header("MICROVEST // MODEL VIEWER", "LSTM RANKING SIGNAL · live console")
st.write("A window into what the LSTM signal actually does. Use the pages on the "
         "left: **Classification** for a single-ticker view + manual backtest, "
         "**Ranking** for the cross-sectional signal that actually matters.")

# Headline signal metrics — OUT-OF-SAMPLE ONLY (after the train cutoff). Including
# the training period would massively inflate these: the model memorized it, so its
# in-sample IC/Sharpe look world-class but mean nothing. This is the honest scorecard.
oos = view[view["date"] > engine.TRAIN_END]
rep = engine.signal_report(oos)
if rep:
    st.caption(f"📏 Out-of-sample only — predictions after the {engine.TRAIN_END.date()} "
               f"train cutoff ({len(oos):,} of {len(view):,} shown). In-sample rows are "
               f"excluded because the model memorized them, which fakes a huge edge.")
    c = st.columns(5)
    c[0].metric("Rank IC", f"{rep.get('ic_spearman', float('nan')):+.3f}", help=ui.HELP["ic"])
    c[1].metric("ICIR (ann.)", f"{rep.get('icir', float('nan')):+.2f}", help=ui.HELP["icir"])
    c[2].metric("Decile spread", f"{rep.get('decile_spread', float('nan')):+.4f}",
                help=ui.HELP["decile"])
    c[3].metric("Long-short Sharpe", f"{rep.get('ls_sharpe', float('nan')):+.2f}",
                help=ui.HELP["ls_sharpe"])
    c[4].metric("Hit rate", f"{rep.get('hit_hit_rate', float('nan')):.1%}", help=ui.HELP["hit_rate"])
else:
    st.info(f"No out-of-sample data in view yet (need predictions after "
            f"{engine.TRAIN_END.date()}). Move the as-of date later.")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Predicted-direction mix")
    st.caption("How often the model says Short / NoTrade / Long in the visible window.")
    counts = view["pred_class"].value_counts().sort_index()
    labels = []
    for i in counts.index:
        labels.append(engine.CLASS_NAMES[i])
    counts.index = labels
    st.bar_chart(counts, height=260, color=ui.CYAN)
with right:
    st.subheader("Rolling OOS rank IC")
    st.caption("21-day mean of the daily cross-sectional IC — is the edge stable?")
    ic = engine.rolling_ic(oos)
    if len(ic):
        roll = ic.rolling(21, min_periods=5).mean().rename("21-day mean IC")
        st.area_chart(roll, height=260, color=ui.CYAN)
    else:
        st.info("Not enough out-of-sample days to chart yet.")

st.divider()

st.subheader("Universe")
ui.chips(cfg["universe"])

legend_colors = {}
legend_items = []
for cls, name in enumerate(engine.CLASS_NAMES):
    label = f"● {name}"
    legend_items.append(label)
    legend_colors[label] = engine.CLASS_COLOR[cls]
ui.chips(legend_items, colors=legend_colors)

st.divider()
ui.caveats()
