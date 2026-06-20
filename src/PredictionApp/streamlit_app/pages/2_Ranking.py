"""Cross-sectional ranking view — the signal that actually matters."""

import pandas as pd
import plotly.express as px
import streamlit as st

import engine
import ui

st.set_page_config(page_title="Ranking", page_icon="🏆", layout="wide")
cfg = ui.controls()
view = cfg["view"]

st.title("🏆 RANKING")
ui.banner("CROSS-SECTIONAL SIGNAL · the score that actually matters")

# ---- Pick which stocks to rank ----
all_tickers = sorted(view["ticker"].unique())
picked = st.multiselect("Stocks to rank", all_tickers, default=all_tickers,
                        help="Subset of the loaded universe to rank against each other.")
if not picked:
    st.warning("Pick at least one stock to rank.")
    st.stop()
view = view[view["ticker"].isin(picked)]

# ---- As-of leaderboard ----
last_date = view["date"].max()
today = view[view["date"] == last_date].sort_values("signal", ascending=False)
half = max(1, len(today) // 2)
n = st.slider("Names per side", 1, half, min(5, half)) if half > 1 else 1

st.subheader(f"Leaderboard — {last_date:%Y-%m-%d}")
lc, rc = st.columns(2)
cols = ["ticker", "signal", "p_long", "p_short", "close"]
with lc:
    st.caption("🟢 Top — go long")
    st.dataframe(today.head(n)[cols].style.format(
        {"signal": "{:+.2f}", "p_long": "{:.0%}", "p_short": "{:.0%}", "close": "${:.2f}"})
        .background_gradient(cmap="Greens", subset=["signal"]),
        width="stretch", hide_index=True)
with rc:
    st.caption("🔴 Bottom — go short")
    st.dataframe(today.tail(n).iloc[::-1][cols].style.format(
        {"signal": "{:+.2f}", "p_long": "{:.0%}", "p_short": "{:.0%}", "close": "${:.2f}"})
        .background_gradient(cmap="Reds_r", subset=["signal"]),
        width="stretch", hide_index=True)

st.divider()

# ---- Signal quality over the visible window ----
rep = engine.signal_report(view)
if not rep:
    st.info("Not enough realized data in this window to score the signal.")
    ui.caveats()
    st.stop()

st.subheader("Signal quality")
m = st.columns(5)
m[0].metric("Rank IC", f"{rep.get('ic_spearman', float('nan')):+.3f}")
m[1].metric("ICIR (ann.)", f"{rep.get('icir', float('nan')):+.2f}")
m[2].metric("IC>0 days", f"{rep.get('ic_hit_rate', float('nan')):.0%}")
m[3].metric("Decile spread", f"{rep.get('decile_spread', float('nan')):+.4f}")
m[4].metric("Long-short Sharpe", f"{rep.get('ls_sharpe', float('nan')):+.2f}")

c1, c2 = st.columns(2)
with c1:
    st.caption("Mean forward return by signal decile (monotone up = good)")
    dec = engine.decile_table(view)
    if not dec.empty:
        fig = px.bar(x=dec.index.astype(int), y=(dec.values * 100),
                     labels={"x": "signal decile (0=bearish, 9=bullish)", "y": "fwd ret %"})
        fig.update_traces(marker_color="#4C72B0")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
with c2:
    q = st.select_slider("Long-short book quantile", [0.1, 0.2, 0.3], value=0.2)
    st.caption(f"Cumulative top-{int(q*100)}% minus bottom-{int(q*100)}% (gross)")
    curve = engine.long_short_curve(view, quantile=q)
    if len(curve):
        st.area_chart(curve.rename("cum. spread"))

st.caption("Daily cross-sectional rank IC (stability of the edge over time)")
ic = engine.rolling_ic(view)
if len(ic):
    roll = ic.rolling(21, min_periods=5).mean().rename("21-day mean IC")
    st.line_chart(pd.concat([ic.rename("daily IC"), roll], axis=1))

ui.caveats()
