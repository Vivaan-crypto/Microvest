"""Chart + prediction — search ANY US ticker, see a candlestick, and (when the
ticker has enough history) the model's directionality on the latest bar and in the
hover. The chart always renders; prediction layers on top when available."""

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import engine
import ui

st.set_page_config(page_title="Chart", page_icon="📈", layout="wide")
ckpt, mtime, info = ui.pick_checkpoint()
st.title("📈 CHART + PREDICTION")
ui.banner("ANY US TICKER · candlestick + model directionality")

SPAN = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260, "max": 10 ** 6}

c1, c2 = st.columns([1, 2])
ticker = c1.text_input("🔎 Ticker", value="AAPL").strip().upper()
period = c2.radio("Range", list(SPAN), index=3, horizontal=True)
if not ticker:
    st.stop()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_raw(ticker, period):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True,
                     progress=False, multi_level_index=False)
    return None if df is None or df.empty else df.reset_index().dropna(subset=["Close"])


def candlestick(d, x, o, h, low, c, v, customdata=None, hover=None, pred_class=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=d[x], open=d[o], high=d[h], low=d[low], close=d[c],
        increasing_line_color="#00e5ff", decreasing_line_color="#ff3b6b",
        name="OHLC", hoverinfo=("skip" if hover else None)), row=1, col=1)
    if hover is not None:
        fig.add_trace(go.Scatter(x=d[x], y=d[h], mode="markers", showlegend=False,
                                 marker=dict(size=0.1, color="rgba(0,0,0,0)"),
                                 customdata=customdata, hovertemplate=hover), row=1, col=1)
    if pred_class is not None:
        fig.add_trace(go.Scatter(x=d[x], y=d[low] * 0.985, mode="markers",
                                 showlegend=False, hoverinfo="skip",
                                 marker=dict(size=5, color=[engine.CLASS_COLOR[k] for k in pred_class])),
                      row=1, col=1)
    vcol = ["#00e5ff" if cc >= oo else "#ff3b6b" for oo, cc in zip(d[o], d[c])]
    fig.add_trace(go.Bar(x=d[x], y=d[v], marker_color=vcol, opacity=0.5,
                         name="Vol", hoverinfo="skip"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0, r=0, t=8, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_rangeslider_visible=False, showlegend=False,
                      hovermode="x unified", font=dict(family="JetBrains Mono"))
    fig.update_yaxes(gridcolor="rgba(0,229,255,.07)")
    # Hide non-trading days: drop weekends, then the specific holidays (business
    # days that have no bar) so the candles sit flush with no gaps.
    xs = pd.to_datetime(d[x])
    missing = sorted(set(pd.date_range(xs.min(), xs.max(), freq="B")) - set(xs.dt.normalize()))
    breaks = [dict(bounds=["sat", "mon"])]
    if missing:
        breaks.append(dict(values=missing))
    fig.update_xaxes(rangebreaks=breaks, gridcolor="rgba(0,229,255,.07)")
    return fig


# ---- Try to score the ticker (needs ~100+ bars of history) ----
pred = None
try:
    with st.spinner(f"Scoring {ticker}…"):
        pred = ui.get_single_prediction(ticker, ckpt, mtime, dt.date.today().isoformat())
except Exception:
    pred = None

if pred is not None and not pred.empty:
    last = pred.iloc[-1]
    cls = int(last["pred_class"])
    st.markdown(
        f"<div style='font-family:Orbitron;font-size:1.5rem;color:{engine.CLASS_COLOR[cls]};"
        f"text-shadow:0 0 16px {engine.CLASS_COLOR[cls]}66'>"
        f"▲ {ticker} → {engine.CLASS_NAMES[cls].upper()}</div>", unsafe_allow_html=True)
    m = st.columns(4)
    m[0].metric("Close", f"${last['close']:.2f}")
    m[1].metric("Signal P(Long)−P(Short)", f"{last['signal']:+.2f}")
    m[2].metric("Confidence", f"{max(last['p_short'], last['p_notrade'], last['p_long']):.0%}")
    m[3].metric("As of", f"{last['date']:%Y-%m-%d}")

    bars = "".join(
        f"<div style='flex:1'><div style='font-size:.7rem;opacity:.7'>{engine.CLASS_NAMES[i]} "
        f"{last[col]:.0%}</div><div style='height:8px;border-radius:4px;background:#0b111c'>"
        f"<div style='height:100%;width:{last[col]*100:.0f}%;border-radius:4px;"
        f"background:{engine.CLASS_COLOR[i]};box-shadow:0 0 10px {engine.CLASS_COLOR[i]}'></div></div></div>"
        for i, col in enumerate(["p_short", "p_notrade", "p_long"]))
    st.markdown(f"<div style='display:flex;gap:14px;margin:8px 0'>{bars}</div>",
                unsafe_allow_html=True)

    d = pred.sort_values("date").tail(SPAN[period])
    names = [engine.CLASS_NAMES[k] for k in d["pred_class"]]
    cd = list(zip(d["open"], d["high"], d["low"], d["close"], d["volume"], d["signal"], names))
    hover = ("<b>%{x|%Y-%m-%d}</b><br>O %{customdata[0]:.2f}  H %{customdata[1]:.2f}<br>"
             "L %{customdata[2]:.2f}  C %{customdata[3]:.2f}<br>Vol %{customdata[4]:,.0f}<br>"
             "<b>Pred %{customdata[6]}</b>  signal %{customdata[5]:+.2f}<extra></extra>")
    st.plotly_chart(candlestick(d, "date", "open", "high", "low", "close", "volume",
                                customdata=cd, hover=hover, pred_class=list(d["pred_class"])),
                    width="stretch")
    st.caption("ℹ️ Single-ticker scoring neutralizes the 3 cross-sectional peer-rank "
               "features; the other 23 are fully active. Read prediction as the hover/ribbon.")
else:
    df = fetch_raw(ticker, period)
    if df is None:
        st.error(f"No data for **{ticker}** — check the symbol.")
        st.stop()
    st.info(f"Showing price only — not enough history to run the model on **{ticker}**.")
    st.plotly_chart(candlestick(df, "Date", "Open", "High", "Low", "Close", "Volume"),
                    width="stretch")
