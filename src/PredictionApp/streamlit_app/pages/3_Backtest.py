"""Strategy backtester — pick a ticker, a starting amount, a date range and a
strategy (candle color, model dots, an indicator threshold, or your own custom
function), then see the percent growth and an equity curve vs. buy-and-hold.

No lookahead: a position decided at the close of day t earns the t -> t+1 return.
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine
import ui

st.set_page_config(page_title="Backtest", page_icon="🧪", layout="wide")
ckpt, mtime, info = ui.pick_checkpoint()
st.title("🧪 STRATEGY BACKTEST")
ui.banner("PICK A RULE · SIMULATE · SEE THE GROWTH")

DEFAULT_CUSTOM = '''def strategy(df):
    """Return a position for each day: 1 = long, -1 = short, 0 = cash.
    Available columns: open, high, low, close, volume, signal, pred_class,
    p_long, p_short, p_notrade.  (pred_class: 0=Short/red, 1=NoTrade, 2=Long/green)
    Use np.nan to mean "hold whatever you held yesterday".
    """
    green_candle = df["close"] >= df["open"]
    green_dot = df["pred_class"] == 2          # Long
    red = (df["close"] < df["open"]) & (df["pred_class"] == 0)

    pos = pd.Series(np.nan, index=df.index)    # default: hold
    pos[green_candle & green_dot] = 1          # buy: green candle + green dot
    pos[red] = 0                               # sell: red candle + red dot
    return pos
'''


# ------------------------------------------------------------------
# Strategy -> daily position in {-1, 0, 1} (np.nan = hold previous day)
# ------------------------------------------------------------------
def build_positions(df, name, threshold, allow_short, custom_code):
    sell = -1 if allow_short else 0
    green_candle = df["close"] >= df["open"]
    red_candle = df["close"] < df["open"]
    green_dot = df["pred_class"] == 2          # Long
    red_dot = df["pred_class"] == 0            # Short

    pos = pd.Series(np.nan, index=df.index)    # np.nan = hold previous position

    if name == "Candle color":
        pos[green_candle] = 1
        pos[red_candle] = sell
    elif name == "Model dots":
        pos[green_dot] = 1
        pos[red_dot] = sell
        # NoTrade (yellow) -> stays np.nan -> holds
    elif name == "Candle + dot":
        pos[green_candle & green_dot] = 1
        pos[red_candle & red_dot] = sell
    elif name == "Signal threshold":
        pos[df["signal"] > threshold] = 1
        pos[df["signal"] < -threshold] = sell
    elif name == "Custom":
        namespace = {"pd": pd, "np": np}
        exec(custom_code, namespace)
        fn = namespace["strategy"]
        raw = fn(df.copy())
        pos = pd.Series(raw, index=df.index).astype(float)

    return pos.ffill().fillna(0).clip(-1, 1)


def simulate(df, positions, start_cash):
    df = df.sort_values("date").reset_index(drop=True)
    positions = positions.reset_index(drop=True)

    # Return earned from day t to day t+1, aligned to the decision day t.
    ret_next = df["close"].pct_change().shift(-1).fillna(0)

    strat_ret = (positions * ret_next).fillna(0)
    equity = (1 + strat_ret).cumprod() * start_cash
    buy_hold = (1 + ret_next).cumprod() * start_cash
    return df["date"], equity, buy_hold, positions


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
c1, c2, c3 = st.columns([1.2, 1, 1])
ticker = c1.text_input("🔎 Ticker", value="AAPL").strip().upper()
start_cash = c2.number_input("Starting amount ($)", min_value=1.0, value=10000.0, step=1000.0)
allow_short = c3.checkbox("Allow shorting", value=False,
                          help="If off, a 'sell' just moves to cash instead of going short.")
if not ticker:
    st.stop()

# Score the whole history once; we filter to the chosen window afterwards.
try:
    with st.spinner(f"Scoring {ticker}…"):
        preds = ui.get_single_prediction(ticker, ckpt, mtime, dt.date.today().isoformat())
except Exception as e:
    st.error(f"Could not score {ticker}: {e}")
    st.stop()

if preds is None or preds.empty:
    st.error(f"Not enough history to run the model on **{ticker}**.")
    st.stop()

preds = preds.sort_values("date").reset_index(drop=True)
min_d, max_d = preds["date"].min().date(), preds["date"].max().date()

d1, d2 = st.columns(2)
start_date = d1.date_input("Start date", value=max(min_d, max_d.replace(year=max_d.year - 2)),
                           min_value=min_d, max_value=max_d)
end_date = d2.date_input("End date", value=max_d, min_value=min_d, max_value=max_d)

strat = st.selectbox(
    "Strategy", ["Candle color", "Model dots", "Candle + dot", "Signal threshold", "Custom"],
    help="Candle color: buy green / sell red.  Model dots: buy Long / sell Short.  "
         "Candle + dot: both must agree.  Signal threshold: long/short on the model's signal.")

threshold = 0.2
custom_code = DEFAULT_CUSTOM
if strat == "Signal threshold":
    threshold = st.slider("Signal threshold", 0.0, 1.0, 0.2, 0.05,
                          help="Go long when signal > +thr, short/cash when signal < -thr.")
if strat == "Custom":
    custom_code = st.text_area("Custom strategy (define a function `strategy(df)`)",
                               value=DEFAULT_CUSTOM, height=260)

# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
window = preds[(preds["date"] >= pd.Timestamp(start_date)) &
               (preds["date"] <= pd.Timestamp(end_date))].reset_index(drop=True)
if len(window) < 2:
    st.warning("Pick a wider date range — not enough bars to simulate.")
    st.stop()

try:
    positions = build_positions(window, strat, threshold, allow_short, custom_code)
except Exception as e:
    st.error(f"Strategy failed: {e}")
    st.stop()

dates, equity, buy_hold, positions = simulate(window, positions, start_cash)

final_val = float(equity.iloc[-1])
pct_growth = final_val / start_cash - 1
bh_growth = float(buy_hold.iloc[-1]) / start_cash - 1
drawdown = float((equity / equity.cummax() - 1).min())
n_trades = int((positions.diff().fillna(positions) != 0).sum())
exposure = float((positions != 0).mean())

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
st.subheader("Results")
m = st.columns(5)
m[0].metric("Final value", f"${final_val:,.0f}", f"{pct_growth:+.1%}")
m[1].metric("Strategy growth", f"{pct_growth:+.1%}")
m[2].metric("Buy & hold", f"{bh_growth:+.1%}", f"{pct_growth - bh_growth:+.1%} vs B&H")
m[3].metric("Max drawdown", f"{drawdown:.1%}")
m[4].metric("Trades · exposure", f"{n_trades} · {exposure:.0%}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=equity, name="Strategy", mode="lines",
                         line=dict(color="#00e5ff", width=2)))
fig.add_trace(go.Scatter(x=dates, y=buy_hold, name="Buy & hold", mode="lines",
                         line=dict(color="#8a93a6", width=1.5, dash="dot")))
fig.add_hline(y=start_cash, line=dict(color="rgba(255,255,255,.2)", dash="dash"))
fig.update_layout(template="plotly_dark", height=420, margin=dict(l=0, r=0, t=10, b=0),
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  hovermode="x unified", font=dict(family="JetBrains Mono"),
                  legend=dict(orientation="h", y=1.02, x=0))
fig.update_yaxes(gridcolor="rgba(0,229,255,.07)", title="Portfolio value ($)")
fig.update_xaxes(gridcolor="rgba(0,229,255,.07)")
st.plotly_chart(fig, width="stretch")

with st.expander("Price with entries / exits"):
    pf = go.Figure()
    pf.add_trace(go.Candlestick(
        x=window["date"], open=window["open"], high=window["high"],
        low=window["low"], close=window["close"], name="OHLC",
        increasing_line_color="#00e5ff", decreasing_line_color="#ff3b6b"))
    longs = window[positions.values > 0]
    shorts = window[positions.values < 0]
    pf.add_trace(go.Scatter(x=longs["date"], y=longs["low"] * 0.985, mode="markers",
                            name="long", marker=dict(symbol="triangle-up", size=8, color="#2ecc71")))
    pf.add_trace(go.Scatter(x=shorts["date"], y=shorts["high"] * 1.015, mode="markers",
                            name="short/cash", marker=dict(symbol="triangle-down", size=8, color="#ff3b6b")))
    pf.update_layout(template="plotly_dark", height=420, margin=dict(l=0, r=0, t=10, b=0),
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     xaxis_rangeslider_visible=False, font=dict(family="JetBrains Mono"))
    pf.update_yaxes(gridcolor="rgba(0,229,255,.07)")
    pf.update_xaxes(gridcolor="rgba(0,229,255,.07)", rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(pf, width="stretch")

st.caption("⚠️ Gross of costs/slippage. A position set at today's close earns "
           "tomorrow's return (no lookahead). 'Sell' = cash unless shorting is on.")
