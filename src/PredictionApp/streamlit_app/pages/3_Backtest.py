"""Strategy backtester — score a ticker once, then run every strategy off that one
prediction and compare them: overlaid equity curves, a summary table, and a
buy/sell chart per strategy, all vs. buy-and-hold and the S&P 500.

No lookahead: a position decided at the close of day t earns the t -> t+1 return.

Performance notes:
- The heavy work (scoring the ticker) is cached in ui.get_single_prediction.
- Everything below the config lives in a @st.fragment, so moving a slider reruns
  only this section — not the whole page.
- The per-strategy candlestick charts render lazily (behind a toggle), so seven
  heavy charts aren't rebuilt on every interaction.
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from src.PredictionApp.streamlit_app import engine, ui

st.set_page_config(page_title="Backtest", page_icon="🧪", layout="wide")
ckpt, mtime, info = ui.pick_checkpoint()
ui.page_header("STRATEGY BACKTEST", "ONE PREDICTION RUN · EVERY STRATEGY OVERLAID")

# Strategies we run for everyone (Custom lives in its own section below).
BUILTIN_STRATEGIES = ["Candle color", "Model dots", "Candle + dot", "Signal threshold",
                      "Probability threshold", "MA crossover", "RSI"]

# One distinct line color per strategy on the overlay chart.
PALETTE = ["#00e5ff", "#2ecc71", "#ff6b9d", "#f1c40f", "#b48cff", "#ff9f43", "#5dd5c4"]

DEFAULT_CUSTOM = '''def strategy(df):
    """Return a position for each day: 1 = long, -1 = short, 0 = cash.

    Columns on df: open, high, low, close, volume, signal, pred_class,
    p_long, p_short, p_notrade.  (pred_class: 0=Short, 1=NoTrade, 2=Long)
    Helpers available: sma(df["close"], n) and rsi(df["close"], n).
    Use np.nan to mean "hold whatever you held yesterday".
    """
    fast = sma(df["close"], 20)
    slow = sma(df["close"], 50)

    pos = pd.Series(np.nan, index=df.index)    # default: hold
    pos[fast > slow] = 1                        # buy when fast MA is above slow MA
    pos[fast < slow] = 0                        # sell to cash when it crosses back
    return pos
'''


# ------------------------------------------------------------------
# Small indicators (computed from close prices, just for strategy rules)
# ------------------------------------------------------------------
def sma(close, window):
    return close.rolling(window).mean()


def rsi(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def add_indicators(df, params):
    """Pre-compute indicators on the full history (so they're warmed up before the
    backtest window starts). Adds the columns the MA/RSI strategies read."""
    df = df.copy()
    df["ma"] = sma(df["close"], params["ma_window"])
    df["rsi"] = rsi(df["close"], params["rsi_period"])
    return df


# ------------------------------------------------------------------
# Strategy -> daily position in {-1, 0, 1} (np.nan = hold previous day)
# ------------------------------------------------------------------
def build_positions(df, name, params, allow_short, custom_code):
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
        thr = params["signal_thr"]
        pos[df["signal"] > thr] = 1
        pos[df["signal"] < -thr] = sell
    elif name == "Probability threshold":
        p = params["prob_thr"]
        pos[df["p_long"] >= p] = 1
        pos[df["p_short"] >= p] = sell
    elif name == "MA crossover":
        pos[df["close"] > df["ma"]] = 1
        pos[df["close"] < df["ma"]] = sell
    elif name == "RSI":
        pos[df["rsi"] < params["rsi_low"]] = 1      # oversold -> buy
        pos[df["rsi"] > params["rsi_high"]] = sell  # overbought -> sell
    elif name == "Custom":
        namespace = {"pd": pd, "np": np, "sma": sma, "rsi": rsi}
        exec(custom_code, namespace)
        fn = namespace["strategy"]
        raw = fn(df.copy())
        pos = pd.Series(raw, index=df.index).astype(float)

    return pos.ffill().fillna(0).clip(-1, 1)


# ------------------------------------------------------------------
# Simulation + stats
# ------------------------------------------------------------------
def equity_curve(positions, ret_next, start_cash):
    """Compound a position series against next-day returns into a $ equity curve."""
    positions = positions.reset_index(drop=True)
    daily_return = (positions * ret_next).fillna(0)
    return (1 + daily_return).cumprod() * start_cash


def strategy_stats(name, equity, positions, start_cash, bh_growth, sp_growth):
    growth = equity.iloc[-1] / start_cash - 1
    drawdown = (equity / equity.cummax() - 1).min()
    position_changes = positions.diff().fillna(positions) != 0
    return {
        "Strategy": name,
        "Growth": growth,
        "vs Buy&Hold": growth - bh_growth,
        "vs S&P": growth - sp_growth,
        "Max drawdown": float(drawdown),
        "Trades": int(position_changes.sum()),
        "Exposure": float((positions != 0).mean()),
    }


def markers_chart(window, positions):
    """Candlestick with up/down triangles where this strategy buys / sells."""
    longs = window[positions.values > 0]
    shorts = window[positions.values < 0]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=window["date"], open=window["open"], high=window["high"],
        low=window["low"], close=window["close"], name="OHLC",
        increasing_line_color=ui.CYAN, decreasing_line_color=ui.RED))
    fig.add_trace(go.Scatter(
        x=longs["date"], y=longs["low"] * 0.985, mode="markers", name="long",
        marker=dict(symbol="triangle-up", size=8, color=ui.GREEN)))
    fig.add_trace(go.Scatter(
        x=shorts["date"], y=shorts["high"] * 1.015, mode="markers", name="short/cash",
        marker=dict(symbol="triangle-down", size=8, color=ui.RED)))
    ui.style_chart(fig, height=380)
    fig.update_layout(xaxis_rangeslider_visible=False)
    ui.hide_nontrading_days(fig, window["date"])
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sp500(start, end):
    df = yf.download("^GSPC", start=start, end=end, interval="1d",
                     auto_adjust=True, progress=False, multi_level_index=False)
    if df is None or df.empty:
        return None
    out = df.reset_index()[["Date", "Close"]]
    out.columns = ["date", "close"]
    return out


def sp500_equity(window_dates, start_cash, start, end):
    """S&P 500 buy-and-hold equity, aligned to the backtest's trading days."""
    sp = fetch_sp500(start, end)
    if sp is None or sp.empty:
        return None
    closes = sp.set_index("date")["close"]
    aligned = closes.reindex(pd.to_datetime(window_dates)).ffill().bfill()
    ret_next = aligned.pct_change().shift(-1).fillna(0)
    return (1 + ret_next).cumprod() * start_cash


def period_overview(window, buy_hold):
    """Strategy-independent facts about the stock over the window — useful context
    before comparing strategies (total return, volatility, drawdown, etc.)."""
    close = window["close"]
    daily_return = close.pct_change()
    return {
        "growth": close.iloc[-1] / close.iloc[0] - 1,
        "volatility": daily_return.std() * (252 ** 0.5),   # annualized
        "max_dd": float((buy_hold / buy_hold.cummax() - 1).min()),
        "best_day": daily_return.max(),
        "worst_day": daily_return.min(),
        "days": len(window),
        "start_price": close.iloc[0],
        "end_price": close.iloc[-1],
    }


# ------------------------------------------------------------------
# Config (the one prediction run happens here, then it's reused by all strategies)
# ------------------------------------------------------------------
c1, c2, c3 = st.columns([1.2, 1, 1])
ticker = c1.text_input("🔎 Ticker", value="AAPL").strip().upper()
start_cash = c2.number_input("Starting amount ($)", min_value=1.0, value=10000.0, step=1000.0)
allow_short = c3.checkbox("Allow shorting", value=False,
                          help="If off, a 'sell' just moves to cash instead of going short.")
if not ticker:
    st.stop()

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


# ------------------------------------------------------------------
# Strategy lab — a fragment: date/slider changes rerun ONLY this section
# ------------------------------------------------------------------
@st.fragment
def strategy_lab(preds, ticker, start_cash, allow_short):
    min_d = preds["date"].min().date()
    max_d = preds["date"].max().date()

    d1, d2 = st.columns(2)
    default_start = max(min_d, (pd.Timestamp(max_d) - pd.DateOffset(years=2)).date())
    start_date = d1.date_input("Start date", value=default_start,
                               min_value=min_d, max_value=max_d, key="bt_start")
    end_date = d2.date_input("End date", value=max_d,
                             min_value=min_d, max_value=max_d, key="bt_end")

    # Each slider has a `key`, so Streamlit keeps its value in session state and the
    # settings stick when you search a different ticker (within this session).
    with st.expander("⚙️ Threshold configuration — saved across searches"):
        a, b = st.columns(2)
        signal_thr = a.slider("Signal threshold", 0.0, 1.0, 0.2, 0.05,
                              key="cfg_signal_thr", help=ui.HELP["signal_thr"])
        prob_thr = b.slider("Probability threshold", 0.34, 0.95, 0.45, 0.01,
                            key="cfg_prob_thr", help=ui.HELP["prob_thr"])
        ma_window = a.slider("MA window (days)", 5, 200, 50, 5,
                             key="cfg_ma_window", help=ui.HELP["ma_window"])
        rsi_period = b.slider("RSI period", 2, 30, 14,
                              key="cfg_rsi_period", help=ui.HELP["rsi_period"])
        rsi_low = a.slider("RSI oversold (buy below)", 5, 45, 30,
                           key="cfg_rsi_low", help=ui.HELP["rsi_band"])
        rsi_high = b.slider("RSI overbought (sell above)", 55, 95, 70,
                            key="cfg_rsi_high", help=ui.HELP["rsi_band"])
    params = {"signal_thr": signal_thr, "prob_thr": prob_thr, "ma_window": ma_window,
              "rsi_period": rsi_period, "rsi_low": rsi_low, "rsi_high": rsi_high}

    # Indicators warm up on the full history, then we slice the chosen window.
    data = add_indicators(preds, params)
    window = data[(data["date"] >= pd.Timestamp(start_date)) &
                  (data["date"] <= pd.Timestamp(end_date))].reset_index(drop=True)
    if len(window) < 2:
        st.warning("Pick a wider date range — not enough bars to simulate.")
        return

    dates = window["date"]
    ret_next = window["close"].pct_change().shift(-1).fillna(0)
    buy_hold = (1 + ret_next).cumprod() * start_cash
    bh_growth = float(buy_hold.iloc[-1]) / start_cash - 1

    sp_equity = sp500_equity(window["date"], start_cash,
                             start_date, end_date + dt.timedelta(days=1))
    if sp_equity is not None:
        sp_growth = float(sp_equity.iloc[-1]) / start_cash - 1
    else:
        sp_growth = float("nan")

    # ---- Stock overview (strategy-independent context) ----
    st.subheader(f"{ticker} over this window")
    overview = period_overview(window, buy_hold)
    o = st.columns(6)
    o[0].metric(f"{ticker} return", f"{overview['growth']:+.1%}",
                help="Buy-and-hold total return of the stock over the window.")
    o[1].metric("S&P 500 return", f"{sp_growth:+.1%}",
                help="Buy-and-hold total return of the S&P 500 over the same window — the market benchmark.")
    o[2].metric(f"{ticker} vs S&P", f"{overview['growth'] - sp_growth:+.1%}", help=ui.HELP["vs_spx"])
    o[3].metric("Ann. volatility", f"{overview['volatility']:.1%}", help=ui.HELP["volatility"])
    o[4].metric("Buy & hold max DD", f"{overview['max_dd']:.1%}", help=ui.HELP["max_dd"])
    o[5].metric("Trading days", f"{overview['days']}",
                help="Number of trading bars in the selected window.")
    st.caption(f"Price {overview['start_price']:.2f} → {overview['end_price']:.2f}  ·  "
               f"best day {overview['best_day']:+.1%}  ·  worst day {overview['worst_day']:+.1%}")

    # ---- Run every strategy off the single prediction ----
    curves = {}   # name -> (equity series, positions series)
    stats = []
    for name in BUILTIN_STRATEGIES:
        positions = build_positions(window, name, params, allow_short, "")
        equity = equity_curve(positions, ret_next, start_cash)
        curves[name] = (equity, positions)
        stats.append(strategy_stats(name, equity, positions, start_cash, bh_growth, sp_growth))

    # ---- Overlay: every strategy on one equity chart ----
    st.subheader("Equity — every strategy")
    fig = go.Figure()
    for i, name in enumerate(BUILTIN_STRATEGIES):
        equity = curves[name][0]
        fig.add_trace(go.Scatter(x=dates, y=equity, name=name, mode="lines",
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=1.8)))
    fig.add_trace(go.Scatter(x=dates, y=buy_hold, name=f"{ticker} buy & hold", mode="lines",
                             line=dict(color=ui.GRAY, width=1.5, dash="dot")))
    if sp_equity is not None:
        fig.add_trace(go.Scatter(x=dates, y=sp_equity, name="S&P 500", mode="lines",
                                 line=dict(color="#ffffff", width=1.2, dash="dash")))
    fig.add_hline(y=start_cash, line=dict(color="rgba(255,255,255,.2)", dash="dash"))
    ui.style_chart(fig, height=460)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.04, x=0))
    fig.update_yaxes(title="Portfolio value ($)")
    ui.show_chart(fig, key="equity_overlay")
    st.caption("Tip: click a name in the legend to hide/show that line.")

    # ---- Comparison table ----
    st.subheader("Comparison")
    table = pd.DataFrame(stats).sort_values("Growth", ascending=False)
    styled = (table.style
              .format({"Growth": "{:+.1%}", "vs Buy&Hold": "{:+.1%}", "vs S&P": "{:+.1%}",
                       "Max drawdown": "{:.1%}", "Exposure": "{:.0%}"})
              .background_gradient(cmap="RdYlGn", subset=["Growth"]))
    st.dataframe(styled, width="stretch", hide_index=True)
    st.caption(
        "**Growth** = strategy's total return · **vs Buy&Hold / vs S&P** = edge over those "
        "benchmarks · **Max drawdown** = worst peak-to-trough drop (less negative is better) · "
        "**Trades** = number of position changes · **Exposure** = share of days in the market "
        "(vs cash). A *low-exposure* strategy matching buy & hold is doing more per unit of risk.")
    st.caption(f"Benchmarks over this window — {ticker} buy & hold {bh_growth:+.1%} · "
               f"S&P 500 {sp_growth:+.1%}")

    # ---- Buy / sell markers per strategy (lazy: chart builds only when toggled) ----
    st.subheader("Entries / exits per strategy")
    for name in BUILTIN_STRATEGIES:
        with st.expander(name):
            show = st.toggle("Render entries/exits chart", key=f"show_chart_{name}")
            if show:
                ui.show_chart(markers_chart(window, curves[name][1]), key=f"markers_{name}")

    # ---- Custom strategy (runs off the same prediction) ----
    with st.expander("✏️ Custom strategy"):
        custom_code = st.text_area("Define a function `strategy(df)`",
                                   value=DEFAULT_CUSTOM, height=260, key="custom_code")
        try:
            positions = build_positions(window, "Custom", params, allow_short, custom_code)
            equity = equity_curve(positions, ret_next, start_cash)
            growth = float(equity.iloc[-1]) / start_cash - 1
            st.metric("Custom growth", f"{growth:+.1%}", f"{growth - bh_growth:+.1%} vs B&H")

            cf = go.Figure()
            cf.add_trace(go.Scatter(x=dates, y=equity, name="Custom", mode="lines",
                                    line=dict(color=ui.CYAN, width=2)))
            cf.add_trace(go.Scatter(x=dates, y=buy_hold, name=f"{ticker} buy & hold",
                                    mode="lines", line=dict(color=ui.GRAY, width=1.5, dash="dot")))
            ui.style_chart(cf, height=360)
            cf.update_layout(hovermode="x unified")
            ui.show_chart(cf, key="custom_equity")
            ui.show_chart(markers_chart(window, positions), key="custom_markers")
        except Exception as e:
            st.error(f"Strategy failed: {e}")

    st.caption("⚠️ Gross of costs/slippage. A position set at today's close earns "
               "tomorrow's return (no lookahead). 'Sell' = cash unless shorting is on.")


strategy_lab(preds, ticker, start_cash, allow_short)
