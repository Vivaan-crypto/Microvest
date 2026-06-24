"""Chart + prediction — search ANY US ticker, see a candlestick, and (when the
ticker has enough history) the model's directionality on the latest bar and in the
hover. The chart always renders; prediction layers on top when available."""

import datetime as dt

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

# How many trading days each range button shows.
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
    if df is None or df.empty:
        return None
    return df.reset_index().dropna(subset=["Close"])


def candle_chart(df, hover=None, customdata=None, pred_class=None):
    """Two-row candlestick (price on top, volume below).

    `df` must have columns: date, open, high, low, close, volume.
    Optional: `hover`/`customdata` for a custom tooltip, `pred_class` for the
    colored direction dots under each candle.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22], vertical_spacing=0.03)

    # The candles. With a custom hover we turn off the candle's own tooltip.
    if hover is not None:
        candle_hover = "skip"
    else:
        candle_hover = None
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="OHLC", increasing_line_color=ui.CYAN, decreasing_line_color=ui.RED,
        hoverinfo=candle_hover), row=1, col=1)

    # Invisible markers that carry the custom OHLCV + prediction tooltip.
    if hover is not None:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["high"], mode="markers", showlegend=False,
            marker=dict(size=0.1, color="rgba(0,0,0,0)"),
            customdata=customdata, hovertemplate=hover), row=1, col=1)

    # A colored dot under each candle for the predicted direction.
    if pred_class is not None:
        dot_colors = []
        for cls in pred_class:
            dot_colors.append(engine.CLASS_COLOR[cls])
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["low"] * 0.985, mode="markers",
            showlegend=False, hoverinfo="skip",
            marker=dict(size=5, color=dot_colors)), row=1, col=1)

    # Volume bars: cyan on up days, red on down days.
    volume_colors = []
    for open_price, close_price in zip(df["open"], df["close"]):
        if close_price >= open_price:
            volume_colors.append(ui.CYAN)
        else:
            volume_colors.append(ui.RED)
    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"], marker_color=volume_colors,
        opacity=0.5, name="Vol", hoverinfo="skip"), row=2, col=1)

    ui.style_chart(fig, height=600)
    fig.update_layout(showlegend=False, xaxis_rangeslider_visible=False,
                      hovermode="x unified")
    ui.hide_nontrading_days(fig, df["date"])
    return fig


def show_headline(ticker, row):
    """Big colored 'TICKER -> DIRECTION' line plus the four summary metrics."""
    cls = int(row["pred_class"])
    color = engine.CLASS_COLOR[cls]
    direction = engine.CLASS_NAMES[cls].upper()
    st.markdown(
        f"<div style='font-family:Orbitron;font-size:1.5rem;color:{color};"
        f"text-shadow:0 0 16px {color}66'>▲ {ticker} → {direction}</div>",
        unsafe_allow_html=True)

    confidence = max(row["p_short"], row["p_notrade"], row["p_long"])
    m = st.columns(4)
    m[0].metric("Close", f"${row['close']:.2f}")
    m[1].metric("Signal P(Long)−P(Short)", f"{row['signal']:+.2f}")
    m[2].metric("Confidence", f"{confidence:.0%}")
    m[3].metric("As of", f"{row['date']:%Y-%m-%d}")


def show_prob_bars(row):
    """A small horizontal probability bar for each class (Short / NoTrade / Long)."""
    prob_cols = ["p_short", "p_notrade", "p_long"]
    bars = ""
    for i, col in enumerate(prob_cols):
        pct = row[col]
        name = engine.CLASS_NAMES[i]
        color = engine.CLASS_COLOR[i]
        bars += (
            f"<div style='flex:1'>"
            f"<div style='font-size:.7rem;opacity:.7'>{name} {pct:.0%}</div>"
            f"<div style='height:8px;border-radius:4px;background:#0b111c'>"
            f"<div style='height:100%;width:{pct * 100:.0f}%;border-radius:4px;"
            f"background:{color};box-shadow:0 0 10px {color}'></div></div></div>")
    st.markdown(f"<div style='display:flex;gap:14px;margin:8px 0'>{bars}</div>",
                unsafe_allow_html=True)


def build_hover(df):
    """Build the tooltip text + matching customdata for the OHLCV + prediction hover."""
    names = []
    for cls in df["pred_class"]:
        names.append(engine.CLASS_NAMES[cls])
    customdata = list(zip(df["open"], df["high"], df["low"], df["close"],
                          df["volume"], df["signal"], names))
    hover = ("<b>%{x|%Y-%m-%d}</b><br>O %{customdata[0]:.2f}  H %{customdata[1]:.2f}<br>"
             "L %{customdata[2]:.2f}  C %{customdata[3]:.2f}<br>Vol %{customdata[4]:,.0f}<br>"
             "<b>Pred %{customdata[6]}</b>  signal %{customdata[5]:+.2f}<extra></extra>")
    return hover, customdata


# ---- Try to score the ticker (needs ~100+ bars of history) ----
prediction = None
try:
    with st.spinner(f"Scoring {ticker}…"):
        prediction = ui.get_single_prediction(ticker, ckpt, mtime, dt.date.today().isoformat())
except Exception:
    prediction = None

if prediction is not None and not prediction.empty:
    last = prediction.iloc[-1]
    show_headline(ticker, last)
    show_prob_bars(last)

    window = prediction.sort_values("date").tail(SPAN[period])
    hover, customdata = build_hover(window)
    chart = candle_chart(window, hover=hover, customdata=customdata,
                         pred_class=list(window["pred_class"]))
    st.plotly_chart(chart, width="stretch")
    st.caption("ℹ️ Single-ticker scoring neutralizes the 3 cross-sectional peer-rank "
               "features; the other 23 are fully active. Read prediction as the hover/ribbon.")
else:
    price = fetch_raw(ticker, period)
    if price is None:
        st.error(f"No data for **{ticker}** — check the symbol.")
        st.stop()
    # Rename the yfinance columns to the lowercase names candle_chart expects.
    price = price.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                  "Low": "low", "Close": "close", "Volume": "volume"})
    st.info(f"Showing price only — not enough history to run the model on **{ticker}**.")
    st.plotly_chart(candle_chart(price), width="stretch")
