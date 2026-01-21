# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import torch
import yfinance as yf
from model import StockLSTMModel
import pandas_ta as ta

# ----------------------
# App / model constants
# ----------------------
WINDOW = 20
PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]
INDICATOR_COLS = ["SMA20", "EMA12", "RSI14", "MACD", "MACD_signal", "MACD_hist"]  # 6 features
PRICE_INPUT_SIZE = 5
INDICATOR_INPUT_SIZE = 6
PATH = "//LightningLogs/version_16/checkpoints/epoch=217-step=299750.ckpt"


# ----------------------
# Small helpers
# ----------------------
def fetch_data(symbol: str, start, end) -> pd.DataFrame:
    """Download daily OHLCV data for a single ticker."""
    _df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False,
                      multi_level_index=False).rename_axis("Date").reset_index().set_index("Date")
    return _df


def compute_indicators(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators using pandas_ta,
    then rename them to match INDICATOR_COLS.
    Mirrors what you do in preprocessing.get_data.
    """
    _df = _df.copy()

    # These use Close by default, like your training script
    _df.ta.sma(length=20, append=True)
    _df.ta.ema(length=12, append=True)
    _df.ta.rsi(length=14, append=True)
    _df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # Rename to the names used in INDICATOR_COLS
    rename_map = {
        "SMA_20": "SMA20",
        "EMA_12": "EMA12",
        "RSI_14": "RSI14",
        "MACD_12_26_9": "MACD",
        "MACDs_12_26_9": "MACD_signal",
        "MACDh_12_26_9": "MACD_hist",
    }
    _df = _df.rename(columns=rename_map)

    return _df


def zscore_window(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True)
    return (arr - mu) / (sd + eps)


def build_model_and_weights() -> StockLSTMModel:
    """Instantiate StockLSTMModel and load trained weights."""
    model = StockLSTMModel()

    # Use relative path: model/model.pth (from lightning_train.py)
    weights_path = os.path.abspath(PATH)

    if os.path.isfile(weights_path):
        ckpt = torch.load(weights_path)

        # This works for both:
        # - raw state_dict
        # - Lightning .ckpt w/ "state_dict" key
        state_dict = ckpt.get("state_dict", ckpt)

        # Strip "model." prefix (since you saved LightningModule.state_dict())
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned[k[len("model."):]] = v
            else:
                cleaned[k] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            st.warning(f"Missing keys when loading model: {missing}")
        if unexpected:
            st.warning(f"Unexpected keys when loading model: {unexpected}")
    else:
        st.warning(
            f"Checkpoint not found at {weights_path}. "
            "Using randomly initialized weights."
        )

    model.eval()
    return model


def latest_window_tensors(_df: pd.DataFrame):
    """
    From a full OHLCV+indicator df, build the tensors the model expects:
      - price_seq: [1, WINDOW, 5] z-scored OHLCV
      - indicators: [1, 6] standardized tech indicators (last bar)
      - last_close: raw last close price
    """
    # Ensure indicators exist
    need_inds = [c for c in INDICATOR_COLS if c not in _df.columns]
    if need_inds:
        _df = compute_indicators(_df)

    _df = _df.dropna().copy()
    if len(_df) < WINDOW:
        raise ValueError(
            f"Not enough rows after indicators to form a {WINDOW}-bar window "
            f"(have {len(_df)})."
        )

    # Last WINDOW bars
    win = _df.iloc[-WINDOW:]

    # OHLCV → LSTM branch: [1, T, 5]
    price_seq_np = win[PRICE_COLS].to_numpy().astype("float32")
    price_seq_np = zscore_window(price_seq_np)
    _price_seq = torch.from_numpy(price_seq_np[None, :, :])

    # Indicators → FF branch: [1, 6]
    # Take last row's indicators and standardize vs trailing window
    ind_np = win.iloc[-1:][INDICATOR_COLS].to_numpy().astype("float32")
    trailing = _df.iloc[-max(100, WINDOW):][INDICATOR_COLS].to_numpy().astype("float32")
    ind_np = (ind_np - trailing.mean(axis=0, keepdims=True)) / (
            trailing.std(axis=0, keepdims=True) + 1e-6
    )

    _indicators = torch.from_numpy(ind_np)
    _last_close = float(win["Close"].iloc[-1])

    return _price_seq, _indicators, _last_close, df


# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Stock Market Predictor", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .reportview-container { background-color: #121212; }
    .sidebar-content { background-color: #1e1e1e; }
    .stTextInput>div>div{ background-color: #2d2d2d; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("📊 Stock Market Predictor")
    stock_symbol = st.text_input("Stock Symbol", "AAPL").upper().strip()
    start_date = st.date_input("Start Date", pd.Timestamp("2020-01-01"))
    end_date = st.date_input("End Date", pd.Timestamp.today())
    run_button = st.button("Run Prediction")

st.title("📈 Stock Market Analysis Dashboard")

model = build_model_and_weights()

if run_button:
    try:
        # Fetch data
        df = fetch_data(stock_symbol, start_date, end_date)
        df = compute_indicators(df)

        # Main chart
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                )
            ]
        )
        fig.update_layout(
            title=f"{stock_symbol} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            font=dict(color="#e0e0e0"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Indicators chart
        st.subheader("Technical Indicators (pandas_ta)")
        available_inds = [c for c in INDICATOR_COLS if c in df.columns]
        if available_inds:
            st.line_chart(df[available_inds].dropna())
        else:
            st.info("No indicators computed (check data length).")

        # Build tensors & run model
        price_seq, indicators, last_close, df_ready = latest_window_tensors(df)
        with torch.no_grad():
            pred_return = model(price_seq, indicators)
            pred_return = float(pred_return.item())  # decimal (e.g. 0.05 = +5%)

        pred_price = last_close * (1.0 + pred_return)

        st.subheader("Prediction Results (5-Day Horizon)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Last Close", f"${last_close:,.2f}")
        c2.metric("Predicted 5-Day Return", f"{pred_return * 100:+.2f}%")
        c3.metric(
            "Projected Price (5d)",
            f"${pred_price:,.2f}",
            f"{(pred_price - last_close):+.2f}",
        )

        # Simple pseudo-confidence gauge (based on magnitude of prediction)
        conf = max(0.0, 1.0 - min(1.0, abs(pred_return) / 0.05)) * 100.0
        fig_conf = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=conf,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Confidence Level"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "teal"},
                    "bgcolor": "gray",
                    "borderwidth": 2,
                    "bordercolor": "white",
                },
            )
        )
        fig_conf.update_layout(template="plotly_dark")
        st.plotly_chart(fig_conf, use_container_width=True)

        st.subheader("Raw Data")
        st.dataframe(df.tail(400).style.format(precision=4))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
