from datetime import datetime, timedelta
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import seaborn as sns
import streamlit as st
import yfinance as yf
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.dates import date2num
from sklearn.preprocessing import StandardScaler

# ======================================
# CONFIGURATION & STYLING
# ======================================
# Visual settings
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# Default parameters
TICKER = "AAPL"
DAYS = 90
MODEL_PATH = "model.pkl"
THRESHOLDS = {
    "STRONG_BUY": 0.75,
    "BUY": 0.65,
    "WEAK_BUY": 0.55,
    "NEUTRAL": (0.45, 0.55),
    "SELL": 0.35,
    "STRONG_SELL": 0.25,
}

# ======================================
# STREAMLIT UI
# ======================================
st.set_page_config(layout="wide", page_title="Stock Predictor Pro")
st.title("📊 Stock Predictor Pro - AI-Driven Analysis")

# Sidebar controls
with st.sidebar:
    st.header("Analysis Parameters")
    ticker = st.text_input("Stock Ticker", TICKER)
    days = st.slider("Lookback Period (days)", 7, 365, DAYS)
    st.markdown("---")
    st.caption("Model Settings:")
    show_confidence = st.checkbox("Show Confidence Metrics", True)
    show_raw = st.checkbox("Show Raw Data", False)

# ======================================
# MAIN ANALYSIS FUNCTION
# ======================================
if st.button("Run Analysis", type="primary"):
    with st.spinner("Crunching market data..."):
        try:
            # --------------------------
            # 1. DATA FETCHING
            # --------------------------
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            df = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval="1h",
                progress=False,
            )

            if df.empty:
                st.error("⚠️ No data returned - check ticker or try again later")
                st.stop()

            # Clean data
            df.columns = [col.lower() for col in df.columns]
            df = df.rename(columns={"adj close": "close"})

            # --------------------------
            # 2. FEATURE ENGINEERING
            # --------------------------
            # Technical Indicators
            df["rsi"] = ta.rsi(df["close"], length=14)
            df["macd"] = ta.ema(df["close"], length=12) - ta.ema(df["close"], length=26)
            df["macd_signal"] = ta.ema(df["macd"], length=9)
            df["ema_200"] = ta.ema(df["close"], length=200)
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            df["returns"] = df["close"].pct_change()
            df["volatility"] = df["returns"].rolling(24).std()

            # Lagged features
            for lag in [1, 3, 6]:
                df[f"rsi_lag{lag}"] = df["rsi"].shift(lag)
                df[f"volume_lag{lag}"] = df["volume"].shift(lag)

            df = df.dropna()

            # --------------------------
            # 3. MODEL PREDICTION
            # --------------------------
            # Load model artifacts
            model, scaler, features = joblib.load(MODEL_PATH)
            # Prepare and scale features
            X = df[features]
            X_scaled = scaler.transform(X)

            # Get predictions
            predictions = model.predict_proba(X_scaled)
            df["prediction"] = predictions[:, 1]
            latest_pred = df["prediction"].iloc[-1]

            # --------------------------
            # 4. TRADING SIGNAL
            # --------------------------
            if latest_pred >= THRESHOLDS["STRONG_BUY"]:
                signal = "STRONG BUY"
                color = "#006400"  # Dark green
                emoji = "🚀"
            elif latest_pred >= THRESHOLDS["BUY"]:
                signal = "BUY"
                color = "#228B22"  # Forest green
                emoji = "📈"
            elif latest_pred >= THRESHOLDS["WEAK_BUY"]:
                signal = "WEAK BUY"
                color = "#7CFC00"  # Lawn green
                emoji = "↗️"
            elif latest_pred <= THRESHOLDS["STRONG_SELL"]:
                signal = "STRONG SELL"
                color = "#8B0000"  # Dark red
                emoji = "⚠️"
            elif latest_pred <= THRESHOLDS["SELL"]:
                signal = "SELL"
                color = "#FF0000"  # Red
                emoji = "📉"
            else:
                signal = "NEUTRAL"
                color = "#696969"  # Dim gray
                emoji = "➖"

            confidence = max(latest_pred, 1 - latest_pred)
            confidence_str = f"{confidence*100:.1f}%"

            # --------------------------
            # 5. VISUALIZATION (CANDLESTICK CHART)
            # --------------------------
            # Create figure with subplots
            fig = plt.figure(figsize=(14, 12), facecolor="white")
            gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
            ax4 = fig.add_subplot(gs[3])

            # Prepare data for candlestick chart
            df_plot = df.tail(100).copy()  # Show last 100 periods
            df_plot['date_num'] = date2num(df_plot.index.to_pydatetime())

            # Plot candlesticks
            candlestick_ohlc(
                ax1,
                df_plot[['date_num', 'open', 'high', 'low', 'close']].values,
                width=0.006,
                colorup='g',
                colordown='r',
                alpha=0.8
            )

            # Plot EMA200
            ax1.plot(df_plot.index, df_plot['ema_200'], color='orange', linestyle='--', label='200 EMA')

            # Add prediction zones
            ax1.fill_between(
                df_plot.index,
                df_plot["close"],
                where=(df_plot["prediction"] > 0.65),
                color="green",
                alpha=0.1,
                label="Buy Zone",
            )
            ax1.fill_between(
                df_plot.index,
                df_plot["close"],
                where=(df_plot["prediction"] < 0.35),
                color="red",
                alpha=0.1,
                label="Sell Zone",
            )

            ax1.set_title(
                f"{ticker} Price Analysis - {emoji} {signal} ({confidence_str} Confidence)",
                fontsize=14,
                fontweight="bold",
                pad=20,
                color=color,
            )
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)

            # RSI subplot
            sns.lineplot(data=df_plot, x=df_plot.index, y="rsi", ax=ax2, color="#9467bd")
            ax2.axhline(70, color="red", linestyle=":", alpha=0.7)
            ax2.axhline(30, color="green", linestyle=":", alpha=0.7)
            ax2.fill_between(df_plot.index, 70, 30, color="gray", alpha=0.1)
            ax2.set_title("RSI (14-period)", fontsize=10)

            # MACD subplot
            sns.lineplot(
                data=df_plot,
                x=df_plot.index,
                y="macd",
                ax=ax3,
                color="#1f77b4",
                label="MACD",
            )
            sns.lineplot(
                data=df_plot,
                x=df_plot.index,
                y="macd_signal",
                ax=ax3,
                color="#ff7f0e",
                label="Signal",
            )
            ax3.bar(
                df_plot.index,
                df_plot["macd"] - df_plot["macd_signal"],
                color=np.where((df_plot["macd"] - df_plot["macd_signal"]) > 0, "green", "red"),
                alpha=0.3,
                width=0.01,
            )
            ax3.axhline(0, color="black", linestyle="-", alpha=0.5)
            ax3.set_title("MACD (12,26,9)", fontsize=10)
            ax3.legend(loc="upper left")

            # Prediction probability subplot
            sns.lineplot(data=df_plot, x=df_plot.index, y="prediction", ax=ax4, color=color)
            ax4.axhline(0.5