from datetime import datetime, timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# Configuration
TICKER = "AAPL"  # Default ticker
DAYS = 30  # Lookback period
MODEL_PATH = r"C:\Users\shahv\OneDrive\Documents\GitHub\Microvest\model\model.pkl"

# Streamlit UI
st.set_page_config(layout="wide", page_title="Stock Analyzer")
st.title("📈 Enhanced Stock Analysis (1-Hour Intervals)")

# User input
ticker = st.text_input("Enter stock ticker:", TICKER)
days = st.slider("Lookback days:", 7, 90, DAYS)

if st.button("Analyze"):
    with st.spinner("Fetching df..."):
        try:
            # 1. df Fetching with yfinance (unchanged)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            df = yf.download(
                ticker, start=start_date, end=end_date, interval="1h", progress=False
            )

            if df.empty:
                st.error("No df returned - check ticker or try again later")
                st.stop()

            # 2. Feature Engineering (keeping all original indicators)
            df.columns = df.columns.droplevel(1)  # Remove ticker level
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            # Original indicator calculations (unchanged)
            df["rsi"] = ta.rsi(df["close"], length=14)
            df["macd"] = ta.ema(close=df["close"], length=12) - ta.ema(
                close=df["close"], length=26
            )
            df["ema_20"] = ta.ema(close=df["close"], length=20)
            # Target Variable
            df["target"] = (df["close"].shift(-3) / df["close"] - 1 > 0.005).astype(int)
            # Momentum Indicators
            df["stoch_k"] = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)[
                "STOCHk_14_3_3"
            ]  # Stochastic %K
            df["stoch_d"] = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)[
                "STOCHd_14_3_3"
            ]  # Stochastic %D
            df["cci"] = ta.cci(
                df["high"], df["low"], df["close"], length=20
            )  # Commodity Channel Index
            df["mom"] = ta.mom(df["close"], length=10)  # Momentum (10-period)

            # Trend Indicators
            df["adx"] = ta.adx(df["high"], df["low"], df["close"], length=14)[
                "ADX_14"
            ]  # Average Directional Index
            df["psar"] = ta.psar(df["high"], df["low"], df["close"])[
                "PSARl_0.02_0.2"
            ]  # Parabolic SAR

            # Volume Indicators
            df["obv"] = ta.obv(df["close"], df["volume"])  # On-Balance Volume
            df["vwap"] = ta.vwap(
                df["high"], df["low"], df["close"], df["volume"]
            )  # Volume Weighted Avg Price

            # Volatility Indicators
            df["atr"] = ta.atr(
                df["high"], df["low"], df["close"], length=14
            )  # Average True Range
            df["bb_width"] = ta.bbands(df["close"], length=20)[
                "BBB_20_2.0"
            ]  # Bollinger Band Width
            df["returns"] = df["close"].pct_change()
            df = df.dropna()

            # 3. Load Model and Predict (unchanged)
            model, scaler, features = joblib.load(MODEL_PATH)
            X = df[features]
            X_scaled = scaler.transform(X)
            predictions = model.predict_proba(X_scaled)
            df["prediction"] = predictions[:, 1]  # Probability of positive return
            # Get latest prediction
            latest_pred = df["prediction"].iloc[-1]

            # 4. Generate Trading Signal (NEW)
            if latest_pred > 0.75:
                signal = "STRONG BUY"
                color = "darkgreen"
                emoji = "🚀"
            elif latest_pred > 0.65:
                signal = "BUY"
                color = "green"
                emoji = "📈"
            elif latest_pred > 0.55:
                signal = "WEAK BUY"
                color = "limegreen"
                emoji = "↗️"
            elif latest_pred < 0.45:
                signal = "SELL"
                color = "red"
                emoji = "📉"
            elif latest_pred < 0.35:
                signal = "STRONG SELL"
                color = "darkred"
                emoji = "⚠️"
            else:
                signal = "NEUTRAL"
                color = "gray"
                emoji = "➖"

            confidence = f"{max(latest_pred, 1-latest_pred)*100:.1f}%"

            # 5. Enhanced Visualization (NEW but keeping original indicators)
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])

            # Price Chart (with original SMA_20)
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(df.index, df["close"], label="Price", color="blue", linewidth=2)
            ax1.plot(
                df.index, df["ema_20"], label="20 EMA", color="orange", linestyle="--"
            )

            # Highlight prediction zones
            ax1.fill_between(
                df.index,
                df["close"],
                where=(df["prediction"] > 0.65),
                color="green",
                alpha=0.1,
                label="Buy Zone",
            )
            ax1.fill_between(
                df.index,
                df["close"],
                where=(df["prediction"] < 0.45),
                color="red",
                alpha=0.1,
                label="Sell Zone",
            )

            ax1.set_title(
                f"{ticker} Price - Current Signal: {emoji} {signal} ({confidence} confidence)",
                fontsize=14,
                fontweight="bold",
                color=color,
            )
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)

            # RSI Chart (unchanged)
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax2.plot(df.index, df["rsi"], label="RSI 14", color="purple")
            ax2.axhline(70, color="red", linestyle="--")
            ax2.axhline(30, color="green", linestyle="--")
            ax2.set_title("RSI Indicator")
            ax2.grid(True, alpha=0.3)

            """
            # MACD Chart (unchanged)
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            ax3.plot(df.index, df["MACD_12_26_9"], label="MACD", color="blue")
            ax3.plot(df.index, df["MACDs_12_26_9"], label="Signal", color="orange")
            ax3.bar(
                df.index,
                df["MACDh_12_26_9"],
                color=np.where(df["MACDh_12_26_9"] > 0, "green", "red"),
                alpha=0.5,
            )
            ax3.set_title("MACD (12,26,9)")
            ax3.legend(loc="upper left")
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            """
            # 6. Display Results (NEW)
            st.pyplot(fig)

            # Signal Summary Card
            st.markdown(
                f"""
            <div style="
                border: 2px solid {color};
                border-radius: 5px;
                padding: 10px;
                text-align: center;
                margin: 10px 0;
                background-color: {color}10;
            ">
                <h2 style="color: {color};">{emoji} {signal}</h2>
                <p style="font-size: 20px;">Confidence: {confidence}</p>
                <p>Current Price: ${df['close'].iloc[-1]:.2f}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Key Metrics Columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "RSI (14)",
                    f"{df['rsi'].iloc[-1]:.1f}",
                    (
                        "Overbought"
                        if df["rsi"].iloc[-1] > 70
                        else "Oversold" if df["rsi"].iloc[-1] < 30 else "Neutral"
                    ),
                )

            """
            with col2:
                st.metric(
                    "MACD",
                    f"{df['MACD_12_26_9'].iloc[-1]:.2f}",
                    (
                        "Bullish"
                        if df["MACD_12_26_9"].iloc[-1] > df["MACDs_12_26_9"].iloc[-1]
                        else "Bearish"
                    ),
                )
            """
            with col3:
                st.metric(
                    "20 SMA vs Price",
                    f"{(df['close'].iloc[-1]/df['ema_20'].iloc[-1]-1)*100:.1f}%",
                    (
                        "Above EMA"
                        if df["close"].iloc[-1] > df["ema_20"].iloc[-1]
                        else "Below EMA"
                    ),
                )

            # Recent df Table (unchanged column names)
            st.subheader("Recent df")
            st.dataframe(
                df[["close", "rsi", "macd", "prediction"]]
                .tail(10)
                .style.format(
                    {
                        "Close": "${:.2f}",
                        "RSI": "{:.1f}",
                        "MACD": "{:.2f}",
                        "Chance of Success": "{:.2%}",
                    }
                )
                .background_gradient(subset=["prediction"], cmap="RdYlGn")
                .highlight_max(subset=["prediction"], color="lightgreen")
                .highlight_min(subset=["prediction"], color="salmon")
            )

        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
