from datetime import datetime, timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import seaborn as sns
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# ======================================
# CONFIGURATION & STYLING
# ======================================
# Visual settings
sns.set_style("whitegrid")
sns.set_palette("deep")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# Default parameters
TICKER = "AAPL"
DAYS = 90
MODEL_PATH = r"C:\Users\shahv\OneDrive\Documents\GitHub\Microvest\model\model.pkl"
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
            df.columns = df.columns.droplevel(1)
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

            # Calculate directional confidence (0-100%)
            if latest_pred > 0.5:
                confidence = (latest_pred - 0.5) * 2  # Buy confidence
            else:
                confidence = (0.5 - latest_pred) * 2  # Sell confidence

            confidence_str = f"{confidence*100:.1f}%"

            # --------------------------
            # 5. VISUALIZATION
            # --------------------------
            # Create figure with subplots
            fig = plt.figure(figsize=(14, 12), facecolor="white")
            gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
            ax4 = fig.add_subplot(gs[3])

            # Main price chart
            sns.lineplot(
                data=df,
                x=df.index,
                y="close",
                ax=ax1,
                color="#1f77b4",
                linewidth=2,
                label="Price",
            )
            sns.lineplot(
                data=df,
                x=df.index,
                y="ema_200",
                ax=ax1,
                color="#ff7f0e",
                linestyle="--",
                label="200 EMA",
            )

            # Add prediction zones
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
                where=(df["prediction"] < 0.35),
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

            # RSI subplot
            sns.lineplot(data=df, x=df.index, y="rsi", ax=ax2, color="#9467bd")
            ax2.axhline(70, color="red", linestyle=":", alpha=0.7)
            ax2.axhline(30, color="green", linestyle=":", alpha=0.7)
            ax2.fill_between(df.index, 70, 30, color="gray", alpha=0.1)
            ax2.set_title("RSI (14-period)", fontsize=10)

            # MACD subplot
            sns.lineplot(
                data=df,
                x=df.index,
                y="macd",
                ax=ax3,
                color="#1f77b4",
                label="MACD",
            )
            sns.lineplot(
                data=df,
                x=df.index,
                y="macd_signal",
                ax=ax3,
                color="#ff7f0e",
                label="Signal",
            )
            ax3.bar(
                df.index,
                df["macd"] - df["macd_signal"],
                color=np.where((df["macd"] - df["macd_signal"]) > 0, "green", "red"),
                alpha=0.3,
                width=0.01,
            )
            ax3.axhline(0, color="black", linestyle="-", alpha=0.5)
            ax3.set_title("MACD (12,26,9)", fontsize=10)
            ax3.legend(loc="upper left")

            # Prediction probability subplot
            sns.lineplot(data=df, x=df.index, y="prediction", ax=ax4, color=color)
            ax4.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
            ax4.set_ylim(0, 1)
            ax4.set_title("Model Prediction Probability", fontsize=10)
            ax4.fill_between(
                df.index,
                df["prediction"],
                where=(df["prediction"] > 0.5),
                color="green",
                alpha=0.1,
            )
            ax4.fill_between(
                df.index,
                df["prediction"],
                where=(df["prediction"] < 0.5),
                color="red",
                alpha=0.1,
            )

            plt.tight_layout()
            st.pyplot(fig)

            # --------------------------
            # 6. DASHBOARD METRICS
            # --------------------------
            # Signal card
            st.markdown(
                f"""
            <div style="
                border-left: 6px solid {color};
                border-radius: 4px;
                padding: 16px;
                background-color: #f8f9fa;
                margin: 16px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 32px;">{emoji}</span>
                    <div>
                        <h2 style="margin: 0; color: {color};">{signal} SIGNAL</h2>
                        <p style="margin: 4px 0; font-size: 18px;">Confidence: <strong>{confidence_str}</strong></p>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 12px;">
                    <div>Current Price: <strong>${df['close'].iloc[-1]:.2f}</strong></div>
                    <div>ATR (Volatility): <strong>{df['atr'].iloc[-1]:.2f}</strong></div>
                    <div>RSI: <strong>{df['rsi'].iloc[-1]:.1f}</strong></div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Key metrics columns
            st.subheader("Technical Snapshot")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Prediction Strength",
                    f"{latest_pred:.1%}",
                    delta=(
                        f"{(latest_pred-0.5)*100:.1f}% from neutral"
                        if latest_pred != 0.5
                        else "0%"
                    ),
                )

            with col2:
                st.metric(
                    "Volatility (ATR)",
                    f"{df['atr'].iloc[-1]:.2f}",
                    delta=f"{df['atr'].pct_change().iloc[-1]:.1%} change",
                )

            with col3:
                st.metric(
                    "RSI (14)",
                    f"{df['rsi'].iloc[-1]:.1f}",
                    (
                        "Overbought"
                        if df["rsi"].iloc[-1] > 70
                        else "Oversold" if df["rsi"].iloc[-1] < 30 else "Neutral"
                    ),
                )

            with col4:
                macd_diff = df["macd"].iloc[-1] - df["macd_signal"].iloc[-1]
                st.metric(
                    "MACD Diff",
                    f"{macd_diff:.2f}",
                    "Bullish" if macd_diff > 0 else "Bearish",
                )

            # Recent predictions
            st.subheader("Recent Predictions")
            recent_data = df[["close", "rsi", "macd", "prediction"]].tail(24)

            # Prediction heatmap
            fig_heat, ax_heat = plt.subplots(figsize=(12, 1))
            sns.heatmap(
                recent_data[["prediction"]].T,
                annot=True,
                fmt=".0%",
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                cbar=False,
                ax=ax_heat,
                annot_kws={"size": 8},
            )
            ax_heat.set_xticklabels(
                [d.strftime("%m/%d %H:%M") for d in recent_data.index], rotation=45
            )
            ax_heat.set_title("Prediction Probability Over Last 24 Periods")
            st.pyplot(fig_heat)

            # Raw data table
            if show_raw:
                st.subheader("Detailed Data")
                st.dataframe(
                    recent_data.style.format(
                        {
                            "close": "${:.2f}",
                            "rsi": "{:.1f}",
                            "macd": "{:.2f}",
                            "prediction": "{:.1%}",
                        }
                    ).background_gradient(
                        subset=["prediction"], cmap="RdYlGn", vmin=0, vmax=1
                    )
                )

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.exception(e)

# Add footer
st.markdown("---")
