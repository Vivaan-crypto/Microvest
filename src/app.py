import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yaml
import torch
import pandas_ta as ta
from datetime import datetime, timedelta
from pathlib import Path
from plotly.subplots import make_subplots
from model import AttentionLSTM

# Page configuration
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


class StockPredictor:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.tickers = self.config['data']['tickers']
        self.model_path = self.config['paths']['combined_model_path']
        self.model = None
        self.scaler = None
        self.ticker_encodings = {}
        self.load_model()

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            st.error(f"Config file not found: {config_path}")
            return {}

    def load_model(self):
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            self.scaler = checkpoint["scaler"]
            self.ticker_encodings = checkpoint["ticker_encodings"]
            self.tickers = checkpoint["tickers"]
            self.feature_columns = checkpoint["feature_columns"]

            hyperparams = checkpoint["hyperparameters"]
            self.model = AttentionLSTM(**hyperparams)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.model = None

    def add_technical_indicators(self, df):
        df["price_change"] = df["Close"].pct_change()
        volatility_window = self.config['data']['volatility_window']
        volume_ratio_window = self.config['data']['volume_ratio_window']

        df["volatility"] = df["price_change"].rolling(volatility_window).std()
        df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(volume_ratio_window).mean()

        # Add indicators (Bollinger Bands removed)
        df.ta.sma(length=self.config['data']['sma_period'], append=True)
        df.ta.ema(length=self.config['data']['ema_period'], append=True)
        df.ta.rsi(length=self.config['data']['rsi_period'], append=True)
        df.ta.macd(append=True)
        df.ta.obv(append=True)

        return df

    def prepare_features(self, df, ticker):
        try:
            df = self.add_technical_indicators(df)
            feature_data = pd.DataFrame()

            # Add base features
            for col in self.feature_columns:
                if not col.startswith('ticker_'):
                    if col in df.columns:
                        feature_data[col] = df[col]
                    else:
                        feature_data[col] = 0

            # Add ticker encodings
            for t in self.tickers:
                feature_data[f"ticker_{t}"] = 1 if t == ticker else 0

            return feature_data[self.feature_columns].dropna()
        except Exception as e:
            st.error(f"Error preparing features: {e}")
            return None

    def predict(self, ticker, data):
        try:
            if self.model is None:
                return {"error": "Model not loaded"}

            features = self.prepare_features(data, ticker)
            if features is None:
                return {"error": "Feature preparation failed"}

            scaled_features = self.scaler.transform(features)
            sequence_length = self.config['model']['sequence_length']
            sequence = scaled_features[-sequence_length:]

            with torch.no_grad():
                input_tensor = torch.FloatTensor(sequence).unsqueeze(0)
                prediction = self.model(input_tensor).squeeze().item()

            confidence = min(abs(prediction) * self.config['prediction']['confidence_multiplier'], 1.0)
            current_price = data["Close"].iloc[-1]
            predicted_price = current_price * (1 + prediction)

            if prediction > 0 and confidence >= self.config['prediction']['buy_threshold']:
                signal = "BUY"
            elif prediction < 0 and confidence >= self.config['prediction']['sell_threshold']:
                signal = "SELL"
            else:
                signal = "HOLD"

            return {
                "predicted_return": prediction,
                "predicted_price": predicted_price,
                "confidence": confidence,
                "signal": signal,
                "current_price": current_price,
            }
        except Exception as e:
            return {"error": f"Prediction error: {e}"}


@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None


def create_price_chart(data, ticker):
    """Create candlestick chart with volume"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker} Price", "Volume"),
        row_heights=[0.7, 0.3]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price"
    ), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(
        x=data.index,
        y=data["Volume"],
        name="Volume",
        marker_color="rgba(158,202,225,0.6)"
    ), row=2, col=1)

    # Moving averages
    if len(data) >= 50:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["Close"].rolling(20).mean(),
            line=dict(color="orange", width=2),
            name="MA20"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["Close"].rolling(50).mean(),
            line=dict(color="red", width=2),
            name="MA50"
        ), row=1, col=1)

    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        height=600,
        xaxis_rangeslider_visible=False
    )

    return fig


def create_technical_chart(data, ticker):
    """Create technical indicators chart"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("RSI", "MACD", "Bollinger Bands", "Volume"),
        vertical_spacing=0.08
    )

    # Add technical indicators
    data_ta = data.copy()
    data_ta.ta.rsi(length=14, append=True)
    data_ta.ta.macd(append=True)
    data_ta.ta.bbands(append=True)

    # RSI
    if "RSI_14" in data_ta.columns:
        fig.add_trace(go.Scatter(
            x=data_ta.index, y=data_ta["RSI_14"], name="RSI"
        ), row=1, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

    # MACD
    if "MACD_12_26_9" in data_ta.columns:
        fig.add_trace(go.Scatter(
            x=data_ta.index, y=data_ta["MACD_12_26_9"], name="MACD"
        ), row=1, col=2)
        if "MACDs_12_26_9" in data_ta.columns:
            fig.add_trace(go.Scatter(
                x=data_ta.index, y=data_ta["MACDs_12_26_9"], name="Signal"
            ), row=1, col=2)

    # Bollinger Bands
    bb_cols = ["BBU_20_2.0", "BBM_20_2.0", "BBL_20_2.0"]
    if all(col in data_ta.columns for col in bb_cols):
        fig.add_trace(go.Scatter(
            x=data_ta.index, y=data_ta["BBU_20_2.0"], name="BB Upper"
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=data_ta.index, y=data_ta["BBM_20_2.0"], name="BB Middle"
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=data_ta.index, y=data_ta["BBL_20_2.0"], name="BB Lower"
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=data_ta.index, y=data_ta["Close"], name="Close"
        ), row=2, col=1)

    # Volume
    fig.add_trace(go.Bar(
        x=data_ta.index, y=data_ta["Volume"], name="Volume"
    ), row=2, col=2)

    fig.update_layout(height=600, title_text="Technical Indicators")
    return fig


def create_confidence_gauge(confidence):
    """Create confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={"text": "Confidence Level"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": (
                "darkgreen" if confidence >= 0.8
                else "orange" if confidence >= 0.6
                else "red"
            )},
            "steps": [
                {"range": [0, 60], "color": "lightgray"},
                {"range": [60, 80], "color": "gray"},
                {"range": [80, 100], "color": "darkgray"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


def get_confidence_style(confidence):
    """Get confidence styling"""
    if confidence >= 0.8:
        return "confidence-high", "High"
    elif confidence >= 0.6:
        return "confidence-medium", "Medium"
    else:
        return "confidence-low", "Low"


def main():
    st.markdown('<h1 class="main-header">📈 Stock Prediction Dashboard</h1>', unsafe_allow_html=True)

    # Initialize predictor
    predictor = StockPredictor(config_path="C:/Users/shahv/OneDrive/Documents/GitHub/Microvest/src/config.yaml")

    if predictor.model is None:
        st.error("❌ Model not loaded. Please check if the model file exists and configuration is correct.")
        st.stop()

    # Sidebar configuration
    st.sidebar.header("📊 Configuration")

    # Stock selection
    selected_ticker = st.sidebar.selectbox(
        "🏢 Select Stock Ticker",
        predictor.tickers,
        index=0
    )

    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    period_options = predictor.config['dashboard']['period_options']

    period_choice = st.sidebar.radio(
        "Select Period",
        list(period_options.keys()),
        index=3
    )

    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_options[period_choice])

    # Custom date range option
    use_custom_dates = st.sidebar.checkbox("Use Custom Date Range")
    if use_custom_dates:
        start_date = st.sidebar.date_input("Start Date", start_date, max_value=end_date)
        end_date = st.sidebar.date_input("End Date", end_date, min_value=start_date, max_value=datetime.now())

    # Analyze button
    if st.sidebar.button("📊 Analyze Stock", type="primary"):
        with st.spinner(f"Fetching data for {selected_ticker}..."):
            data = fetch_stock_data(selected_ticker, start_date, end_date)
            if data is not None and len(data) > 0:
                st.session_state.stock_data = data
                st.session_state.current_ticker = selected_ticker

    # Main content
    if hasattr(st.session_state, 'stock_data') and st.session_state.stock_data is not None:
        data = st.session_state.stock_data
        ticker = st.session_state.current_ticker

        # Current metrics
        current_price = data["Close"].iloc[-1]
        prev_price = data["Close"].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}",
                      f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
        with col2:
            volume_change = 0
            if len(data) > 1:
                volume_change = (
                        (data["Volume"].iloc[-1] - data["Volume"].iloc[-2])
                        / data["Volume"].iloc[-2] * 100
                )
            st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}", f"{volume_change:+.1f}%")
        with col3:
            st.metric("52W High", f"${data['High'].rolling(252).max().iloc[-1]:.2f}")
        with col4:
            st.metric("52W Low", f"${data['Low'].rolling(252).min().iloc[-1]:.2f}")

        # Prediction section
        st.markdown("### 🔮 Prediction")

        prediction = predictor.predict(ticker, data)
        if "error" not in prediction:
            col1, col2 = st.columns([2, 1])

            with col1:
                conf_style, conf_text = get_confidence_style(prediction["confidence"])

                st.markdown(f"""
                <div class="prediction-box">
                    <h3>🎯 Prediction Results</h3>
                    <p><strong>Predicted Return:</strong> {prediction["predicted_return"]:.2%}</p>
                    <p><strong>Predicted Price:</strong> ${prediction["predicted_price"]:.2f}</p>
                    <p><strong>Signal:</strong> {prediction["signal"]}</p>
                    <p><strong>Confidence:</strong> <span class="{conf_style}">{prediction["confidence"]:.1%} ({conf_text})</span></p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.plotly_chart(
                    create_confidence_gauge(prediction["confidence"]),
                    use_container_width=True
                )
        else:
            st.error(f"Prediction error: {prediction['error']}")

        # Charts
        st.markdown("### 📊 Price Chart")
        st.plotly_chart(create_price_chart(data, ticker), use_container_width=True)

        st.markdown("### 📈 Technical Indicators")
        st.plotly_chart(create_technical_chart(data, ticker), use_container_width=True)

        # Recent data table
        st.markdown("### 📋 Recent Data")
        st.dataframe(data.tail(10).round(2))

        # Additional analysis
        with st.expander("📊 Additional Analysis"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 Price Statistics")
                st.write(f"**Average Price (Period):** ${data['Close'].mean():.2f}")
                st.write(f"**Volatility:** {data['Close'].pct_change().std() * 100:.2f}%")
                max_drawdown = ((data["Close"].cummax() - data["Close"]) / data["Close"].cummax()).max() * 100
                st.write(f"**Max Drawdown:** {max_drawdown:.2f}%")

            with col2:
                st.subheader("📊 Volume Statistics")
                st.write(f"**Average Volume:** {data['Volume'].mean():,.0f}")
                volume_trend = ("📈" if data["Volume"].iloc[-5:].mean() > data["Volume"].iloc[-10:-5].mean() else "📉")
                st.write(f"**Volume Trend:** {volume_trend}")
                st.write(f"**Highest Volume:** {data['Volume'].max():,.0f}")

    else:
        st.info("👆 Please select a ticker and date range from the sidebar to get started!")

        # Show available tickers
        st.markdown("### 🏢 Available Tickers")
        st.write("The model supports predictions for the following stocks:")
        cols = st.columns(5)
        for i, ticker in enumerate(predictor.tickers):
            with cols[i % 5]:
                st.write(f"• {ticker}")


if __name__ == "__main__":
    main()