import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

# Import from the main module
from stock_predictor import (
    StockDataLoader,
    StockPredictor,
    TechnicalIndicators,
    create_output_directory,
    save_predictions,
)

# Configure Streamlit
st.set_page_config(
    page_title="Stock Market Prediction Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5a8a;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    return {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corp.",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corp.",
        "META": "Meta Platforms Inc.",
        "NFLX": "Netflix Inc.",
        "SPY": "SPDR S&P 500 ETF",
        "QQQ": "Invesco QQQ ETF",
    }


def create_price_chart(
    data: pd.DataFrame, ticker: str, selected_indicators: List[str]
) -> go.Figure:
    """Create an interactive price chart with selected indicators"""
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            f"{ticker} Price & Indicators",
            "Volume",
            "Technical Indicators",
        ],
        vertical_spacing=0.1,
        row_heights=[0.6, 0.2, 0.2],
    )

    # Price chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # Add selected indicators to price chart
    for indicator in selected_indicators:
        if indicator in data.columns:
            if indicator.startswith("BB_"):
                # Bollinger Bands
                if indicator == "BB_Upper":
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data["BB_Upper"],
                            name="BB Upper",
                            line=dict(dash="dash"),
                        ),
                        row=1,
                        col=1,
                    )
                elif indicator == "BB_Lower":
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data["BB_Lower"],
                            name="BB Lower",
                            line=dict(dash="dash"),
                        ),
                        row=1,
                        col=1,
                    )
                elif indicator == "BB_Middle":
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data["BB_Middle"],
                            name="BB Middle",
                            line=dict(dash="dot"),
                        ),
                        row=1,
                        col=1,
                    )
            elif indicator.startswith("SMA_") or indicator.startswith("EMA_"):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[indicator],
                        name=indicator,
                        line=dict(width=2),
                    ),
                    row=1,
                    col=1,
                )

    # Volume chart
    fig.add_trace(
        go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color="lightblue"),
        row=2,
        col=1,
    )

    # Technical indicators subplot
    oscillators = ["RSI", "Stoch_K", "Stoch_D"]
    for osc in oscillators:
        if osc in selected_indicators and osc in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[osc], name=osc), row=3, col=1)

    # Add RSI levels
    if "RSI" in selected_indicators:
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        xaxis_title="Date",
        height=800,
        showlegend=True,
        template="plotly_white",
    )

    return fig


def create_prediction_visualization(
    data: pd.DataFrame, prediction: float, confidence: float, prediction_days: int
) -> go.Figure:
    """Create visualization showing prediction vs historical data"""
    fig = go.Figure()

    # Historical prices
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Historical Price",
            line=dict(color="blue", width=2),
        )
    )

    # Prediction point
    last_price = data["Close"].iloc[-1]
    future_date = data.index[-1] + pd.Timedelta(days=prediction_days)
    predicted_price = last_price * (1 + prediction / 100)

    # Add prediction point
    fig.add_trace(
        go.Scatter(
            x=[future_date],
            y=[predicted_price],
            mode="markers",
            name=f"Prediction ({prediction_days}d)",
            marker=dict(size=15, color="red", symbol="star"),
            text=[
                f"Predicted: ${predicted_price:.2f}<br>Return: {prediction:.2f}%<br>Confidence: {confidence:.2f}"
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Add prediction line
    fig.add_trace(
        go.Scatter(
            x=[data.index[-1], future_date],
            y=[last_price, predicted_price],
            mode="lines",
            name="Prediction Line",
            line=dict(color="red", dash="dash", width=3),
        )
    )

    # Add confidence interval
    error_margin = abs(predicted_price - last_price) * (1 - confidence) * 0.5
    fig.add_trace(
        go.Scatter(
            x=[future_date, future_date],
            y=[predicted_price - error_margin, predicted_price + error_margin],
            mode="lines",
            name="Confidence Interval",
            line=dict(color="rgba(255,0,0,0.3)", width=8),
            fill="tonexty",
        )
    )

    fig.update_layout(
        title=f"Stock Price Prediction - {prediction_days} Day Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        height=500,
    )

    return fig


def create_indicator_correlation_heatmap(data: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap of technical indicators"""
    # Select numeric columns only
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    correlation_matrix = data[numeric_cols].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=correlation_matrix.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 8},
        )
    )

    fig.update_layout(
        title="Technical Indicators Correlation Matrix",
        height=600,
        template="plotly_white",
    )

    return fig


def display_model_performance(train_losses: List[float], val_losses: List[float]):
    """Display model training performance"""
    fig = go.Figure()

    epochs = list(range(len(train_losses)))

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_losses,
            mode="lines",
            name="Training Loss",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=val_losses,
            mode="lines",
            name="Validation Loss",
            line=dict(color="red"),
        )
    )

    fig.update_layout(
        title="Model Training Performance",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
        height=400,
    )

    return fig


def main():
    """Main Streamlit application"""

    # Header
    st.markdown(
        '<div class="main-header">🤖 AI Stock Market Prediction Tool</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Initialize session state
    if "predictor" not in st.session_state:
        st.session_state.predictor = StockPredictor()
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    if "training_data" not in st.session_state:
        st.session_state.training_data = None

    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")

    # Stock selection
    sample_stocks = load_sample_data()
    selected_ticker = st.sidebar.selectbox(
        "Select Stock Ticker",
        options=list(sample_stocks.keys()),
        format_func=lambda x: f"{x} - {sample_stocks[x]}",
    )

    # Custom ticker input
    custom_ticker = st.sidebar.text_input(
        "Or enter custom ticker:", placeholder="e.g., AAPL"
    )

    if custom_ticker:
        ticker = custom_ticker.upper()
    else:
        ticker = selected_ticker

    # Prediction parameters
    st.sidebar.subheader("📊 Prediction Settings")
    prediction_days = st.sidebar.slider(
        "Prediction Period (days)",
        min_value=1,
        max_value=30,
        value=5,
        help="Number of days to predict into the future",
    )

    # Model parameters
    st.sidebar.subheader("🧠 Model Settings")
    use_lstm = st.sidebar.checkbox("Use LSTM layers", value=True)
    hidden_sizes = st.sidebar.multiselect(
        "Hidden layer sizes", options=[64, 128, 256, 512], default=[256, 128, 64]
    )
    dropout_rate = st.sidebar.slider(
        "Dropout rate", min_value=0.1, max_value=0.5, value=0.3, step=0.1
    )

    # Training parameters
    st.sidebar.subheader("⚙️ Training Settings")
    epochs = st.sidebar.slider("Training epochs", 50, 200, 100)
    learning_rate = st.sidebar.select_slider(
        "Learning rate", options=[0.0001, 0.001, 0.01, 0.1], value=0.001
    )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header(f"📈 Analysis for {ticker}")

        # Data loading and training section
        if st.button("🔄 Load Data & Train Model", type="primary"):
            try:
                with st.spinner("Loading data and training model..."):
                    # Update model parameters
                    st.session_state.predictor.model_params = {
                        "hidden_sizes": sorted(hidden_sizes, reverse=True),
                        "dropout_rate": dropout_rate,
                        "use_lstm": use_lstm,
                    }

                    # Prepare data
                    train_loader, val_loader, features = (
                        st.session_state.predictor.prepare_data(ticker, prediction_days)
                    )

                    # Train model
                    train_losses, val_losses = st.session_state.predictor.train_model(
                        train_loader,
                        val_loader,
                        input_size=features.shape[1],
                        epochs=epochs,
                        learning_rate=learning_rate,
                    )

                    st.session_state.model_trained = True
                    st.session_state.training_data = {
                        "features": features,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                    }

                    # Save model
                    st.session_state.predictor.save_model()

                    st.success("✅ Model trained successfully!")

            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")
                st.exception(e)

    with col2:
        # Model status
        if st.session_state.model_trained:
            st.success("🤖 Model Ready")
        else:
            st.warning("⚠️ Model Not Trained")

        # Quick stats
        if st.session_state.training_data:
            st.metric(
                "Features Used", st.session_state.training_data["features"].shape[1]
            )
            st.metric(
                "Training Samples", len(st.session_state.training_data["features"])
            )

    # Display results if model is trained
    if st.session_state.model_trained and st.session_state.training_data:

        # Load fresh data for prediction
        try:
            data_loader = StockDataLoader([ticker])
            recent_data = data_loader.fetch_data(ticker)

            if not recent_data.empty:
                recent_data = data_loader.add_technical_indicators(recent_data)

                # Make prediction
                prediction, confidence = st.session_state.predictor.predict(
                    recent_data, prediction_days
                )

                # Display prediction
                st.markdown("## 🎯 Prediction Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(
                        f"""
                    <div class="prediction-box">
                        <h3>Predicted Return</h3>
                        <h2>{prediction:.2f}%</h2>
                        <p>over {prediction_days} days</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        f"""
                    <div class="prediction-box">
                        <h3>Confidence</h3>
                        <h2>{confidence:.1%}</h2>
                        <p>model certainty</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col3:
                    direction = "📈 Long" if prediction > 0 else "📉 Short"
                    st.markdown(
                        f"""
                    <div class="prediction-box">
                        <h3>Recommendation</h3>
                        <h2>{direction}</h2>
                        <p>trade direction</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Save prediction
                save_predictions(ticker, prediction, confidence, prediction_days)

                # Prediction visualization
                st.markdown("## 📊 Prediction Visualization")
                pred_fig = create_prediction_visualization(
                    recent_data, prediction, confidence, prediction_days
                )
                st.plotly_chart(pred_fig, use_container_width=True)

                # Technical analysis section
                st.markdown("## 📈 Technical Analysis")

                # Indicator selection
                all_indicators = [
                    "SMA_10",
                    "SMA_20",
                    "SMA_50",
                    "EMA_12",
                    "EMA_26",
                    "RSI",
                    "MACD",
                    "MACD_Signal",
                    "MACD_Histogram",
                    "BB_Upper",
                    "BB_Middle",
                    "BB_Lower",
                    "Stoch_K",
                    "Stoch_D",
                    "ATR",
                    "Volume_Ratio",
                    "Volatility",
                ]

                selected_indicators = st.multiselect(
                    "Select indicators to display:",
                    options=all_indicators,
                    default=["SMA_20", "RSI", "MACD", "BB_Upper", "BB_Lower"],
                )

                if selected_indicators:
                    # Create and display chart
                    chart_fig = create_price_chart(
                        recent_data, ticker, selected_indicators
                    )
                    st.plotly_chart(chart_fig, use_container_width=True)

                # Model performance
                st.markdown("## 📊 Model Performance")

                col1, col2 = st.columns(2)

                with col1:
                    # Training performance
                    perf_fig = display_model_performance(
                        st.session_state.training_data["train_losses"],
                        st.session_state.training_data["val_losses"],
                    )
                    st.plotly_chart(perf_fig, use_container_width=True)

                with col2:
                    # Correlation heatmap
                    corr_fig = create_indicator_correlation_heatmap(recent_data)
                    st.plotly_chart(corr_fig, use_container_width=True)

                # Data table
                st.markdown("## 📋 Recent Data")

                display_columns = [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                ] + selected_indicators
                available_columns = [
                    col for col in display_columns if col in recent_data.columns
                ]

                st.dataframe(
                    recent_data[available_columns].tail(10).round(4),
                    use_container_width=True,
                )

                # Download section
                st.markdown("## 💾 Export Data")

                col1, col2 = st.columns(2)

                with col1:
                    # Download predictions
                    if os.path.exists("./output/predictions.csv"):
                        predictions_df = pd.read_csv("./output/predictions.csv")
                        csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name=f"{ticker}_predictions.csv",
                            mime="text/csv",
                        )

                with col2:
                    # Download model
                    if os.path.exists("model.pt"):
                        with open("model.pt", "rb") as f:
                            st.download_button(
                                label="🤖 Download Model",
                                data=f.read(),
                                file_name=f"{ticker}_model.pt",
                                mime="application/octet-stream",
                            )

        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. 
        Trading decisions should not be based solely on these predictions. 
        Always consult with a financial advisor before making investment decisions.</p>
        <p>Built with ❤️ using Streamlit, PyTorch, and advanced deep learning techniques.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
