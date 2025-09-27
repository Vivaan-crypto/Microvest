# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import torch
from model import lstm_model
from preprocessing import get_historical_data
import numpy as np

# Load your model
model = lstm_model()
model.load_state_dict(torch.load('model/model.pth'))  # Make sure to have this file

st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide"
)

# Define the UI style
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #121212;
    }
    .sidebar-content {
        background-color: #1e1e1e;
    }
    .stTextInput>div>div{
        background-color: #2d2d2d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar controls
with st.sidebar:
    st.title("📊 Stock Market Predictor")
    stock_symbol = st.text_input("Stock Symbol", "AAPL")
    start_date = st.date_input("Start Date", pd.Timestamp("2020-01-01"))
    end_date = st.date_input("End Date", pd.Timestamp.today())
    model_type = st.selectbox(" Prediction Model", ["LSTM", "GRU", "Transformer"])
    st.button("Run Prediction", on_click=lambda: (st.session_state.run_id + 1))

    # Main content
    st.title("📈 Stock Market Analysis Dashboard")

    # Get historical data
    if 'data' not in st.session_state or st.session_state.run_id != st.session_state.get('current_run_id', 1):
        df = get_historical_data(stock_symbol, start_date, end_date)
        st.session_state.data = df
        st.session_state.current_run_id = st.session_state.run_id

    df = st.session_state.data

    # Plot candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    fig.update_layout(
        title=f"{stock_symbol} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        font=dict(color="#e0e0e0")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show indicators
    st.subheader("Technical Indicators")
    indicators = df[['rsi', 'macd', 'macd_signal']]
    st.line_chart(indicators)

    # Make prediction
    if st.sidebar.button("Run Prediction"):
        # Preprocess data and make prediction
        # This is a placeholder - implement your actual prediction logic
        latest_price = df['Close'].iloc[-1]
        prediction = latest_price * (1 + np.random.uniform(-0.02, 0.02))  # Random prediction

        # Calculate confidence (example: based on prediction difference)
        confidence = abs(1 - abs((prediction - latest_price) / latest_price))

        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        col1.metric("Latest Price", f"${latest_price:.2f}")
        col2.metric("Predicted Price", f"${prediction:.2f}",
                    f"{(prediction - latest_price):.2f}" if prediction > latest_price else f"-{(latest_price - prediction):.2f}")

        # Confidence visualization
        st.subheader("Model Confidence")
        fig_conf = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Level"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "teal"},
                'bgcolor': "gray",
                'borderwidth': 2,
                'bordercolor': "white"
            }))
        fig_conf.update_layout(template="plotly_dark")
        st.plotly_chart(fig_conf)

    # Show raw data
    if st.checkbox("Show Raw Data"):
        st.dataframe(df.style.format(precision=2))