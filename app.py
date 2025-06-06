import os
from datetime import datetime, timedelta

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
from mplfinance.original_flavor import candlestick_ohlc

load_dotenv()

# Alpaca configuration
ALPACA_API_KEY = "PKI3PV5XYFEHV9DGHJP6"
ALPACA_SECRET_KEY = "3M8gQOcLFXLqmfWALhzJGBs6aCNPaPhWyz2MpnBh"
client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
MODEL_PATH = r"C:\Users\shahv\OneDrive\Documents\GitHub\Microvest\model\model.pkl"


def get_1h_data(ticker, days=30):
    """Consistent with model.py implementation"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Hour,
            start=start_date,
            end=end_date,
        )
        bars = client.get_stock_bars(request).df
        if isinstance(bars.index, pd.MultiIndex):
            return bars.xs(ticker, level=0)
        return bars
    except Exception:
        # Fallback to minute resampling
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=start_date,
            end=end_date,
        )
        minute_bars = client.get_stock_bars(request).df
        if isinstance(minute_bars.index, pd.MultiIndex):
            minute_bars = minute_bars.xs(ticker, level=0)
        return (
            minute_bars.resample("1H")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )


def add_features(df):
    """Consistent with model.py"""
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.sma(length=20, append=True)
    df["Returns"] = df["Close"].pct_change()
    return df.dropna()


# Streamlit app setup
st.set_page_config(layout="wide", page_title="Alpaca Stock Analyzer")

st.title("📈 Alpaca Stock Analysis (1-Hour Intervals)")
ticker = st.text_input("Enter stock ticker:", "AAPL")

if st.button("Analyze"):
    with st.spinner("Fetching data and analyzing..."):
        try:
            # Get data and features
            df = get_1h_data(ticker)
            df = add_features(df)

            # Load model
            model, scaler, features = joblib.load(MODEL_PATH)

            # Rest of your Streamlit app logic...
            # (Same visualization and prediction code as before)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error("Please check your Alpaca credentials and internet connection")
