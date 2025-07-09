# ✅ Revamped Stock Predictor Pro App with Enhancements
# Fixes: Model/Scaler Sync, PyTorch Saving, Robust UI, Interval Dropdown, Financial Fundamentals

import time
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import pytz
import seaborn as sns
import streamlit as st
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup
import requests

warnings.filterwarnings("ignore")


# ===============================
# NEURAL NETWORK MODEL
# ===============================
class StockPredictionNN(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.network(x)


# ===============================
# STYLE AND CONFIG
# ===============================
st.set_page_config(page_title="Stock Predictor Pro", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        background: linear-gradient(to bottom right, #f0f2f5, #ffffff);
        border-radius: 8px;
        box-shadow: 0 0 8px rgba(0,0,0,0.05);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ===============================
# LOADING MODEL
# ===============================
@st.cache_resource
def load_model(model_path, input_size):
    model = StockPredictionNN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# ===============================
# FETCH DATA
# ===============================
def fetch_data(ticker, interval="1d"):
    data = yf.download(ticker, period="1y", interval=interval, progress=False)
    data = data.dropna()
    data.reset_index(inplace=True)
    return data


# ===============================
# ADD FUNDAMENTALS
# ===============================
def fetch_fundamentals(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        tables = soup.find_all("table")
        rows = []
        for table in tables:
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) == 2:
                    label, value = tds[0].text, tds[1].text
                    rows.append((label, value))
        return rows[:10]  # return top 10 stats
    except:
        return []


# ===============================
# ENGINEER FEATURES
# ===============================
def engineer_features(df):
    df = df.copy()
    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(20).std()
    df["rsi"] = ta.rsi(df["Close"], length=14)
    df["sma_20"] = ta.sma(df["Close"], length=20)
    df = df.dropna()
    return df


# ===============================
# PREDICT
# ===============================
def predict(model, scaler, df, features):
    X = scaler.transform(df[features])
    X_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        y_pred = model(X_tensor).numpy().flatten()
    return y_pred


# ===============================
# MAIN APP
# ===============================
st.title("📈 Stock Predictor Pro - Enhanced Edition")

# Sidebar Inputs
with st.sidebar:
    st.header("🛠 Settings")
    ticker = st.text_input("Enter Ticker", "AAPL")
    interval = st.selectbox("Interval", ["1d", "1h", "15m"])
    show_fundamentals = st.checkbox("Show Company Fundamentals", value=True)
    st.markdown("---")
    st.write("💡 Pro Tips:")
    st.caption("- RSI > 70 = Overbought | < 30 = Oversold")
    st.caption("- High volatility = high risk & reward")
    st.caption("- SMA crossovers often signal trends")

# Load model artifacts
FEATURES = ["returns", "volatility", "rsi"]
scaler = StandardScaler()
data = fetch_data(ticker, interval)
data = engineer_features(data)
scaler.fit(data[FEATURES])
model = load_model("model_nn.pt", len(FEATURES))

# Predict
predictions = predict(model, scaler, data, FEATURES)
data["prediction"] = predictions

# Show chart
st.subheader(f"📉 {ticker} Stock Price with AI Prediction")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(data["Date"], data["Close"], label="Actual Price", linewidth=2)
ax.plot(
    data["Date"],
    data["Close"] * (1 + data["prediction"]),
    label="Predicted",
    linestyle="--",
)
ax.set_title(f"{ticker} Price vs Predicted Trend")
ax.legend()
st.pyplot(fig)

# Show fundamentals
if show_fundamentals:
    st.subheader(f"📊 Top Financials: {ticker}")
    fundamentals = fetch_fundamentals(ticker)
    for label, value in fundamentals:
        st.write(f"**{label}**: {value}")

# Metrics
st.subheader("📌 Latest Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Latest Price", f"${data['Close'].iloc[-1]:.2f}")
col2.metric("Latest RSI", f"{data['rsi'].iloc[-1]:.2f}")
col3.metric("Volatility", f"{data['volatility'].iloc[-1]:.4f}")

st.markdown("---")
st.markdown(
    """
<p style='text-align: center; color: gray;'>⚠️ This is not financial advice. Always DYOR before investing.</p>
""",
    unsafe_allow_html=True,
)
