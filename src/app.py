"""
Streamlit dashboard – 100 lines, no fat.

Key concepts shown:
  • session-state to cache heavy downloads
  • plotly for interactive charts
  • human-readable signal logic
"""

import streamlit as st, yfinance as yf, pandas_ta as ta, plotly.graph_objects as go
from datetime import datetime, timedelta
import torch, yaml
from model import AttentionLSTM
from data_module import StockDataModule  # for scaling

st.set_page_config(page_title="📈 StockOracle", layout="wide")
st.title("📈 StockOracle – minimal & teachable")

# --- load config once ---
@st.cache_resource
def load_cfg():
    return yaml.safe_load(open("config.yaml"))
cfg = load_cfg()
tickers = cfg['data']['tickers']

# --- sidebar ---
ticker = st.sidebar.selectbox("Choose ticker", tickers)
period = st.sidebar.select_slider("Look-back", options=["1M","3M","6M","1Y","2Y"])
start = datetime.today() - timedelta(days={"1M":30,"3M":90,"6M":180,"1Y":365,"2Y":730}[period])

# --- download ---
@st.cache_data
def get_data(ticker): return yf.download(ticker, start=start)
price = get_data(ticker)

# --- inference ---
@st.cache_resource
def load_model():
    model = AttentionLSTM(argparse.Namespace(**cfg))
    model.load_state_dict(torch.load(cfg['paths']['model'], map_location='cpu'))
    model.eval()
    return model
model = load_model()

# prepare last 30 days & scale
feat_cols = ['Close','Volume','SMA_20','EMA_12','RSI_14','MACD_12_26_9']
df_ta = price.ta.sma(length=20).ta.ema(length=12).ta.rsi(length=14).ta.macd().dropna()
X = df_ta[feat_cols].iloc[-30:].values
X = (X - X.mean(axis=0)) / (X.std(axis=0)+1e-8)  # quick z-score on the fly
X = torch.FloatTensor(X).unsqueeze(0)
with torch.no_grad():
    pred_ret = model(X).item()

# --- present ---
col1, col2 = st.columns([2,1])
with col1:
    fig = go.Figure(data=[go.Candlestick(x=price.index,
        open=price.Open, high=price.High, low=price.Low, close=price.Close)])
    fig.update_layout(height=400, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Current", f"${price.Close[-1]:.2f}")
    st.metric("Predicted return", f"{pred_ret*100:+.2f}%")
    signal = "🔴 SELL" if pred_ret < -0.005 else "🟢 BUY" if pred_ret > 0.005 else "⚪ HOLD"
    st.subheader(signal)