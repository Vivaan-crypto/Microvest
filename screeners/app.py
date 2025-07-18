# app.py  – SMA-only + date range + session cache
import streamlit as st, yfinance as yf, pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="SMA-Only Screener", layout="wide")
st.title("📈 Stocks above 50-day SMA")

# ---------- 1. Pick date range ----------
col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start", datetime.today() - timedelta(days=90))
with col2:
    end = st.date_input("End", datetime.today())
if start >= end:
    st.error("End date must be after start date.")
    st.stop()

# ---------- 2. Cache the download ----------
@st.cache_data(ttl=3600, show_spinner="Downloading prices…")
def fetch_prices(ticker_list, start_date, end_date):
    data = {}
    for sym in ticker_list:
        try:
            df = yf.download(sym, start=start_date, end=end_date, progress=False)
            if len(df) >= 50:
                df["SMA50"] = df["Close"].rolling(50).mean()
                data[sym] = df
        except Exception:
            pass
    return data

# ---------- 3. Load universe ----------
@st.cache_data(ttl=3600)
def get_sp500():
    return pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].tolist()

tickers = get_sp500()

# ---------- 4. Run or clear cache ----------
if st.button("🔄 Re-download data"):
    st.cache_data.clear()          # force fresh download
data = fetch_prices(tickers, start, end)

# ---------- 5. Filters ----------
volume_toggle = st.checkbox("🔊 Require volume surge?", value=False)

passed = []
for t, df in data.items():
    last_close = df["Close"].iloc[-1].item()
    last_sma   = df["SMA50"].iloc[-1].item()
    vol_ok     = True

    if volume_toggle:
        avg_vol = df["Volume"].rolling(20).mean().iloc[-1].item()
        vol_ok  = df["Volume"].iloc[-1].item() > 1.3 * avg_vol

    if pd.notna(last_sma) and last_close > last_sma and vol_ok:
        passed.append(t)

st.subheader(f"✅ {len(passed)} tickers passed filters")
st.write(passed)