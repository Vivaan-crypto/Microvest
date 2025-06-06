import os
from datetime import datetime, timedelta

import joblib
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration
TICKER = "AMZN"
DAYS = 500
MODEL_PATH = r"C:\Users\shahv\OneDrive\Documents\GitHub\Microvest\model\model.pkl"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# 1. Data Fetching
print(f"Fetching {TICKER} data...")
end_date = datetime.now()
start_date = end_date - timedelta(days=DAYS)
data = yf.download(
    tickers=TICKER, start=start_date, end=end_date, interval="1h", progress=False
)

if data.empty:
    raise ValueError("No data returned - check ticker or internet connection")
# 2. Feature Engineering
print("Calculating indicators...")
data.columns = data.columns.droplevel(1)  # Remove ticker level
data = data.rename(
    columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
)
for i in data.columns:
    data[i] = round(data[i], 2)


# Technical Indicators
data["rsi"] = ta.rsi(data["close"], length=14)
data["macd"] = ta.ema(close=data["close"], length=12) - ta.ema(
    close=data["close"], length=26
)
data["ema_20"] = ta.ema(close=data["close"], length=20)
# Target Variable
data["target"] = (data["close"].shift(-3) / data["close"] - 1 > 0.005).astype(int)
# Momentum Indicators
data["stoch_k"] = ta.stoch(data["high"], data["low"], data["close"], k=14, d=3)[
    "STOCHk_14_3_3"
]  # Stochastic %K
data["stoch_d"] = ta.stoch(data["high"], data["low"], data["close"], k=14, d=3)[
    "STOCHd_14_3_3"
]  # Stochastic %D
data["cci"] = ta.cci(
    data["high"], data["low"], data["close"], length=20
)  # Commodity Channel Index
data["mom"] = ta.mom(data["close"], length=10)  # Momentum (10-period)

# Trend Indicators
data["adx"] = ta.adx(data["high"], data["low"], data["close"], length=14)[
    "ADX_14"
]  # Average Directional Index
data["psar"] = ta.psar(data["high"], data["low"], data["close"])[
    "PSARl_0.02_0.2"
]  # Parabolic SAR

# Volume Indicators
data["obv"] = ta.obv(data["close"], data["volume"])  # On-Balance Volume
data["vwap"] = ta.vwap(
    data["high"], data["low"], data["close"], data["volume"]
)  # Volume Weighted Avg Price

# Volatility Indicators
data["atr"] = ta.atr(
    data["high"], data["low"], data["close"], length=14
)  # Average True Range
data["bb_width"] = ta.bbands(data["close"], length=20)[
    "BBB_20_2.0"
]  # Bollinger Band Width
data = data.dropna()
# 3. Model Training
print("Training model...")
features = [
    "rsi",
    "macd",
    "ema_20",
    "stoch_k",
    "stoch_d",
    "cci",
    "mom",
    "adx",
    "psar",
    "obv",
    "vwap",
    "atr",
    "bb_width",
]
X = data[features]
y = data["target"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.fit_transform(x_test)


# Train model
model = RandomForestClassifier(
    n_estimators=400, max_depth=10, random_state=42, n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# 4. Save Model
print(f"Saving model to {MODEL_PATH}")
joblib.dump((model, scaler, features), MODEL_PATH)

# 5. Quick Validation
train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)
print(f"Training accuracy: {train_acc:.2%}")
print(f"Testing accuracy: {test_acc:.2%}")


print("Model training complete!")
