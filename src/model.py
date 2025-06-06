import os
from datetime import datetime, timedelta

import joblib
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Configuration
TICKER = "AAPL"
DAYS = 100
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
data = data.rename(
    columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
)

# Technical Indicators
temp = pd.DataFrame()
temp["12"] = ta.ema(close=data["close"], length=12)
temp["26"] = ta.ema(close=data["close"], length=26)

temp["12_26"]
data["rsi"] = ta.rsi(close=data["close"], length=14, append=True)
data["macd_12_26_slow"]
data["macd_9_fast"] = ta.ema(close=data["macd_12_26"], lengt=9)
data["ema"] = ta.ema(close=data["close"], length=20)

# Target Variable
data["target"] = (data["close"].shift(-3) / data["close"] - 1 > 0.005).astype(int)
data = data.dropna()
data
# 3. Model Training
print("Training model...")
features = ["RSI_14", "MACD_12_26_9", "SMA_20"]
X = data[features]
y = data["target"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
model.fit(X_scaled, y)

# 4. Save Model
print(f"Saving model to {MODEL_PATH}")
joblib.dump((model, scaler, features), MODEL_PATH)

# 5. Quick Validation
train_acc = model.score(X_scaled, y)
print(f"Training accuracy: {train_acc:.2%}")
print("Model training complete!")
