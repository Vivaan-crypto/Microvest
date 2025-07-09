from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ======================
# CONFIGURATION
# ======================
TICKER = "AMZN"
DAYS = 500
MODEL_PATH = r"C:\Users\shahv\OneDrive\Documents\GitHub\Microvest\model\model.pkl"
INTERVAL = "1h"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ======================
# DATA FETCHING + CLEANING
# ======================
def load_data() -> np.ndarray:
    print(f"Fetching {TICKER} data...")

    # Get data
    data = yf.download(
        tickers=TICKER,
        start=datetime.now() - timedelta(days=DAYS),
        end=datetime.now(),
        interval=INTERVAL,
        progress=False,
    )

    # Clean column names
    data.columns = data.columns.droplevel(1)
    data = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    # ======================
    # FEATURE ENGINEERING
    # ======================
    # Price Features
    data["returns"] = data["close"].pct_change()
    data["volatility"] = data["returns"].rolling(24).std()

    # Volume Features
    data["volume_ma"] = data["volume"].rolling(24).mean()
    data["volume_z"] = (data["volume"] - data["volume_ma"]) / data["volume_ma"].replace(
        0, 1e-6
    )

    # Technical Indicators
    data["rsi"] = ta.rsi(data["close"], length=14)
    data["macd"] = ta.ema(data["close"], length=12) - ta.ema(data["close"], length=26)
    data["atr"] = ta.atr(data["high"], data["low"], data["close"], length=14)

    # Lagged Features
    for lag in [1, 3, 6]:
        data[f"rsi_lag{lag}"] = data["rsi"].shift(lag)

    # Target (1 if price increases >0.5% in next 3 periods)
    data["target"] = (data["close"].shift(-3) > data["close"] * 1.005).astype(int)

    # Clean data
    data.dropna()

    array = np.array(data)
    return array


def main():
    data = load_data()
    print(data)


if __name__ == "main":
    main()

"""
# ======================
# MODEL TRAINING
# ======================
# Selected Features
features = ["rsi", "macd", "atr", "volatility", "volume_z", "rsi_lag1", "rsi_lag3"]
X = data[features]
y = data["target"]

# Train-Test Split (time-series aware)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_STATE
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train model
model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    reg_alpha=0.1,  # L1 regularization
)
model.fit(X_train_scaled, y_train)

# ======================
# EVALUATION
# ======================
train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

print("\n=== Model Performance ===")
print(f"Training Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test_scaled)))

# ======================
# SAVE MODEL
# ======================
joblib.dump({"model": model, "scaler": scaler, "features": features}, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
"""
