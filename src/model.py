import os
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

load_dotenv()

MODEL_PATH = r"C:\Users\shahv\OneDrive\Documents\GitHub\Microvest\model\model.pkl"
os.makedirs("model", exist_ok=True)


def get_1h_data(ticker, days=30, max_retries=3):
    """Get hourly data using yfinance with retry logic"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    for attempt in range(max_retries):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1h",
                progress=False,
                timeout=10,  # Add timeout
            )

            if data.empty:
                raise ValueError(f"No data returned for {ticker}")

            return data

        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Failed to download data after {max_retries} attempts: {str(e)}"
                )
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2)  # Wait before retrying


def add_features(df):
    """Add technical indicators with validation"""
    if df.empty:
        raise ValueError("Empty DataFrame passed to add_features")

    try:
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        # Basic indicators
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.sma(length=20, append=True)

        # Validate we got the expected columns
        required_cols = ["RSI_14", "MACD_12_26_9", "SMA_20"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing expected column: {col}")

        return df.dropna()

    except Exception as e:
        raise ValueError(f"Error in add_features: {str(e)}")


def train_model(df):
    """Model training with validation"""
    if df.empty:
        raise ValueError("Cannot train model on empty data")

    try:
        features = ["RSI_14", "MACD_12_26_9", "SMA_20"]
        X = df[features]
        y = (df["close"].shift(-3) / df["close"] - 1 > 0.005).astype(int)

        # Rest of training logic...

    except Exception as e:
        raise RuntimeError(f"Model training failed: {str(e)}")


def train_and_save(ticker="AAPL", days=30):
    """Main training function with error handling"""
    try:
        print(f"Fetching data for {ticker}...")
        df = get_1h_data(ticker, days)

        print("Adding features...")
        df = add_features(df)

        print("Training model...")
        model, scaler = train_model(df)

        print(f"Model saved to {MODEL_PATH}")
        return model, scaler

    except Exception as e:
        print(f"Error in train_and_save: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        train_and_save()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print("Try again later or check your internet connection")
