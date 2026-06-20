import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import preprocess as pp
from model import StockLSTMModel
import yfinance as yf
FEATURE_COLS = pp.FEATURE_COLS
WINDOW = pp.WINDOW
TRAIN_END = pd.Timestamp(pp.TRAIN_END)


def load_model(path):
    # Load a trained checkpoint and put the weights into a fresh LSTM model.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    # Lightning saves the weights under "model." — strip that off every key.
    weights = {}
    for key in state_dict:
        clean_key = key.replace("model.", "")
        weights[clean_key] = state_dict[key]

    # Figure out the model size from the shape of the LSTM weight matrix.
    lstm_weight = weights["LSTM.weight_ih_l0"]
    hidden_size = lstm_weight.shape[0] // 4   # LSTM has 4 gates
    input_size = lstm_weight.shape[1]

    model = StockLSTMModel(input_size=input_size, lstm_hidden_size=hidden_size)
    model.load_state_dict(weights)
    model.eval()
    return model


@torch.no_grad()
def predict(panel, model):
    rows = []

    for ticker, df in panel.groupby("ticker"):
        df = df.sort_values("Date")
        df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

        # Standardize features using only the training period's mean and std.
        train_rows = df[df["Date"] <= TRAIN_END]
        scaler = StandardScaler()
        scaler.fit(train_rows[FEATURE_COLS])
        feats = scaler.transform(df[FEATURE_COLS]).astype(np.float32)

        # Build one rolling window of WINDOW days for each day we can score.
        windows = []
        for i in range(WINDOW - 1, len(df)):
            window = feats[i - WINDOW + 1:i + 1]
            windows.append(window)
        windows = np.stack(windows)

        # Run the model and turn the logits into probabilities.
        logits = model(torch.from_numpy(windows))
        proba = torch.softmax(logits, dim=1).numpy()

        # Record one prediction row per day.
        scored_days = df.iloc[WINDOW - 1:].reset_index(drop=True)
        for j in range(len(scored_days)):
            p = proba[j]
            rows.append({
                "date": scored_days["Date"][j],
                "ticker": ticker,
                "p_short": p[0],
                "p_notrade": p[1],
                "p_long": p[2],
                "signal": p[2] - p[0],
                "pred_class": int(p.argmax()),
            })

    return pd.DataFrame(rows)

# Cache the S&P 500 + VIX download so it isn't re-fetched for every ticker.
_market_cache = {}


def get_market_features(start, end):
    key = (start, end)
    if key not in _market_cache:
        _market_cache[key] = pp.build_market_features(start, end)
    return _market_cache[key]


def get_data(ticker: str, start: str, end: str):
    # 1. Download the raw daily price bars and put Date back as a column.
    raw = pd.DataFrame(yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True,
                      progress=False, threads=True, multi_level_index=False))
    raw["ticker"] = ticker

    # 2. Per-ticker technical features (returns, vol, RSI, MACD, moving averages)
    #    plus the forward return / label columns.
    main_df = pp.build_features(raw)
    market = get_market_features(start, end)
    main_df = main_df.merge(market, on="Date", how="left")
    main_df = pp.add_relative_features(main_df)
    main_df = pp.add_cross_sectional_features(main_df)

    return main_df.replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":
    # Use a long history: the features need ~100+ days of warm-up, and VIX
    # needs ~1 year, so a short range comes back all-NaN.
    df = get_data("AAPL", "2018-01-01", "2024-01-01")
    print(df.dropna().tail())