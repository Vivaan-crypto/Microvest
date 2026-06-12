import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import StandardScaler
# ta-lib backtrader
from indicators import (
    sma,
    ema,
    rsi,
    macd,
    pct_return,
    forward_return,
    realized_vol,
    volume_zscore,
)


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_panel(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    dfs = []
    for t in tickers:
        df = yf.download(
            t,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )

        df = df.reset_index()
        df["ticker"] = t
        df = df[["Date", "Open", "High", "Low", "Close", "Volume", "ticker"]]
        dfs.append(df)

    panel = pd.concat(dfs, ignore_index=True)
    panel = panel.sort_values(["ticker", "Date"]).reset_index(drop=True)
    return panel


# ============================================================
# FEATURE ENGINEERING (PER TICKER)
# ============================================================

def add_features_for_ticker(
        df_ticker: pd.DataFrame,
        target_horizon: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    df = df_ticker.sort_values("Date").copy()
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    daily_ret = pct_return(close, periods=1)
    macd_line, signal_line, hist = macd(close)

    df["ret_1d"] = daily_ret
    df["ret_5d"] = pct_return(close, 5)
    df["ret_20d"] = pct_return(close, 20)

    df["vol_20d"] = realized_vol(daily_ret, 20)
    df["vol_60d"] = realized_vol(daily_ret, 60)

    df["volume_zscore"] = volume_zscore(volume, 20)

    df["sma_20"] = sma(close, 20)
    df["sma_50"] = sma(close, 50)

    df["ema_20"] = ema(close, 20)
    df["ema_50"] = ema(close, 50)
    df["ema_100"] = ema(close, 100)

    # ---------- TARGET ----------
    df["target"] = forward_return(close, horizon=target_horizon)

    # 🔑 SCALE TARGET (prevents collapse to zero)
    df["target"] = df["target"] * 100.0

    # ---------- HARD NUMERICAL CLEANUP ----------
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    feature_cols = list(df.columns.drop(["Date", "ticker", "target"]))
    return df, feature_cols


# ============================================================
# PANEL BUILD
# ============================================================

def build_feature_panel(
        panel: pd.DataFrame,
        target_horizon: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    out = []
    feature_cols = None

    for t, grp in panel.groupby("ticker"):
        feat_df, cols = add_features_for_ticker(grp, target_horizon)
        out.append(feat_df)
        feature_cols = cols

    df = pd.concat(out, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df, feature_cols


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def time_split_train_test(
        df: pd.DataFrame,
        train_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    train = df[df["Date"] <= train_end].copy()
    test = df[df["Date"] > train_end].copy()
    return train, test


# ============================================================
# SCALING (PER TICKER)
# ============================================================

def scale_features_by_ticker_train_test(
        train: pd.DataFrame,
        test: pd.DataFrame,
        feature_cols: List[str],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, StandardScaler]]:
    scalers = {}
    out = {"train": [], "test": []}

    for t in sorted(train["ticker"].unique()):
        train_t = train[train["ticker"] == t].copy()

        scaler = StandardScaler()
        scaler.fit(train_t[feature_cols].values)
        scalers[t] = scaler

        for name, split in [("train", train), ("test", test)]:
            df_t = split[split["ticker"] == t].copy()
            if df_t.empty:
                continue
            df_t[feature_cols] = scaler.transform(df_t[feature_cols].values)
            out[name].append(df_t)

    return {
        k: pd.concat(v, ignore_index=True) if v else pd.DataFrame()
        for k, v in out.items()
    }, scalers


# ============================================================
# WINDOWING
# ============================================================

def build_windows_for_split(
        df_split: pd.DataFrame,
        feature_cols: List[str],
        window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X, y = [], []

    df_split = df_split.sort_values(["ticker", "Date"])

    for _, grp in df_split.groupby("ticker"):
        feat = grp[feature_cols].values
        tgt = grp["target"].values

        for i in range(window - 1, len(grp)):
            X.append(feat[i - window + 1: i + 1])
            y.append([tgt[i]])

    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32),
    )


# ============================================================
# ENTRYPOINT
# ============================================================

def get_data(
        tickers: List[str],
        start_date: str,
        end_date: str,
        target_horizon: int = 5,
        window: int = 20,
        train_end: str = "2023-12-31",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    panel = download_panel(tickers, start_date, end_date)
    df_feat, feature_cols = build_feature_panel(panel, target_horizon)

    train_df, test_df = time_split_train_test(df_feat, train_end)
    scaled, _ = scale_features_by_ticker_train_test(train_df, test_df, feature_cols)

    X_train, y_train = build_windows_for_split(scaled["train"], feature_cols, window)
    X_test, y_test = build_windows_for_split(scaled["test"], feature_cols, window)

    return X_train, y_train, X_test, y_test


def get_data_new(
        tickers: List[str],
        start_date: str,
        end_date: str,
        target_horizon: int = 5,
        train_end: str = "2023-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    panel = download_panel(tickers, start_date, end_date)
    df_feat, feature_cols = build_feature_panel(panel, target_horizon)

    train_df, test_df = time_split_train_test(df_feat, train_end)
    scaled, _ = scale_features_by_ticker_train_test(train_df, test_df, feature_cols)

    return train_df, test_df


if __name__ == "__main__":
    TICKERS = [
        "AAPL", "MSFT", "GOOGL", "NVDA", "AVGO", "AMD", "INTC", "ASML", "CRM", "ADBE",
        "META", "NFLX", "DIS",
        "AMZN", "TSLA", "HD",
        "JPM", "BAC", "WFC", "GS", "MS", "C",
        "XOM", "CVX", "COP", "EOG", "SLB",
        "JNJ", "PFE", "MRK", "UNH", "ABBV",
        "CAT", "BA", "GE", "HON", "MMM",
        "PG", "KO", "PEP", "COST", "WMT",
        "NEE", "DUK", "SO", "EXC",
        "LIN", "SHW", "APD", "FCX"
    ]

    # X_train, y_train, X_test, y_test = get_data(
    #     tickers=TICKERS,
    #     start_date="2021-01-01",
    #     end_date="2026-01-01",
    #     target_horizon=5,
    #     window=20,
    #     train_end="2024-12-31",
    # )
    train, test = get_data_new(
        tickers=TICKERS,
        start_date="2021-01-01",
        end_date="2026-01-01",
        target_horizon=5,
        train_end="2024-12-31",
    )

    os.makedirs("data/CSV", exist_ok=True)
    train.to_csv("data/CSV/train.csv")
    test.to_csv("data/CSV/test.csv")
    print("Saved train/test tensors")
