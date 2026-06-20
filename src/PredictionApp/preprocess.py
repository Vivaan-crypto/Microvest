"""
Unified, leak-free preprocessing pipeline.

Replaces the fragmented Helpers chain (data_to_csv -> auto_label -> csv_to_pt),
which produced flat [N, F] tensors with raw price levels, the ticker id as a
feature, and no scaling.

This module produces properly windowed [N, T, F] tensors with:
    - stationary, causal features only (no raw price/MA levels, no ticker id)
    - fixed-percentage 5-day-forward labels (short / no-trade / long)
    - per-ticker StandardScaler fit on the TRAIN period only
    - a purge gap between train and test so no train label peeks into test

Run directly to (re)build the .pt tensors under Data/CSV/.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from indicators import (
    ema,
    macd,
    pct_return,
    realized_vol,
    rsi,
    sma,
    volume_zscore,
)
# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
TICKERS: List[str] = [
    "AAPL",
    "ABBV", "ADBE", "AMD", "AMZN", "APD", "ASML", "AVGO", "BA", "BAC",
        "C", "CAT", "COP", "COST", "CRM", "CVX", "DIS", "DUK", "EOG", "EXC",
        "FCX", "GE", "GOOGL", "GS", "HD", "HON", "INTC", "JNJ", "JPM", "KO",
        "LIN", "META", "MMM", "MRK", "MS", "MSFT", "NEE", "NFLX", "NVDA", "PEP",
        "PFE", "PG", "SHW", "SLB", "SO", "TSLA", "UNH", "WFC", "WMT", "XOM",
        "ACN", "ADP", "AIG", "AMAT", "AMGN", "AMT", "AXP", "BK", "BKNG", "BLK",
        "BMY", "BX", "CB", "CL", "CMCSA", "CME", "CSCO", "DE", "DHR", "EMR",
        "F", "FDX", "GD", "GILD", "GM", "IBM", "ITW", "LLY", "LMT", "LOW",
        "MA", "MCD", "MDT", "MO", "NKE", "ORCL", "PM", "QCOM", "RTX", "SBUX",
        "SPG", "T", "TGT", "TMO", "TXN", "UNP", "UPS", "USB", "V", "VZ",
]

START_DATE = "2014-06-01"   # extra history so the 100-day EMA warms up
END_DATE = "2026-01-01"
TRAIN_END = "2021-08-16"    # last decision date that may land in train
HORIZON = 5                 # forward-return horizon (trading days)
WINDOW = 20                 # sequence length fed to the LSTM
LABEL_PCT = 0.05          # fixed +/-4% threshold (long / short)

# Market-context symbols, downloaded once and broadcast across every ticker.
# These give the model a "regime" view so it can price a name relative to the
# broad market and the volatility environment instead of in isolation.
MARKET_SYMBOL = "^GSPC"     # S&P 500 index -> market return / vol features
VIX_SYMBOL = "^VIX"         # CBOE volatility index -> fear/regime features

OUT_DIR = "Data"
#TODO: Migrate plot to this script
color_map = {1: "green", -1: "red", 0: "yellow"}

label_dir = "label_charts"
# Class mapping: ordered so down < flat < up.
#   short (-1) -> 0,  no-trade (0) -> 1,  long (+1) -> 2
LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_NAMES = ["Short", "NoTrade", "Long"]

# Stationary feature columns produced by build_features (order matters).
# Per-ticker, causal features (computed in isolation from one name's own history).
PER_TICKER_COLS = [
    "ret_1d", "ret_5d", "ret_20d",
    "vol_20d", "vol_60d",
    "volume_zscore",
    "rsi_14",
    "macd_hist_norm",
    "dist_sma_20", "dist_sma_50",
    "dist_ema_20", "dist_ema_50", "dist_ema_100",
    "intraday_range",
    "overnight_gap",
]

# Market-wide features: identical for every ticker on a given date (regime context).
MARKET_COLS = [
    "mkt_ret_5d", "mkt_ret_20d", "mkt_vol_20d",
    "vix_z", "vix_chg_5d",
]

# Market-relative features: this name's behavior net of the market (relative value).
RELATIVE_COLS = [
    "excess_ret_5d", "excess_ret_20d", "beta_60d",
]

# Cross-sectional features: this name's standing vs. all peers on the same date.
# Rank-percentile in [0, 1] -> robust, bounded, and inherently "priced vs. others".
CROSS_SECTIONAL_COLS = [
    "xs_rank_ret_5d", "xs_rank_ret_20d", "xs_rank_vol_20d",
]

FEATURE_COLS = PER_TICKER_COLS + MARKET_COLS + RELATIVE_COLS + CROSS_SECTIONAL_COLS
INPUT_SIZE = len(FEATURE_COLS)  # 26


# ------------------------------------------------------------------
# Download
# ------------------------------------------------------------------
def download_panel(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for t in tickers:
        df = yf.download(
            t, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False, multi_level_index=False,
        )
        if df.empty:
            print(f"  WARN: no data for {t}, skipping")
            continue
        df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df["ticker"] = t
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["ticker", "Date"]).reset_index(drop=True)


# ------------------------------------------------------------------
# Features (per ticker, all causal / stationary)
# ------------------------------------------------------------------
def build_features(df_ticker: pd.DataFrame) -> pd.DataFrame:
    df = df_ticker.sort_values("Date").copy()
    close, high, low, op, volume = (
        df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    )

    daily_ret = pct_return(close, 1)
    macd_line, _signal, hist = macd(close)

    df["ret_1d"] = daily_ret
    df["ret_5d"] = pct_return(close, 5)
    df["ret_20d"] = pct_return(close, 20)

    df["vol_20d"] = realized_vol(daily_ret, 20)
    df["vol_60d"] = realized_vol(daily_ret, 60)

    df["volume_zscore"] = volume_zscore(volume, 20)

    df["rsi_14"] = rsi(close, 14) / 100.0          # bounded -> ~[0,1]
    df["macd_hist_norm"] = hist / close            # price-normalized

    df["dist_sma_20"] = close / sma(close, 20) - 1.0
    df["dist_sma_50"] = close / sma(close, 50) - 1.0
    df["dist_ema_20"] = close / ema(close, 20) - 1.0
    df["dist_ema_50"] = close / ema(close, 50) - 1.0
    df["dist_ema_100"] = close / ema(close, 100) - 1.0

    df["intraday_range"] = (high - low) / close
    df["overnight_gap"] = op / close.shift(1) - 1.0

    # ---- label: fixed % forward return ----
    fwd_ret = (close.shift(-HORIZON) - close) / close
    label = pd.Series(0, index=df.index)
    label[fwd_ret >= LABEL_PCT] = 1
    label[fwd_ret <= -LABEL_PCT] = -1
    df["fwd_ret"] = fwd_ret          # continuous target, kept for ranking metrics (IC)
    df["label_raw"] = label
    df["fwd_valid"] = fwd_ret.notna()  # False for the last HORIZON rows

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# ------------------------------------------------------------------
# Market context: VIX + broad-market index (downloaded once, broadcast)
# ------------------------------------------------------------------
def build_market_features(start: str, end: str) -> pd.DataFrame:
    """Download S&P 500 + VIX and return a [Date, MARKET_COLS] frame.

    All columns are causal/stationary and identical across tickers on a given
    date. Merged onto every name so the model sees the regime it is trading in.
    """
    mkt = yf.download(
        MARKET_SYMBOL, start=start, end=end, interval="1d",
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    vix = yf.download(
        VIX_SYMBOL, start=start, end=end, interval="1d",
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    if mkt.empty or vix.empty:
        raise RuntimeError("Failed to download market/VIX context data")

    out = pd.DataFrame({"Date": pd.to_datetime(mkt.index)}).reset_index(drop=True)
    mkt_close = mkt["Close"].reset_index(drop=True)
    mkt_daily = pct_return(mkt_close, 1)

    out["mkt_ret_5d"] = pct_return(mkt_close, 5).values
    out["mkt_ret_20d"] = pct_return(mkt_close, 20).values
    out["mkt_vol_20d"] = realized_vol(mkt_daily, 20).values
    # Carry the market daily/close series so per-ticker beta can use them.
    out["mkt_ret_1d"] = mkt_daily.values

    # VIX: level z-scored over a rolling year (stationary regime indicator),
    # plus its 5-day change (vol spiking vs. calming).
    vix_close = vix["Close"]
    vix_close = vix_close.reindex(mkt.index)  # align to market trading days
    vix_z = (vix_close - vix_close.rolling(252, min_periods=60).mean()) \
        / vix_close.rolling(252, min_periods=60).std()
    out["vix_z"] = vix_z.values
    out["vix_chg_5d"] = vix_close.pct_change(5).values

    return out.replace([np.inf, -np.inf], np.nan)


def add_relative_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Market-relative features (per row): excess returns + rolling beta.

    Requires MARKET_COLS already merged in (needs mkt_ret_1d / mkt_ret_5d / ...).
    """
    panel = panel.sort_values(["ticker", "Date"]).copy()
    panel["excess_ret_5d"] = panel["ret_5d"] - panel["mkt_ret_5d"]
    panel["excess_ret_20d"] = panel["ret_20d"] - panel["mkt_ret_20d"]

    # Rolling 60d beta per ticker. Built by concatenating per-group Series (instead
    # of groupby.apply) so it's robust to a single-ticker panel — apply() returns a
    # DataFrame for one group and breaks the assignment. Output is identical for many.
    betas = []
    for _, grp in panel.groupby("ticker"):
        cov = grp["ret_1d"].rolling(60, min_periods=60).cov(grp["mkt_ret_1d"])
        var = grp["mkt_ret_1d"].rolling(60, min_periods=60).var()
        betas.append(cov / var.replace(0, np.nan))
    panel["beta_60d"] = pd.concat(betas)
    return panel.replace([np.inf, -np.inf], np.nan)


def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank features: where this name sits among all peers today.

    For each date, rank a feature across the universe and map to a [0, 1]
    percentile. This is the "price relative to other names/indices" signal:
    a 0.95 momentum rank means top-5% mover that day regardless of absolute level.
    """
    panel = panel.copy()
    rank_src = {
        "xs_rank_ret_5d": "ret_5d",
        "xs_rank_ret_20d": "ret_20d",
        "xs_rank_vol_20d": "vol_20d",
    }
    for out_col, src_col in rank_src.items():
        panel[out_col] = (
            panel.groupby("Date")[src_col].rank(pct=True, method="average")
        )
    return panel


# ------------------------------------------------------------------
# Scaling + windowing (per ticker, scaler fit on train only)
# ------------------------------------------------------------------
def windows_for_ticker(
    df: pd.DataFrame,
) -> Tuple[list, list, list, list, list, list]:
    """(X_train, y_train, X_test, y_test, fwd_ret_test, date_test) for one ticker.

    The last two align 1:1 with the test windows (realized forward return + decision
    date) and feed the ranking metrics in `metrics.py`.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Drop NaN warmup rows (e.g. 100-day EMA needs 100 days to fill)
    df = df[df[FEATURE_COLS].notna().all(axis=1)].reset_index(drop=True)

    train_end = pd.Timestamp(TRAIN_END)
    train_mask = df["Date"] <= train_end

    # Fit scaler on train rows only, apply to all
    scaler = StandardScaler()
    feats = scaler.fit(df.loc[train_mask, FEATURE_COLS]).transform(df[FEATURE_COLS]).astype(np.float32)
    labels = df["label_raw"].map(LABEL_TO_CLASS).values
    valid = df["fwd_valid"].values
    fwd = df["fwd_ret"].values.astype(np.float32)
    dates = df["Date"].values  # datetime64[ns]

    # last_train_i is the last row index that belongs to train.
    # Rows [last_train_i - HORIZON + 1 : last_train_i + 1] have labels that
    # peek into test-period prices, so we purge them.
    last_train_i = int(np.where(train_mask.values)[0][-1])
    purge_start_i = last_train_i - HORIZON + 1

    Xtr, ytr, Xte, yte, fwd_te, date_te = [], [], [], [], [], []
    for i in range(WINDOW - 1, len(df)):
        if not valid[i]:
            continue
        window = feats[i - WINDOW + 1 : i + 1]   # [WINDOW, F]
        label = labels[i]
        if i < purge_start_i:
            Xtr.append(window); ytr.append(label)
        elif i > last_train_i:
            Xte.append(window); yte.append(label)
            fwd_te.append(fwd[i]); date_te.append(dates[i])
        # purge_start_i <= i <= last_train_i: dropped

    return Xtr, ytr, Xte, yte, fwd_te, date_te


def ticker_graph(feat: pd.DataFrame, ticker: str) -> None:
    data = feat[feat["fwd_valid"]].reset_index(drop=True)
    colors = data["label_raw"].map(color_map)

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(data["Date"], data["Close"], color="#CCCCCC", linewidth=0.8, zorder=1)
    ax.scatter(data["Date"], data["Close"], c=colors, s=12, zorder=2, linewidths=0)

    ax.set_facecolor("#0A0A0F")
    fig.patch.set_facecolor("#0A0A0F")
    ax.tick_params(colors="#888888")
    ax.spines["bottom"].set_color("#222233")
    ax.spines["top"].set_color("#222233")
    ax.spines["left"].set_color("#222233")
    ax.spines["right"].set_color("#222233")
    ax.yaxis.label.set_color("#888888")
    ax.xaxis.label.set_color("#888888")
    ax.title.set_color("#FFFFFF")

    ax.set_title(f"{ticker}", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")

    patches = [
        mpatches.Patch(color="green", label="Long (+1)"),
        mpatches.Patch(color="red", label="Short (-1)"),
        mpatches.Patch(color="yellow", label="No Trade (0)"),
    ]
    ax.legend(handles=patches, facecolor="#0F0F1A", edgecolor="#222233",
              labelcolor="#CCCCCC", fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{label_dir}/{ticker}.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {ticker}")
# ------------------------------------------------------------------
# Panel assembly: per-ticker features + market context + cross-section
# ------------------------------------------------------------------
def build_panel() -> pd.DataFrame:
    """Download everything and return one fully-featured long panel.

    Steps: per-ticker causal features -> merge market/VIX regime -> market-
    relative features -> cross-sectional ranks across the universe per date.
    """
    print(f"Downloading {len(TICKERS)} tickers...")
    raw = download_panel(TICKERS, START_DATE, END_DATE)

    print("Building per-ticker features...")
    frames = [build_features(grp) for _, grp in raw.groupby("ticker")]
    panel = pd.concat(frames, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])

    print(f"Downloading market context ({MARKET_SYMBOL}, {VIX_SYMBOL})...")
    market = build_market_features(START_DATE, END_DATE)
    panel = panel.merge(market, on="Date", how="left")

    print("Adding market-relative + cross-sectional features...")
    panel = add_relative_features(panel)
    panel = add_cross_sectional_features(panel)
    return panel.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    panel = build_panel()

    # ---- windowed [N, T, F] tensors for the LSTM / Transformer ----
    Xtr_all, ytr_all, Xte_all, yte_all = [], [], [], []
    fwd_te_all, date_te_all = [], []
    for t, grp in panel.groupby("ticker"):
        Xtr, ytr, Xte, yte, fwd_te, date_te = windows_for_ticker(grp)
        Xtr_all += Xtr; ytr_all += ytr; Xte_all += Xte; yte_all += yte
        fwd_te_all += fwd_te; date_te_all += date_te
        ticker_graph(grp, t)

    X_train = torch.tensor(np.array(Xtr_all), dtype=torch.float32)
    y_train = torch.tensor(np.array(ytr_all), dtype=torch.long).unsqueeze(-1)
    X_test = torch.tensor(np.array(Xte_all), dtype=torch.float32)
    y_test = torch.tensor(np.array(yte_all), dtype=torch.long).unsqueeze(-1)

    torch.save(X_train, f"{OUT_DIR}/X_train.pt")
    torch.save(y_train, f"{OUT_DIR}/y_train.pt")
    torch.save(X_test, f"{OUT_DIR}/X_test.pt")
    torch.save(y_test, f"{OUT_DIR}/y_test.pt")

    # Ranking-metric side data, aligned 1:1 with X_test (order preserved above).
    torch.save(torch.tensor(np.array(fwd_te_all), dtype=torch.float32),
               f"{OUT_DIR}/fwd_ret_test.pt")
    np.save(f"{OUT_DIR}/dates_test.npy", np.array(date_te_all, dtype="datetime64[ns]"))

    def dist(y):
        vals, counts = np.unique(y.numpy(), return_counts=True)
        return {CLASS_NAMES[int(v)]: int(c) for v, c in zip(vals, counts)}

    print("\n=== Done ===")
    print(f"X_train {tuple(X_train.shape)}  y_train {tuple(y_train.shape)}")
    print(f"X_test  {tuple(X_test.shape)}  y_test  {tuple(y_test.shape)}")
    print(f"Features ({INPUT_SIZE}): {FEATURE_COLS}")
    print(f"Train label dist: {dist(y_train)}")
    print(f"Test  label dist: {dist(y_test)}")


if __name__ == "__main__":
    main()
