"""
Inference engine for the Streamlit app — pure (no streamlit), so it stays testable.

Reuses the training feature pipeline (`preprocess.py`) so what the app feeds the
model is identical to what it was trained on. Produces one tidy predictions frame
that every page reads:

    date | ticker | close | p_short | p_notrade | p_long | signal | pred_class
         | fwd_ret | true_class

`signal = P(Long) - P(Short)` is the continuous ranking score; `pred_class` is the
argmax (for the classification view). Features are causal and the per-ticker scaler
is fit on data up to TRAIN_END, so filtering by an "as-of" date downstream is a
leak-free backtest — no recompute needed.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# Make the parent PredictionApp package importable when run from the app folder.
PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import metrics as M  # noqa: E402
import preprocess as pp  # noqa: E402
from model import StockLSTMModel, StockTransformerModel  # noqa: E402

FEATURE_COLS = pp.FEATURE_COLS
WINDOW = pp.WINDOW
HORIZON = pp.HORIZON
TRAIN_END = pd.Timestamp(pp.TRAIN_END)
START_DATE = pp.START_DATE
UNIVERSE = list(pp.TICKERS)

CLASS_NAMES = ["Short", "NoTrade", "Long"]
CLASS_COLOR = {0: "#e74c3c", 1: "#f1c40f", 2: "#2ecc71"}  # red / yellow / green


# ------------------------------------------------------------------
# Checkpoint loading (architecture derived from the weights, so any ckpt loads)
# ------------------------------------------------------------------
def list_checkpoints():
    # Find all best-model .ckpt files across all timestamp subdirectories (recursive)
    cks = sorted(glob.glob(os.path.join(PARENT, "checkpoints", "**", "best-model-*.ckpt"), recursive=True),
                 key=os.path.getmtime, reverse=True)
    # final = os.path.join(PARENT, "model_final.pth")
    # if os.path.exists(final):
    #     cks.append(final)
    return cks


def _clean_state_dict(raw):
    # A Lightning checkpoint is a dict with a "state_dict" key; a plain .pth is
    # already the weights. Pull out the weights either way.
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    else:
        state_dict = raw

    # Strip the wrapper prefixes Lightning (and torch.compile) add to every key.
    clean = {}
    for key in state_dict:
        new_key = key
        new_key = new_key.replace("model._orig_mod.", "")
        new_key = new_key.replace("model.", "")
        new_key = new_key.replace("_orig_mod.", "")
        clean[new_key] = state_dict[key]
    return clean


def load_model(path):
    """Load a checkpoint into a StockLSTMModel sized to match its own weights."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    weights = _clean_state_dict(raw)

    # Read the model size straight off the weight shapes.
    lstm_weight = weights["LSTM.weight_ih_l0"]
    hidden = lstm_weight.shape[0] // 4   # LSTM has 4 gates
    inp = lstm_weight.shape[1]
    n_classes = weights["classification_head.2.weight"].shape[0]

    # Count how many LSTM layers the checkpoint has.
    layers = 0
    for key in weights:
        if key.startswith("LSTM.weight_ih_l"):
            layers = layers + 1

    model = StockLSTMModel(input_size=inp, lstm_hidden_size=hidden,
                           lstm_layers=layers, num_classes=n_classes)
    model.load_state_dict(weights)
    model.eval()

    info = {"input": inp, "hidden": hidden, "layers": layers,
            "classes": n_classes, "missing": [], "path": path}
    return model, info


# ------------------------------------------------------------------
# Feature panel + prediction
# ------------------------------------------------------------------
_MARKET_CACHE = {}


def _market_features(start, end):
    """Cache S&P/VIX context per (start, end) so single-ticker searches don't
    re-download it every time — a big chunk of the per-search lag."""
    key = (start, end)
    if key not in _MARKET_CACHE:
        _MARKET_CACHE[key] = pp.build_market_features(start, end)
    return _MARKET_CACHE[key]


def _download_bulk(tickers, start, end):
    """One threaded yfinance call for the whole list (vs. a serial per-ticker loop)."""
    tickers = list(tickers)
    raw = yf.download(tickers, start=start, end=end, interval="1d", auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=cols + ["ticker"])
    multi = isinstance(raw.columns, pd.MultiIndex)
    frames = []
    for t in tickers:
        if multi:
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t].copy()
        else:
            df = raw.copy()
        df = df.reset_index()
        if not set(cols).issubset(df.columns):
            continue
        df = df[cols].dropna(subset=["Close"])
        df["ticker"] = t
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=cols + ["ticker"])
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "Date"]).reset_index(drop=True)


def build_panel(tickers, start, end):
    """Download + featurize a universe (mirrors preprocess.build_panel, but fast)."""
    raw = _download_bulk(tickers, start, end)
    if raw.empty:
        raise ValueError(f"No price data for {list(tickers)}")
    frames = [pp.build_features(grp) for _, grp in raw.groupby("ticker")]
    panel = pd.concat(frames, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel = panel.merge(_market_features(start, end), on="Date", how="left")
    panel = pp.add_relative_features(panel)
    panel = pp.add_cross_sectional_features(panel)
    return panel.replace([np.inf, -np.inf], np.nan)


@torch.no_grad()
def predict(panel, model):
    rows = []
    for ticker, df in panel.groupby("ticker"):
        df = df.sort_values("Date")
        df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        if len(df) < WINDOW:
            continue

        # Standardize features using only the training period's mean and std.
        train_rows = df[df["Date"] <= TRAIN_END]
        if len(train_rows) < WINDOW:
            train_rows = df
        scaler = StandardScaler()
        scaler.fit(train_rows[FEATURE_COLS])
        feats = scaler.transform(df[FEATURE_COLS]).astype(np.float32)

        # Build one rolling window of WINDOW days for each day we can score.
        windows = []
        for i in range(WINDOW - 1, len(df)):
            windows.append(feats[i - WINDOW + 1:i + 1])
        windows = np.stack(windows)

        # Run the model and turn the logits into probabilities.
        logits = model(torch.from_numpy(windows))
        proba = torch.softmax(logits, dim=1).numpy()

        # Record one prediction row per scored day.
        scored_days = df.iloc[WINDOW - 1:].reset_index(drop=True)
        for j in range(len(scored_days)):
            day = scored_days.iloc[j]
            p = proba[j]

            # Forward return + true label only exist when the future is known.
            if day["fwd_valid"]:
                fwd_ret = float(day["fwd_ret"])
                true_class = int(pp.LABEL_TO_CLASS[day["label_raw"]])
            else:
                fwd_ret = np.nan
                true_class = np.nan

            rows.append({
                "date": day["Date"],
                "ticker": ticker,
                "open": float(day["Open"]),
                "high": float(day["High"]),
                "low": float(day["Low"]),
                "close": float(day["Close"]),
                "volume": float(day["Volume"]),
                "p_short": float(p[0]),
                "p_notrade": float(p[1]),
                "p_long": float(p[2]),
                "signal": float(p[2] - p[0]),
                "pred_class": int(p.argmax()),
                "fwd_ret": fwd_ret,
                "true_class": true_class,
            })

    out = pd.DataFrame(rows)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


# ------------------------------------------------------------------
# Ranking helpers (long-short book + deciles), reused by the Ranking page
# ------------------------------------------------------------------
def long_short_curve(df, quantile=0.2):
    """Daily top-minus-bottom basket return -> cumulative (illustrative, gross)."""
    d = df.dropna(subset=["signal", "fwd_ret"])
    pnl = {}
    for date, g in d.groupby("date"):
        if len(g) < 5 or g["signal"].std() == 0:
            continue
        hi, lo = g["signal"].quantile(1 - quantile), g["signal"].quantile(quantile)
        longs, shorts = g.loc[g["signal"] >= hi, "fwd_ret"], g.loc[g["signal"] <= lo, "fwd_ret"]
        if len(longs) and len(shorts):
            pnl[date] = longs.mean() - shorts.mean()
    s = pd.Series(pnl).sort_index()
    return s.cumsum()


def decile_table(df, n=10):
    d = df.dropna(subset=["signal", "fwd_ret"])
    if len(d) < n * 2 or d["signal"].std() == 0:
        return pd.Series(dtype=float)
    d = d.assign(bucket=pd.qcut(d["signal"], n, labels=False, duplicates="drop"))
    return d.groupby("bucket")["fwd_ret"].mean()


def signal_report(df):
    d = df.dropna(subset=["signal", "fwd_ret"])
    if d.empty:
        return {}
    return M.evaluate_signal(d["fwd_ret"].values, signal=d["signal"].values,
                             dates=d["date"].values, verbose=False)


def rolling_ic(df):
    d = df.dropna(subset=["signal", "fwd_ret"])
    return M.cross_sectional_ic(d["date"].values, d["signal"].values, d["fwd_ret"].values)
