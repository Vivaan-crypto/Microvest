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
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    out = {}
    for k, v in sd.items():
        for prefix in ("model._orig_mod.", "model.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        out[k] = v
    return out


def load_model(path):
    """Load a checkpoint — auto-detects LSTM vs Transformer from weight keys."""
    sd = _clean_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    n_classes = sd["classification_head.2.weight"].shape[0]

    if "LSTM.weight_ih_l0" in sd:
        # LSTM checkpoint
        ih = sd["LSTM.weight_ih_l0"]
        hidden, inp = ih.shape[0] // 4, ih.shape[1]
        layers = len([k for k in sd if k.startswith("LSTM.weight_ih_l")])
        model = StockLSTMModel(input_size=inp, lstm_hidden_size=hidden,
                               lstm_layers=layers, num_classes=n_classes)
        arch = f"LSTM · {layers}L · {hidden}h · {inp}f · {n_classes}c"
        info = {"type": "LSTM", "input": inp, "hidden": hidden, "layers": layers,
                "classes": n_classes, "path": path}
    else:
        # Transformer checkpoint — infer d_model and layers from projection weight
        inp = sd["input_to_transformer_linear.weight"].shape[1]
        d_model = sd["input_to_transformer_linear.weight"].shape[0]
        layers = len([k for k in sd if k.startswith("Transformer.layers.") and k.endswith(".self_attn.in_proj_weight")])
        model = StockTransformerModel(input_size=inp, d_model=d_model,
                                      transformer_layers=layers, num_classes=n_classes)
        arch = f"Transformer · {layers}L · d{d_model} · {inp}f · {n_classes}c"
        info = {"type": "Transformer", "input": inp, "hidden": d_model, "layers": layers,
                "classes": n_classes, "path": path}

    model_keys = set(model.state_dict())
    missing = model_keys - set(sd)
    model.load_state_dict({k: v for k, v in sd.items() if k in model_keys}, strict=False)
    model.eval()
    info["missing"] = sorted(missing)
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
        df = df[df[FEATURE_COLS].notna().all(axis=1)].reset_index(drop=True)
        if len(df) < WINDOW:
            continue

        scaler = StandardScaler()
        train_rows = df[df["Date"] <= TRAIN_END]
        scaler.fit((train_rows if len(train_rows) >= WINDOW else df)[FEATURE_COLS])
        feats = scaler.transform(df[FEATURE_COLS]).astype(np.float32)

        idxs = np.arange(WINDOW - 1, len(df))
        windows = np.stack([feats[i - WINDOW + 1:i + 1] for i in idxs])
        proba = torch.softmax(model(torch.from_numpy(windows)), dim=1).numpy()

        sub = df.iloc[idxs].reset_index(drop=True)
        valid = sub["fwd_valid"].values
        true_cls = sub["label_raw"].map(pp.LABEL_TO_CLASS).values
        for j in range(len(idxs)):
            p = proba[j]
            rows.append({
                "date": sub["Date"].iloc[j], "ticker": ticker,
                "open": float(sub["Open"].iloc[j]), "high": float(sub["High"].iloc[j]),
                "low": float(sub["Low"].iloc[j]), "close": float(sub["Close"].iloc[j]),
                "volume": float(sub["Volume"].iloc[j]),
                "p_short": float(p[0]), "p_notrade": float(p[1]), "p_long": float(p[2]),
                "signal": float(p[2] - p[0]), "pred_class": int(p.argmax()),
                "fwd_ret": float(sub["fwd_ret"].iloc[j]) if valid[j] else np.nan,
                "true_class": int(true_cls[j]) if valid[j] else np.nan,
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
