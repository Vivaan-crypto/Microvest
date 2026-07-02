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
import re
import sys
from datetime import datetime

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


# Metrics in the filename, e.g.: best-model-23-acc0.62-ic+0.065.ckpt
#                                          ^epoch  ^acc    ^ic
_CKPT_RE = re.compile(r"best-model-(\d+)-acc(\d+\.\d+)-ic([+-]?\d+\.\d+)")


def _version_start_time(version_dir):
    """When a training run started: prefer the epoch embedded in the TensorBoard
    events file name, fall back to the folder's mtime."""
    for events in glob.glob(os.path.join(version_dir, "events.out.tfevents.*")):
        parts = os.path.basename(events).split(".")
        if len(parts) > 3:
            try:
                return float(parts[3])
            except ValueError:
                pass
    return os.path.getmtime(version_dir)


def _version_index():
    """[(start_time, version_number)] for every run logged under lightning_logs.
    These version_N folders are the source of truth for the version numbers."""
    base = os.path.join(PARENT, "lightning_logs", "stock_prediction_model")
    index = []
    for path in glob.glob(os.path.join(base, "version_*")):
        try:
            number = int(os.path.basename(path).split("_")[1])
        except (IndexError, ValueError):
            continue
        index.append((_version_start_time(path), number))
    return index


def list_versions():
    """Every version number available in lightning_logs, sorted."""
    return sorted(number for _, number in _version_index())


def _version_for_run(run, version_index):
    """Map a checkpoints/<run> folder (named YYYYMMDD_HHMMSS) to its lightning_logs
    version by matching the run's start time to the nearest version folder."""
    try:
        run_time = datetime.strptime(run, "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return None

    best_version = None
    best_gap = None
    for start_time, number in version_index:
        gap = abs(start_time - run_time)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_version = number

    # Only trust the match if the times line up (same run, within ~5 minutes).
    if best_gap is not None and best_gap <= 300:
        return best_version
    return None


def parse_checkpoint(path, version_index=None):
    """Pull (epoch, accuracy, ic, version, run) out of a checkpoint path. The
    version comes from lightning_logs (matched by run time), not the filename.
    Metric fields are None for older / differently-named checkpoints."""
    if version_index is None:
        version_index = _version_index()
    name = os.path.basename(path)
    run = os.path.basename(os.path.dirname(path))   # the timestamp folder
    match = _CKPT_RE.search(name)
    if match:
        epoch = int(match.group(1))
        acc = float(match.group(2))
        ic = float(match.group(3))
    else:
        epoch = None
        acc = None
        ic = None
    version = _version_for_run(run, version_index)
    return {"path": path, "name": name, "run": run,
            "epoch": epoch, "acc": acc, "ic": ic, "version": version}


def list_checkpoint_metas():
    """Parse every checkpoint, tagging each with its lightning_logs version.
    Builds the version index once and reuses it for all checkpoints."""
    version_index = _version_index()
    return [parse_checkpoint(p, version_index) for p in list_checkpoints()]


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

    # Keep only the weights the model actually has. A Lightning checkpoint also
    # stores non-model tensors (e.g. "criterion.weight", the loss class weights)
    # that the bare model doesn't know about.
    model_keys = model.state_dict().keys()
    model_weights = {}
    for key in weights:
        if key in model_keys:
            model_weights[key] = weights[key]

    model.load_state_dict(model_weights)
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
    frames = []
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

        # Assemble one output row per scored day — vectorized (no per-row loop).
        scored = df.iloc[WINDOW - 1:].reset_index(drop=True)
        valid = scored["fwd_valid"].to_numpy()
        true_cls = scored["label_raw"].map(pp.LABEL_TO_CLASS).to_numpy(dtype=float)
        frames.append(pd.DataFrame({
            "date": scored["Date"].to_numpy(),
            "ticker": ticker,
            "open": scored["Open"].to_numpy(),
            "high": scored["High"].to_numpy(),
            "low": scored["Low"].to_numpy(),
            "close": scored["Close"].to_numpy(),
            "volume": scored["Volume"].to_numpy(),
            "p_short": proba[:, 0],
            "p_notrade": proba[:, 1],
            "p_long": proba[:, 2],
            "signal": proba[:, 2] - proba[:, 0],
            "pred_class": proba.argmax(axis=1),
            # Forward return + true label only exist when the future is known.
            "fwd_ret": np.where(valid, scored["fwd_ret"].to_numpy(), np.nan),
            "true_class": np.where(valid, true_cls, np.nan),
        }))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
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
