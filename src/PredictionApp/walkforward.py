"""
Walk-forward training + prediction (the honest evaluation loop).

For each fold in config.WALK_FORWARD_FOLDS:
    train  : everything up to (train_end - 1 year), purged at the boundary
    val    : the final year before train_end — used ONLY for early stopping /
             checkpoint selection, so the test year never influences training
    test   : the following calendar year, never seen by anything

Per fold, per ticker, the feature scaler is fit on train-period rows only.
Purge: any row whose label window (HORIZON days forward) crosses a boundary is
dropped, so no label ever peeks across train/val/test lines.

Outputs one parquet of test-year predictions per fold under
Data/walkforward/run_<timestamp>/, plus meta.json. Grade a run with:

    python scorecard.py Data/walkforward/run_<timestamp>

Usage:
    python walkforward.py                     # all folds, config defaults
    python walkforward.py --folds 2022 2023   # a subset
    python walkforward.py --universe legacy100 --epochs 10   # quick pass
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import config
import data_store
import preprocess as pp
from dataset import WindowDataset
from lightning_modules import LightningModule

WINDOW = pp.WINDOW
HORIZON = pp.HORIZON
FEATURE_COLS = pp.FEATURE_COLS


# ------------------------------------------------------------------
# Fold assembly
# ------------------------------------------------------------------
def build_fold(panel: pd.DataFrame, train_end: str, test_start: str, test_end: str):
    """Slice the panel into purged train/val/test WindowDatasets for one fold.

    Returns (datasets, side) where side carries the arrays the metrics need,
    aligned 1:1 with the val/test dataset order.
    """
    train_end = pd.Timestamp(train_end)
    test_end = pd.Timestamp(test_end)
    val_start = train_end - pd.DateOffset(years=1)

    feats_list, labels_list = [], []
    idx_tr, idx_va, idx_te = [], [], []
    va_side = {"fwd_z": [], "fwd_ret": [], "date": [], "ticker": []}
    te_side = {"fwd_z": [], "fwd_ret": [], "date": [], "ticker": [],
               "y_true": [], "close": []}

    for ticker, df in panel.groupby("ticker"):
        df = df[df[FEATURE_COLS].notna().all(axis=1)].sort_values("Date").reset_index(drop=True)
        dates = pd.to_datetime(df["Date"])

        train_rows = df[dates <= train_end]
        if len(train_rows) < 252:      # need at least a year to scale/train on
            continue

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(train_rows[FEATURE_COLS])
        feats = scaler.transform(df[FEATURE_COLS]).astype(np.float32)
        labels = df["label_raw"].map(pp.LABEL_TO_CLASS).values
        valid = df["fwd_valid"].values
        fwd_ret = df["fwd_ret"].values.astype(np.float32)
        fwd_z = df["fwd_z"].values.astype(np.float32)
        closes = df["Close"].values.astype(np.float32)

        arr_id = len(feats_list)
        feats_list.append(feats)
        labels_list.append(labels)

        d = dates.values
        iv = int(np.searchsorted(d, np.datetime64(val_start), side="right"))
        it = int(np.searchsorted(d, np.datetime64(train_end), side="right"))
        ie = int(np.searchsorted(d, np.datetime64(test_end), side="right"))

        for i in range(WINDOW - 1, ie):
            if not valid[i]:
                continue
            if i < iv - HORIZON:                       # train (purged at val line)
                idx_tr.append((arr_id, i))
            elif iv <= i < it - HORIZON:               # val (purged at test line)
                idx_va.append((arr_id, i))
                va_side["fwd_z"].append(fwd_z[i]); va_side["fwd_ret"].append(fwd_ret[i])
                va_side["date"].append(d[i]); va_side["ticker"].append(ticker)
            elif it <= i:                              # test
                idx_te.append((arr_id, i))
                te_side["fwd_z"].append(fwd_z[i]); te_side["fwd_ret"].append(fwd_ret[i])
                te_side["date"].append(d[i]); te_side["ticker"].append(ticker)
                te_side["y_true"].append(int(labels[i])); te_side["close"].append(closes[i])

    ds_tr = WindowDataset(feats_list, labels_list, idx_tr, WINDOW)
    ds_va = WindowDataset(feats_list, labels_list, idx_va, WINDOW)
    ds_te = WindowDataset(feats_list, labels_list, idx_te, WINDOW)
    return (ds_tr, ds_va, ds_te), (va_side, te_side)


def train_labels(ds_tr: WindowDataset) -> np.ndarray:
    return np.array([ds_tr.labels[a][i] for a, i in ds_tr.index])


# ------------------------------------------------------------------
# One fold: train -> predict test year -> parquet
# ------------------------------------------------------------------
@torch.no_grad()
def predict_dataset(model: LightningModule, ds: WindowDataset, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    for x, _y in loader:
        probs.append(torch.softmax(model(x), dim=1).cpu())
    return torch.cat(probs).numpy()


def run_fold(panel, fold, run_dir, args):
    name, train_end, test_start, test_end = fold
    print(f"\n=== Fold {name}: train<={train_end} | test {test_start}..{test_end} ===")

    (ds_tr, ds_va, ds_te), (va_side, te_side) = build_fold(panel, train_end, test_start, test_end)
    print(f"  windows: train {len(ds_tr):,} | val {len(ds_va):,} | test {len(ds_te):,}")
    if len(ds_tr) == 0 or len(ds_va) == 0 or len(ds_te) == 0:
        print("  SKIP: an empty split")
        return None

    # Same seed per fold: fold-to-fold and run-to-run comparisons in scorecard.py
    # are only meaningful if init/shuffle noise is fixed.
    L.seed_everything(config.SEED, workers=True)

    ytr = train_labels(ds_tr)
    model = LightningModule(
        ytr, lr=args.lr, weight_decay=config.WEIGHT_DECAY,
        val_fwd_ret=np.array(va_side["fwd_z"]),      # grade IC on the v2 target
        val_dates=np.array(va_side["date"]),
        input_size=len(FEATURE_COLS),
        hidden_size=config.LSTM_HIDDEN,
        num_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT,
    )

    # num_workers=0 on purpose: Windows spawn-workers each copy the fold's
    # feature arrays (hundreds of MB on sp500) and __getitem__ is a trivial
    # array slice, so main-process loading is faster AND lighter here.
    loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True,
                           num_workers=config.WF_NUM_WORKERS)
    loader_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                           num_workers=config.WF_NUM_WORKERS)

    ckpt_dir = os.path.join(run_dir, f"ckpt_{name}")
    checkpoint = ModelCheckpoint(monitor="val/ic_spearman", mode="max",
                                 dirpath=ckpt_dir, filename="best", save_top_k=1)
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        logger=TensorBoardLogger(run_dir, name=f"tb_{name}"),
        callbacks=[checkpoint,
                   EarlyStopping(monitor="val/ic_spearman", mode="max",
                                 patience=args.patience, min_delta=0.001)],
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
    )
    trainer.fit(model, train_dataloaders=loader_tr, val_dataloaders=loader_va)

    # Restore the best-val-IC weights before scoring the test year.
    if checkpoint.best_model_path:
        state = torch.load(checkpoint.best_model_path, map_location="cpu",
                           weights_only=False)["state_dict"]
        model.load_state_dict(state)

    proba = predict_dataset(model, ds_te, args.batch_size)
    out = pd.DataFrame({
        "date": pd.to_datetime(np.array(te_side["date"])),
        "ticker": te_side["ticker"],
        "close": te_side["close"],
        "p_short": proba[:, 0], "p_notrade": proba[:, 1], "p_long": proba[:, 2],
        "signal": proba[:, 2] - proba[:, 0],
        "pred_class": proba.argmax(1),
        "y_true": te_side["y_true"],
        "fwd_ret": te_side["fwd_ret"],
        "fwd_z": te_side["fwd_z"],
        "fold": name,
    })
    path = os.path.join(run_dir, f"fold_{name}.parquet")
    out.to_parquet(path, index=False)
    print(f"  saved {len(out):,} test predictions -> {path}")
    return path


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Walk-forward training/evaluation")
    ap.add_argument("--universe", default=config.UNIVERSE_NAME,
                    choices=["sp500", "legacy100"])
    ap.add_argument("--folds", nargs="*", default=None,
                    help="fold names to run (default: all), e.g. --folds 2022 2023")
    ap.add_argument("--epochs", type=int, default=config.WF_MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=config.WF_PATIENCE)
    ap.add_argument("--batch-size", type=int, default=config.WF_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.WF_LR)
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild the cached feature panel from fresh downloads")
    args = ap.parse_args()

    tickers = data_store.resolve_universe(args.universe)
    print(f"Universe: {args.universe} ({len(tickers)} tickers)")
    panel = pp.build_panel(tickers, refresh=args.refresh)
    print(f"Panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['Date'].min():%Y-%m-%d}..{panel['Date'].max():%Y-%m-%d}")

    folds = config.WALK_FORWARD_FOLDS
    if args.folds:
        folds = [f for f in folds if f[0] in set(args.folds)]
        if not folds:
            raise SystemExit(f"No folds matched {args.folds}")

    run_dir = os.path.join(config.WF_DIR, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({
            "config_version": config.CONFIG_VERSION,
            "universe": args.universe,
            "n_tickers": len(tickers),
            "folds": [f[0] for f in folds],
            "z_threshold": config.Z_THRESHOLD,
            "horizon": config.HORIZON,
            "window": config.WINDOW,
            "features": FEATURE_COLS,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        }, f, indent=2)

    for fold in folds:
        run_fold(panel, fold, run_dir, args)

    print(f"\nRun complete: {run_dir}")
    print(f"Grade it with:  python scorecard.py {run_dir}")


if __name__ == "__main__":
    main()
