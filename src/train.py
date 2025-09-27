"""
One-command training:

$ python train.py --cfg config.yaml
"""

import yaml, torch, argparse
import pandas as pd
import pytorch_lightning as L
from model import AttentionLSTM
from lightning_modules import StockDataModule  # see below
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import yfinance as yf
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config.yaml")
    args = parser.parse_args()

    cfg = argparse.Namespace(**yaml.safe_load(open(args.cfg)))

    # --- load raw prices & compute tech indicators ---
    df = pd.concat([
        yf.download(ticker, period=cfg.period)
        .assign(ticker=ticker)
        for ticker in cfg.tickers
    ])
    # helper returns scaled sequences ready for torch
    dm = StockDataModule.from_dataframe(df, cfg)

    model = AttentionLSTM(cfg)
    trainer = L.Trainer(
        max_epochs=2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ModelCheckpoint(dirpath="models", filename="complete_model", save_top_k=1)
        ]
    )
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()