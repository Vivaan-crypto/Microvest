"""
LightningDataModule + feature engineering in ~60 lines.
Concepts:
  - We create **log returns** as regression target.
  - We z-score every feature per ticker.
  - We slice sliding windows (seq_len) for LSTM consumption.
"""

import yfinance as yf, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as L

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = map(torch.FloatTensor, (X, y))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class StockDataModule(L.LightningDataModule):
    def __init__(self, cfg, X_train, y_train, X_val, y_val):
        super().__init__()
        self.cfg, self.train_ds, self.val_ds = cfg, StockDataset(X_train, y_train), StockDataset(X_val, y_val)

    @classmethod
    def from_dataframe(cls, df, cfg):
        df = df.copy()
        # 1) compute technicals
        import pandas_ta as ta
        techs = df.groupby('ticker').apply(lambda d: d.ta.sma(length=cfg.techs['sma'])
                                                      .ta.ema(length=cfg.techs['ema'])
                                                      .ta.rsi(length=cfg.techs['rsi'])
                                                      .ta.macd())
        df = pd.concat([df, techs], axis=1).dropna()

        # 2) target = next-day log return
        df['target'] = df.groupby('ticker')['Close'].pct_change().shift(-1)
        df = df.dropna()

        # 3) scale features per ticker
        scalers = {t: StandardScaler() for t in cfg.tickers}
        feats = ['Close','Volume','SMA_20','EMA_12','RSI_14','MACD_12_26_9']
        for t in cfg.tickers:
            mask = df.ticker==t
            df.loc[mask, feats] = scalers[t].fit_transform(df.loc[mask, feats])

        # 4) sliding windows
        def to_windows(tdf):
            X, y = [], []
            for i in range(cfg.seq_len, len(tdf)):
                X.append(tdf[feats].iloc[i-cfg.seq_len:i].values)
                y.append(tdf['target'].iloc[i])
            return X, y
        X, y = map(torch.tensor, zip(*(to_windows(df[df.ticker==t]) for t in cfg.tickers)))
        X, y = X.reshape(-1, cfg.seq_len, len(feats)), y.flatten()

        # 5) train/val split
        split = int(len(X)*0.8)
        return cls(cfg, X[:split], y[:split], X[split:], y[split:])

    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.cfg.batch_size, shuffle=True)
    def val_dataloader(self):   return DataLoader(self.val_ds, batch_size=self.cfg.batch_size)