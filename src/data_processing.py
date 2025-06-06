import os
from datetime import datetime

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
from alpaca.data import StockHistoricalDataClient, StockTradesRequest

df = pd.read_csv("../data/RKLB_stock_data_3-1-25_5-29-25.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize(None)
df.set_index("Timestamp", inplace=True)

df.head()


def resample_to_ohlcv(trade_df, interval="1min"):
    ohlcv = round(trade_df["Price"].resample(interval).ohlc(), 2)
    ohlcv["Volume"] = trade_df["Size"].resample(interval).sum()
    ohlcv["Average Price"] = round(
        trade_df["Price"].resample(interval).mean(), ndigits=2
    )
    return ohlcv.dropna()


ohlcv_df = resample_to_ohlcv(df, interval="4H")


ohlcv_df["rsi"] = ta.rsi(ohlcv_df["close"], length=14)
ohlcv_df["macd"] = ta.macd(ohlcv_df["close"])["MACD_12_26_9"]
ohlcv_df["sma_20"] = ta.sma(
    ohlcv_df["close"], length=20
)  # SMA = Standard Moving Average
ohlcv_df["sma_50"] = ta.sma(ohlcv_df["close"], length=50)
ohlcv_df["sma_100"] = ta.sma(ohlcv_df["close"], length=100)

plt.figure(figsize=(15, 10))
plt.plot(ohlcv_df["Average Price"], label="Average Price ($)")
plt.plot(ohlcv_df["sma_20"], color="red", label="SMA 20")
plt.plot(ohlcv_df["sma_50"], color="green", label="SMA 50")
plt.plot(ohlcv_df["sma_100"], color="blue", label="SMA 100")
plt.legend()
plt.title(
    "RKLB Stock data with SMA 20/50/100 (3/1/2025) - (5/29/2025) - 4 Hour intervals"
)
plt.show()
plt.figure(figsize=(15, 10))
plt.ylim(-0.8, 1)
plt.plot(ohlcv_df["macd"])
plt.show()
