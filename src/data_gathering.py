import os
from datetime import datetime
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
from alpaca.data import StockHistoricalDataClient, StockTradesRequest

dotenv.load_dotenv("../.env.local")

# API INFO
api_key = os.getenv("ALPACA_MARKETS_API_KEY")
secret_key = os.getenv("ALPACA_MARKETS_API_SECRET")
endpoint = os.getenv("ALPACA_MARKETS_ENDPOINT")

# data client initialization
data_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

params_RKLB = StockTradesRequest(
    symbol_or_symbols="RKLB",
    start=datetime(2025, 3, 1, 13, 30),
    end=datetime(2025, 5, 30, 20, 00),
)


def trades_to_dataframe(trade_info, symbol):
    df = pd.DataFrame(
        [
            {
                "Symbol": t.symbol,
                "Timestamp": t.timestamp,
                "Exchange": t.exchange,
                "Price": t.price,
                "Size": t.size,
                "Conditions": t.conditions,
            }
            for t in trade_info.data[symbol]
        ]
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize(None)
    df.set_index("Timestamp", inplace=True)
    return df


def filter_dataf(df_to_filter, start, end):
    df_filtered = df_to_filter.between_time(start, end)  # 9:30 AM to 4:00 PM ET in UTC

    plt.figure(figsize=(15, 10))
    plt.plot(df_filtered["Price"])
    plt.title("RKLB Stock data (5/29/2025)")
    plt.show()
    return df_filtered


trades = data_client.get_stock_trades(request_params=params_RKLB)


df = trades_to_dataframe(trades, "RKLB")
df.head()
df.to_csv("../data/df.csv")


plt.figure(figsize=(15, 10))
plt.plot(df["Price"])
plt.title("RKLB Stock data (5/29/2025)")
plt.show()
