import os
from datetime import datetime

import dotenv
import pandas as pd
import pandas_ta as ta
from alpaca.data import StockHistoricalDataClient, StockTradesRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import Stock

dotenv.load_dotenv("../.env.local")

# API INFO
api_key = os.getenv("ALPACA_MARKETS_API_KEY")
secret_key = os.getenv("ALPACA_MARKETS_API_SECRET")
endpoint = os.getenv("ALPACA_MARKETS_ENDPOINT")


# Getting hisotorical data
dc = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

params = StockTradesRequest(
    symbol_or_symbols="RKLB",
    start=datetime(2025, 5, 29, 15, 30),
    end=datetime(2025, 5, 29, 15, 45),
)

trades = dc.get_stock_trades(request_params=params)

counter = 0
for i in trades.data["RKLB"]:
    print(i)
    counter += 1
    if counter >= 2:
        break
