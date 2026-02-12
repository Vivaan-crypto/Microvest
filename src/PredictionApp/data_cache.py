import json
import yfinance
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import tqdm

ticker = ["MSFT", "AAPL", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "DIS"]
dict_path = "Data/stock_dict.json"

def fetch_stock_data(ticker_list, start, end) -> pd.DataFrame:
    df = pd.DataFrame(yfinance.download(ticker_list, start=start, end=end, interval='1d', group_by='ticker', auto_adjust=True))
    return df


def main():
    save_ssd("Data/stock_data.pt")


def save_ssd(filename: str):
    data = fetch_stock_data(ticker, "2020-01-01", datetime.today())
    tensors = []
    for t in tqdm.tqdm(ticker):
        x = torch.from_numpy(data[t].to_numpy().astype(np.float64))
        with open(dict_path, 'r') as f:
            dict = json.load(f)
            x[:, -1] = dict[t]
        tensors.append(x)
    stock_data = torch.stack(tensors)
    torch.save(stock_data, filename)

def save_ram():
    print("name")



if __name__ == "__main__":
    main()
