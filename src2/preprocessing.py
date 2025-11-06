import yfinance as yf
import pandas as pd
import torch
from pyarrow import float32
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta


def get_data(tickers, start_date, end_date, target_len=5, ticker_necessary=False):
    # Download and concat all tickers
    df = pd.concat(
        [yf.download(t, start=start_date, end=end_date, interval='1d', multi_level_index=False)
         .assign(ticker=t) for t in tickers]
    )

    # Add indicators per ticker
    new_df = pd.DataFrame()
    for t in tickers:
        tdf = df[df['ticker'] == t].copy()
        tdf.ta.sma(length=20, append=True)
        tdf.ta.ema(length=12, append=True)
        tdf.ta.rsi(length=14, append=True)
        tdf.ta.macd(fast=12, slow=26, signal=9, append=True)
        new_df = pd.concat([new_df, tdf])

    # Drop NaNs after indicators
    new_df.dropna(inplace=True)

    # Compute 5-day change as target
    new_df['target'] = new_df.groupby('ticker')['Close'].pct_change(periods=target_len).shift(-target_len)

    # Select features
    features = new_df.drop(columns=['target', 'ticker']).columns

    # Scale features per ticker
    scalers = {t: StandardScaler() for t in tickers}
    new_df[features] = new_df[features].astype('float32')
    for t in tickers:
        mask = new_df['ticker'] == t
        new_df.loc[mask, features] = scalers[t].fit_transform(new_df.loc[mask, features])

    new_df.dropna(inplace=True)
    ticker = pd.DataFrame(new_df['ticker'])
    y = pd.DataFrame(new_df['target'])
    x = new_df.drop(columns=['target', 'ticker'])
    # Convert to torch tensors
    x_tensor = torch.tensor(x.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32)

    print(f"X shape: {x_tensor.shape}, y shape: {y_tensor.shape}")
    if ticker_necessary:
        return x_tensor, y_tensor, ticker
    else:
        return x_tensor, y_tensor


# Usage
if __name__ == '__main__':
    ticker_list = ['AAPL', 'TSLA', 'MSFT', 'SOFI', 'RKLB', 'MOD', 'INOD', 'VOO', 'AMZN', 'GOOGL', 'NFLX', 'NVDA', 'INTC', 'CSCO', 'AMD']
    X_data, y_data = get_data(ticker_list, '2015-01-01', '2025-01-01', target_len=5)
    torch.save(X_data, 'data/x_tensors.pt')
    torch.save(y_data, 'data/y_tensors.pt')
    #TODO: Stock splits are not taken into account!
