import yfinance as yf

print(yf.__version__)

data = yf.download(tickers="AAPL", interval="1h", period="1mo")
data
