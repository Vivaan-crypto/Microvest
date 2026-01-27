import torch
import tqdm

from model import StockLSTMModel
import os
import pandas as pd
from datetime import timedelta, datetime
import yfinance as yf
import numpy as np
import talib
import math
WINDOW = 20
PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]
INDICATOR_COLS = ["SMA20", "EMA12", "RSI14", "MACD", "MACD_signal", "MACD_hist"]  # 6 features
PRICE_INPUT_SIZE = 5
INDICATOR_INPUT_SIZE = 6
PATH = "C:/Users/shahv/OneDrive/Documents/GitHub/Microvest/src/PredictionApp/Model/model.ckpt"
"""
sp500 = [
    "NVDA", "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "AVGO", "META", "TSLA",
    "LLY", "JPM", "WMT", "ORCL", "V", "XOM", "MA", "JNJ", "NFLX", "PLTR",
    "ABBV", "COST", "AMD", "BAC", "HD", "PG", "GE", "CVX", "CSCO", "KO",
    "UNH", "IBM", "MU", "WFC", "MS", "CAT", "GS", "AXP", "PM", "TMUS",
    "RTX", "CRM", "MRK", "ABT", "MCD", "TMO", "PEP", "LIN", "ISRG", "UBER",
    "DIS", "APP", "QCOM", "LRCX", "INTU", "T", "AMGN", "AMAT", "C", "NOW",
    "NEE", "BX", "VZ", "BLK", "INTC", "SCHW", "ANET", "APH", "BKNG", "TJX",
    "GEV", "DHR", "GILD", "BSX", "ACN", "SPGI", "KLAC", "BA", "TXN", "PFE",
    "PANW", "ADBE", "SYK", "ETN", "CRWD", "COF", "WELL", "UNP", "PGR", "DE",
    "LOW", "HON", "MDT", "PLD", "CB", "ADI", "COP", "VRTX", "HOOD", "HCA",
    "LMT", "KKR", "CEG", "PH", "MCK", "CME", "ADP", "CMCSA", "SO", "CVS",
    "MO", "SBUX", "NEM", "DUK", "BMY", "NKE", "GD", "TT", "DELL", "MMC",
    "DASH", "MMM", "ICE", "AMT", "CDNS", "MCO", "WM", "ORLY", "SHW", "HWM",
    "UPS", "NOC", "JCI", "EQIX", "BK", "MAR", "COIN", "APO", "TDG", "AON",
    "CTAS", "WMB", "ABNB", "MDLZ", "ECL", "USB", "ELV", "SNPS", "PNC", "CI",
    "EMR", "REGN", "ITW", "GLW", "COR", "TEL", "MNST", "RCL", "SPG", "AJG",
    "GM", "CSX", "RSG", "DDOG", "AEP", "AZO", "TRV", "PWR", "CMI", "NSC",
    "ADSK", "MSI", "FDX", "CL", "HLT", "WDAY", "FTNT", "KMI", "MPC", "SRE",
    "AFL", "EOG", "VST", "PYPL", "APD", "FCX", "TFC", "PSX", "WBD", "STX",
    "ALL", "VLO", "BDX", "DLR", "SLB", "IDXX", "LHX", "WDC", "ZTS", "URI",
    "F", "O", "ROST", "MET", "D", "PCAR", "EA", "EW", "NDAQ", "NXPI",
    "CAH", "ROP", "PSA", "BKR", "XEL", "FAST", "EXC", "CARR", "CBRE", "CTVA",
    "AME", "OKE", "KR", "LVS", "MPWR", "GWW", "AXON", "TTWO", "ETR", "FANG",
    "AMP", "MSCI", "ROK", "OXY", "AIG", "DHI", "CMG", "A", "YUM", "PEG",
    "FICO", "TGT", "PAYX", "CCI", "CPRT", "EBAY", "DAL", "IQV", "PRU", "EQT",
    "GRMN", "HIG", "TRGP", "VMC", "VTR", "KDP", "XYZ", "ED", "HSY", "PCG",
    "WEC", "MLM", "TKO", "SYY", "RMD", "CTSH", "WAB", "XYL", "OTIS", "KMB",
    "CCL", "NUE", "ACGL", "GEHC", "FIS", "STT", "VICI", "EXPE", "KVUE", "EL",
    "NRG", "LYV", "RJF", "LEN", "WTW", "KEYS", "UAL", "HPE", "VRSK", "IR",
    "CHTR", "EXR", "KHC", "IBKR", "TSCO", "WRB", "K", "MCHP", "CSGP", "FOXA",
    "MTB", "MTD", "HUM", "DTE", "AEE", "ADM", "FITB", "ATO", "ROL", "EXE",
    "EME", "ODFL", "BRO", "ES", "FOX", "PPL", "FSLR", "CBOE", "IRM", "TER",
    "FE", "BR", "SYF", "CNP", "AWK", "CINF", "STE", "EFX", "GIS", "AVB",
    "DOV", "HBAN", "BIIB", "VLTO", "LDOS", "NTRS", "ULTA", "TDY", "TPL", "VRSN",
    "PODD", "EQR", "PHM", "HUBB", "HAL", "DG", "HPQ", "STLD", "DXCM", "EIX",
    "WAT", "CMS", "DVN", "STZ", "CFG", "TROW", "WSM", "LH", "RF", "NTAP",
    "PPG", "SMCI", "L", "JBL", "PTC", "DLTR", "SBAC", "DGX", "TPR", "NVR",
    "NI", "INCY", "TTD", "LULU", "DRI", "CHD", "TYL", "RL", "CTRA", "IP",
    "AMCR", "CPAY", "KEY", "TSN", "CDW", "ON", "WST", "BG", "PFG", "EXPD",
    "J", "TRMB", "CHRW", "CNC", "SW", "ZBH", "GPC", "PKG", "EVRG", "GPN",
    "MKC", "GDDY", "Q", "INVH", "LNT", "PSKY", "SNA", "PNR", "APTV", "LUV",
    "ESS", "IFF", "IT", "DD", "LII", "HOLX", "GEN", "FTV", "DOW", "WY",
    "BBY", "MAA", "JBHT", "NWS", "ERIE", "NWSA", "LYB", "COO", "TXT", "UHS",
    "OMC", "ALLE", "KIM", "DPZ", "EG", "ALB", "FFIV", "AVY", "CF",
    "SOLV", "REG", "NDSN", "BALL", "CLX", "MAS", "UDR", "AKAM", "BXP", "HRL",
    "WYNN", "VTRS", "HII", "IEX", "DOC", "HST", "ZBRA", "DECK", "JKHY", "SJM",
    "BEN", "AIZ", "BLDR", "CPT", "DAY", "HAS", "PNW", "RVTY", "GL", "IVZ",
    "FDS", "SWK", "SWKS", "EPAM", "AES", "ALGN", "MRNA", "BAX", "CPB", "TECH",
    "TAP", "PAYC", "ARE", "POOL", "AOS", "IPG", "MGM", "GNRC", "APA", "DVA",
    "FRT", "HSIC", "CAG", "NCLH", "MOS", "CRL", "LW", "LKQ", "MTCH", "MOH",
    "SOLS", "MHK"
]
"""
sp500 = [    "NVDA", "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "AVGO", "META", "TSLA",
    "LLY", "JPM", "WMT", "ORCL", "V", "XOM", "MA", "JNJ", "NFLX", "PLTR",
    "ABBV", "COST", "AMD", "BAC", "HD", "PG", "GE", "CVX", "CSCO", "KO",
    "UNH", "IBM", "MU", "WFC", "MS", "CAT", "GS", "AXP", "PM", "TMUS",
    "RTX", "CRM", "MRK", "ABT", "MCD", "TMO", "PEP", "LIN", "ISRG", "UBER",
    "DIS", "APP", "QCOM", "LRCX", "INTU", "T", "AMGN", "AMAT", "C", "NOW",
    "NEE", "BX", "VZ", "BLK", "INTC", "SCHW", "ANET", "APH", "BKNG", "TJX",
    "GEV", "DHR", "GILD", "BSX", "ACN", "SPGI", "KLAC", "BA", "TXN", "PFE",
    "PANW", "ADBE", "SYK", "ETN", "CRWD", "COF", "WELL", "UNP", "PGR", "DE"]
TIMEDELTA = 5


def zscore_window(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True)
    return (arr - mu) / (sd + eps)


def fetch_process_data(ticker: str, start, end) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    # Dummy implementation for testing
    """Download daily OHLCV data for a single ticker."""
    _df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False,
                      multi_level_index=False).rename_axis("Date").reset_index().set_index("Date")
    current_price = _df["Close"].iloc[-1]
    current_return = 100 * ((_df["Close"].iloc[-1] - _df["Close"].iloc[-TIMEDELTA])/(_df["Close"].iloc[-TIMEDELTA]))
    _df = _df.iloc[:-7]
    prev_price = _df["Close"].iloc[-1]

    # indicators
    _df["SMA20"] = talib.SMA(_df["Close"], timeperiod=20)
    _df["EMA12"] = talib.EMA(_df["Close"], timeperiod=12)
    _df["RSI14"] = talib.RSI(_df["Close"], timeperiod=14)
    _df["MACD"], _df["MACD_signal"], _df["MACD_hist"] = talib.MACD(
        _df["Close"], fastperiod=12, slowperiod=26, signalperiod=9
    )

    _df.dropna(inplace=True)
    # TODO: df -> tensor
    ohclvNumpy = zscore_window(_df[PRICE_COLS].to_numpy().astype("float32"))
    ohclvTensor = torch.from_numpy(ohclvNumpy[None, :, :])

    indicatorsNumpy = _df[INDICATOR_COLS].to_numpy().astype("float32")
    indicatorsTensor = torch.from_numpy(indicatorsNumpy)

    return ohclvTensor, indicatorsTensor[None, -1], prev_price, current_price, current_return


def load_model() -> StockLSTMModel:
    weights_path = os.path.abspath(PATH)
    model = StockLSTMModel()

    if os.path.isfile(weights_path):
        ckpt = torch.load(weights_path)

        state_dict = ckpt.get("state_dict", ckpt)

        # Strip "model." prefix (since you saved LightningModule.state_dict())
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned[k[len("model."):]] = v
            else:
                cleaned[k] = v
        model.load_state_dict(cleaned, strict=False)
    else:
        print(
            f"Checkpoint not found at {weights_path}. "
            "Using randomly initialized weights."
        )
    model.eval()
    return model


def main():
    model = load_model()
    direction_counter = 0
    pbar = tqdm.tqdm(sp500)
    for index, ticker in enumerate(pbar):
        ohclv_data, indicators_data, prev_price, current_price, current_return = fetch_process_data(ticker, pd.Timestamp("2020-01-01"), datetime.date(datetime.now()))
        pred = model(ohclv_data, indicators_data)
        pbar.set_description(f"Ticker: {ticker}, Predicted Price: {(pred.item() + 1) * prev_price}, Last price: {current_price} || Predicted Return: {pred.item() * 100}%, Actual Return: {current_return}%")
        if math.copysign(1, pred.item()) == math.copysign(1, current_return):
            direction_counter += 1

    print(f"Directional Accuracy: {direction_counter / len(sp500) * 100}%")
    # TODO: SP500 stocks prediction vs actual (directionally). Maybe even add weights based on differences.


if __name__ == "__main__":
    main()
