"""
Technical indicator calculations for the stock workspace.

The chart only computes indicators that the user selects, which keeps the
dashboard responsive and avoids precalculating a huge batch of unused series.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


INDICATOR_SPECS: Dict[str, dict] = {
    "sma20": {"label": "SMA 20", "pane": "overlay", "color": "#00f5ff"},
    "sma50": {"label": "SMA 50", "pane": "overlay", "color": "#fb923c"},
    "sma100": {"label": "SMA 100", "pane": "overlay", "color": "#a78bfa"},
    "sma200": {"label": "SMA 200", "pane": "overlay", "color": "#f59e0b"},
    "ema9": {"label": "EMA 9", "pane": "overlay", "color": "#22c55e"},
    "ema21": {"label": "EMA 21", "pane": "overlay", "color": "#14b8a6"},
    "ema50": {"label": "EMA 50", "pane": "overlay", "color": "#8b5cf6"},
    "vwap": {"label": "VWAP", "pane": "overlay", "color": "#f472b6"},
    "bbands": {"label": "Bollinger Bands", "pane": "overlay", "color": "#c084fc"},
    "ichimoku": {"label": "Ichimoku", "pane": "overlay", "color": "#60a5fa"},
    "donchian": {"label": "Donchian", "pane": "overlay", "color": "#38bdf8"},
    "psar": {"label": "Parabolic SAR", "pane": "overlay", "color": "#facc15"},
    "supertrend": {"label": "Supertrend", "pane": "overlay", "color": "#34d399"},
    "rsi14": {"label": "RSI 14", "pane": "lower", "color": "#a78bfa"},
    "rsi29": {"label": "RSI 29", "pane": "lower", "color": "#facc15"},
    "macd": {"label": "MACD", "pane": "lower", "color": "#00e5ff"},
    "stochastic": {"label": "Stochastic", "pane": "lower", "color": "#f97316"},
    "adx": {"label": "ADX", "pane": "lower", "color": "#e879f9"},
    "cci": {"label": "CCI", "pane": "lower", "color": "#fb7185"},
    "atr": {"label": "ATR", "pane": "lower", "color": "#60a5fa"},
    "roc": {"label": "ROC", "pane": "lower", "color": "#84cc16"},
    "mfi": {"label": "MFI", "pane": "lower", "color": "#22d3ee"},
    "obv": {"label": "OBV", "pane": "lower", "color": "#f472b6"},
    "volume_ma": {"label": "Volume MA", "pane": "volume", "color": "#cbd5e1"},
}

# Compatibility alias for older code paths and cached imports.
indicator_specs = INDICATOR_SPECS

INDICATOR_GROUPS = [
    {
        "label": "Trend",
        "items": ["sma20", "sma50", "sma100", "sma200", "ema9", "ema21", "ema50", "vwap", "supertrend"],
    },
    {
        "label": "Volatility",
        "items": ["bbands", "ichimoku", "donchian", "psar", "atr"],
    },
    {
        "label": "Momentum",
        "items": ["rsi14", "rsi29", "macd", "stochastic", "adx", "cci", "roc"],
    },
    {
        "label": "Volume / Flow",
        "items": ["mfi", "obv", "volume_ma"],
    },
]

DEFAULT_INDICATORS = ["sma20", "sma50", "ema21", "vwap", "bbands", "rsi14", "macd"]


def build_indicator_options(grouped: bool = True) -> list[dict]:
    if not grouped:
        return [{"label": spec["label"], "value": key} for key, spec in INDICATOR_SPECS.items()]

    return [
        {
            "label": group["label"],
            "options": [
                {"label": INDICATOR_SPECS[key]["label"], "value": key}
                for key in group["items"]
                if key in INDICATOR_SPECS
            ],
        }
        for group in INDICATOR_GROUPS
    ]


def group_selected_indicators(selected: Iterable[str] | None) -> list[dict]:
    selected_list = normalize_indicator_selection(selected)
    grouped: list[dict] = []

    for group in INDICATOR_GROUPS:
        items = [key for key in group["items"] if key in selected_list]
        if items:
            grouped.append(
                {
                    "label": group["label"],
                    "items": items,
                }
            )

    return grouped


def normalize_indicator_selection(selected: Iterable[str] | None) -> list[str]:
    if not selected:
        return list(DEFAULT_INDICATORS)
    valid = [value for value in selected if value in INDICATOR_SPECS]
    return valid or list(DEFAULT_INDICATORS)


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def _macd(series: pd.Series) -> pd.DataFrame:
    ema12 = _ema(series, 12)
    ema26 = _ema(series, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    hist = macd - signal
    return pd.DataFrame({"MACD": macd, "MACD_SIGNAL": signal, "MACD_HIST": hist})


def _stochastic(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    low_min = df["Low"].rolling(window=period).min()
    high_max = df["High"].rolling(window=period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df["Close"] - low_min) / denom
    d = k.rolling(window=3).mean()
    return pd.DataFrame({"STOCH_K": k.fillna(50), "STOCH_D": d.fillna(50)})


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_diff = df["High"].diff()
    low_diff = -df["Low"].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    atr = _atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)


def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = tp.rolling(window=period).mean()
    mad = (tp - sma).abs().rolling(window=period).mean()
    return ((tp - sma) / (0.015 * mad.replace(0, np.nan))).fillna(0)


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum().fillna(0)


def _mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    positive = mf.where(tp > tp.shift(), 0.0)
    negative = mf.where(tp < tp.shift(), 0.0)
    pos_sum = positive.rolling(window=period).sum()
    neg_sum = negative.rolling(window=period).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return (100 - (100 / (1 + ratio))).fillna(50)


def _roc(series: pd.Series, period: int = 12) -> pd.Series:
    return ((series / series.shift(period)) - 1) * 100


def _vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    return (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, np.nan)


def _donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DONCHIAN_HIGH": df["High"].rolling(window=period).max(),
            "DONCHIAN_LOW": df["Low"].rolling(window=period).min(),
        }
    )


def _ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    tenkan = (df["High"].rolling(9).max() + df["Low"].rolling(9).min()) / 2
    kijun = (df["High"].rolling(26).max() + df["Low"].rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((df["High"].rolling(52).max() + df["Low"].rolling(52).min()) / 2).shift(26)
    chikou = df["Close"].shift(-26)
    return pd.DataFrame(
        {
            "TENKAN": tenkan,
            "KIJUN": kijun,
            "SENKOU_A": senkou_a,
            "SENKOU_B": senkou_b,
            "CHIKOU": chikou,
        }
    )


def _psar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    high = df["High"].values
    low = df["Low"].values
    psar = np.zeros(len(df))
    bull = True
    af = step
    ep = low[0]
    psar[0] = low[0]

    for i in range(1, len(df)):
        prev = psar[i - 1]
        if bull:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], low[i - 1], low[i])
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                ep = low[i]
                af = step
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], high[i - 1], high[i])
            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                ep = high[i]
                af = step

    return pd.Series(psar, index=df.index)


def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    atr = _atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    trend = pd.Series(index=df.index, dtype=float)
    trend.iloc[0] = lowerband.iloc[0]

    for i in range(1, len(df)):
        prev = trend.iloc[i - 1]
        if df["Close"].iloc[i] > upperband.iloc[i - 1]:
            trend.iloc[i] = lowerband.iloc[i]
        elif df["Close"].iloc[i] < lowerband.iloc[i - 1]:
            trend.iloc[i] = upperband.iloc[i]
        else:
            trend.iloc[i] = prev
            if prev == upperband.iloc[i - 1] and lowerband.iloc[i] > prev:
                trend.iloc[i] = lowerband.iloc[i]
            if prev == lowerband.iloc[i - 1] and upperband.iloc[i] < prev:
                trend.iloc[i] = upperband.iloc[i]
    return trend.fillna(method="bfill")


def compute_indicators(df: pd.DataFrame, selected: Iterable[str] | None = None) -> pd.DataFrame:
    selected_list = normalize_indicator_selection(selected)
    result = df.copy()
    close = result["Close"]

    if any(name in selected_list for name in ["sma20", "sma50", "sma100", "sma200"]):
        if "sma20" in selected_list:
            result["SMA20"] = _sma(close, 20)
        if "sma50" in selected_list:
            result["SMA50"] = _sma(close, 50)
        if "sma100" in selected_list:
            result["SMA100"] = _sma(close, 100)
        if "sma200" in selected_list:
            result["SMA200"] = _sma(close, 200)

    if any(name in selected_list for name in ["ema9", "ema21", "ema50"]):
        if "ema9" in selected_list:
            result["EMA9"] = _ema(close, 9)
        if "ema21" in selected_list:
            result["EMA21"] = _ema(close, 21)
        if "ema50" in selected_list:
            result["EMA50"] = _ema(close, 50)

    if "vwap" in selected_list:
        result["VWAP"] = _vwap(result)

    if "bbands" in selected_list:
        mid = _sma(close, 20)
        std = close.rolling(window=20).std()
        result["BB_MID"] = mid
        result["BB_UP"] = mid + 2 * std
        result["BB_LOW"] = mid - 2 * std

    if "ichimoku" in selected_list:
        ichimoku = _ichimoku(result)
        result = result.join(ichimoku)

    if "donchian" in selected_list:
        result = result.join(_donchian(result))

    if "psar" in selected_list:
        result["PSAR"] = _psar(result)

    if "supertrend" in selected_list:
        result["SUPERTREND"] = _supertrend(result)

    if "rsi14" in selected_list:
        result["RSI14"] = _rsi(close, 14)
    if "rsi29" in selected_list:
        result["RSI29"] = _rsi(close, 29)
    if "macd" in selected_list:
        macd_df = _macd(close)
        result = result.join(macd_df)
    if "stochastic" in selected_list:
        result = result.join(_stochastic(result))
    if "adx" in selected_list:
        result["ADX"] = _adx(result)
    if "cci" in selected_list:
        result["CCI"] = _cci(result)
    if "atr" in selected_list:
        result["ATR"] = _atr(result)
    if "roc" in selected_list:
        result["ROC"] = _roc(close)
    if "mfi" in selected_list:
        result["MFI"] = _mfi(result)
    if "obv" in selected_list:
        result["OBV"] = _obv(result)
    if "volume_ma" in selected_list:
        result["VOLUME_MA"] = result["Volume"].rolling(window=20).mean()

    return result
