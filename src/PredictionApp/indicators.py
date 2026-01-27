"""
Pure helper functions for feature and target calculations.

Each function:
    - Takes pandas.Series as input
    - Returns pandas.Series (or tuple of Series)
    - Contains no DataFrame / groupby / ticker logic
"""

import pandas as pd
import numpy as np


# -----------------------------
# Basic technical indicators
# -----------------------------

def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder)."""
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(length, min_periods=length).mean()
    avg_loss = loss.rolling(length, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))

    return rsi_val


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
):
    """
    MACD, signal line, and histogram.
    Returns:
        macd_line, signal_line, hist
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# -----------------------------
# Return and volatility helpers
# -----------------------------

def pct_return(series: pd.Series, periods: int = 1) -> pd.Series:
    """Backward-looking percentage return over 'periods' steps."""
    return series.pct_change(periods=periods)


def forward_return(series: pd.Series, horizon: int) -> pd.Series:
    """
    Forward percentage return:
        (P_{t+horizon} / P_t) - 1
    """
    future = series.shift(-horizon)
    return (future / series) - 1.0


def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """
    Rolling realized volatility (std) of returns over a window.
    'returns' should already be something like series.pct_change().
    """
    return returns.rolling(window, min_periods=window).std()


# -----------------------------
# Volume-based features
# -----------------------------

def volume_zscore(volume: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score of volume:
        (V_t - mean) / std over 'window'.
    """
    roll_mean = volume.rolling(window, min_periods=window).mean()
    roll_std = volume.rolling(window, min_periods=window).std()
    return (volume - roll_mean) / roll_std.replace(0, np.nan)
