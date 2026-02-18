"""
Data Layer
Handles all data fetching, processing, and calculations

Industry pattern: "Repository Pattern" / "Data Access Layer"
- Separates data logic from UI
- Single source of truth for data operations
- Easy to swap data sources (yfinance → API)
- Easy to add caching
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Tuple, Optional
from config import SYMBOLS, SECTORS


# =============================================================================
# DATA FETCHING
# =============================================================================

def all_stock_data() -> pd.DataFrame:
    """
    Fetch current snapshot of all stocks

    Returns:
        DataFrame with columns: Ticker, Sector, Last, Change, Size, Open, High, Low

    Industry note: In production, this would:
    - Use a proper financial API (not yfinance)
    - Implement caching
    - Handle retries and errors gracefully
    """
    try:
        data = yf.download(
            " ".join(SYMBOLS),
            period="2d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        rows = []
        for symbol in SYMBOLS:
            try:
                # Handle both multi-index and single ticker DataFrames
                cols = data[symbol] if isinstance(data.columns, pd.MultiIndex) else data
                closes = cols["Close"]
                volumes = cols["Volume"]

                last_price = float(closes.iloc[-1])
                prev_price = float(closes.iloc[-2])
                pct_change = ((last_price - prev_price) / prev_price) * 100

                # Calculate size for treemap (log scale for better visualization)
                market_weight = np.log10(max(last_price * float(volumes.iloc[-1]), 1))

                rows.append({
                    "Ticker": symbol,
                    "Sector": SECTORS.get(symbol, "Unknown"),
                    "Last": round(last_price, 2),
                    "Change": round(pct_change, 2),
                    "Size": market_weight,
                    "Open": cols["Open"],
                    "High": cols["High"],
                    "Low": cols["Low"],
                })
            except Exception as e:
                # Skip stocks that fail to fetch
                continue

        return pd.DataFrame(rows)

    except Exception as e:
        # Return empty DataFrame on complete failure
        return pd.DataFrame()


def single_stock_data(ticker: str, period_days: int = 365) -> pd.DataFrame:
    """
    Fetch historical data for a single stock

    Args:
        ticker: Stock symbol (e.g., "AAPL")
        period_days: Time period in days (int)

    Returns:
        DataFrame with OHLCV data

    Industry note: Cache this data since historical data doesn't change
    """

    hist = yf.download(
        ticker,
        start=pd.Timestamp.now() - pd.Timedelta(days=period_days),
        end=pd.Timestamp.now(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    return hist


# =============================================================================
# DATA PROCESSING & CALCULATIONS
# =============================================================================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators (SMA, Bollinger Bands, RSI)

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added indicator columns

    Industry note: These calculations are "pure functions" -
    they don't modify the original data, making them easy to test
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Simple Moving Averages
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()

    # Bollinger Bands
    df["BB_mid"] = df["Close"].rolling(window=20).mean()
    df["BB_std"] = df["Close"].rolling(window=20).std()
    df["BB_up"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_low"] = df["BB_mid"] - 2 * df["BB_std"]

    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def calculate_market_stats(df: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate overall market statistics

    Args:
        df: DataFrame with stock data including 'Change' column

    Returns:
        Dictionary with keys: gainers, losers, avg_change

    Example:
        stats = calculate_market_stats(snapshot_df)
        print(f"Gainers: {stats['gainers']}")
    """
    if df.empty:
        return {"gainers": 0, "losers": 0, "avg_change": 0.0}

    return {
        "gainers": int((df["Change"] > 0).sum()),
        "losers": int((df["Change"] < 0).sum()),
        "avg_change": float(df["Change"].mean())
    }


def calculate_watchlist_stats(df: pd.DataFrame, watchlist_tickers: list) -> Dict[str, any]:
    """
    Calculate statistics for watchlist stocks only

    Args:
        df: DataFrame with all stock data
        watchlist_tickers: List of ticker symbols to filter by

    Returns:
        Dictionary with keys: gainers, losers, avg_change
    """
    if df.empty:
        return {"gainers": 0, "losers": 0, "avg_change": 0.0}

    watchlist_df = df[df["Ticker"].isin(watchlist_tickers)]

    if watchlist_df.empty:
        return {"gainers": 0, "losers": 0, "avg_change": 0.0}

    return {
        "gainers": int((watchlist_df["Change"] > 0).sum()),
        "losers": int((watchlist_df["Change"] < 0).sum()),
        "avg_change": float(watchlist_df["Change"].mean())
    }


def get_stock_by_ticker(df: pd.DataFrame, ticker: str) -> Optional[Dict]:
    """
    Get data for a specific ticker

    Args:
        df: DataFrame with stock data
        ticker: Stock symbol to retrieve

    Returns:
        Dictionary with stock data, or None if not found

    Example:
        stock = get_stock_by_ticker(df, "AAPL")
        if stock:
            print(f"Price: ${stock['Last']}")
    """
    if df.empty or ticker not in df["Ticker"].values:
        return None

    return df[df["Ticker"] == ticker].iloc[0].to_dict()


# =============================================================================
# VOLUME ANALYSIS
# =============================================================================

def calculate_volume_colors(df: pd.DataFrame) -> list:
    """
    Calculate colors for volume bars based on price movement

    Args:
        df: DataFrame with OHLCV data

    Returns:
        List of colors (green for up days, red for down days)

    Industry pattern: Separating visual logic from data logic
    keeps your code maintainable
    """
    from config import Colors

    return [
        Colors.SUCCESS if df["Close"].iloc[i] >= df["Open"].iloc[i] else Colors.DANGER
        for i in range(len(df))
    ]


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_snapshot_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate snapshot data quality

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, error_message)

    Industry best practice: Always validate external data

    Example:
        is_valid, error = validate_snapshot_data(df)
        if not is_valid:
            print(f"Data error: {error}")
    """
    if df.empty:
        return False, "No data available"

    required_columns = ["Ticker", "Sector", "Last", "Change", "Size"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return False, f"Missing columns: {', '.join(missing_columns)}"

    if len(df) < 5:
        return False, f"Insufficient data: only {len(df)} stocks fetched"

    return True, ""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_percentage(value: float, include_sign: bool = True) -> str:
    """
    Format percentage for display

    Args:
        value: Percentage value (e.g., 2.5)
        include_sign: Whether to include +/- sign

    Returns:
        Formatted string (e.g., "+2.50%")
    """
    sign = "+" if value > 0 and include_sign else ""
    return f"{sign}{value:.2f}%"


def format_price(value: float, currency: str = "$") -> str:
    """
    Format price for display

    Args:
        value: Price value (e.g., 175.50)
        currency: Currency symbol

    Returns:
        Formatted string (e.g., "$175.50")
    """
    return f"{currency}{value:.2f}"
