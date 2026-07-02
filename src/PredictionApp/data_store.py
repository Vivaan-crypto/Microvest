"""
Disk cache for everything expensive: raw prices, featurized panels, universes.

Design: download/compute ONCE, save to parquet under Data/cache/, reuse forever.
Cache keys hash the inputs that define the artifact (tickers + dates + config
version), so changing the universe or bumping CONFIG_VERSION naturally produces
a fresh file instead of silently reusing stale data. Pass refresh=True to force
a rebuild of any artifact.
"""

import hashlib
import os
from typing import Callable, List

import pandas as pd
import yfinance as yf

import config

# The original hand-picked 100 mega-caps (fallback universe).
LEGACY_100: List[str] = [
    "AAPL",
    "ABBV", "ADBE", "AMD", "AMZN", "APD", "ASML", "AVGO", "BA", "BAC",
    "C", "CAT", "COP", "COST", "CRM", "CVX", "DIS", "DUK", "EOG", "EXC",
    "FCX", "GE", "GOOGL", "GS", "HD", "HON", "INTC", "JNJ", "JPM", "KO",
    "LIN", "META", "MMM", "MRK", "MS", "MSFT", "NEE", "NFLX", "NVDA", "PEP",
    "PFE", "PG", "SHW", "SLB", "SO", "TSLA", "UNH", "WFC", "WMT", "XOM",
    "ACN", "ADP", "AIG", "AMAT", "AMGN", "AMT", "AXP", "BK", "BKNG", "BLK",
    "BMY", "BX", "CB", "CL", "CMCSA", "CME", "CSCO", "DE", "DHR", "EMR",
    "F", "FDX", "GD", "GILD", "GM", "IBM", "ITW", "LLY", "LMT", "LOW",
    "MA", "MCD", "MDT", "MO", "NKE", "ORCL", "PM", "QCOM", "RTX", "SBUX",
    "SPG", "T", "TGT", "TMO", "TXN", "UNP", "UPS", "USB", "V", "VZ",
]

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _ensure_cache_dir():
    os.makedirs(config.CACHE_DIR, exist_ok=True)


def _key(*parts) -> str:
    """Stable short hash of whatever defines a cached artifact."""
    blob = "|".join(str(p) for p in parts)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def cached_frame(name: str, key_parts: tuple, builder: Callable[[], pd.DataFrame],
                 refresh: bool = False) -> pd.DataFrame:
    """Generic parquet cache: return the saved frame if it exists, else build,
    save, and return. `name` is a human-readable prefix so the cache dir is
    browsable; `key_parts` are hashed into the filename."""
    _ensure_cache_dir()
    path = os.path.join(config.CACHE_DIR, f"{name}_{_key(*key_parts)}.parquet")
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path)
    df = builder()
    df.to_parquet(path, index=False)
    print(f"  cached {name} -> {os.path.basename(path)} ({len(df):,} rows)")
    return df


# ------------------------------------------------------------------
# Universe
# ------------------------------------------------------------------
def sp500_tickers(refresh: bool = False) -> List[str]:
    """Current S&P 500 constituents from Wikipedia, cached to disk.

    NOTE: this is TODAY'S list — survivorship bias (see config.py). Falls back
    to LEGACY_100 if the fetch fails (no internet / page format change).
    """
    def _fetch() -> pd.DataFrame:
        # Wikipedia 403s bare urllib (pandas' default); send a browser UA and
        # hand the fetched HTML to read_html via StringIO.
        import io
        import urllib.request
        req = urllib.request.Request(_SP500_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html))
        symbols = tables[0]["Symbol"].astype(str).str.strip()
        # Wikipedia uses BRK.B / BF.B; yfinance wants BRK-B / BF-B.
        symbols = symbols.str.replace(".", "-", regex=False)
        return pd.DataFrame({"ticker": sorted(symbols.unique())})

    try:
        df = cached_frame("universe_sp500", ("sp500",), _fetch, refresh=refresh)
        tickers = df["ticker"].tolist()
        if len(tickers) < 400:   # sanity: a mangled scrape shouldn't pass
            raise ValueError(f"only {len(tickers)} symbols parsed")
        return tickers
    except Exception as e:
        print(f"WARN: S&P 500 fetch failed ({e}); falling back to LEGACY_100")
        return list(LEGACY_100)


def resolve_universe(name: str = None) -> List[str]:
    name = name or config.UNIVERSE_NAME
    if name == "sp500":
        return sp500_tickers()
    if name == "legacy100":
        return list(LEGACY_100)
    raise ValueError(f"Unknown universe: {name}")


# ------------------------------------------------------------------
# Raw prices
# ------------------------------------------------------------------
def _download_chunk(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """One threaded yfinance call for a batch of tickers -> long frame."""
    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    raw = yf.download(tickers, start=start, end=end, interval="1d",
                      auto_adjust=True, progress=False, group_by="ticker",
                      threads=True)
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=cols + ["ticker"])
    multi = isinstance(raw.columns, pd.MultiIndex)
    frames = []
    for t in tickers:
        if multi:
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t].copy()
        else:
            df = raw.copy()
        df = df.reset_index()
        if not set(cols).issubset(df.columns):
            continue
        df = df[cols].dropna(subset=["Close"])
        df["ticker"] = t
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=cols + ["ticker"])
    return pd.concat(frames, ignore_index=True)


def get_raw_panel(tickers: List[str], start: str = None, end: str = None,
                  refresh: bool = False, chunk_size: int = 50) -> pd.DataFrame:
    """Long OHLCV frame (Date, O, H, L, C, V, ticker) for a universe — cached.

    First call for a big universe downloads in chunks (yfinance is flaky on
    500-name single calls); every call after that is a ~1s parquet read.
    """
    start = start or config.START_DATE
    end = end or config.END_DATE
    tickers = sorted(set(tickers))

    def _build() -> pd.DataFrame:
        frames = []
        for i in range(0, len(tickers), chunk_size):
            batch = tickers[i:i + chunk_size]
            print(f"  downloading {i + 1}-{i + len(batch)} of {len(tickers)}...")
            frames.append(_download_chunk(batch, start, end))
        panel = pd.concat(frames, ignore_index=True)
        panel["Date"] = pd.to_datetime(panel["Date"])
        got = panel["ticker"].nunique()
        if got < len(tickers):
            missing = sorted(set(tickers) - set(panel["ticker"].unique()))
            print(f"  WARN: no data for {len(missing)} tickers: {missing[:10]}...")
        return panel.sort_values(["ticker", "Date"]).reset_index(drop=True)

    return cached_frame("raw", (tuple(tickers), start, end), _build, refresh=refresh)
