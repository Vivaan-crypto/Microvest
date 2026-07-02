"""
Offline news-sentiment scorer (one-time preprocessing, cached).

Turns the FNSPID financial-news corpus into a compact per-(ticker, day) sentiment
table that `preprocess.py` merges into the feature panel.

Pipeline:
    FNSPID (streamed in chunks)  ->  filter to our TICKERS
      ->  FinBERT score of each UNIQUE headline (p_positive - p_negative)
      ->  aggregate to (ticker, calendar_date): summed polarity + article count
      ->  write Data/news_daily.parquet

Design notes:
    * We DELIBERATELY do not shift dates or compute EWMA/buzz here. The causal
      "news is only actionable on the next trading day" roll needs the trading
      calendar, which lives in preprocess.py -- so this file stays a dumb,
      cacheable scorer keyed by the news calendar date.
    * Headlines are de-duplicated before scoring (FNSPID repeats a lot of wire
      copy), and unique-headline scores are cached, so a re-run is nearly free.

Run:
    pip install transformers            # torch + pyarrow already present
    python news_sentiment.py --source path/to/fnspid.csv        # or .parquet
    python news_sentiment.py --source ... --limit 50000         # quick smoke test

Get FNSPID from Hugging Face (Zihan1004/FNSPID); download the news file locally
and pass it with --source (streaming a 20GB file straight from HF is slow/fragile).
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
OUT_DIR = "Data"
MODEL = "ProsusAI/finbert"          # 3-class finance sentiment (pos/neg/neutral)

# FNSPID column names. Adjust here if your download uses different headers --
# the script prints the columns it actually sees if these are missing.
COL_DATE = "Date"
COL_SYMBOL = "Stock_symbol"
COL_TITLE = "Article_title"

OUT_PATH = os.path.join(OUT_DIR, "news_daily.parquet")
CACHE_PATH = os.path.join(OUT_DIR, "news_title_scores.parquet")

CHUNK_ROWS = 200_000                # rows per streaming chunk
SCORE_BATCH = 64                    # headlines per FinBERT forward pass
MAX_TOKENS = 64                     # headlines are short; cap for speed


def _load_tickers() -> List[str]:
    """Single source of truth is preprocess.TICKERS; import it lazily so this
    module doesn't drag in yfinance/matplotlib just to be imported."""
    try:
        from preprocess import TICKERS
    except ImportError as e:
        raise ImportError(
            "Could not import TICKERS from preprocess.py -- run this in the project "
            "environment (the one with yfinance installed) where preprocess.py imports."
        ) from e
    return list(TICKERS)


# --------------------------------------------------------------------------
# 1. Stream FNSPID and keep only our tickers' (date, symbol, title)
# --------------------------------------------------------------------------
def _iter_chunks(path: str):
    """Yield DataFrames of the needed columns, whatever the file format."""
    cols = [COL_DATE, COL_SYMBOL, COL_TITLE]
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        available = set(pf.schema.names)
        _check_cols(available)
        for batch in pf.iter_batches(batch_size=CHUNK_ROWS, columns=cols):
            yield batch.to_pandas()
    else:
        # Peek at the header so we can fail loudly with the real column names.
        head = pd.read_csv(path, nrows=0)
        _check_cols(set(head.columns))
        for chunk in pd.read_csv(path, usecols=cols, chunksize=CHUNK_ROWS):
            yield chunk


def _check_cols(available: set) -> None:
    missing = [c for c in (COL_DATE, COL_SYMBOL, COL_TITLE) if c not in available]
    if missing:
        raise ValueError(
            f"FNSPID file is missing expected columns {missing}. "
            f"Columns present: {sorted(available)}. "
            f"Edit COL_DATE / COL_SYMBOL / COL_TITLE at the top of this file to match."
        )


def load_news(path: str, tickers: List[str], limit: int | None = None) -> pd.DataFrame:
    tickset = set(tickers)
    kept, seen = [], 0
    for chunk in _iter_chunks(path):
        chunk = chunk[chunk[COL_SYMBOL].isin(tickset)]
        chunk = chunk.dropna(subset=[COL_TITLE, COL_DATE])
        if len(chunk):
            kept.append(chunk)
        seen += CHUNK_ROWS
        if limit is not None and seen >= limit:
            break

    if not kept:
        raise RuntimeError("No rows matched our tickers -- check the symbol column/format.")

    df = pd.concat(kept, ignore_index=True)
    df = df.rename(columns={COL_DATE: "date", COL_SYMBOL: "ticker", COL_TITLE: "title"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].str.len() > 0]
    print(f"Kept {len(df):,} news rows across {df['ticker'].nunique()} tickers "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")
    return df


# --------------------------------------------------------------------------
# 2. FinBERT: score each UNIQUE headline once (cached)
# --------------------------------------------------------------------------
def score_unique_titles(titles: np.ndarray) -> pd.DataFrame:
    """Return DataFrame[title, polarity] for the given unique titles, using a cache."""
    cache = pd.DataFrame(columns=["title", "polarity"])
    if os.path.exists(CACHE_PATH):
        cache = pd.read_parquet(CACHE_PATH)

    known = set(cache["title"])
    todo = [t for t in titles if t not in known]
    print(f"{len(titles):,} unique titles; {len(todo):,} new to score, "
          f"{len(titles) - len(todo):,} from cache.")

    if todo:
        scores = _finbert_scores(todo)
        new = pd.DataFrame({"title": todo, "polarity": scores})
        cache = pd.concat([cache, new], ignore_index=True).drop_duplicates("title", keep="last")
        cache.to_parquet(CACHE_PATH, index=False)
        print(f"Cached {len(cache):,} title scores -> {CACHE_PATH}")

    return cache[cache["title"].isin(set(titles))]


def _finbert_scores(titles: List[str]) -> np.ndarray:
    """FinBERT polarity = P(positive) - P(negative), batched on CPU."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise ImportError("FinBERT needs `transformers` (pip install transformers).") from e

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()

    # Read label order from the model config instead of hardcoding indices.
    id2label = {i: l.lower() for i, l in model.config.id2label.items()}
    pos_i = next(i for i, l in id2label.items() if l == "positive")
    neg_i = next(i for i, l in id2label.items() if l == "negative")

    out = np.empty(len(titles), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(titles), SCORE_BATCH):
            batch = titles[s:s + SCORE_BATCH]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=MAX_TOKENS, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=1).cpu().numpy()
            out[s:s + len(batch)] = probs[:, pos_i] - probs[:, neg_i]
            if s % (SCORE_BATCH * 50) == 0:
                print(f"  scored {s + len(batch):,}/{len(titles):,}", end="\r")
    print()
    return out


# --------------------------------------------------------------------------
# 3. Aggregate to (ticker, calendar_date)
# --------------------------------------------------------------------------
def aggregate(df: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(scored, on="title", how="left")
    grp = df.groupby(["ticker", "date"], sort=True)
    daily = grp.agg(sent_sum=("polarity", "sum"),
                    n_articles=("polarity", "size")).reset_index()
    print(f"Aggregated to {len(daily):,} (ticker, day) rows.")
    return daily


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Score FNSPID news with FinBERT.")
    ap.add_argument("--source", required=True, help="Path to local FNSPID .csv/.parquet")
    ap.add_argument("--end-date", default=None, help="Drop news after this date (YYYY-MM-DD)")
    ap.add_argument("--limit", type=int, default=None, help="Cap rows read (smoke test)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_news(args.source, _load_tickers(), limit=args.limit)
    if args.end_date:
        df = df[df["date"] <= pd.Timestamp(args.end_date)]
        print(f"After end-date filter: {len(df):,} rows.")

    scored = score_unique_titles(df["title"].unique())
    daily = aggregate(df, scored)

    daily.to_parquet(OUT_PATH, index=False)
    print(f"\n=== Done ===\nWrote {OUT_PATH}  ({len(daily):,} rows)")
    print(daily["n_articles"].describe().to_string())


if __name__ == "__main__":
    main()
