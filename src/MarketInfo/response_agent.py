"""
Compatibility helpers for legacy imports.

The original version of this module mixed RSS fetching, SEC scraping, and LLM
stubs in one place. That made the app harder to reason about and introduced a
few broken code paths. The news utilities now live in `news.py`; this module is
kept as a thin shim so older imports do not explode.
"""

from __future__ import annotations

from news import RSS_FEEDS, build_news_context, fetch_all_news, fetch_rss


def fetch_company_financials(ticker: str) -> dict:
    """Legacy placeholder for SEC lookup."""
    return {}


def get_response(message: str) -> str:
    """Legacy response hook.

    The AI page now calls OpenAI directly, so this stays as a lightweight shim
    for any code that still imports it.
    """
    return message
