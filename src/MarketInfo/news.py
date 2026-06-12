"""
News utilities for AI-assisted market context.

This keeps RSS fetching and context building in one place so the AI page does
not have to own the data plumbing.
"""

from __future__ import annotations

import feedparser

RSS_FEEDS = {
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Reuters": "https://feeds.reuters.com/reuters/businessNews",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
}


def fetch_rss(feed_name: str, url: str, limit: int = 10) -> list[dict]:
    """Fetch a single RSS feed and return normalized article dictionaries."""
    feed = feedparser.parse(url)
    if feed.bozo:
        return []

    articles: list[dict] = []
    for entry in feed.entries[:limit]:
        articles.append(
            {
                "source": feed_name,
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
            }
        )
    return articles


def fetch_all_news(limit: int = 10) -> list[dict]:
    """Fetch and flatten all configured RSS feeds."""
    articles: list[dict] = []
    for name, url in RSS_FEEDS.items():
        articles.extend(fetch_rss(name, url, limit))
    return articles


def build_news_context(ticker: str = "", limit: int = 10) -> str:
    """Build a short text block for LLM context."""
    articles = fetch_all_news(limit=limit)
    if ticker:
        ticker = ticker.upper().strip()
        relevant = [
            article
            for article in articles
            if ticker in f"{article.get('title', '')} {article.get('summary', '')}".upper()
        ]
        if relevant:
            articles = relevant + [article for article in articles if article not in relevant]

    lines = []
    for article in articles[:limit]:
        lines.append(
            f"[{article.get('source', '')}] {article.get('title', '')} - "
            f"{article.get('summary', '')[:200]}"
        )
    return "\n".join(lines)
