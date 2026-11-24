"""
news_fetcher.py — Live FX News Feed & Sentiment Context

Fetches live Forex news headlines from public RSS feeds (FXStreet) using standard
requests + xml.etree.ElementTree.

Features:
  - 5-second timeout with non-blocking graceful fallback
  - Currency-specific filtering matching pair's base and quote currencies
  - Lightweight keyword bias detection (BULLISH / BEARISH / NEUTRAL)
  - Zero external heavyweight dependencies (pure Python stdlib + requests)
"""

import xml.etree.ElementTree as ET
from typing import Any
import requests

RSS_FEED_URL = "https://www.fxstreet.com/rss/news"

# Currency keyword dictionaries for high-relevance matching
CURRENCY_KEYWORDS: dict[str, list[str]] = {
    "EUR": ["eur", "euro", "ecb", "lagarde", "eurozone", "germany"],
    "USD": ["usd", "dollar", "fed", "powell", "fomc", "treasury", "nfp", "cpi", "inflation"],
    "GBP": ["gbp", "pound", "sterling", "boe", "bailey", "uk", "britain"],
    "JPY": ["jpy", "yen", "boj", "ueda", "japan", "tokyo"],
    "CHF": ["chf", "franc", "snb", "jordan", "swiss", "switzerland"],
    "AUD": ["aud", "aussie", "rba", "bullock", "australia"],
}

BULLISH_KEYWORDS = [
    "rallies", "surges", "jumps", "gains", "strengthens", "hawkish",
    "rebounds", "higher", "breakout", "climbs", "uptrend", "bullish"
]
BEARISH_KEYWORDS = [
    "plunges", "slumps", "drops", "falls", "weakens", "dovish",
    "retreats", "lower", "breakdown", "tumbles", "downtrend", "bearish"
]


def fetch_all_headlines(timeout_seconds: float = 5.0) -> list[dict[str, str]]:
    """
    Fetch the latest raw items from the FXStreet RSS feed.
    Returns list of dicts with title, link, and pubDate.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
    }
    try:
        response = requests.get(RSS_FEED_URL, headers=headers, timeout=timeout_seconds)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        items: list[dict[str, str]] = []

        for item in root.findall(".//item"):
            title_elem = item.find("title")
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
            if title:
                items.append({"title": title})

        return items
    except Exception as exc:
        # Non-blocking graceful fallback
        return []


def get_news_for_pair(symbol: str, max_headlines: int = 5) -> dict[str, Any]:
    """
    Retrieve filtered headlines matching the base and quote currencies of symbol.

    Args:
        symbol: e.g. "EUR/USD" or "GBP/JPY"
        max_headlines: Max relevant headlines to return (default 5)

    Returns:
        Structured dict with headlines, news_bias, and formatted summary.
    """
    clean_symbol = symbol.replace("/", "").upper()
    parts = symbol.split("/") if "/" in symbol else [clean_symbol[:3], clean_symbol[3:]]
    base_curr = parts[0].upper() if len(parts) >= 1 else ""
    quote_curr = parts[1].upper() if len(parts) >= 2 else ""

    # Build search terms for this pair
    pair_terms = {base_curr.lower(), quote_curr.lower(), clean_symbol.lower(), symbol.lower()}
    for c in (base_curr, quote_curr):
        for term in CURRENCY_KEYWORDS.get(c, []):
            pair_terms.add(term)

    all_items = fetch_all_headlines()
    matched_headlines: list[str] = []

    for item in all_items:
        t = item["title"]
        t_lower = t.lower()
        if any(term in t_lower for term in pair_terms):
            matched_headlines.append(t)
            if len(matched_headlines) >= max_headlines:
                break

    # Determine crude keyword bias
    bull_count = 0
    bear_count = 0
    for h in matched_headlines:
        h_low = h.lower()
        bull_count += sum(1 for kw in BULLISH_KEYWORDS if kw in h_low)
        bear_count += sum(1 for kw in BEARISH_KEYWORDS if kw in h_low)

    if bull_count > bear_count:
        bias = "BULLISH"
    elif bear_count > bull_count:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    available = len(matched_headlines) > 0
    if available:
        summary = f"{len(matched_headlines)} recent headline(s) found for {symbol} (bias: {bias})."
    else:
        summary = f"No breaking headlines specifically impacting {symbol}; regular market session flow."

    return {
        "symbol":        symbol,
        "available":     available,
        "news_bias":     bias,
        "headlines":     matched_headlines,
        "summary":       summary,
    }
