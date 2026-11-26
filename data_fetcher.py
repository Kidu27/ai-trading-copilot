"""
data_fetcher.py — Gate 1: Market Data

Fetches OHLC candle data from Twelve Data REST API.
Provides a mock-data generator for offline unit testing.

Rate limit awareness: Twelve Data free tier allows 8 req/min.
We always sleep briefly between requests to stay safe.
"""

import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import config


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_candles(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    """
    Fetch OHLC candles from Twelve Data /time_series endpoint.

    Args:
        symbol:     Pair string, e.g. "EUR/USD"
        interval:   "1h" or "15min"
        outputsize: Number of candles (max 5000, free tier ~800)

    Returns:
        DataFrame with columns [open, high, low, close] as float64,
        DatetimeIndex sorted oldest → newest.

    Raises:
        RuntimeError: on API errors or missing API key.
    """
    if not config.TWELVE_DATA_API_KEY:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY is not set. "
            "Copy .env.example → .env and add your key, "
            "or use --mock mode."
        )

    params = {
        "symbol":     symbol,
        "interval":   interval,
        "outputsize": outputsize,
        "apikey":     config.TWELVE_DATA_API_KEY,
        "format":     "JSON",
        "timezone":   "UTC",
    }

    url = f"{config.TD_BASE_URL}/time_series"
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error fetching {symbol} {interval}: {exc}") from exc

    payload = response.json()

    # Twelve Data signals errors inside the JSON body
    if payload.get("status") == "error":
        code    = payload.get("code", "unknown")
        message = payload.get("message", "No message")
        raise RuntimeError(
            f"Twelve Data API error [{code}] for {symbol} {interval}: {message}"
        )

    raw_values = payload.get("values")
    if not raw_values:
        raise RuntimeError(
            f"No candle data returned for {symbol} {interval}. "
            "Check symbol format (e.g. 'EUR/USD') and your API plan."
        )

    return _parse_candles(raw_values)


def fetch_pair_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: fetches both timeframes for one pair.

    Returns:
        (df_1h, df_15m) — both DataFrames ready for market_math.py

    Note:
        Sleeps 8 seconds between the two requests to respect the
        Twelve Data free-tier rate limit of 8 requests/minute.
    """
    print(f"  [data_fetcher] Fetching 1H candles for {symbol} …")
    df_1h = fetch_candles(symbol, "1h", config.TD_1H_CANDLES)

    print(f"  [data_fetcher] Sleeping 8 s (rate-limit guard) …")
    time.sleep(8)

    print(f"  [data_fetcher] Fetching 15m candles for {symbol} …")
    df_15m = fetch_candles(symbol, "15min", config.TD_15M_CANDLES)

    return df_1h, df_15m


# ─────────────────────────────────────────────────────────────────────────────
# Mock data generator — zero API key required
# ─────────────────────────────────────────────────────────────────────────────

def generate_mock_data(
    n_1h: int = 100,
    n_15m: int = 50,
    seed: int = 42,
    regime: str = "uptrend",   # "uptrend" | "downtrend" | "ranging"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic synthetic Forex OHLC data for offline testing.

    The generator builds a price walk with configurable regime so
    market_math.py can be tested against known expected outcomes.

    Args:
        n_1h:    Number of 1H candles.
        n_15m:   Number of 15m candles.
        seed:    RNG seed for reproducibility.
        regime:  Price regime to simulate.

    Returns:
        (df_1h, df_15m) with the same schema as fetch_candles().
    """
    rng = np.random.default_rng(seed)

    def _build_ohlc(
        n: int,
        start_price: float,
        drift: float,
        volatility: float,
        freq: str,
    ) -> pd.DataFrame:
        """Build a synthetic OHLC DataFrame."""
        # Generate log-return walk
        returns = rng.normal(loc=drift, scale=volatility, size=n)
        closes  = start_price * np.exp(np.cumsum(returns))

        # Construct realistic H/L/O around close
        hl_range = np.abs(rng.normal(loc=0, scale=volatility * 2, size=n)) * closes
        highs    = closes + hl_range * rng.uniform(0.3, 1.0, size=n)
        lows     = closes - hl_range * rng.uniform(0.3, 1.0, size=n)
        # Open is previous close (with tiny gap noise)
        opens    = np.roll(closes, 1)
        opens[0] = start_price

        # Build datetime index (end at "now", step backwards)
        end_time = pd.Timestamp("2026-11-23 12:00:00", tz="UTC")
        idx = pd.date_range(end=end_time, periods=n, freq=freq)

        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes},
            index=idx,
        )
        df = df.astype(float)
        return df

    regime_params = {
        "uptrend":   {"drift":  0.0003, "vol": 0.0015},
        "downtrend": {"drift": -0.0003, "vol": 0.0015},
        "ranging":   {"drift":  0.0000, "vol": 0.0008},
    }
    p = regime_params.get(regime, regime_params["uptrend"])

    df_1h  = _build_ohlc(n_1h,  1.0850, p["drift"], p["vol"],  "1h")
    df_15m = _build_ohlc(n_15m, df_1h["close"].iloc[-1], p["drift"], p["vol"] / 2, "15min")

    return df_1h, df_15m


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_candles(raw_values: list[dict]) -> pd.DataFrame:
    """
    Convert Twelve Data raw JSON list → clean DataFrame.

    Twelve Data returns newest-first; we reverse to oldest-first.
    """
    df = pd.DataFrame(raw_values)

    # Keep only needed columns
    df = df[["datetime", "open", "high", "low", "close"]]

    # Parse datetime — Twelve Data format: "2024-01-15 09:00:00"
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime")

    # Twelve Data returns newest → oldest; flip to oldest → newest
    df = df.iloc[::-1].copy()

    # Cast prices to float
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    return df
