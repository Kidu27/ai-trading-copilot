"""
journal.py — Trade logging & daily circuit breaker.

Every AI-audited setup (regardless of decision) is appended to
trades_log.csv. The circuit breaker checks today's losing trades
and blocks further alerts if MAX_DAILY_LOSSES is reached.
"""

import csv
import os
from datetime import datetime
from typing import Any

import pytz

import config

# ── File path ─────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(__file__), "trades_log.csv")

_CSV_HEADERS = [
    "timestamp_eat",
    "pair",
    "decision",
    "confluence_score",
    "entry_price",
    "stop_loss",
    "take_profit",
    "risk_reward",
    "risk_usd",
    "outcome",      # filled manually later: WIN / LOSS / OPEN / SKIPPED
    "notes",
]


def log_setup(
    symbol: str,
    math_result: dict[str, Any],
    ai_result: dict[str, Any],
    outcome: str = "OPEN",
    notes: str = "",
) -> None:
    """
    Append one setup to trades_log.csv.

    Args:
        symbol:      e.g. "EUR/USD"
        math_result: Gate 2 output dict.
        ai_result:   Gate 3 output dict.
        outcome:     Filled in later; default "OPEN".
        notes:       Free-text annotation.
    """
    _ensure_headers()

    eat_tz = pytz.timezone(config.TIMEZONE)
    now_eat = datetime.now(tz=eat_tz).strftime("%Y-%m-%d %H:%M:%S")

    trade_plan  = ai_result.get("trade_plan", {})

    row = {
        "timestamp_eat":   now_eat,
        "pair":            symbol,
        "decision":        ai_result.get("decision", "NO_TRADE"),
        "confluence_score": ai_result.get("confluence_score", 0),
        "entry_price":     trade_plan.get("entry",        ""),
        "stop_loss":       trade_plan.get("stop_loss",    ""),
        "take_profit":     trade_plan.get("take_profit",  ""),
        "risk_reward":     trade_plan.get("risk_reward",  ""),
        "risk_usd":        trade_plan.get("risk_usd",     math_result.get("max_risk_usd", "")),
        "outcome":         outcome,
        "notes":           notes,
    }

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS)
        writer.writerow(row)

    print(f"  [journal] Logged setup for {symbol} → {row['decision']}")


def check_daily_loss_limit() -> tuple[bool, int]:
    """
    Count today's LOSS outcomes in trades_log.csv.

    Returns:
        (limit_reached: bool, loss_count: int)
        If limit_reached is True, main.py should stop alerting.
    """
    if not os.path.exists(LOG_FILE):
        return False, 0

    eat_tz    = pytz.timezone(config.TIMEZONE)
    today_str = datetime.now(tz=eat_tz).strftime("%Y-%m-%d")

    loss_count = 0
    with open(LOG_FILE, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("timestamp_eat", "").startswith(today_str):
                if row.get("outcome", "").upper() == "LOSS":
                    loss_count += 1

    limit_reached = loss_count >= config.MAX_DAILY_LOSSES
    return limit_reached, loss_count


def _ensure_headers() -> None:
    """Write CSV header row if the file does not exist yet."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS)
            writer.writeheader()
