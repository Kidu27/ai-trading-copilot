"""
journal.py — Trade logging & daily circuit breaker.

Every AI-audited setup (regardless of decision) is appended to
trades_log.csv. The circuit breaker checks today's losing trades
and blocks further alerts if MAX_DAILY_LOSSES is reached.

Cooldown guard: check_pair_cooldown() prevents duplicate alerts
for the same currency pair within a configurable rolling window.
"""

import csv
import os
from datetime import datetime, timedelta
from typing import Any

import pytz

import config

# ── File path ─────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(__file__), "trades_log.csv")

_CSV_HEADERS = [
    "timestamp_eat",
    "signal_id",
    "pair",
    "decision",
    "confluence_score",
    "entry_price",
    "stop_loss",
    "take_profit",
    "standard_lots",
    "risk_reward",
    "risk_usd",
    "outcome",      # WIN / LOSS / OPEN / EXECUTED / SKIPPED / DECLINED
    "exit_price",
    "exit_time",
    "pnl_usd",
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
    standard_lots = math_result.get("position_size_lots", trade_plan.get("lots", ""))

    row = {
        "timestamp_eat":   now_eat,
        "signal_id":       math_result.get("signal_id", ""),
        "pair":            symbol,
        "decision":        ai_result.get("decision", "NO_TRADE"),
        "confluence_score": ai_result.get("confluence_score", 0),
        "entry_price":     trade_plan.get("entry",        ""),
        "stop_loss":       trade_plan.get("stop_loss",    ""),
        "take_profit":     trade_plan.get("take_profit",  ""),
        "standard_lots":   standard_lots,
        "risk_reward":     trade_plan.get("risk_reward",  ""),
        "risk_usd":        trade_plan.get("risk_usd",     math_result.get("max_risk_usd", "")),
        "outcome":         outcome,
        "exit_price":      "",
        "exit_time":       "",
        "pnl_usd":         "",
        "notes":           notes,
    }

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS, extrasaction="ignore")
        writer.writerow(row)

    print(f"  [journal] Logged setup for {symbol} → {row['decision']}")


def log_executed_trade(
    signal: dict[str, Any],
    fill_price: float | None = None,
    notes: str = "Order executed",
) -> None:
    """
    Log or update an accepted/executed trade in trades_log.csv in-place.

    Called by telegram_listener when user taps [✅ Accept & Enter] or [✅ Entered Manually].
    Updates existing row matching signal_id (outcome="EXECUTED", entry_price=fill_price, notes)
    to eliminate duplicate rows.
    """
    _ensure_headers()

    target_id = signal.get("signal_id", "")
    rows: list[dict[str, Any]] = []
    updated = False

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if target_id and row.get("signal_id") == target_id:
                    row["outcome"] = "EXECUTED"
                    if fill_price is not None:
                        row["entry_price"] = str(fill_price)
                    existing_notes = (row.get("notes") or "").strip()
                    row["notes"] = f"{existing_notes} | {notes}" if existing_notes else notes
                    updated = True
                rows.append(row)

    if updated:
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  [journal] Updated in-place EXECUTED trade for {signal.get('pair', '')} (ID: {target_id[:8]}…)")
    else:
        # Fallback: append if signal_id row not previously logged
        eat_tz = pytz.timezone(config.TIMEZONE)
        now_eat = datetime.now(tz=eat_tz).strftime("%Y-%m-%d %H:%M:%S")
        new_row = {
            "timestamp_eat":   signal.get("timestamp_eat") or now_eat,
            "signal_id":       target_id,
            "pair":            signal.get("pair", ""),
            "decision":        signal.get("direction", "EXECUTED"),
            "confluence_score": signal.get("confluence_score", 0),
            "entry_price":     fill_price or signal.get("entry", ""),
            "stop_loss":       signal.get("stop_loss", ""),
            "take_profit":     signal.get("take_profit", ""),
            "standard_lots":   signal.get("standard_lots", ""),
            "risk_reward":     signal.get("risk_reward", ""),
            "risk_usd":        signal.get("risk_usd", ""),
            "outcome":         "EXECUTED",
            "exit_price":      "",
            "exit_time":       "",
            "pnl_usd":         "",
            "notes":           notes,
        }
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS, extrasaction="ignore")
            writer.writerow(new_row)
        print(f"  [journal] Appended new EXECUTED trade for {new_row['pair']} (ID: {target_id[:8]}…)")


def update_trade_outcome(
    signal_id: str,
    exit_price: float,
    exit_time: str,
    pnl_usd: float,
    outcome: str,
    notes: str = "",
) -> bool:
    """
    Update trades_log.csv when a trade closes.

    Finds the row by signal_id and updates exit_price, exit_time, pnl_usd,
    and outcome ('WIN' or 'LOSS').
    """
    _ensure_headers()
    if not os.path.exists(LOG_FILE):
        return False

    rows: list[dict[str, Any]] = []
    updated = False

    with open(LOG_FILE, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("signal_id", "").strip() == signal_id.strip():
                row["exit_price"] = str(exit_price)
                row["exit_time"]  = exit_time
                row["pnl_usd"]    = str(pnl_usd)
                row["outcome"]    = outcome.upper()
                if notes:
                    existing = row.get("notes", "").strip()
                    row["notes"] = f"{existing} | {notes}".strip(" |")
                updated = True
            rows.append(row)

    if not updated:
        # If signal_id wasn't found in log, append a new completed trade record
        row = {
            "timestamp_eat":   exit_time,
            "signal_id":       signal_id,
            "pair":            "",
            "decision":        "",
            "confluence_score": 0,
            "entry_price":     "",
            "stop_loss":       "",
            "take_profit":     "",
            "standard_lots":   "",
            "risk_reward":     "",
            "risk_usd":        "",
            "outcome":         outcome.upper(),
            "exit_price":      str(exit_price),
            "exit_time":       exit_time,
            "pnl_usd":         str(pnl_usd),
            "notes":           notes,
        }
        rows.append(row)

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [journal] Updated trade {signal_id[:8]}… outcome → {outcome.upper()} (${pnl_usd})")
    return True


def check_pair_cooldown(pair: str, hours: float | None = None) -> tuple[bool, float]:
    """Return (is_on_cooldown, hours_remaining) based ONLY on actionable trade alerts, ignoring NO_TRADE."""
    cooldown_hrs = float(hours if hours is not None else getattr(config, "PAIR_COOLDOWN_HOURS", 1.0))
    if getattr(config, "DEMO_DISCOVERY_MODE", False):
        cooldown_hrs = min(cooldown_hrs, 1.0)
    if cooldown_hrs <= 0 or not os.path.exists(LOG_FILE):
        return False, 0.0

    eat_tz = pytz.timezone(config.TIMEZONE)
    now = datetime.now(tz=eat_tz)
    latest_dt = None

    try:
        with open(LOG_FILE, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if (row.get("pair") or "").strip() != pair:
                    continue
                decision = (row.get("decision") or "").strip().upper()
                outcome = (row.get("outcome") or "").strip().upper()
                if decision not in {"ENTER_LONG", "ENTER_SHORT"} and outcome not in {"EXECUTED", "WIN", "LOSS"}:
                    continue
                ts_raw = (row.get("timestamp_eat") or "").strip()
                if not ts_raw:
                    continue
                try:
                    dt = eat_tz.localize(datetime.strptime(ts_raw[:19], "%Y-%m-%d %H:%M:%S"))
                    if latest_dt is None or dt > latest_dt:
                        latest_dt = dt
                except ValueError:
                    continue
    except OSError:
        return False, 0.0

    if latest_dt is None:
        return False, 0.0

    elapsed_hrs = (now - latest_dt).total_seconds() / 3600.0
    if elapsed_hrs < cooldown_hrs:
        return True, round(cooldown_hrs - elapsed_hrs, 2)
    return False, 0.0



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
    """Write or update CSV header row if the file does not exist or has outdated headers."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS)
            writer.writeheader()
    else:
        # Check if existing headers contain our required fields
        try:
            with open(LOG_FILE, "r", newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                first_row = next(reader, None)
            if first_row and "signal_id" not in first_row:
                # Upgrade headers on existing file
                with open(LOG_FILE, "r", newline="", encoding="utf-8") as fh:
                    dict_reader = csv.DictReader(fh)
                    rows = list(dict_reader)
                with open(LOG_FILE, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS, extrasaction="ignore")
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(r)
        except Exception as exc:
            print(f"  [journal] Warning during header verification: {exc}")
