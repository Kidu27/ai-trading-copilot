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
        "risk_usd":        trade_plan.get("risk_usd") or math_result.get("actual_risk_usd") or math_result.get("max_risk_usd", ""),
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
    notes: str = "Manually entered on MT5 Mobile — auto-tracking SL/TP",
) -> None:
    """
    Log or update an accepted/executed trade in trades_log.csv in-place.

    Called by telegram_listener when user taps [✅ Accept & Enter] or [📱 I Entered on MT5 Mobile].
    Updates existing row matching signal_id (decision, outcome="EXECUTED", entry_price, notes)
    to eliminate duplicate rows and ensure proper direction & level tracking.
    """
    _ensure_headers()

    target_id = signal.get("signal_id", "")
    rows: list[dict[str, Any]] = []
    updated = False

    entry_val = fill_price if fill_price is not None else signal.get("entry")
    sl_val = signal.get("stop_loss")
    tp_val = signal.get("take_profit")

    try:
        e = float(entry_val or 0.0)
        s = float(sl_val or 0.0)
    except (ValueError, TypeError):
        e, s = 0.0, 0.0

    raw_dir = str(signal.get("direction", "")).strip().upper()
    if "LONG" in raw_dir and "SHORT" not in raw_dir:
        true_dir = "ENTER_LONG"
    elif "SHORT" in raw_dir:
        true_dir = "ENTER_SHORT"
    elif e > 0 and s > 0:
        true_dir = "ENTER_LONG" if s < e else "ENTER_SHORT"
    else:
        true_dir = "ENTER_LONG"

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if target_id and row.get("signal_id") == target_id:
                    row["outcome"] = "EXECUTED"
                    row["decision"] = true_dir
                    if entry_val is not None:
                        row["entry_price"] = str(entry_val)
                    if sl_val is not None and str(sl_val).strip():
                        row["stop_loss"] = str(sl_val)
                    if tp_val is not None and str(tp_val).strip():
                        row["take_profit"] = str(tp_val)
                    if signal.get("standard_lots"):
                        row["standard_lots"] = str(signal.get("standard_lots"))
                    if signal.get("risk_reward"):
                        row["risk_reward"] = str(signal.get("risk_reward"))
                    if signal.get("risk_usd"):
                        row["risk_usd"] = str(signal.get("risk_usd"))
                    row["exit_price"] = ""
                    row["exit_time"] = ""
                    row["pnl_usd"] = ""
                    row["notes"] = notes
                    updated = True
                rows.append(row)

    if updated:
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  [journal] Updated in-place EXECUTED trade for {signal.get('pair', '')} ({true_dir}) (ID: {target_id[:8]}…)")
    else:
        # Fallback: append if signal_id row not previously logged
        eat_tz = pytz.timezone(config.TIMEZONE)
        now_eat = datetime.now(tz=eat_tz).strftime("%Y-%m-%d %H:%M:%S")
        new_row = {
            "timestamp_eat":   signal.get("timestamp_eat") or now_eat,
            "signal_id":       target_id,
            "pair":            signal.get("pair", ""),
            "decision":        true_dir,
            "confluence_score": signal.get("confluence_score", 0),
            "entry_price":     entry_val or "",
            "stop_loss":       sl_val or "",
            "take_profit":     tp_val or "",
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
        print(f"  [journal] Appended new EXECUTED trade for {new_row['pair']} ({true_dir}) (ID: {target_id[:8]}…)")


def mark_shadow_trade(
    signal_id: str,
    signal: dict[str, Any] | None = None,
    notes: str = "User declined/skipped — shadow tracked for AI learning",
) -> bool:
    """
    Mark a declined, skipped, or expired setup as SHADOW_OPEN in trades_log.csv.
    Infers direction ('SHADOW_LONG' if stop_loss < entry_price else 'SHADOW_SHORT').
    """
    _ensure_headers()
    if not os.path.exists(LOG_FILE):
        return False

    rows: list[dict[str, Any]] = []
    updated = False
    sig = signal or {}

    with open(LOG_FILE, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_id = (row.get("signal_id") or "").strip()
            if signal_id and row_id == signal_id:
                try:
                    entry = float(row.get("entry_price") or sig.get("entry") or 0.0)
                    sl = float(row.get("stop_loss") or sig.get("stop_loss") or 0.0)
                    tp = float(row.get("take_profit") or sig.get("take_profit") or 0.0)
                except (ValueError, TypeError):
                    entry, sl, tp = 0.0, 0.0, 0.0

                raw_dir = (row.get("decision") or sig.get("direction") or "").strip().upper()
                if "LONG" in raw_dir and "SHORT" not in raw_dir:
                    shadow_dir = "SHADOW_LONG"
                elif "SHORT" in raw_dir:
                    shadow_dir = "SHADOW_SHORT"
                elif entry > 0 and sl > 0:
                    shadow_dir = "SHADOW_LONG" if sl < entry else "SHADOW_SHORT"
                else:
                    shadow_dir = "SHADOW_LONG"

                row["decision"] = shadow_dir
                row["outcome"] = "SHADOW_OPEN"
                if entry > 0 and not row.get("entry_price"):
                    row["entry_price"] = str(entry)
                if sl > 0 and not row.get("stop_loss"):
                    row["stop_loss"] = str(sl)
                if tp > 0 and not row.get("take_profit"):
                    row["take_profit"] = str(tp)
                if sig.get("risk_reward") and not row.get("risk_reward"):
                    row["risk_reward"] = str(sig.get("risk_reward"))
                if sig.get("risk_usd") and not row.get("risk_usd"):
                    row["risk_usd"] = str(sig.get("risk_usd"))
                row["exit_price"] = ""
                row["exit_time"] = ""
                row["pnl_usd"] = ""
                row["notes"] = notes
                updated = True
            rows.append(row)

    if not updated and sig:
        eat_tz = pytz.timezone(config.TIMEZONE)
        now_eat = datetime.now(tz=eat_tz).strftime("%Y-%m-%d %H:%M:%S")
        try:
            entry = float(sig.get("entry") or 0.0)
            sl = float(sig.get("stop_loss") or 0.0)
        except (ValueError, TypeError):
            entry, sl = 0.0, 0.0

        raw_dir = (sig.get("direction") or "").strip().upper()
        if "LONG" in raw_dir and "SHORT" not in raw_dir:
            shadow_dir = "SHADOW_LONG"
        elif "SHORT" in raw_dir:
            shadow_dir = "SHADOW_SHORT"
        elif entry > 0 and sl > 0:
            shadow_dir = "SHADOW_LONG" if sl < entry else "SHADOW_SHORT"
        else:
            shadow_dir = "SHADOW_LONG"

        new_row = {
            "timestamp_eat":   sig.get("timestamp_eat") or now_eat,
            "signal_id":       signal_id,
            "pair":            sig.get("pair", ""),
            "decision":        shadow_dir,
            "confluence_score": sig.get("confluence_score", 0),
            "entry_price":     sig.get("entry", ""),
            "stop_loss":       sig.get("stop_loss", ""),
            "take_profit":     sig.get("take_profit", ""),
            "standard_lots":   sig.get("standard_lots", ""),
            "risk_reward":     sig.get("risk_reward", ""),
            "risk_usd":        sig.get("risk_usd", ""),
            "outcome":         "SHADOW_OPEN",
            "exit_price":      "",
            "exit_time":       "",
            "pnl_usd":         "",
            "notes":           notes,
        }
        rows.append(new_row)
        updated = True

    if updated:
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  [journal] Marked signal {signal_id[:8]}… as SHADOW_OPEN ({notes})")
        return True

    return False


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
