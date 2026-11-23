"""
notifier.py — Gate 4: Trade Alert Delivery

Formats a clean, readable trade card and sends it via Telegram.

For actionable signals (ENTER_LONG / ENTER_SHORT):
  - Saves the signal to pending_signals.json (keyed by signal_id).
  - Sends a Telegram message with InlineKeyboardMarkup:
      [✅ Accept & Enter]   callback_data="accept:<signal_id>"
      [❌ Decline]          callback_data="decline:<signal_id>"

For NO_TRADE signals:
  - Sends a plain informational card (no buttons).

If Telegram is not configured, falls back to console print.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any

import pytz
import requests

import config

# ── Pending signals store ──────────────────────────────────────────────────────
PENDING_SIGNALS_FILE = os.path.join(os.path.dirname(__file__), "pending_signals.json")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def send_alert(
    symbol: str,
    math_result: dict[str, Any],
    ai_result: dict[str, Any],
) -> None:
    """
    Format and deliver a trade alert card.

    For ENTER_LONG / ENTER_SHORT: saves to pending_signals.json and sends
    an interactive Telegram card with Accept / Decline buttons.
    For NO_TRADE: sends a plain informational card.

    Falls back to console print if Telegram is not configured.
    """
    decision = ai_result.get("decision", "NO_TRADE")

    if decision in ("ENTER_LONG", "ENTER_SHORT"):
        signal = _build_signal(symbol, math_result, ai_result)
        _save_pending_signal(signal)
        card = _format_card(symbol, math_result, ai_result, signal_id=signal["signal_id"])
        if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
            _send_telegram_with_buttons(card, signal["signal_id"])
        else:
            _print_console(card)
    else:
        card = _format_card(symbol, math_result, ai_result, signal_id=None)
        if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
            _send_telegram(card)
        else:
            _print_console(card)


# ─────────────────────────────────────────────────────────────────────────────
# Signal builder & persistence
# ─────────────────────────────────────────────────────────────────────────────

def _build_signal(
    symbol: str,
    math_result: dict[str, Any],
    ai_result: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact, serialisable signal dict for pending_signals.json."""
    eat_tz  = pytz.timezone(config.TIMEZONE)
    now_eat = datetime.now(tz=eat_tz)
    trade_plan = ai_result.get("trade_plan", {})

    # Prefer standard lots from math_result; fall back to AI trade_plan lots
    standard_lots = (
        math_result.get("position_size_lots")
        or trade_plan.get("lots")
        or 0.01
    )
    units = math_result.get("position_size_units") or int(standard_lots * 100_000)

    return {
        "signal_id":       str(uuid.uuid4()),
        "timestamp_eat":   now_eat.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_iso":   now_eat.isoformat(),
        "pair":            symbol,
        "direction":       ai_result.get("decision", "NO_TRADE"),   # ENTER_LONG / ENTER_SHORT
        "entry":           trade_plan.get("entry")    or math_result.get("entry_price"),
        "stop_loss":       trade_plan.get("stop_loss")  or math_result.get("stop_loss"),
        "take_profit":     trade_plan.get("take_profit") or math_result.get("take_profit"),
        "standard_lots":   standard_lots,
        "units":           units,
        "confluence_score": ai_result.get("confluence_score", 0),
        "status":          "PENDING",   # PENDING | ACCEPTED | DECLINED | EXPIRED | EXECUTED
    }


def _save_pending_signal(signal: dict[str, Any]) -> None:
    """Append (or update) the signal in pending_signals.json."""
    store = _load_pending_signals()
    store[signal["signal_id"]] = signal
    with open(PENDING_SIGNALS_FILE, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)
    print(f"  [notifier] Signal saved → {signal['signal_id']}")


def _load_pending_signals() -> dict[str, Any]:
    """Load the pending signals store (returns empty dict if file missing)."""
    if not os.path.exists(PENDING_SIGNALS_FILE):
        return {}
    try:
        with open(PENDING_SIGNALS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def update_signal_status(signal_id: str, status: str) -> None:
    """Update the status field of an existing pending signal (called by telegram_listener)."""
    store = _load_pending_signals()
    if signal_id in store:
        store[signal_id]["status"] = status
        with open(PENDING_SIGNALS_FILE, "w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2)


def get_pending_signal(signal_id: str) -> dict[str, Any] | None:
    """Retrieve a single pending signal by ID (called by telegram_listener)."""
    return _load_pending_signals().get(signal_id)


# ─────────────────────────────────────────────────────────────────────────────
# Card formatter
# ─────────────────────────────────────────────────────────────────────────────

def _format_card(
    symbol: str,
    math_result: dict[str, Any],
    ai_result: dict[str, Any],
    signal_id: str | None,
) -> str:
    """Build a human-readable trade card in plain HTML for Telegram."""
    decision    = ai_result.get("decision", "NO_TRADE")
    score       = ai_result.get("confluence_score", 0)
    trend       = ai_result.get("trend_1h", math_result.get("trend_1h", "?"))
    summary     = ai_result.get("plain_english_summary", "")
    trade_plan  = ai_result.get("trade_plan", {})
    devil_risks = ai_result.get("devils_advocate_risks", [])
    checklist   = ai_result.get("checklist_audit", {})

    decision_emoji = {
        "ENTER_LONG":  "🟢 ENTER LONG",
        "ENTER_SHORT": "🔴 ENTER SHORT",
        "NO_TRADE":    "⚫ NO TRADE",
    }.get(decision, "⚫ NO TRADE")

    score_bar = _score_bar(score)
    demo_tag  = " [DEMO]" if config.DEMO_MODE else ""

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🤖 AI FOREX CO-PILOT{demo_tag}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📌 Pair    : {symbol}",
        f"📊 Signal  : {decision_emoji}",
        f"🎯 Score   : {score}/100  {score_bar}",
        f"📈 Trend   : {trend}",
        f"💰 Price   : {math_result.get('current_price', '?')}",
        "",
    ]

    if decision != "NO_TRADE" and trade_plan.get("entry"):
        std_lots   = math_result.get("position_size_lots",       trade_plan.get("lots", "?"))
        micro_lots = math_result.get("position_size_micro_lots", "?")
        lines += [
            "── TRADE PLAN ─────────────────",
            f"  Entry       : {trade_plan.get('entry', '?')}",
            f"  Stop Loss   : {trade_plan.get('stop_loss', '?')}",
            f"  Take Profit : {trade_plan.get('take_profit', '?')}",
            f"  R:R Ratio   : 1 : {trade_plan.get('risk_reward', '?')}",
            f"  Risk (USD)  : ${trade_plan.get('risk_usd', '?')}",
            f"  Std Lots    : {std_lots} lots  (MT5)",
            f"  Micro Lots  : {micro_lots} µlots (ref)",
            "",
        ]
        if signal_id:
            lines += [
                f"  🔑 Signal ID: {signal_id[:8]}…",
                f"  ⏱  Expires  : {config.SIGNAL_EXPIRY_MINUTES} min after receipt",
                "",
            ]

    # 5-step checklist
    if checklist:
        lines.append("── AUDIT CHECKLIST ─────────────")
        step_labels = {
            "step1_trend":      "1H Trend      ",
            "step2_zone_fib":   "Zone / Fib    ",
            "step3_candle":     "15m Candle    ",
            "step4_math":       "Math Gate     ",
            "step5_volatility": "Volatility    ",
        }
        for key, label in step_labels.items():
            step = checklist.get(key, {})
            icon = "✅" if step.get("result") == "PASS" else "❌"
            lines.append(f"  {icon} {label}: {step.get('reasoning', '')}")
        lines.append("")

    # Devil's advocate
    if devil_risks:
        lines.append("── DEVIL'S ADVOCATE ────────────")
        for risk in devil_risks[:3]:
            lines.append(f"  ⚠️  {risk}")
        lines.append("")

    # Summary
    if summary:
        lines += [
            "── SUMMARY ─────────────────────",
            summary,
            "",
        ]

    # Math gate failures
    if not math_result.get("passed_math_gate", True):
        lines.append("── MATH GATE FAILURES ──────────")
        for reason in math_result.get("fail_reasons", []):
            lines.append(f"  🚫 {reason}")
        lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("⚠️  Analysis only. NOT financial advice.")
    lines.append("   YOU make all trading decisions.")

    return "\n".join(lines)


def _score_bar(score: int) -> str:
    """Generate a visual progress bar for the confluence score."""
    filled = round(score / 10)
    empty  = 10 - filled
    return "█" * filled + "░" * empty


# ─────────────────────────────────────────────────────────────────────────────
# Delivery methods
# ─────────────────────────────────────────────────────────────────────────────

def _send_telegram_with_buttons(message: str, signal_id: str) -> None:
    """
    Send a Telegram message with InlineKeyboardMarkup:
      [✅ Accept & Enter]  callback_data="accept:<signal_id>"
      [❌ Decline]         callback_data="decline:<signal_id>"
    """
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    config.TELEGRAM_CHAT_ID,
        "text":       f"<pre>{message}</pre>",
        "parse_mode": "HTML",
        "reply_markup": {
            "inline_keyboard": [[
                {
                    "text":          "✅ Accept & Enter",
                    "callback_data": f"accept:{signal_id}",
                },
                {
                    "text":          "❌ Decline",
                    "callback_data": f"decline:{signal_id}",
                },
            ]]
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        print(f"  [notifier] Interactive alert sent via Telegram ✅  (signal {signal_id[:8]}…)")
    except requests.RequestException as exc:
        print(f"  [notifier] Telegram send failed: {exc}")
        print("  [notifier] Falling back to console output:")
        _print_console(message)


def _send_telegram(message: str) -> None:
    """Send a plain (no-button) message via Telegram Bot API."""
    url     = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    config.TELEGRAM_CHAT_ID,
        "text":       f"<pre>{message}</pre>",
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        print("  [notifier] Alert sent via Telegram ✅")
    except requests.RequestException as exc:
        print(f"  [notifier] Telegram send failed: {exc}")
        print("  [notifier] Falling back to console output:")
        _print_console(message)


def _print_console(message: str) -> None:
    """Print the trade card to stdout."""
    print("\n" + message + "\n")
