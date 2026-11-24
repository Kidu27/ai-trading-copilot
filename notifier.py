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

import html
import json
import os
import uuid
from datetime import datetime
from typing import Any

import pytz
import requests

import config
import pattern_stats

# ── Pending signals store ──────────────────────────────────────────────────────
PENDING_SIGNALS_FILE = os.path.join(os.path.dirname(__file__), "pending_signals.json")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def _print_ticket_box(
    symbol: str,
    direction: str,
    score: int,
    entry: float | None,
    stop_loss: float | None,
    take_profit: float | None,
    std_lots: float,
    units: int,
    risk_reward: float | None,
    risk_usd: float | None,
    signal_id: str,
) -> None:
    """Print a clearly labeled multi-line ticket box to the terminal so logs are never unlabeled numbers."""
    rr_str = f"1 : {risk_reward:.2f}" if risk_reward is not None else "N/A"
    risk_str = f"${risk_usd:.2f}" if risk_usd is not None else "N/A"
    print(
        f"\n┌────────────────────────────────────────────────────────┐\n"
        f"│               📋 ACTIONABLE TRADE TICKET               │\n"
        f"├────────────────────────────────────────────────────────┤\n"
        f"│  📌 Pair         : {symbol:<35}│\n"
        f"│  📊 Direction    : {direction:<35}│\n"
        f"│  🎯 Score        : {f'{score}/100':<35}│\n"
        f"│  💵 Entry        : {str(entry):<35}│\n"
        f"│  🛑 Stop-Loss    : {str(stop_loss):<35}│\n"
        f"│  🎯 Take-Profit  : {str(take_profit):<35}│\n"
        f"│  📦 Volume (Lots): {f'{std_lots} std ({units:,} units)':<35}│\n"
        f"│  ⚖️  R:R Ratio    : {rr_str:<35}│\n"
        f"│  🛡️  Risk (USD)   : {risk_str:<35}│\n"
        f"│  🔑 Signal ID    : {signal_id[:8]}...{'':<28}│\n"
        f"└────────────────────────────────────────────────────────┘\n"
    )


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

    # Math-approved zero-loss fail-safe: if AI was busy/failed but math gate passed with valid plan
    if decision == "NO_TRADE" and ai_result.get("_fail_safe") and math_result.get("passed_math_gate"):
        math_dir = str(math_result.get("direction", "LONG")).upper()
        direction = "ENTER_LONG" if "LONG" in math_dir else "ENTER_SHORT"

        fallback_ai = dict(ai_result)
        fallback_ai["decision"] = direction
        fallback_ai["confluence_score"] = 65
        fallback_ai["_math_approved_fallback"] = True

        signal = _build_signal(symbol, math_result, fallback_ai)
        signal["direction"] = direction
        signal["confluence_score"] = 65
        _save_pending_signal(signal)

        _print_ticket_box(
            symbol=symbol,
            direction=signal["direction"],
            score=signal.get("confluence_score", 65),
            entry=signal.get("entry"),
            stop_loss=signal.get("stop_loss"),
            take_profit=signal.get("take_profit"),
            std_lots=signal.get("standard_lots", 0.01),
            units=signal.get("units", 1000),
            risk_reward=math_result.get("risk_reward_ratio"),
            risk_usd=math_result.get("max_risk_usd"),
            signal_id=signal["signal_id"],
        )

        pattern_stats.record_fingerprint(
            signal_id=signal["signal_id"],
            pair=symbol,
            direction=signal["direction"],
            math_result=math_result,
            ai_result=fallback_ai,
            news_bias=fallback_ai.get("news_bias", "NEUTRAL"),
            outcome="PENDING",
            pnl_usd=0.0,
        )

        card = _format_card(symbol, math_result, fallback_ai, signal_id=signal["signal_id"])
        if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
            _send_telegram_with_buttons(card, signal["signal_id"])
        else:
            _print_console(card)
        return

    if decision in ("ENTER_LONG", "ENTER_SHORT"):
        signal = _build_signal(symbol, math_result, ai_result)
        _save_pending_signal(signal)

        # Print clearly labeled multi-line ticket box to terminal so logs are never unlabeled numbers
        trade_plan = ai_result.get("trade_plan", {})
        _print_ticket_box(
            symbol=symbol,
            direction=signal["direction"],
            score=signal.get("confluence_score", 0),
            entry=signal.get("entry"),
            stop_loss=signal.get("stop_loss"),
            take_profit=signal.get("take_profit"),
            std_lots=signal.get("standard_lots", 0.01),
            units=signal.get("units", 1000),
            risk_reward=trade_plan.get("risk_reward") or math_result.get("risk_reward_ratio"),
            risk_usd=trade_plan.get("risk_usd") or math_result.get("max_risk_usd"),
            signal_id=signal["signal_id"],
        )

        # Record indicator fingerprint in pattern_history.json
        pattern_stats.record_fingerprint(
            signal_id=signal["signal_id"],
            pair=symbol,
            direction=signal["direction"],
            math_result=math_result,
            ai_result=ai_result,
            news_bias=ai_result.get("news_bias", "NEUTRAL"),
            outcome="PENDING",
            pnl_usd=0.0,
        )
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

    # Use signal_id attached by main.py, or generate new UUID if absent
    signal_id = math_result.get("signal_id") or str(uuid.uuid4())
    math_result["signal_id"] = signal_id

    return {
        "signal_id":       signal_id,
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
        "risk_reward":      trade_plan.get("risk_reward") or math_result.get("risk_reward_ratio", 2.0),
        "risk_usd":         trade_plan.get("risk_usd") or math_result.get("max_risk_usd", 10.0),
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

    is_math_fallback = bool(ai_result.get("_math_approved_fallback"))

    if is_math_fallback:
        decision_emoji = "🟢 ENTER LONG (MATH)" if "LONG" in decision else "🔴 ENTER SHORT (MATH)"
    else:
        decision_emoji = {
            "ENTER_LONG":  "🟢 ENTER LONG",
            "ENTER_SHORT": "🔴 ENTER SHORT",
            "NO_TRADE":    "⚫ NO TRADE",
        }.get(decision, "⚫ NO TRADE")

    score_bar = _score_bar(score)
    if config.DEMO_MODE and getattr(config, "DEMO_DISCOVERY_MODE", False):
        demo_tag = " [DEMO DISCOVERY]"
    elif config.DEMO_MODE:
        demo_tag = " [DEMO]"
    else:
        demo_tag = ""

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🤖 AI FOREX CO-PILOT{demo_tag}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📌 Pair    : {symbol}",
        f"📊 Signal  : {decision_emoji}",
        f"🎯 Score   : {score}/100  {score_bar}",
        f"📈 Trend   : {trend}",
        f"💰 Price   : {math_result.get('current_price', '?')}",
    ]

    news_bias = ai_result.get("news_bias")
    if news_bias and news_bias != "NEUTRAL":
        lines.append(f"📰 News    : {news_bias} bias")
    lines.append("")

    if (decision != "NO_TRADE" or is_math_fallback) and trade_plan.get("entry"):
        std_lots   = math_result.get("position_size_lots",       trade_plan.get("lots", "?"))
        micro_lots = math_result.get("position_size_micro_lots", "?")
        header_text = "── MATH-APPROVED PLAN (AI BUSY) ──" if is_math_fallback else "── TRADE PLAN ─────────────────"
        lines += [
            header_text,
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
        "text":       f"<pre>{html.escape(message)}</pre>",
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
        "text":       f"<pre>{html.escape(message)}</pre>",
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


# ─────────────────────────────────────────────────────────────────────────────
# Post-Mortem Delivery
# ─────────────────────────────────────────────────────────────────────────────

def send_post_mortem(report: dict[str, Any]) -> None:
    """
    Format and deliver a post-mortem diagnostic card via Telegram
    (or console fallback).

    Args:
        report: Post-mortem report dict generated by post_mortem.py.
    """
    card = _format_post_mortem_card(report)

    if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        _send_telegram(card)
    else:
        _print_console(card)


def _format_post_mortem_card(report: dict[str, Any]) -> str:
    """Build a human-readable post-mortem diagnostic card."""
    pair         = report.get("pair", "UNKNOWN")
    outcome      = report.get("outcome", "UNKNOWN").upper()
    pnl_usd      = report.get("pnl_usd", 0.0)
    entry        = report.get("entry", "?")
    exit_price   = report.get("exit_price", "?")
    root_cause   = report.get("root_cause", "N/A")
    rule_status  = report.get("rule_followed", "N/A")
    lesson       = report.get("actionable_lesson", "N/A")
    signal_id    = report.get("signal_id", "")
    duration     = report.get("duration", "")

    emoji = "🟢 WIN" if outcome == "WIN" else "🔴 LOSS"
    pnl_val = float(pnl_usd) if pnl_usd != "" else 0.0
    pnl_sign = "+" if pnl_val > 0 else ""

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🔬 TRADE POST-MORTEM DIAGNOSTIC",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📌 Pair     : {pair}",
        f"📊 Outcome  : {emoji} ({pnl_sign}${pnl_val:.2f})",
        f"💵 Levels   : Entry {entry} → Exit {exit_price}",
    ]

    if duration:
        lines.append(f"⏱  Duration : {duration}")
    if signal_id:
        lines.append(f"🔑 ID       : {signal_id[:8]}…")

    # Fetch Winning DNA and Readiness Metrics from pattern_stats
    try:
        win_dna = pattern_stats.get_win_signature_summary()
    except Exception:
        win_dna = "Collecting trade data for empirical pattern synthesis."

    try:
        readiness = pattern_stats.get_readiness_metrics()
        readiness_text = readiness.get("summary_text", "")
    except Exception:
        readiness_text = "Readiness: Gathering closed trade statistics."

    lines += [
        "",
        "── ROOT CAUSE ANALYSIS ──────",
        root_cause,
        "",
        "── PLAYBOOK COMPLIANCE ──────",
        rule_status,
        "",
        "── ACTIONABLE LESSON ────────",
        f"💡 {lesson}",
        "",
        "── WINNING DNA SIGNATURE ────",
        win_dna,
        "",
        "── 30-TRADE READINESS METER ─",
        readiness_text,
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🧠 Saved to rolling memory (last 10 lessons applied to future audits).",
    ]

    return "\n".join(lines)


def send_outcome_update(card: str) -> None:
    """Send an automated candle outcome or shadow trade card via Telegram (or print to console)."""
    if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        _send_telegram(card)
    else:
        _print_console(card)
