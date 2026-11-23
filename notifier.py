"""
notifier.py — Gate 4: Trade Alert Delivery

Formats a clean, readable trade card and sends it via Telegram.
If Telegram is not configured, prints to the console instead.
"""

import requests
from typing import Any

import config


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

    If Telegram credentials are configured → send via Bot API.
    Otherwise → print to console.
    """
    card = _format_card(symbol, math_result, ai_result)

    if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        _send_telegram(card)
    else:
        _print_console(card)


# ─────────────────────────────────────────────────────────────────────────────
# Card formatter
# ─────────────────────────────────────────────────────────────────────────────

def _format_card(
    symbol: str,
    math_result: dict[str, Any],
    ai_result: dict[str, Any],
) -> str:
    """Build a human-readable trade card in Telegram MarkdownV2 format."""
    decision     = ai_result.get("decision", "NO_TRADE")
    score        = ai_result.get("confluence_score", 0)
    trend        = ai_result.get("trend_1h", math_result.get("trend_1h", "?"))
    summary      = ai_result.get("plain_english_summary", "")
    trade_plan   = ai_result.get("trade_plan", {})
    devil_risks  = ai_result.get("devils_advocate_risks", [])
    checklist    = ai_result.get("checklist_audit", {})

    # Emoji decorators
    decision_emoji = {
        "ENTER_LONG":  "🟢 ENTER LONG",
        "ENTER_SHORT": "🔴 ENTER SHORT",
        "NO_TRADE":    "⚫ NO TRADE",
    }.get(decision, "⚫ NO TRADE")

    score_bar = _score_bar(score)

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🤖 AI FOREX CO-PILOT ALERT",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📌 Pair    : {symbol}",
        f"📊 Signal  : {decision_emoji}",
        f"🎯 Score   : {score}/100  {score_bar}",
        f"📈 Trend   : {trend}",
        f"💰 Price   : {math_result.get('current_price', '?')}",
        "",
    ]

    if decision != "NO_TRADE" and trade_plan.get("entry"):
        lines += [
            "── TRADE PLAN ─────────────────",
            f"  Entry       : {trade_plan.get('entry', '?')}",
            f"  Stop Loss   : {trade_plan.get('stop_loss', '?')}",
            f"  Take Profit : {trade_plan.get('take_profit', '?')}",
            f"  R:R Ratio   : 1 : {trade_plan.get('risk_reward', '?')}",
            f"  Risk (USD)  : ${trade_plan.get('risk_usd', '?')}",
            f"  Lot Size    : {trade_plan.get('lots', '?')} micro-lots",
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

    # Math gate status
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

def _send_telegram(message: str) -> None:
    """Send message via Telegram Bot API."""
    url     = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    config.TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "HTML",           # HTML is more forgiving than MarkdownV2
    }
    # Strip markdown-specific characters for plain HTML delivery
    clean_text = message.replace("*", "").replace("_", "").replace("`", "")
    payload["text"] = f"<pre>{clean_text}</pre>"

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
