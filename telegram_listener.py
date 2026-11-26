"""
telegram_listener.py — Interactive Telegram Button Handler

Listens for Telegram InlineKeyboard callback_query events using lightweight
long-polling (no webhooks, no asyncio, no extra framework — just requests).

Workflow
────────
  1. User receives the trade card with [✅ Accept & Enter] / [❌ Decline] buttons.
  2. User taps a button on their phone.
  3. This listener receives the callback_query via /getUpdates.
  4. It immediately calls /answerCallbackQuery to clear the loading spinner
     (Telegram expires callbacks after 10 seconds).
  5a. DECLINE:  Marks signal DECLINED in pending_signals.json and updates the
                Telegram message to show "❌ Setup Declined by User".
  5b. ACCEPT :  Checks signal age vs SIGNAL_EXPIRY_MINUTES.
                - Expired  → marks EXPIRED and edits message accordingly.
                - Valid    → calls broker_executor.execute_bracket_order(),
                             marks EXECUTED in pending_signals.json, and edits
                             the Telegram message with the execution confirmation.

Security
────────
  Every callback_query.message.chat.id is verified against TELEGRAM_CHAT_ID.
  Any callback from an unauthorized chat is silently ignored.

Usage
─────
  python telegram_listener.py              # listen indefinitely (Ctrl-C to stop)
  python telegram_listener.py --timeout 30 # listen for 30 seconds then exit
"""

import argparse
import html
import sys
import time
from datetime import datetime
from typing import Any

import pytz
import requests

import config
import notifier
import broker_executor
import journal


# ─────────────────────────────────────────────────────────────────────────────
# Telegram Bot API helpers
# ─────────────────────────────────────────────────────────────────────────────

_TELEGRAM_BASE = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"


def _tg_get(endpoint: str, params: dict | None = None) -> dict[str, Any]:
    """GET to Telegram Bot API; returns the JSON body."""
    resp = requests.get(f"{_TELEGRAM_BASE}/{endpoint}", params=params, timeout=35)
    resp.raise_for_status()
    return resp.json()


def _tg_post(endpoint: str, payload: dict) -> dict[str, Any]:
    """POST to Telegram Bot API; returns the JSON body."""
    resp = requests.post(f"{_TELEGRAM_BASE}/{endpoint}", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _answer_callback(callback_query_id: str, text: str = "", alert: bool = False) -> None:
    """Answer a callback_query to clear Telegram's loading spinner."""
    try:
        _tg_post("answerCallbackQuery", {
            "callback_query_id": callback_query_id,
            "text":              text,
            "show_alert":        alert,
        })
    except requests.RequestException as exc:
        print(f"  [listener] answerCallbackQuery failed: {exc}")


def _edit_message(
    chat_id: int | str,
    message_id: int,
    new_text: str,
    reply_markup: dict | None = None,
) -> None:
    """Edit the original Telegram message to reflect the action taken."""
    if reply_markup is None:
        reply_markup = {"inline_keyboard": []}

    inner = new_text
    if inner.startswith("<pre>") and inner.endswith("</pre>"):
        inner = inner[5:-6]
    text_to_send = f"<pre>{html.escape(inner)}</pre>"
    try:
        _tg_post("editMessageText", {
            "chat_id":      chat_id,
            "message_id":   message_id,
            "text":         text_to_send,
            "parse_mode":   "HTML",
            "reply_markup": reply_markup,
        })
    except requests.RequestException as exc:
        print(f"  [listener] editMessageText failed: {exc}")


def _get_base_card(cq: dict[str, Any], signal: dict[str, Any]) -> str:
    """
    Retrieve or reconstruct the full trade card text so trade numbers
    are NEVER deleted or overwritten on Telegram.
    """
    orig = cq.get("message", {}).get("text", "")
    if orig and len(orig.strip()) > 35:
        return orig.strip()

    pair = signal.get("pair", "?")
    direction = signal.get("direction", "?")
    score = signal.get("confluence_score", 0)
    entry = signal.get("entry", "?")
    sl = signal.get("stop_loss", "?")
    tp = signal.get("take_profit", "?")
    lots = signal.get("standard_lots", 0.01)
    units = signal.get("units", int(lots * 100_000))
    risk_usd = signal.get("risk_usd", round(config.ACCOUNT_BALANCE * config.MAX_RISK_PERCENT, 2))
    rr = signal.get("risk_reward", 2.0)

    dir_emoji = "🟢 ENTER LONG" if "LONG" in direction else ("🔴 ENTER SHORT" if "SHORT" in direction else direction)

    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🤖 AI FOREX CO-PILOT [DEMO DISCOVERY]\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 Pair    : {pair}\n"
        f"📊 Signal  : {dir_emoji}\n"
        f"🎯 Score   : {score}/100\n"
        f"💰 Price   : {entry}\n\n"
        "── TRADE PLAN ─────────────────\n"
        f"  Entry       : {entry}\n"
        f"  Stop Loss   : {sl}\n"
        f"  Take Profit : {tp}\n"
        f"  R:R Ratio   : 1 : {rr}\n"
        f"  Risk (USD)  : ${risk_usd}\n"
        f"  Std Lots    : {lots} lots  (MT5)\n"
        f"  Units       : {units:,} units\n"
        f"  Signal ID   : {signal.get('signal_id', '')[:8]}…"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Callback handlers
# ─────────────────────────────────────────────────────────────────────────────

def _handle_decline(
    signal_id: str,
    signal: dict[str, Any],
    chat_id: int | str,
    message_id: int,
    callback_query_id: str,
    cq: dict[str, Any],
) -> None:
    """User tapped ❌ Decline — shadow-track signal for AI learning and update Telegram message."""
    _answer_callback(callback_query_id, text="Signal declined — shadow tracked for AI learning.")
    notifier.update_signal_status(signal_id, "DECLINED")
    base_card = _get_base_card(cq, signal)

    try:
        journal.mark_shadow_trade(
            signal_id=signal_id,
            signal=signal,
            notes="User declined/skipped — shadow tracked for AI learning",
        )
    except Exception as exc:
        print(f"  [listener] Warning: failed to mark shadow trade in journal: {exc}")

    try:
        import pattern_stats
        hist = pattern_stats.load_pattern_history()
        if signal_id in hist:
            hist[signal_id]["outcome"] = "SHADOW_OPEN"
            hist[signal_id]["is_shadow_candidate"] = True
            pattern_stats.save_pattern_history(hist)
    except Exception as exc:
        print(f"  [listener] Warning: failed to update pattern_history for shadow trade: {exc}")

    edit_text = (
        f"{base_card}\n\n"
        "❌ Setup Declined by User\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Declined. 15m candles are being shadow-tracked for AI learning."
    )
    _edit_message(chat_id, message_id, edit_text, reply_markup={"inline_keyboard": []})
    print(f"  [listener] Signal {signal_id[:8]}… DECLINED by user — marked SHADOW_OPEN.")


def _handle_accept(
    signal_id: str,
    signal: dict[str, Any],
    chat_id: int | str,
    message_id: int,
    callback_query_id: str,
    cq: dict[str, Any],
) -> None:
    """
    User tapped ✅ Accept & Enter.

    1. Preserves full trade card numbers at the top.
    2. Checks signal expiry.
    3. If MT5MacBridgeExecutor connected and succeeds -> confirms fill and attaches [Report WIN] / [Report LOSS].
    4. If MT5MacBridgeExecutor fails or MT5_FILES_DIR disconnected (cloud mode) -> appends Manual MT5 Mobile Ticket
       and attaches [Entered Manually] / [Skipped].
    """
    eat_tz  = pytz.timezone(config.TIMEZONE)
    now_eat = datetime.now(tz=eat_tz)
    base_card = _get_base_card(cq, signal)

    # Parse the signal timestamp
    ts_str = signal.get("timestamp_eat", "")
    try:
        signal_ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        signal_ts = eat_tz.localize(signal_ts)
        age_minutes = (now_eat - signal_ts).total_seconds() / 60.0
    except ValueError:
        age_minutes = float("inf")

    pair      = signal.get("pair", "?")
    direction = signal.get("direction", "?")
    std_lots  = signal.get("standard_lots", 0.01)
    units     = signal.get("units", int(std_lots * 100_000))
    entry     = signal.get("entry", "?")
    sl        = signal.get("stop_loss", "?")
    tp        = signal.get("take_profit", "?")

    # ── Expiry check ──────────────────────────────────────────────────────────
    if age_minutes > config.SIGNAL_EXPIRY_MINUTES:
        _answer_callback(callback_query_id, text="⚠️ Signal expired! Auto-tracked as shadow setup.", alert=True)
        notifier.update_signal_status(signal_id, "EXPIRED")
        try:
            journal.mark_shadow_trade(
                signal_id=signal_id,
                signal=signal,
                notes="User declined/skipped — shadow tracked for AI learning",
            )
        except Exception as exc:
            print(f"  [listener] Warning: failed to mark shadow trade in journal on expiry: {exc}")

        edit_text = (
            f"{base_card}\n\n"
            f"⚠️ Signal Expired (>{config.SIGNAL_EXPIRY_MINUTES} mins old) — Order Cancelled\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏱  Age     : {age_minutes:.1f} min  (limit: {config.SIGNAL_EXPIRY_MINUTES} min)\n"
            "Order cancelled. 15m candles are being shadow-tracked for AI learning."
        )
        _edit_message(chat_id, message_id, edit_text, reply_markup={"inline_keyboard": []})
        print(
            f"  [listener] Signal {signal_id[:8]}… EXPIRED "
            f"({age_minutes:.1f} min > {config.SIGNAL_EXPIRY_MINUTES} min limit) — marked SHADOW_OPEN."
        )
        return

    # ── Check if MT5 bridge is connected on disk ─────────────────────────────
    import os
    mt5_dir = getattr(config, "MT5_FILES_DIR", "") or os.getenv("MT5_FILES_DIR", "")
    is_mt5_connected = bool(mt5_dir and os.path.isdir(mt5_dir))

    if is_mt5_connected:
        _answer_callback(callback_query_id, text="⚙️ Submitting to MT5 bridge…")
        executor = broker_executor.get_executor()
        result   = executor.execute_bracket_order(signal)

        if result["success"]:
            notifier.update_signal_status(signal_id, "EXECUTED")
            fill_price = result.get("fill_price", entry)

            try:
                journal.log_executed_trade(
                    signal,
                    fill_price=float(fill_price) if fill_price is not None else None,
                    notes="Executed via MT5 Bridge",
                )
            except Exception as exc:
                print(f"  [listener] Warning: failed to log execution to journal: {exc}")

            edit_text = (
                f"{base_card}\n\n"
                f"✅ ORDER EXECUTED [MT5 BRIDGE]\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎫 Ticket : {result.get('mt5_ticket', 'N/A')}\n"
                f"💵 Fill   : {fill_price}\n"
                f"📦 Volume : {std_lots} std lots ({units:,} units)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Position is LIVE. When it closes, tap below to report outcome:"
            )
            report_markup = {
                "inline_keyboard": [
                    [
                        {"text": "🏆 Report WIN", "callback_data": f"win:{signal_id}"},
                        {"text": "❌ Report LOSS", "callback_data": f"loss:{signal_id}"},
                    ]
                ]
            }
            _edit_message(chat_id, message_id, edit_text, reply_markup=report_markup)
            print(f"  [listener] Signal {signal_id[:8]}… EXECUTED on MT5 bridge.")
            return

    # ── Fallback: Manual MT5 Mobile Ticket ────────────────────────────────────
    notifier.update_signal_status(signal_id, "WAITING_MANUAL")
    _answer_callback(callback_query_id, text="📱 MT5 Bridge unavailable — manual ticket ready!", alert=False)

    manual_ticket = (
        f"{base_card}\n\n"
        f"📱 MANUAL MT5 MOBILE TICKET\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 Pair        : {pair}\n"
        f"📊 Side        : {direction}\n"
        f"📦 Volume      : {std_lots} Lots ({units:,} units)\n"
        f"💵 Entry Level : {entry}\n"
        f"🛑 Stop Loss   : {sl}\n"
        f"🎯 Take Profit : {tp}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"ℹ️ MT5 Bridge disconnected (Cloud/Mobile Mode). Open MT5 on your phone, enter the ticket above, then tap below:"
    )
    manual_markup = {
        "inline_keyboard": [
            [
                {"text": "📱 I Entered on MT5 Mobile", "callback_data": f"manual:{signal_id}"},
                {"text": "⏭️ Skipped", "callback_data": f"skip:{signal_id}"},
            ]
        ]
    }
    _edit_message(chat_id, message_id, manual_ticket, reply_markup=manual_markup)
    print(f"  [listener] Signal {signal_id[:8]}… manual ticket presented to user.")


def _handle_manual(
    signal_id: str,
    signal: dict[str, Any],
    chat_id: int | str,
    message_id: int,
    callback_query_id: str,
    cq: dict[str, Any],
) -> None:
    """User tapped [📱 I Entered on MT5 Mobile]."""
    _answer_callback(callback_query_id, text="✅ Trade recorded! Auto-tracking SL & TP via 15m candles.")
    notifier.update_signal_status(signal_id, "EXECUTED")
    base_card = _get_base_card(cq, signal)

    eat_tz  = pytz.timezone(config.TIMEZONE)
    now_eat = datetime.now(tz=eat_tz)
    now_eat_str = now_eat.strftime("%Y-%m-%d %H:%M:%S")

    entry = float(signal.get("entry") or signal.get("fill_price") or 0.0)
    sl = float(signal.get("stop_loss") or 0.0)
    tp = float(signal.get("take_profit") or 0.0)
    risk_usd = float(signal.get("risk_usd") or round(config.ACCOUNT_BALANCE * config.MAX_RISK_PERCENT, 2))
    risk_reward = float(signal.get("risk_reward") or 2.0)
    pair = signal.get("pair", "")

    # Infer true direction from entry/SL if needed
    raw_dir = str(signal.get("direction", "")).strip().upper()
    if "LONG" in raw_dir and "SHORT" not in raw_dir:
        true_dir = "ENTER_LONG"
    elif "SHORT" in raw_dir:
        true_dir = "ENTER_SHORT"
    elif entry > 0 and sl > 0:
        true_dir = "ENTER_LONG" if sl < entry else "ENTER_SHORT"
    else:
        true_dir = "ENTER_LONG"

    signal["direction"] = true_dir
    signal["entry"] = entry
    signal["stop_loss"] = sl
    signal["take_profit"] = tp

    # Log EXECUTED row to trades_log.csv
    try:
        journal.log_executed_trade(
            signal,
            fill_price=entry,
            notes="Manually entered on MT5 Mobile — auto-tracking SL/TP",
        )
    except Exception as exc:
        print(f"  [listener] Warning: failed to log manual trade to journal: {exc}")

    # Register the trade into active_trades.json
    trade_record = {
        "signal_id":     signal_id,
        "pair":          pair,
        "direction":     true_dir,
        "entry_price":   entry,
        "entry":         entry,
        "fill_price":    entry,
        "stop_loss":     sl,
        "take_profit":   tp,
        "risk_usd":      risk_usd,
        "risk_reward":   risk_reward,
        "standard_lots": signal.get("standard_lots", 0.01),
        "units":         signal.get("units", 1000),
        "mt5_ticket":    "MANUAL_ENTRY",
        "source":        "MANUAL_MT5",
        "status":        "OPEN",
        "executor":      "manual",
        "demo_mode":     config.DEMO_MODE,
        "timestamp_eat": signal.get("timestamp_eat") or now_eat_str,
        "opened_eat":    now_eat_str,
        "closed_eat":    None,
        "pnl_usd":       None,
    }
    broker_executor._append_active_trade(trade_record)

    # Immediately sync active_trades.json and trades_log.csv to GitHub's memory branch
    try:
        import github_sync
        github_sync.push_memory_files(f"Manual MT5 trade registered: {pair} ({true_dir})")
    except Exception as exc:
        print(f"  [listener] Warning: failed to push manual trade to GitHub: {exc}")

    confirm_note = (
        f"{base_card}\n\n"
        f"📱 MANUALLY ENTERED ON MT5 MOBILE\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Side       : {true_dir}\n"
        f"Entry      : {entry}\n"
        f"Stop Loss  : {sl}\n"
        f"Take Profit: {tp}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Position is LIVE. Auto-tracking SL & TP via 15m candles.\n"
        f"You will be alerted instantly when position hits SL or TP.\n"
        f"Or report early manual close below:"
    )
    report_markup = {
        "inline_keyboard": [
            [
                {"text": "🏆 Report WIN", "callback_data": f"win:{signal_id}"},
                {"text": "❌ Report LOSS", "callback_data": f"loss:{signal_id}"},
            ]
        ]
    }
    _edit_message(chat_id, message_id, confirm_note, reply_markup=report_markup)
    print(f"  [listener] Signal {signal_id[:8]}… confirmed as MANUAL MT5 trade ({true_dir}).")


def _handle_skip(
    signal_id: str,
    signal: dict[str, Any],
    chat_id: int | str,
    message_id: int,
    callback_query_id: str,
    cq: dict[str, Any],
) -> None:
    """User tapped [⏭️ Skipped] — shadow-track signal for AI learning."""
    _answer_callback(callback_query_id, text="Trade skipped — shadow tracked for AI learning.")
    notifier.update_signal_status(signal_id, "SKIPPED")
    base_card = _get_base_card(cq, signal)

    try:
        journal.mark_shadow_trade(
            signal_id=signal_id,
            signal=signal,
            notes="User declined/skipped — shadow tracked for AI learning",
        )
    except Exception as exc:
        print(f"  [listener] Warning: failed to mark shadow trade in journal on skip: {exc}")

    try:
        import pattern_stats
        hist = pattern_stats.load_pattern_history()
        if signal_id in hist:
            hist[signal_id]["outcome"] = "SHADOW_OPEN"
            hist[signal_id]["is_shadow_candidate"] = True
            pattern_stats.save_pattern_history(hist)
    except Exception as exc:
        print(f"  [listener] Warning: failed to update pattern_history for shadow trade: {exc}")

    skip_note = (
        f"{base_card}\n\n"
        "⏭️ Manual Entry Skipped by User\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Skipped. 15m candles are being shadow-tracked for AI learning."
    )
    _edit_message(chat_id, message_id, skip_note, reply_markup={"inline_keyboard": []})
    print(f"  [listener] Signal {signal_id[:8]}… marked SKIPPED — marked SHADOW_OPEN.")


def _handle_report_outcome(
    action: str,  # "win" | "loss"
    signal_id: str,
    signal: dict[str, Any],
    chat_id: int | str,
    message_id: int,
    callback_query_id: str,
    cq: dict[str, Any],
) -> None:
    """User tapped [🏆 Report WIN] or [❌ Report LOSS]."""
    import pattern_stats
    import post_mortem

    outcome = "WIN" if action == "win" else "LOSS"
    _answer_callback(callback_query_id, text=f"Recorded {outcome}! Generating diagnostic...")
    base_card = _get_base_card(cq, signal)

    eat_tz = pytz.timezone(config.TIMEZONE)
    now_eat_str = datetime.now(tz=eat_tz).strftime("%Y-%m-%d %H:%M:%S")

    pair = signal.get("pair", "UNKNOWN")
    entry = float(signal.get("entry") or 0.0)
    sl = float(signal.get("stop_loss") or 0.0)
    tp = float(signal.get("take_profit") or 0.0)
    risk_usd = float(signal.get("risk_usd") or round(config.ACCOUNT_BALANCE * config.MAX_RISK_PERCENT, 2))
    rr = float(signal.get("risk_reward") or 2.0)

    if outcome == "WIN":
        exit_price = tp if tp > 0 else entry
        pnl_usd = round(risk_usd * rr, 2)
    else:
        exit_price = sl if sl > 0 else entry
        pnl_usd = -round(risk_usd, 2)

    pnl_sign = "+" if pnl_usd > 0 else ""
    emoji = "🏆 WIN" if outcome == "WIN" else "❌ LOSS"

    # 1. Update pending_signals.json
    notifier.update_signal_status(signal_id, f"CLOSED_{outcome}")

    # 2. Update trades_log.csv
    try:
        journal.update_trade_outcome(
            signal_id=signal_id,
            exit_price=exit_price,
            exit_time=now_eat_str,
            pnl_usd=pnl_usd,
            outcome=outcome,
            notes=f"User reported {outcome} via Telegram button",
        )
    except Exception as exc:
        print(f"  [listener] Warning: failed to update journal outcome: {exc}")

    # 3. Update pattern_history.json
    try:
        pattern_stats.update_outcome(
            signal_id=signal_id,
            outcome=outcome,
            pnl_usd=pnl_usd,
        )
    except Exception as exc:
        print(f"  [listener] Warning: failed to update pattern_stats: {exc}")

    # 4. Close trade in active_trades.json if present
    try:
        trades = post_mortem.load_active_trades()
        if signal_id in trades:
            trades[signal_id]["status"] = "CLOSED"
            trades[signal_id]["exit_price"] = exit_price
            trades[signal_id]["closed_eat"] = now_eat_str
            trades[signal_id]["pnl_usd"] = pnl_usd
            trades[signal_id]["outcome"] = outcome
            post_mortem.save_active_trades(trades)
    except Exception as exc:
        print(f"  [listener] Warning: failed to close in active_trades: {exc}")

    # 5. Trigger post-mortem analysis and send diagnostic card
    try:
        import data_fetcher
        candles = None
        if config.TWELVE_DATA_API_KEY:
            try:
                candles = data_fetcher.fetch_candles(pair, "15min", 20)
            except Exception:
                candles = None

        fake_trade = {
            "signal_id": signal_id,
            "pair": pair,
            "entry": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "direction": signal.get("direction", "ENTER_LONG"),
        }
        analysis = post_mortem.analyze_trade_with_ai(fake_trade, exit_price, outcome, candles)

        pm_report = {
            "signal_id": signal_id,
            "pair": pair,
            "outcome": outcome,
            "entry": entry,
            "exit_price": exit_price,
            "pnl_usd": pnl_usd,
            "duration": now_eat_str,
            "root_cause": analysis["root_cause"],
            "rule_followed": analysis["rule_followed"],
            "actionable_lesson": analysis["actionable_lesson"],
            "timestamp_eat": now_eat_str,
        }
        post_mortem.save_lesson({
            "timestamp_eat": now_eat_str,
            "pair": pair,
            "outcome": outcome,
            "actionable_lesson": analysis["actionable_lesson"],
            "root_cause": analysis["root_cause"],
        })
        notifier.send_post_mortem(pm_report)
    except Exception as exc:
        print(f"  [listener] Error running post-mortem for {signal_id[:8]}…: {exc}")

    # 6. Update the original Telegram card
    outcome_note = (
        f"{base_card}\n\n"
        f"{emoji} ({pnl_sign}${pnl_usd:.2f})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 Exit Price : {exit_price}\n"
        f"⏱  Closed EAT  : {now_eat_str}\n"
        f"🧠 Diagnostic feedback delivered below."
    )
    _edit_message(chat_id, message_id, outcome_note, reply_markup={"inline_keyboard": []})
    print(f"  [listener] Signal {signal_id[:8]}… outcome recorded as {outcome} ({pnl_sign}${pnl_usd:.2f}).")


# ─────────────────────────────────────────────────────────────────────────────
# Core polling loop
# ─────────────────────────────────────────────────────────────────────────────

def _process_callback_query(cq: dict[str, Any]) -> None:
    """Route a single Telegram callback_query to the right handler."""
    # ── Security: verify the chat matches TELEGRAM_CHAT_ID ───────────────────
    cq_chat_id = cq.get("message", {}).get("chat", {}).get("id")
    try:
        authorized_chat = int(config.TELEGRAM_CHAT_ID)
    except (ValueError, TypeError):
        print("  [listener] ⚠️  TELEGRAM_CHAT_ID is not set or not an integer. Ignoring.")
        return

    if cq_chat_id != authorized_chat:
        print(
            f"  [listener] ⛔ Unauthorized callback from chat_id={cq_chat_id}. Ignored."
        )
        return

    callback_query_id = cq.get("id", "")
    message_id        = cq.get("message", {}).get("message_id")
    chat_id           = cq_chat_id
    data              = cq.get("data", "")

    if ":" not in data:
        print(f"  [listener] Unrecognised callback data: {data!r}")
        _answer_callback(callback_query_id, text="Unknown action.")
        return

    action, signal_id = data.split(":", 1)
    signal = notifier.get_pending_signal(signal_id)

    if signal is None:
        _answer_callback(callback_query_id, text="Signal not found.", alert=True)
        _edit_message(chat_id, message_id, f"⚠️ Signal not found (ID: {signal_id[:8]}…).")
        print(f"  [listener] Signal {signal_id[:8]}… not found in pending_signals.json.")
        return

    current_status = signal.get("status", "PENDING")

    # Route based on action
    if action in ("accept", "decline"):
        if current_status not in ("PENDING",):
            _answer_callback(
                callback_query_id,
                text=f"Already {current_status}.",
                alert=True,
            )
            print(
                f"  [listener] Signal {signal_id[:8]}… already in state '{current_status}'. Skipped."
            )
            return

        if action == "decline":
            _handle_decline(signal_id, signal, chat_id, message_id, callback_query_id, cq)
        elif action == "accept":
            _handle_accept(signal_id, signal, chat_id, message_id, callback_query_id, cq)

    elif action in ("manual", "manual_mt5", "mt5_manual", "skip"):
        if current_status not in ("PENDING", "WAITING_MANUAL"):
            _answer_callback(
                callback_query_id,
                text=f"Already {current_status}.",
                alert=True,
            )
            return
        if action in ("manual", "manual_mt5", "mt5_manual"):
            _handle_manual(signal_id, signal, chat_id, message_id, callback_query_id, cq)
        elif action == "skip":
            _handle_skip(signal_id, signal, chat_id, message_id, callback_query_id, cq)

    elif action in ("win", "loss"):
        if current_status in ("CLOSED_WIN", "CLOSED_LOSS"):
            _answer_callback(
                callback_query_id,
                text=f"Outcome already recorded ({current_status}).",
                alert=True,
            )
            return
        _handle_report_outcome(action, signal_id, signal, chat_id, message_id, callback_query_id, cq)

    else:
        _answer_callback(callback_query_id, text="Unknown action.")
        print(f"  [listener] Unknown action: {action!r}")


def listen(timeout_seconds: int | None = None) -> None:
    """
    Long-poll Telegram's /getUpdates for callback_query events.

    Args:
        timeout_seconds: Stop after this many real-time seconds (None = run forever).
    """
    if not config.TELEGRAM_BOT_TOKEN:
        print("  [listener] TELEGRAM_BOT_TOKEN is not set. Cannot listen. Exiting.")
        return
    if not config.TELEGRAM_CHAT_ID:
        print("  [listener] TELEGRAM_CHAT_ID is not set. Cannot verify senders. Exiting.")
        return

    print(f"\n  [listener] 🎧 Listening for Telegram button taps …")
    if timeout_seconds:
        print(f"  [listener]    Will stop after {timeout_seconds}s (Ctrl-C to stop early).")
    else:
        print("  [listener]    Running indefinitely (Ctrl-C to stop).")
    print()

    offset        = None
    start_time    = time.monotonic()
    long_poll_timeout = 20   # seconds for each /getUpdates call

    try:
        while True:
            # Wall-clock timeout guard
            if timeout_seconds and (time.monotonic() - start_time) >= timeout_seconds:
                print(f"\n  [listener] ⏱  {timeout_seconds}s elapsed. Stopping listener.")
                break

            params: dict[str, Any] = {
                "allowed_updates": ["callback_query"],
                "timeout":         long_poll_timeout,
            }
            if offset is not None:
                params["offset"] = offset

            try:
                body = _tg_get("getUpdates", params=params)
            except requests.RequestException as exc:
                print(f"  [listener] getUpdates error: {exc} — retrying in 3s…")
                time.sleep(3)
                continue

            updates = body.get("result", [])
            for update in updates:
                offset = update["update_id"] + 1   # advance offset past this update
                cq = update.get("callback_query")
                if cq:
                    print(
                        f"  [listener] Received callback: {cq.get('data', '?')} "
                        f"from chat_id={cq.get('message', {}).get('chat', {}).get('id')}"
                    )
                    _process_callback_query(cq)

    except KeyboardInterrupt:
        print("\n  [listener] Stopped by user (Ctrl-C).")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram Approval Button Listener")
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Stop listening after N seconds (default: run forever).",
    )
    args = parser.parse_args()
    listen(timeout_seconds=args.timeout)


if __name__ == "__main__":
    main()
