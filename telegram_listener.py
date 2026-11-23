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
import sys
import time
from datetime import datetime
from typing import Any

import pytz
import requests

import config
import notifier
import broker_executor


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


def _edit_message(chat_id: int | str, message_id: int, new_text: str) -> None:
    """Edit the original Telegram message to reflect the action taken."""
    try:
        _tg_post("editMessageText", {
            "chat_id":    chat_id,
            "message_id": message_id,
            "text":       f"<pre>{new_text}</pre>",
            "parse_mode": "HTML",
            # Remove inline keyboard after action is taken
            "reply_markup": {"inline_keyboard": []},
        })
    except requests.RequestException as exc:
        print(f"  [listener] editMessageText failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Callback handlers
# ─────────────────────────────────────────────────────────────────────────────

def _handle_decline(
    signal_id: str,
    signal: dict[str, Any],
    chat_id: int | str,
    message_id: int,
    callback_query_id: str,
) -> None:
    """User tapped ❌ Decline — mark signal declined, update Telegram message."""
    _answer_callback(callback_query_id, text="Signal declined.")
    notifier.update_signal_status(signal_id, "DECLINED")

    pair      = signal.get("pair", "?")
    direction = signal.get("direction", "?")
    score     = signal.get("confluence_score", 0)

    edit_text = (
        "❌ Setup Declined by User\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 Pair    : {pair}\n"
        f"📊 Signal  : {direction}\n"
        f"🎯 Score   : {score}/100\n"
        f"🔑 ID      : {signal_id[:8]}…\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Declined. No order was placed."
    )
    _edit_message(chat_id, message_id, edit_text)
    print(f"  [listener] Signal {signal_id[:8]}… DECLINED by user.")


def _handle_accept(
    signal_id: str,
    signal: dict[str, Any],
    chat_id: int | str,
    message_id: int,
    callback_query_id: str,
) -> None:
    """
    User tapped ✅ Accept & Enter.

    1. Check signal expiry.
    2. If expired → reject and update message.
    3. If valid   → execute bracket order and confirm.
    """
    eat_tz  = pytz.timezone(config.TIMEZONE)
    now_eat = datetime.now(tz=eat_tz)

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
    score     = signal.get("confluence_score", 0)
    std_lots  = signal.get("standard_lots", 0.01)
    units     = signal.get("units", int(std_lots * 100_000))

    # ── Expiry check ──────────────────────────────────────────────────────────
    if age_minutes > config.SIGNAL_EXPIRY_MINUTES:
        _answer_callback(callback_query_id, text="⚠️ Signal expired!", alert=True)
        notifier.update_signal_status(signal_id, "EXPIRED")
        edit_text = (
            f"⚠️ Signal Expired (>{config.SIGNAL_EXPIRY_MINUTES} mins old) — Order Cancelled\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 Pair    : {pair}\n"
            f"📊 Signal  : {direction}\n"
            f"🎯 Score   : {score}/100\n"
            f"⏱  Age     : {age_minutes:.1f} min  (limit: {config.SIGNAL_EXPIRY_MINUTES} min)\n"
            f"🔑 ID      : {signal_id[:8]}…\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Order cancelled — price may have moved significantly."
        )
        _edit_message(chat_id, message_id, edit_text)
        print(
            f"  [listener] Signal {signal_id[:8]}… EXPIRED "
            f"({age_minutes:.1f} min > {config.SIGNAL_EXPIRY_MINUTES} min limit)."
        )
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    _answer_callback(callback_query_id, text="⚙️ Executing order…")
    executor = broker_executor.get_executor()
    result   = executor.execute_bracket_order(signal)

    if result["success"]:
        notifier.update_signal_status(signal_id, "EXECUTED")
        exec_tag   = "[DEMO MOCK]" if config.DEMO_MODE else "[LIVE]"
        fill_price = result.get("fill_price", signal.get("entry", "?"))
        edit_text  = (
            f"✅ Order Executed {exec_tag}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 Pair    : {pair}\n"
            f"📊 Signal  : {direction}\n"
            f"🎯 Score   : {score}/100\n"
            f"💵 Fill    : {fill_price}\n"
            f"🛑 SL      : {signal.get('stop_loss', '?')}\n"
            f"🎯 TP      : {signal.get('take_profit', '?')}\n"
            f"📦 Lots    : {std_lots} std  ({units:,} units)\n"
            f"🔑 ID      : {signal_id[:8]}…\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + result.get("message", "Order placed successfully.")
        )
        print(f"  [listener] Signal {signal_id[:8]}… EXECUTED → {result.get('message')}")
    else:
        notifier.update_signal_status(signal_id, "FAILED")
        edit_text = (
            f"❌ Order Execution Failed\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 Pair    : {pair}\n"
            f"📊 Signal  : {direction}\n"
            f"🔑 ID      : {signal_id[:8]}…\n"
            f"⚠️  Error   : {result.get('error', 'Unknown error')}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "No order was placed. Check broker_executor logs."
        )
        print(f"  [listener] Signal {signal_id[:8]}… EXECUTION FAILED: {result.get('error')}")

    _edit_message(chat_id, message_id, edit_text)


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

    # Guard against double-tapping
    current_status = signal.get("status", "PENDING")
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
        _handle_decline(signal_id, signal, chat_id, message_id, callback_query_id)
    elif action == "accept":
        _handle_accept(signal_id, signal, chat_id, message_id, callback_query_id)
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
