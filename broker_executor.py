"""
broker_executor.py — Demo & MT5 macOS File-Bridge Execution Engine

IMPORTANT: This module runs strictly in DEMO_MODE = True.
No live orders are submitted unless you explicitly flip DEMO_MODE = False
and configure a working MT5_FILES_DIR path pointing to MetaTrader 5's
MQL5/Files directory on macOS.

The Windows-only 'MetaTrader5' Python package is intentionally NOT used here
because this project runs on macOS. The MT5MacBridgeExecutor communicates
with MT5-for-Mac via JSON files read by a lightweight MQL5 file-bridge EA.

Architecture
────────────
  BrokerExecutor (abstract base)
    ├── MockDemoExecutor      ← always used when DEMO_MODE = True
    │     Simulates instant bracket-order fill.
    │     Writes open trade to active_trades.json.
    └── MT5MacBridgeExecutor  ← used when DEMO_MODE = False and MT5_FILES_DIR is set
          Writes order_<signal_id>.json to MT5_FILES_DIR/pending_orders/
          Polls MT5_FILES_DIR/executions/ for fill confirmation (up to 15 s).

Usage (from telegram_listener.py)
─────────────────────────────────
    from broker_executor import get_executor
    executor = get_executor()
    result   = executor.execute_bracket_order(signal)
"""

import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pytz

import config

# ── Shared file paths ──────────────────────────────────────────────────────────
_BASE_DIR        = os.path.dirname(__file__)
ACTIVE_TRADES_FILE = os.path.join(_BASE_DIR, "active_trades.json")


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BrokerExecutor(ABC):
    """Abstract base for all execution backends."""

    @abstractmethod
    def execute_bracket_order(self, signal: dict[str, Any]) -> dict[str, Any]:
        """
        Submit a bracket order (Entry + Stop-Loss + Take-Profit).

        Args:
            signal: The pending signal dict from pending_signals.json.

        Returns:
            Execution result dict:
              {
                "success":      bool,
                "executor":     str,       # "mock" | "mt5_mac_bridge"
                "signal_id":    str,
                "pair":         str,
                "direction":    str,
                "entry":        float | None,
                "stop_loss":    float | None,
                "take_profit":  float | None,
                "standard_lots": float,
                "units":        int,
                "fill_price":   float | None,
                "timestamp_eat": str,
                "message":      str,
                "error":        str | None,
              }
        """


# ─────────────────────────────────────────────────────────────────────────────
# MockDemoExecutor
# ─────────────────────────────────────────────────────────────────────────────

class MockDemoExecutor(BrokerExecutor):
    """
    Simulates instant bracket-order fill for offline / demo testing.

    Does NOT touch any real broker. Records the simulated open trade to
    active_trades.json so the full approval → execution workflow can be
    verified end-to-end without a live broker connection.

    Lot-size safety: enforces minimum 0.01 standard lots (1,000 units).
    """

    def execute_bracket_order(self, signal: dict[str, Any]) -> dict[str, Any]:
        eat_tz    = pytz.timezone(config.TIMEZONE)
        now_eat   = datetime.now(tz=eat_tz)

        pair         = signal.get("pair", "UNKNOWN")
        direction    = signal.get("direction", "UNKNOWN")
        entry        = signal.get("entry")
        stop_loss    = signal.get("stop_loss")
        take_profit  = signal.get("take_profit")
        signal_id    = signal.get("signal_id", "")

        # Enforce minimum lot size: 0.01 standard lots = 1,000 units
        raw_lots  = float(signal.get("standard_lots", 0.01))
        std_lots  = max(round(raw_lots, 2), 0.01)
        units     = int(std_lots * 100_000)

        # Simulate fill at entry price (no slippage in mock mode)
        fill_price = entry

        trade_record = {
            "signal_id":    signal_id,
            "pair":         pair,
            "direction":    direction,
            "entry":        entry,
            "fill_price":   fill_price,
            "stop_loss":    stop_loss,
            "take_profit":  take_profit,
            "standard_lots": std_lots,
            "units":        units,
            "status":       "OPEN",
            "executor":     "mock",
            "demo_mode":    True,
            "opened_eat":   now_eat.strftime("%Y-%m-%d %H:%M:%S"),
            "closed_eat":   None,
            "pnl_usd":      None,
        }

        # Persist to active_trades.json
        _append_active_trade(trade_record)

        msg = (
            f"[DEMO MOCK] 🟢 Order simulated — {direction} {pair}\n"
            f"  Entry: {fill_price}  SL: {stop_loss}  TP: {take_profit}\n"
            f"  Lots: {std_lots} std ({units:,} units)"
        )
        print(f"  [broker_executor] {msg}")

        return {
            "success":       True,
            "executor":      "mock",
            "signal_id":     signal_id,
            "pair":          pair,
            "direction":     direction,
            "entry":         entry,
            "stop_loss":     stop_loss,
            "take_profit":   take_profit,
            "standard_lots": std_lots,
            "units":         units,
            "fill_price":    fill_price,
            "timestamp_eat": now_eat.strftime("%Y-%m-%d %H:%M:%S"),
            "message":       msg,
            "error":         None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MT5MacBridgeExecutor
# ─────────────────────────────────────────────────────────────────────────────

class MT5MacBridgeExecutor(BrokerExecutor):
    """
    macOS-compatible MetaTrader 5 file-bridge executor.

    How it works:
      1. Writes a JSON command file to MT5_FILES_DIR/pending_orders/order_<signal_id>.json.
         A lightweight MQL5 Expert Advisor (file-bridge EA) running inside MT5
         reads this file, places the bracket order, and writes a fill-confirmation
         JSON to MT5_FILES_DIR/executions/fill_<signal_id>.json.
      2. This executor polls the executions directory for up to `poll_timeout_s`
         seconds (default 15 s) for the fill confirmation.
      3. If no confirmation arrives in time, returns success=False with an error message.

    MQL5 file-bridge EA contract (the EA must write this structure):
      {
        "signal_id": "<uuid>",
        "ticket":    <MT5 order ticket int>,
        "fill_price": <float>,
        "status":    "FILLED" | "REJECTED" | "ERROR",
        "error_msg": ""
      }
    """

    def __init__(self, mt5_files_dir: str, poll_timeout_s: int = 15) -> None:
        self.mt5_files_dir   = mt5_files_dir
        self.poll_timeout_s  = poll_timeout_s
        self._pending_dir    = os.path.join(mt5_files_dir, "pending_orders")
        self._executions_dir = os.path.join(mt5_files_dir, "executions")
        os.makedirs(self._pending_dir,    exist_ok=True)
        os.makedirs(self._executions_dir, exist_ok=True)

    def execute_bracket_order(self, signal: dict[str, Any]) -> dict[str, Any]:
        eat_tz   = pytz.timezone(config.TIMEZONE)
        now_eat  = datetime.now(tz=eat_tz)

        pair        = signal.get("pair", "UNKNOWN")
        direction   = signal.get("direction", "UNKNOWN")
        entry       = signal.get("entry")
        stop_loss   = signal.get("stop_loss")
        take_profit = signal.get("take_profit")
        signal_id   = signal.get("signal_id", "")

        raw_lots = float(signal.get("standard_lots", 0.01))
        std_lots = max(round(raw_lots, 2), 0.01)
        units    = int(std_lots * 100_000)

        # Build MT5 order direction string from our internal direction label
        mt5_direction = "BUY" if direction == "ENTER_LONG" else "SELL"

        # ── Write the command file for the MQL5 bridge EA ─────────────────────
        command = {
            "signal_id":    signal_id,
            "action":       "BRACKET_ORDER",
            "symbol":       pair.replace("/", ""),   # e.g. "EURUSD"
            "direction":    mt5_direction,
            "volume":       std_lots,
            "entry":        entry,
            "stop_loss":    stop_loss,
            "take_profit":  take_profit,
            "comment":      f"AGY_{signal_id[:8]}",
            "issued_eat":   now_eat.strftime("%Y-%m-%d %H:%M:%S"),
        }
        order_file = os.path.join(self._pending_dir, f"order_{signal_id}.json")
        with open(order_file, "w", encoding="utf-8") as fh:
            json.dump(command, fh, indent=2)
        print(f"  [broker_executor] MT5 command written → {order_file}")

        # ── Poll for fill confirmation ─────────────────────────────────────────
        fill_file = os.path.join(self._executions_dir, f"fill_{signal_id}.json")
        deadline  = time.monotonic() + self.poll_timeout_s
        fill_data: dict[str, Any] | None = None

        while time.monotonic() < deadline:
            if os.path.exists(fill_file):
                try:
                    with open(fill_file, encoding="utf-8") as fh:
                        fill_data = json.load(fh)
                    break
                except (json.JSONDecodeError, OSError):
                    pass
            time.sleep(0.5)

        if fill_data is None:
            return {
                "success":       False,
                "executor":      "mt5_mac_bridge",
                "signal_id":     signal_id,
                "pair":          pair,
                "direction":     direction,
                "entry":         entry,
                "stop_loss":     stop_loss,
                "take_profit":   take_profit,
                "standard_lots": std_lots,
                "units":         units,
                "fill_price":    None,
                "timestamp_eat": now_eat.strftime("%Y-%m-%d %H:%M:%S"),
                "message":       "MT5 bridge timeout — no fill confirmation received.",
                "error":         f"No fill_{signal_id}.json after {self.poll_timeout_s}s.",
            }

        status    = fill_data.get("status", "ERROR")
        fill_price = fill_data.get("fill_price")
        ticket     = fill_data.get("ticket")
        error_msg  = fill_data.get("error_msg", "")

        if status == "FILLED":
            trade_record = {
                "signal_id":    signal_id,
                "pair":         pair,
                "direction":    direction,
                "entry":        entry,
                "fill_price":   fill_price,
                "stop_loss":    stop_loss,
                "take_profit":  take_profit,
                "standard_lots": std_lots,
                "units":        units,
                "mt5_ticket":   ticket,
                "status":       "OPEN",
                "executor":     "mt5_mac_bridge",
                "demo_mode":    config.DEMO_MODE,
                "opened_eat":   now_eat.strftime("%Y-%m-%d %H:%M:%S"),
                "closed_eat":   None,
                "pnl_usd":      None,
            }
            _append_active_trade(trade_record)
            msg = (
                f"[MT5 BRIDGE] ✅ Order FILLED — {direction} {pair}\n"
                f"  Ticket: {ticket}  Fill: {fill_price}  SL: {stop_loss}  TP: {take_profit}\n"
                f"  Lots: {std_lots} std ({units:,} units)"
            )
            print(f"  [broker_executor] {msg}")
            return {
                "success":       True,
                "executor":      "mt5_mac_bridge",
                "signal_id":     signal_id,
                "pair":          pair,
                "direction":     direction,
                "entry":         entry,
                "stop_loss":     stop_loss,
                "take_profit":   take_profit,
                "standard_lots": std_lots,
                "units":         units,
                "fill_price":    fill_price,
                "mt5_ticket":    ticket,
                "timestamp_eat": now_eat.strftime("%Y-%m-%d %H:%M:%S"),
                "message":       msg,
                "error":         None,
            }
        else:
            msg = f"[MT5 BRIDGE] ❌ Order {status}: {error_msg}"
            print(f"  [broker_executor] {msg}")
            return {
                "success":       False,
                "executor":      "mt5_mac_bridge",
                "signal_id":     signal_id,
                "pair":          pair,
                "direction":     direction,
                "entry":         entry,
                "stop_loss":     stop_loss,
                "take_profit":   take_profit,
                "standard_lots": std_lots,
                "units":         units,
                "fill_price":    None,
                "timestamp_eat": now_eat.strftime("%Y-%m-%d %H:%M:%S"),
                "message":       msg,
                "error":         error_msg,
            }


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def get_executor() -> BrokerExecutor:
    """
    Return the appropriate executor based on config.

    Rules:
      - DEMO_MODE = True  → always MockDemoExecutor (safe default).
      - DEMO_MODE = False AND MT5_FILES_DIR is set → MT5MacBridgeExecutor.
      - DEMO_MODE = False AND MT5_FILES_DIR is empty → MockDemoExecutor + warning.
    """
    if config.DEMO_MODE:
        print("  [broker_executor] DEMO_MODE=True → using MockDemoExecutor.")
        return MockDemoExecutor()

    if config.MT5_FILES_DIR:
        print(
            f"  [broker_executor] DEMO_MODE=False + MT5_FILES_DIR set → "
            f"using MT5MacBridgeExecutor."
        )
        return MT5MacBridgeExecutor(config.MT5_FILES_DIR)

    print(
        "  [broker_executor] ⚠️  DEMO_MODE=False but MT5_FILES_DIR is not set. "
        "Falling back to MockDemoExecutor."
    )
    return MockDemoExecutor()


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper — active trades persistence
# ─────────────────────────────────────────────────────────────────────────────

def _append_active_trade(trade: dict[str, Any]) -> None:
    """Add a trade record to active_trades.json (keyed by signal_id)."""
    if os.path.exists(ACTIVE_TRADES_FILE):
        try:
            with open(ACTIVE_TRADES_FILE, encoding="utf-8") as fh:
                store = json.load(fh)
        except (json.JSONDecodeError, OSError):
            store = {}
    else:
        store = {}

    store[trade["signal_id"]] = trade
    with open(ACTIVE_TRADES_FILE, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)
    print(f"  [broker_executor] Trade recorded → active_trades.json  ({trade['signal_id'][:8]}…)")
