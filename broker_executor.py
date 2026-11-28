"""
broker_executor.py — Cloud-Native cTrader Open API & Mock Execution Engine

Architecture
────────────
  BrokerExecutor (abstract base)
    ├── CTraderExecutor    ← used when cTrader Open API credentials are configured
    │     Connects directly over TLS WebSocket (JSON protocol) to Spotware cTrader
    │     Open API (demo or live endpoint based on config.DEMO_MODE).
    │     Submits MARKET orders with attached relativeStopLoss & relativeTakeProfit
    │     and records fills in active_trades.json.
    └── MockDemoExecutor   ← used for offline testing (--mock) or when cTrader
          credentials are not configured. Simulates instant bracket-order fill
          and writes open trade to active_trades.json.

Usage (from telegram_listener.py)
─────────────────────────────────
    from broker_executor import get_executor
    executor = get_executor()
    result   = executor.execute_bracket_order(signal)
"""

import asyncio
import json
import os
import ssl
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import certifi
import pytz
import requests
import websockets

import config

# ── Shared file paths ──────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(__file__)
ACTIVE_TRADES_FILE = os.path.join(_BASE_DIR, "active_trades.json")

# ── cTrader Open API constants ─────────────────────────────────────────────────
_CTRADER_DEMO_WS = "wss://demo.ctraderapi.com:5036"
_CTRADER_LIVE_WS = "wss://live.ctraderapi.com:5036"
_CTRADER_TOKEN_URL = "https://openapi.ctrader.com/apps/token"

_PT_HEARTBEAT = 51
_PT_PROTO_ERROR_RES = 50
_PT_APP_AUTH_REQ = 2100
_PT_APP_AUTH_RES = 2101
_PT_ACCOUNT_AUTH_REQ = 2102
_PT_ACCOUNT_AUTH_RES = 2103
_PT_NEW_ORDER_REQ = 2106
_PT_SYMBOLS_LIST_REQ = 2114
_PT_SYMBOLS_LIST_RES = 2115
_PT_EXECUTION_EVENT = 2126
_PT_ORDER_ERROR_EVENT = 2132
_PT_ERROR_RES = 2142

# Per-account normalized symbol cache: {ctidTraderAccountId: {"EURUSD": 1, ...}}
_SYMBOL_CACHE: dict[int, dict[str, int]] = {}


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
                "success":             bool,
                "executor":            str,       # "mock" | "ctrader_demo" | "ctrader_live"
                "signal_id":           str,
                "pair":                str,
                "direction":           str,
                "entry":               float | None,
                "stop_loss":           float | None,
                "take_profit":         float | None,
                "standard_lots":       float,
                "units":               int,
                "fill_price":          float | None,
                "ctrader_position_id": int | None,
                "timestamp_eat":       str,
                "message":             str,
                "error":               str | None,
              }
        """


# ─────────────────────────────────────────────────────────────────────────────
# MockDemoExecutor
# ─────────────────────────────────────────────────────────────────────────────

class MockDemoExecutor(BrokerExecutor):
    """
    Simulates instant bracket-order fill for offline / demo testing.

    Does NOT touch any real broker. Records the simulated open trade to
    active_trades.json so the full approval -> execution workflow can be
    verified end-to-end without a live broker connection.

    Lot-size safety: enforces minimum 0.01 standard lots (1,000 units).
    """

    def execute_bracket_order(self, signal: dict[str, Any]) -> dict[str, Any]:
        eat_tz = pytz.timezone(config.TIMEZONE)
        now_eat = datetime.now(tz=eat_tz)

        pair = signal.get("pair", "UNKNOWN")
        direction = signal.get("direction", "UNKNOWN")
        entry = signal.get("entry")
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")
        signal_id = signal.get("signal_id", "")

        # Enforce minimum lot size: 0.01 standard lots = 1,000 units
        raw_lots = float(signal.get("standard_lots", 0.01))
        std_lots = max(round(raw_lots, 2), 0.01)
        units = int(round(std_lots * 100_000))

        # Simulate fill at entry price (no slippage in mock mode)
        fill_price = entry

        trade_record = {
            "signal_id": signal_id,
            "pair": pair,
            "direction": direction,
            "entry": entry,
            "fill_price": fill_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "standard_lots": std_lots,
            "units": units,
            "status": "OPEN",
            "executor": "mock",
            "demo_mode": True,
            "opened_eat": now_eat.strftime("%Y-%m-%d %H:%M:%S"),
            "closed_eat": None,
            "pnl_usd": None,
        }

        _append_active_trade(trade_record)

        msg = (
            f"[DEMO MOCK] 🟢 Order simulated — {direction} {pair}\n"
            f"  Entry: {fill_price}  SL: {stop_loss}  TP: {take_profit}\n"
            f"  Lots: {std_lots} std ({units:,} units)"
        )
        print(f"  [broker_executor] {msg}")

        return {
            "success": True,
            "executor": "mock",
            "signal_id": signal_id,
            "pair": pair,
            "direction": direction,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "standard_lots": std_lots,
            "units": units,
            "fill_price": fill_price,
            "ctrader_position_id": None,
            "timestamp_eat": now_eat.strftime("%Y-%m-%d %H:%M:%S"),
            "message": msg,
            "error": None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CTraderExecutor (JSON over WebSocket)
# ─────────────────────────────────────────────────────────────────────────────

def _get_ctrader_env() -> dict[str, str]:
    """Read cTrader Open API credentials from config with env fallback."""
    return {
        "client_id": (getattr(config, "CTRADER_CLIENT_ID", "") or os.getenv("CTRADER_CLIENT_ID", "")).strip(),
        "client_secret": (getattr(config, "CTRADER_CLIENT_SECRET", "") or os.getenv("CTRADER_CLIENT_SECRET", "")).strip(),
        "access_token": (getattr(config, "CTRADER_ACCESS_TOKEN", "") or os.getenv("CTRADER_ACCESS_TOKEN", "")).strip(),
        "refresh_token": (getattr(config, "CTRADER_REFRESH_TOKEN", "") or os.getenv("CTRADER_REFRESH_TOKEN", "")).strip(),
        "account_id": (getattr(config, "CTRADER_ACCOUNT_ID", "") or os.getenv("CTRADER_ACCOUNT_ID", "")).strip(),
    }


def has_ctrader_credentials() -> bool:
    """Return True when all required cTrader Open API credentials are configured."""
    creds = _get_ctrader_env()
    return bool(
        creds["client_id"]
        and creds["client_secret"]
        and creds["access_token"]
        and creds["account_id"]
    )


def _run_coro(coro_fn) -> Any:
    """Run an async coroutine cleanly from sync code, even if an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_fn())

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro_fn())).result()


def _format_ctrader_error(payload: dict[str, Any], default_code: str = "CTRADER_ERROR") -> str:
    """Format exact errorCode and description from cTrader error payloads."""
    code = payload.get("errorCode") or default_code
    desc = payload.get("description") or payload.get("message") or ""
    return f"{code}: {desc}".strip(": ")


class CTraderExecutor(BrokerExecutor):
    """
    Cloud-native cTrader Open API executor (JSON over TLS WebSocket).

    Connects to Spotware's JSON WebSocket gateway:
      - Demo: wss://demo.ctraderapi.com:5036 (when config.DEMO_MODE is True)
      - Live: wss://live.ctraderapi.com:5036 (when config.DEMO_MODE is False)

    Authenticates application + trading account (with automatic in-memory OAuth
    token refresh on expiry), resolves symbolId, places a MARKET order with
    attached relativeStopLoss / relativeTakeProfit, waits up to 15s for fill
    confirmation, and persists filled trades to active_trades.json.
    """

    def __init__(self, timeout_s: float = 15.0) -> None:
        creds = _get_ctrader_env()
        self.client_id = creds["client_id"]
        self.client_secret = creds["client_secret"]
        self.access_token = creds["access_token"]
        self.refresh_token = creds["refresh_token"]
        self.account_id = creds["account_id"]
        self.timeout_s = float(timeout_s)

    @property
    def endpoint(self) -> str:
        return _CTRADER_DEMO_WS if config.DEMO_MODE else _CTRADER_LIVE_WS

    @property
    def executor_name(self) -> str:
        return "ctrader_demo" if config.DEMO_MODE else "ctrader_live"

    def _refresh_access_token_sync(self) -> tuple[bool, str]:
        """
        Refresh cTrader OAuth access token in memory via Spotware's token endpoint.
        Never writes tokens to disk or GitHub.
        """
        if not self.refresh_token:
            return False, "CTRADER_REFRESH_TOKEN is not set."

        try:
            resp = requests.post(
                _CTRADER_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=self.timeout_s,
            )
            if resp.status_code != 200:
                return False, f"Token refresh HTTP {resp.status_code}: {resp.text[:200]}"

            body = resp.json()
            new_access = (body.get("accessToken") or body.get("access_token") or "").strip()
            new_refresh = (body.get("refreshToken") or body.get("refresh_token") or "").strip()
            if not new_access:
                err_desc = body.get("description") or body.get("error_description") or body.get("errorCode") or "Missing accessToken in refresh response"
                return False, str(err_desc)

            self.access_token = new_access
            config.CTRADER_ACCESS_TOKEN = new_access
            os.environ["CTRADER_ACCESS_TOKEN"] = new_access

            if new_refresh:
                self.refresh_token = new_refresh
                config.CTRADER_REFRESH_TOKEN = new_refresh
                os.environ["CTRADER_REFRESH_TOKEN"] = new_refresh

            print(
                "  [broker_executor] 🔄 cTrader access token refreshed in memory! "
                "Reminder: update CTRADER_ACCESS_TOKEN / CTRADER_REFRESH_TOKEN in your Render environment variables."
            )
            return True, "Refreshed"
        except Exception as exc:
            return False, f"Token refresh failed: {exc}"

    async def _send_json(self, ws: Any, payload_type: int, payload: dict[str, Any]) -> str:
        msg_id = str(uuid.uuid4())
        frame = {
            "clientMsgId": msg_id,
            "payloadType": payload_type,
            "payload": payload,
        }
        await ws.send(json.dumps(frame))
        return msg_id

    async def _recv_until(
        self,
        ws: Any,
        expected_types: set[int],
        deadline: float,
    ) -> dict[str, Any]:
        """Receive JSON frames until one of expected_types or an error payload arrives."""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"cTrader response timeout after {self.timeout_s:.0f}s")

            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            msg = json.loads(raw)
            ptype = msg.get("payloadType")

            if ptype == _PT_HEARTBEAT:
                continue

            if ptype in expected_types or ptype in (_PT_ERROR_RES, _PT_PROTO_ERROR_RES, _PT_ORDER_ERROR_EVENT):
                return msg

    async def _authenticate_and_load_symbols(
        self,
        ws: Any,
        acct_id: int,
        force_symbol_fetch: bool = False,
        needed_symbol: str = "",
    ) -> tuple[dict[str, int] | None, str | None]:
        """Perform App Auth (2100), Account Auth (2102 with refresh retry), and Symbols List (2114)."""
        deadline = time.monotonic() + self.timeout_s

        # 1. Application Auth
        await self._send_json(ws, _PT_APP_AUTH_REQ, {
            "clientId": self.client_id,
            "clientSecret": self.client_secret,
        })
        app_res = await self._recv_until(ws, {_PT_APP_AUTH_RES}, deadline)
        if app_res.get("payloadType") != _PT_APP_AUTH_RES:
            return None, _format_ctrader_error(app_res.get("payload", {}), "APP_AUTH_ERROR")

        # 2. Account Auth (with automatic token refresh retry on expired/invalid token)
        for attempt in range(2):
            await self._send_json(ws, _PT_ACCOUNT_AUTH_REQ, {
                "ctidTraderAccountId": acct_id,
                "accessToken": self.access_token,
            })
            acct_res = await self._recv_until(ws, {_PT_ACCOUNT_AUTH_RES}, deadline)
            if acct_res.get("payloadType") == _PT_ACCOUNT_AUTH_RES:
                break

            err_payload = acct_res.get("payload", {})
            err_code = str(err_payload.get("errorCode", "")).upper()
            err_desc = str(err_payload.get("description", "")).upper()
            is_token_issue = any(k in err_code or k in err_desc for k in ("TOKEN", "EXPIRED", "INVALID", "AUTH"))

            if attempt == 0 and is_token_issue and self.refresh_token:
                ok, refresh_err = await asyncio.to_thread(self._refresh_access_token_sync)
                if ok:
                    deadline = time.monotonic() + self.timeout_s
                    continue
                return None, f"{_format_ctrader_error(err_payload)} (refresh error: {refresh_err})"

            return None, _format_ctrader_error(err_payload, "ACCOUNT_AUTH_ERROR")

        # 3. Symbol ID Cache
        cached = _SYMBOL_CACHE.get(acct_id)
        if force_symbol_fetch or cached is None or (needed_symbol and needed_symbol not in cached):
            await self._send_json(ws, _PT_SYMBOLS_LIST_REQ, {
                "ctidTraderAccountId": acct_id,
                "includeArchivedSymbols": False,
            })
            sym_res = await self._recv_until(ws, {_PT_SYMBOLS_LIST_RES}, deadline)
            if sym_res.get("payloadType") != _PT_SYMBOLS_LIST_RES:
                return None, _format_ctrader_error(sym_res.get("payload", {}), "SYMBOLS_LIST_ERROR")

            symbol_map: dict[str, int] = {}
            for item in sym_res.get("payload", {}).get("symbol", []):
                name = str(item.get("symbolName", "")).replace("/", "").upper()
                sid = item.get("symbolId")
                if name and sid is not None:
                    symbol_map[name] = int(sid)
            _SYMBOL_CACHE[acct_id] = symbol_map

        return _SYMBOL_CACHE.get(acct_id, {}), None

    async def _verify_connection_async(self) -> tuple[bool, str]:
        if not has_ctrader_credentials():
            return False, "cTrader Open API credentials are not configured."

        try:
            acct_id = int(self.account_id)
        except ValueError:
            return False, f"Invalid CTRADER_ACCOUNT_ID: {self.account_id!r}"

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        env_label = "DEMO" if config.DEMO_MODE else "LIVE"
        try:
            async with websockets.connect(
                self.endpoint,
                ssl=ssl_ctx,
                open_timeout=self.timeout_s,
                close_timeout=5,
            ) as ws:
                symbols, err = await self._authenticate_and_load_symbols(
                    ws, acct_id=acct_id, force_symbol_fetch=True
                )
                if err is not None or symbols is None:
                    return False, err or "Authentication failed."
                return True, (
                    f"Connected to cTrader {env_label} "
                    f"(Account {acct_id}, {len(symbols)} symbols available)"
                )
        except Exception as exc:
            return False, f"cTrader connection error: {exc}"

    def verify_connection(self) -> tuple[bool, str]:
        """Authenticate and fetch symbol count without placing any order."""
        return _run_coro(self._verify_connection_async)

    async def _execute_bracket_order_async(
        self,
        signal: dict[str, Any],
        acct_id: int,
        entry_val: float,
        sl_val: float,
        tp_val: float,
        std_lots: float,
        units: int,
        ct_volume: int,
        rel_sl: int,
        rel_tp: int,
        trade_side: int,
    ) -> dict[str, Any]:
        pair = signal.get("pair", "UNKNOWN")
        direction = signal.get("direction", "UNKNOWN")
        signal_id = signal.get("signal_id", "")
        norm_pair = pair.split(" ")[0].replace("/", "").upper()

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        try:
            async with websockets.connect(
                self.endpoint,
                ssl=ssl_ctx,
                open_timeout=self.timeout_s,
                close_timeout=5,
            ) as ws:
                symbols, err = await self._authenticate_and_load_symbols(
                    ws, acct_id=acct_id, needed_symbol=norm_pair
                )
                if err is not None or symbols is None:
                    return self._fail_dict(signal, entry_val, sl_val, tp_val, std_lots, units, err or "Auth failed")

                symbol_id = symbols.get(norm_pair)
                if symbol_id is None:
                    return self._fail_dict(
                        signal, entry_val, sl_val, tp_val, std_lots, units,
                        f"SYMBOL_NOT_FOUND: Pair '{norm_pair}' not found on cTrader account {acct_id}.",
                    )

                order_payload = {
                    "ctidTraderAccountId": acct_id,
                    "symbolId": symbol_id,
                    "orderType": 1,  # MARKET
                    "tradeSide": trade_side,
                    "volume": ct_volume,
                    "relativeStopLoss": rel_sl,
                    "relativeTakeProfit": rel_tp,
                    "label": "VanguardFX",
                    "comment": f"VFX_{signal_id[:8]}",
                }
                await self._send_json(ws, _PT_NEW_ORDER_REQ, order_payload)

                deadline = time.monotonic() + self.timeout_s
                while True:
                    event = await self._recv_until(ws, {_PT_EXECUTION_EVENT}, deadline)
                    ptype = event.get("payloadType")
                    payload = event.get("payload", {})

                    if ptype in (_PT_ERROR_RES, _PT_PROTO_ERROR_RES, _PT_ORDER_ERROR_EVENT):
                        err_str = _format_ctrader_error(payload, "ORDER_ERROR")
                        return self._fail_dict(signal, entry_val, sl_val, tp_val, std_lots, units, err_str)

                    if ptype == _PT_EXECUTION_EVENT:
                        exec_type = payload.get("executionType")

                        # Skip intermediate ORDER_ACCEPTED
                        if exec_type in (2, "ORDER_ACCEPTED"):
                            continue

                        # Order filled
                        if exec_type in (3, "ORDER_FILLED"):
                            position = payload.get("position") or {}
                            order = payload.get("order") or {}
                            deal = payload.get("deal") or {}

                            position_id = position.get("positionId") or order.get("positionId")
                            raw_fill = (
                                deal.get("executionPrice")
                                or order.get("executionPrice")
                                or position.get("price")
                                or entry_val
                            )
                            fill_price = float(raw_fill)
                            return self._success_dict(
                                signal=signal,
                                entry_val=entry_val,
                                sl_val=sl_val,
                                tp_val=tp_val,
                                std_lots=std_lots,
                                units=units,
                                fill_price=fill_price,
                                position_id=position_id,
                            )

                        # Order rejected / cancelled / expired
                        if exec_type in (7, "ORDER_REJECTED", 5, "ORDER_CANCELLED", 6, "ORDER_EXPIRED"):
                            err_str = _format_ctrader_error(
                                payload,
                                default_code=str(exec_type) if isinstance(exec_type, str) else "ORDER_REJECTED",
                            )
                            return self._fail_dict(signal, entry_val, sl_val, tp_val, std_lots, units, err_str)

        except Exception as exc:
            return self._fail_dict(
                signal, entry_val, sl_val, tp_val, std_lots, units,
                f"cTrader execution error: {exc}",
            )

    def _success_dict(
        self,
        signal: dict[str, Any],
        entry_val: float,
        sl_val: float,
        tp_val: float,
        std_lots: float,
        units: int,
        fill_price: float,
        position_id: Any,
    ) -> dict[str, Any]:
        eat_tz = pytz.timezone(config.TIMEZONE)
        now_eat_str = datetime.now(tz=eat_tz).strftime("%Y-%m-%d %H:%M:%S")

        signal_id = signal.get("signal_id", "")
        pair = signal.get("pair", "UNKNOWN")
        direction = signal.get("direction", "UNKNOWN")
        risk_usd = float(signal.get("risk_usd") or round(config.ACCOUNT_BALANCE * config.MAX_RISK_PERCENT, 2))
        calc_rr = round(abs(tp_val - entry_val) / abs(entry_val - sl_val), 2) if sl_val != entry_val else 2.0
        risk_reward = float(signal.get("risk_reward") or calc_rr)

        trade_record = {
            "signal_id": signal_id,
            "pair": pair,
            "direction": direction,
            "executor": self.executor_name,
            "demo_mode": config.DEMO_MODE,
            "ctrader_position_id": position_id,
            "fill_price": fill_price,
            "entry_price": fill_price,
            "entry": entry_val,
            "stop_loss": sl_val,
            "take_profit": tp_val,
            "risk_usd": risk_usd,
            "risk_reward": risk_reward,
            "standard_lots": std_lots,
            "units": units,
            "status": "OPEN",
            "timestamp_eat": signal.get("timestamp_eat") or now_eat_str,
            "opened_eat": now_eat_str,
            "closed_eat": None,
            "pnl_usd": None,
        }
        _append_active_trade(trade_record)

        env_tag = "cTRADER DEMO" if config.DEMO_MODE else "cTRADER LIVE"
        msg = (
            f"[{env_tag}] ✅ Order FILLED — {direction} {pair}\n"
            f"  Position ID: {position_id}  Fill: {fill_price}  SL: {sl_val}  TP: {tp_val}\n"
            f"  Lots: {std_lots} std ({units:,} units)"
        )
        print(f"  [broker_executor] {msg}")

        return {
            "success": True,
            "executor": self.executor_name,
            "signal_id": signal_id,
            "pair": pair,
            "direction": direction,
            "entry": entry_val,
            "stop_loss": sl_val,
            "take_profit": tp_val,
            "standard_lots": std_lots,
            "units": units,
            "fill_price": fill_price,
            "ctrader_position_id": position_id,
            "timestamp_eat": now_eat_str,
            "message": msg,
            "error": None,
        }

    def _fail_dict(
        self,
        signal: dict[str, Any],
        entry_val: float | None,
        sl_val: float | None,
        tp_val: float | None,
        std_lots: float,
        units: int,
        error: str,
    ) -> dict[str, Any]:
        eat_tz = pytz.timezone(config.TIMEZONE)
        now_eat_str = datetime.now(tz=eat_tz).strftime("%Y-%m-%d %H:%M:%S")
        env_tag = "cTRADER DEMO" if config.DEMO_MODE else "cTRADER LIVE"
        msg = f"[{env_tag}] ❌ Order failed: {error}"
        print(f"  [broker_executor] {msg}")

        return {
            "success": False,
            "executor": self.executor_name,
            "signal_id": signal.get("signal_id", ""),
            "pair": signal.get("pair", "UNKNOWN"),
            "direction": signal.get("direction", "UNKNOWN"),
            "entry": entry_val,
            "stop_loss": sl_val,
            "take_profit": tp_val,
            "standard_lots": std_lots,
            "units": units,
            "fill_price": None,
            "ctrader_position_id": None,
            "timestamp_eat": now_eat_str,
            "message": msg,
            "error": error,
        }

    def execute_bracket_order(self, signal: dict[str, Any]) -> dict[str, Any]:
        raw_lots = float(signal.get("standard_lots", 0.01))
        std_lots = max(round(raw_lots, 2), 0.01)
        units = int(round(std_lots * 100_000))
        ct_volume = max(int(round(std_lots * 10_000_000)), 100_000)

        if not has_ctrader_credentials():
            return self._fail_dict(
                signal,
                signal.get("entry"),
                signal.get("stop_loss"),
                signal.get("take_profit"),
                std_lots,
                units,
                "cTrader Open API credentials are not configured.",
            )

        try:
            acct_id = int(self.account_id)
        except ValueError:
            return self._fail_dict(
                signal,
                signal.get("entry"),
                signal.get("stop_loss"),
                signal.get("take_profit"),
                std_lots,
                units,
                f"Invalid CTRADER_ACCOUNT_ID: {self.account_id!r}",
            )

        try:
            entry_val = float(signal.get("entry"))
            sl_val = float(signal.get("stop_loss"))
            tp_val = float(signal.get("take_profit"))
        except (TypeError, ValueError):
            return self._fail_dict(
                signal,
                signal.get("entry"),
                signal.get("stop_loss"),
                signal.get("take_profit"),
                std_lots,
                units,
                "Invalid entry, stop_loss, or take_profit price in signal.",
            )

        rel_sl = int(round(abs(entry_val - sl_val) * 100_000))
        rel_tp = int(round(abs(tp_val - entry_val) * 100_000))
        if rel_sl <= 0 or rel_tp <= 0:
            return self._fail_dict(
                signal,
                entry_val,
                sl_val,
                tp_val,
                std_lots,
                units,
                "Stop-loss and take-profit distances must be greater than zero.",
            )

        dir_upper = str(signal.get("direction", "")).upper()
        trade_side = 1 if ("LONG" in dir_upper or "BUY" in dir_upper) else 2

        return _run_coro(
            lambda: self._execute_bracket_order_async(
                signal=signal,
                acct_id=acct_id,
                entry_val=entry_val,
                sl_val=sl_val,
                tp_val=tp_val,
                std_lots=std_lots,
                units=units,
                ct_volume=ct_volume,
                rel_sl=rel_sl,
                rel_tp=rel_tp,
                trade_side=trade_side,
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory & verification helpers
# ─────────────────────────────────────────────────────────────────────────────

def verify_connection() -> tuple[bool, str]:
    """Verify cTrader Open API connectivity and symbol access without placing an order."""
    if not has_ctrader_credentials():
        return False, "cTrader Open API credentials are not configured."
    return CTraderExecutor().verify_connection()


def get_executor() -> BrokerExecutor:
    """Return CTraderExecutor when cTrader credentials are configured, else MockDemoExecutor."""
    if has_ctrader_credentials():
        env_label = "DEMO" if config.DEMO_MODE else "LIVE"
        print(f"  [broker_executor] cTrader credentials active ({env_label}) → using CTraderExecutor.")
        return CTraderExecutor()
    print(f"  [broker_executor] DEMO_MODE={config.DEMO_MODE} (no cTrader credentials) → using MockDemoExecutor.")
    return MockDemoExecutor()


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
