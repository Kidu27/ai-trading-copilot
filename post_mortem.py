"""
post_mortem.py — Automated Trade Post-Mortem & Rolling Memory System

When a demo or live trade closes (hits Stop-Loss or Take-Profit):
  1. Inspects active_trades.json (or MT5_FILES_DIR/closed_trades/).
  2. Updates trades_log.csv with exit_price, exit_time, pnl_usd, and outcome (WIN / LOSS).
  3. Gathers trade details and 15min candle context covering the trade duration.
  4. Calls GEMINI_MODEL (gemini-4-hydrogen) via the official google-genai SDK
     with structured JSON output to analyze:
       - root_cause: Why the trade hit Take-Profit or Stop-Loss against our 5-step playbook.
       - rule_followed: Whether price respected S/R + Fib zone or violated it due to volatility/news.
       - actionable_lesson: Exactly ONE concrete, 1-sentence filter rule for future setups.
  5. Appends actionable_lesson to lessons_learned.json (retaining the most recent 10 lessons).
  6. Sends a concise diagnostic summary card to Telegram via notifier.py.
"""

import json
import os
import re
from datetime import datetime
from typing import Any

import requests
import pytz
import pandas as pd

import config
import notifier
import journal
import data_fetcher
import pattern_stats

# Official Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

# Pydantic schema
try:
    from pydantic import BaseModel
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False

if _PYDANTIC_AVAILABLE:
    class PostMortemAnalysis(BaseModel):
        root_cause: str
        rule_followed: str
        actionable_lesson: str


# ── File paths ────────────────────────────────────────────────────────────────
_BASE_DIR          = os.path.dirname(__file__)
ACTIVE_TRADES_FILE = os.path.join(_BASE_DIR, "active_trades.json")
LESSONS_FILE       = os.path.join(_BASE_DIR, "lessons_learned.json")


# ─────────────────────────────────────────────────────────────────────────────
# System prompt for post-mortem analysis
# ─────────────────────────────────────────────────────────────────────────────

_POST_MORTEM_SYSTEM_PROMPT = """
You are an elite Forex risk management auditor and trade psychologist conducting an \
objective post-mortem on a recently closed trade.

Your goal is to extract ONE actionable lesson to improve our automated 5-step trading playbook:
1. 1H Trend structure (HH/HL or LH/LL)
2. S/R zones & Fibonacci Golden Zone alignment
3. 15m closed candle trigger confirmation
4. Strict math confirmation (R:R >= 2.0, position sizing)
5. Volatility & news spike filter

Return strictly valid JSON matching this schema:
{
  "root_cause": "<1-2 clear sentences explaining why price reached the stop-loss or take-profit>",
  "rule_followed": "<1-2 sentences on whether price respected our S/R + Fib zone or violated it>",
  "actionable_lesson": "<Exactly ONE concrete 1-sentence rule to filter or improve future setups>"
}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Memory Store
# ─────────────────────────────────────────────────────────────────────────────

def load_lessons() -> list[dict[str, Any]]:
    """Load lessons_learned.json (returns empty list if missing or empty)."""
    if not os.path.exists(LESSONS_FILE):
        return []
    try:
        with open(LESSONS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_lesson(lesson_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Append a new lesson entry to lessons_learned.json.
    Enforces a rolling window of the MOST RECENT 10 lessons.
    """
    lessons = load_lessons()
    lessons.append(lesson_entry)
    # Retain only the most recent 10 lessons
    lessons = lessons[-10:]

    with open(LESSONS_FILE, "w", encoding="utf-8") as fh:
        json.dump(lessons, fh, indent=2)

    print(f"  [post_mortem] Lesson recorded → lessons_learned.json (rolling total: {len(lessons)}/10)")
    return lessons


def get_recent_lessons() -> list[dict[str, Any]]:
    """Retrieve the current rolling list of lessons."""
    return load_lessons()


# ─────────────────────────────────────────────────────────────────────────────
# Active Trades Store Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_active_trades() -> dict[str, Any]:
    """Load active_trades.json."""
    if not os.path.exists(ACTIVE_TRADES_FILE):
        return {}
    try:
        with open(ACTIVE_TRADES_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_active_trades(trades: dict[str, Any]) -> None:
    """Save active_trades.json."""
    with open(ACTIVE_TRADES_FILE, "w", encoding="utf-8") as fh:
        json.dump(trades, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# AI Post-Mortem Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _clean_json_text(text: str) -> str:
    """
    Strip markdown code fences (e.g. ```json ... ```) and leading/trailing whitespace
    so json.loads() receives clean JSON.
    """
    raw = text.strip()
    if "```" in raw:
        start_fence = raw.find("```")
        end_fence = raw.rfind("```")
        if start_fence != -1 and end_fence != -1 and end_fence > start_fence:
            inner = raw[start_fence:end_fence]
            first_nl = inner.find("\n")
            if first_nl != -1:
                return inner[first_nl + 1:].strip()
            return inner.lstrip("`").strip()
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        if first_newline != -1:
            raw = raw[first_newline + 1:]
        else:
            raw = raw.lstrip("`")
    if raw.endswith("```"):
        last_fence = raw.rfind("```")
        if last_fence != -1:
            raw = raw[:last_fence]
    return raw.strip()


def analyze_trade_with_ai(
    trade: dict[str, Any],
    exit_price: float,
    outcome: str,
    candles_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Diagnose a closed trade using multi-provider AI risk managers:
      Tier 1: GroqCloud (Primary)
      Tier 2: OpenRouter (Secondary)
      Tier 3: Gemini Fallback (Single model)
    Falls back to a deterministic, conservative diagnosis if all AI providers fail.
    """
    pair        = trade.get("pair", "EUR/USD")
    direction   = trade.get("direction", "ENTER_LONG")
    entry       = trade.get("entry", trade.get("fill_price", exit_price))
    stop_loss   = trade.get("stop_loss", "?")
    take_profit = trade.get("take_profit", "?")
    opened_eat  = trade.get("opened_eat", "unknown")
    closed_eat  = trade.get("closed_eat", "unknown")
    std_lots    = trade.get("standard_lots", 0.01)

    # Format recent candles if provided
    candle_summary: list[dict[str, Any]] = []
    if candles_df is not None and not candles_df.empty:
        tail = candles_df.tail(8)
        for ts, row in tail.iterrows():
            candle_summary.append({
                "time": str(ts),
                "open": round(float(row["open"]), 5),
                "high": round(float(row["high"]), 5),
                "low": round(float(row["low"]), 5),
                "close": round(float(row["close"]), 5),
            })

    # Prepare prompt
    user_prompt = (
        f"## Closed Trade Diagnostic Request\n\n"
        f"Pair: {pair}\n"
        f"Direction: {direction}\n"
        f"Entry Price: {entry}\n"
        f"Stop Loss: {stop_loss}\n"
        f"Take Profit: {take_profit}\n"
        f"Exit Price: {exit_price}\n"
        f"Outcome: {outcome.upper()}\n"
        f"Position Size: {std_lots} standard lots\n"
        f"Opened (EAT): {opened_eat}\n"
        f"Closed (EAT): {closed_eat}\n\n"
        f"### Recent 15m Candle Action:\n"
        f"```json\n{json.dumps(candle_summary, indent=2)}\n```\n\n"
        f"Provide your 3-field post-mortem JSON now."
    )

    # ── Tier 1: Primary — GroqCloud ──────────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY", "").strip() or getattr(config, "GROQ_API_KEY", "").strip()
    if groq_key:
        groq_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
        for model_id in groq_models:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "temperature": 0.1,
                        "max_tokens": 600,
                        "reasoning_effort": "low",
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": _POST_MORTEM_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    timeout=(5, 25),
                )
                if resp.status_code != 200:
                    print(f"  [post_mortem] Warning: Groq ({model_id}) returned HTTP {resp.status_code}")
                    continue

                data = resp.json()
                choices = data.get("choices")
                if not choices:
                    continue
                content = choices[0].get("message", {}).get("content", "")
                raw_text = _clean_json_text(content)
                if not raw_text:
                    continue

                parsed = json.loads(raw_text)
                if parsed.get("root_cause") and parsed.get("actionable_lesson"):
                    print(f"  [post_mortem] ✅ Diagnosed via Groq ({model_id})")
                    return {
                        "root_cause": str(parsed.get("root_cause", "")).strip(),
                        "rule_followed": str(parsed.get("rule_followed", "")).strip(),
                        "actionable_lesson": str(parsed.get("actionable_lesson", "")).strip(),
                    }
            except Exception as err:
                print(f"  [post_mortem] Warning: Groq ({model_id}) failed: {err}")
                continue

    # ── Tier 2: Secondary — OpenRouter ───────────────────────────────────────
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip() or getattr(config, "OPENROUTER_API_KEY", "").strip()
    if or_key:
        openrouter_models = ["nvidia/nemotron-3-ultra-550b-a55b:free", "openrouter/free"]
        for model_id in openrouter_models:
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {or_key}",
                        "Content-Type": "application/json",
                        "X-Title": "VanguardFX Post-Mortem",
                    },
                    json={
                        "model": model_id,
                        "temperature": 0.1,
                        "max_tokens": 600,
                        "messages": [
                            {"role": "system", "content": _POST_MORTEM_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    timeout=(5, 25),
                )
                if resp.status_code != 200:
                    print(f"  [post_mortem] Warning: OpenRouter ({model_id}) returned HTTP {resp.status_code}")
                    continue

                data = resp.json()
                choices = data.get("choices")
                if not choices:
                    continue
                content = choices[0].get("message", {}).get("content", "")
                raw_text = _clean_json_text(content)
                if not raw_text:
                    continue

                parsed = json.loads(raw_text)
                if parsed.get("root_cause") and parsed.get("actionable_lesson"):
                    print(f"  [post_mortem] ✅ Diagnosed via OpenRouter ({model_id})")
                    return {
                        "root_cause": str(parsed.get("root_cause", "")).strip(),
                        "rule_followed": str(parsed.get("rule_followed", "")).strip(),
                        "actionable_lesson": str(parsed.get("actionable_lesson", "")).strip(),
                    }
            except Exception as err:
                print(f"  [post_mortem] Warning: OpenRouter ({model_id}) failed: {err}")
                continue

    # ── Tier 3: Tertiary — Single Gemini Fallback ────────────────────────────
    gemini_key = getattr(config, "GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "").strip()
    if _GENAI_AVAILABLE and gemini_key:
        try:
            http_opts = (
                types.HttpOptions(timeout=15000, retry_options=types.HttpRetryOptions(attempts=1))
                if hasattr(types, "HttpOptions") and hasattr(types, "HttpRetryOptions")
                else None
            )
            client = genai.Client(api_key=gemini_key, http_options=http_opts) if http_opts else genai.Client(api_key=gemini_key)
            gen_kwargs: dict[str, Any] = {
                "system_instruction": _POST_MORTEM_SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "max_output_tokens": 600,
            }
            if _PYDANTIC_AVAILABLE:
                gen_kwargs["response_schema"] = PostMortemAnalysis

            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=user_prompt,
                    config=types.GenerateContentConfig(**gen_kwargs),
                )
            except Exception as call_err:
                err_str = str(call_err).upper()
                if (
                    "400" in err_str
                    or "INVALID_ARGUMENT" in err_str
                    or "INVALIDARGUMENT" in err_str
                    or "SCHEMA" in err_str
                ) and "response_schema" in gen_kwargs:
                    gen_kwargs.pop("response_schema", None)
                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=user_prompt,
                        config=types.GenerateContentConfig(**gen_kwargs),
                    )
                else:
                    raise call_err

            if response and response.text and response.text.strip():
                raw_text = _clean_json_text(response.text)
                parsed = json.loads(raw_text)
                if parsed.get("root_cause") and parsed.get("actionable_lesson"):
                    print("  [post_mortem] ✅ Diagnosed via Gemini (gemini-3.5-flash-lite)")
                    return {
                        "root_cause": str(parsed.get("root_cause", "")).strip(),
                        "rule_followed": str(parsed.get("rule_followed", "")).strip(),
                        "actionable_lesson": str(parsed.get("actionable_lesson", "")).strip(),
                    }
        except Exception as exc:
            print(f"  [post_mortem] Gemini fallback failed: {exc}")

    print("  [post_mortem] All AI providers exhausted, using rule-based diagnostic.")

    # Deterministic fallback diagnostic
    if outcome.upper() == "LOSS":
        return {
            "root_cause": (
                f"Price breached Stop-Loss at {exit_price} because counter-trend volatility "
                f"overwhelmed the key S/R zone on {pair}."
            ),
            "rule_followed": "The S/R boundary was violated without a second confirmation rejection.",
            "actionable_lesson": (
                f"For {pair}, require a confirmed 15m closed rejection candle before entering "
                "near S/R when market volatility (ATR) is expanding."
            ),
        }
    else:
        return {
            "root_cause": (
                f"Price followed the 1H trend cleanly from {entry} to Take-Profit at {exit_price} "
                "with minimal drawdown."
            ),
            "rule_followed": "The key S/R zone and Fibonacci retracement held firmly as anticipated.",
            "actionable_lesson": (
                f"On {pair}, continue enforcing minimum 1:2 R:R targets when 1H trend and "
                "15m rejection candle align with Fibonacci Golden Zone."
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public Operations: Check Closed Trades & Simulate Close
# ─────────────────────────────────────────────────────────────────────────────

def check_closed_trades(mock_candles: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    """
    Inspect active_trades.json (and MT5_FILES_DIR/closed_trades/) for any trades
    that have closed but have not yet undergone post-mortem analysis.

    For each closed trade:
      - Runs post-mortem AI audit
      - Updates trades_log.csv
      - Records lesson in lessons_learned.json (rolling 10)
      - Sends Telegram diagnostic alert
      - Marks post_mortem_done = True

    Returns:
        List of generated post-mortem report dicts.
    """
    trades = load_active_trades()
    reports: list[dict[str, Any]] = []

    # Check for MT5 bridge closed order JSON files if configured
    if config.MT5_FILES_DIR:
        closed_dir = os.path.join(config.MT5_FILES_DIR, "closed_trades")
        if os.path.exists(closed_dir):
            for fname in os.listdir(closed_dir):
                if fname.startswith("close_") and fname.endswith(".json"):
                    fpath = os.path.join(closed_dir, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as fh:
                            cdata = json.load(fh)
                        sig_id = cdata.get("signal_id")
                        if sig_id and sig_id in trades:
                            trades[sig_id]["status"] = "CLOSED"
                            trades[sig_id]["exit_price"] = cdata.get("exit_price")
                            trades[sig_id]["closed_eat"] = cdata.get("exit_time")
                            trades[sig_id]["pnl_usd"] = cdata.get("pnl_usd")
                            trades[sig_id]["outcome"] = cdata.get("outcome", "LOSS")
                    except Exception as exc:
                        print(f"  [post_mortem] Error reading MT5 closed file {fname}: {exc}")

    eat_tz = pytz.timezone(config.TIMEZONE)
    now_eat_str = datetime.now(tz=eat_tz).strftime("%Y-%m-%d %H:%M:%S")

    for sig_id, trade in list(trades.items()):
        if trade.get("status") == "CLOSED" and not trade.get("post_mortem_done"):
            pair = trade.get("pair", "EUR/USD")
            exit_price = trade.get("exit_price") or trade.get("stop_loss", 0.0)
            outcome = trade.get("outcome", "LOSS").upper()
            pnl_usd = trade.get("pnl_usd", 0.0)

            # Get candle context
            candles = mock_candles
            if candles is None and config.TWELVE_DATA_API_KEY:
                try:
                    candles = data_fetcher.fetch_candles(pair, "15min", 20)
                except Exception:
                    candles = None

            print(f"  [post_mortem] 🔬 Running diagnostic for {pair} ({sig_id[:8]}…) → {outcome}")
            analysis = analyze_trade_with_ai(trade, exit_price, outcome, candles)

            report = {
                "signal_id": sig_id,
                "pair": pair,
                "outcome": outcome,
                "entry": trade.get("entry", trade.get("fill_price")),
                "exit_price": exit_price,
                "pnl_usd": pnl_usd,
                "duration": trade.get("closed_eat", now_eat_str),
                "root_cause": analysis["root_cause"],
                "rule_followed": analysis["rule_followed"],
                "actionable_lesson": analysis["actionable_lesson"],
                "timestamp_eat": now_eat_str,
            }

            # 1. Update trades_log.csv
            journal.update_trade_outcome(
                signal_id=sig_id,
                exit_price=float(exit_price),
                exit_time=trade.get("closed_eat", now_eat_str),
                pnl_usd=float(pnl_usd),
                outcome=outcome,
                notes=analysis["actionable_lesson"][:100],
            )

            # 2. Append to lessons_learned.json (rolling 10)
            lesson_entry = {
                "timestamp_eat": now_eat_str,
                "pair": pair,
                "outcome": outcome,
                "actionable_lesson": analysis["actionable_lesson"],
                "root_cause": analysis["root_cause"],
            }
            save_lesson(lesson_entry)

            # 3. Update pattern_history.json with final outcome & PnL
            try:
                pattern_stats.update_outcome(
                    signal_id=sig_id,
                    outcome=outcome,
                    pnl_usd=float(pnl_usd),
                )
            except Exception as exc:
                print(f"  [post_mortem] Warning: failed to update pattern_history: {exc}")

            # 4. Send Telegram card (including Winning DNA & 30-Trade Readiness Meter)
            notifier.send_post_mortem(report)

            # 5. Mark post_mortem_done
            trade["post_mortem_done"] = True
            reports.append(report)

    save_active_trades(trades)
    return reports


def simulate_mock_close(
    outcome: str = "LOSS",
    signal_id: str | None = None,
    mock_candles: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Simulate an open trade hitting Stop-Loss ('LOSS') or Take-Profit ('WIN')
    and run the entire post-mortem diagnostic and memory-updating loop.

    If active_trades.json has no open trade, synthesizes one first.
    """
    trades = load_active_trades()
    target_trade: dict[str, Any] | None = None
    target_sig_id: str | None = None

    eat_tz = pytz.timezone(config.TIMEZONE)
    now_eat_str = datetime.now(tz=eat_tz).strftime("%Y-%m-%d %H:%M:%S")

    # Find open trade
    if signal_id and signal_id in trades:
        target_trade = trades[signal_id]
        target_sig_id = signal_id
    else:
        for sid, t in trades.items():
            if t.get("status") == "OPEN":
                target_trade = t
                target_sig_id = sid
                break

    # If no open trade exists in store, create a realistic mock open trade
    if target_trade is None:
        import uuid
        target_sig_id = str(uuid.uuid4())
        entry_price = 1.08500
        sl_price    = 1.08300
        tp_price    = 1.08950
        std_lots    = 0.01
        units       = 1000

        target_trade = {
            "signal_id": target_sig_id,
            "pair": "EUR/USD",
            "direction": "ENTER_LONG",
            "entry": entry_price,
            "fill_price": entry_price,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "standard_lots": std_lots,
            "units": units,
            "status": "OPEN",
            "executor": "mock",
            "demo_mode": True,
            "opened_eat": now_eat_str,
            "closed_eat": None,
            "pnl_usd": None,
            "post_mortem_done": False,
        }
        trades[target_sig_id] = target_trade
        # Also log to journal so the full chain exists
        journal.log_executed_trade(target_trade, fill_price=entry_price, notes="Mock setup for close simulation")

    # Determine exit price & PnL
    direction = target_trade.get("direction", "ENTER_LONG")
    entry     = float(target_trade.get("entry", target_trade.get("fill_price", 1.0850)))
    std_lots  = float(target_trade.get("standard_lots", 0.01))

    if outcome.upper() == "LOSS":
        exit_price = float(target_trade.get("stop_loss", entry - 0.0020))
        # PnL in USD (approx $10 per pip per standard lot)
        if direction == "ENTER_LONG":
            pips = (exit_price - entry) / 0.0001
        else:
            pips = (entry - exit_price) / 0.0001
        pnl_usd = round(pips * 10.0 * std_lots, 2)
    else:  # WIN
        exit_price = float(target_trade.get("take_profit", entry + 0.0045))
        if direction == "ENTER_LONG":
            pips = (exit_price - entry) / 0.0001
        else:
            pips = (entry - exit_price) / 0.0001
        pnl_usd = round(pips * 10.0 * std_lots, 2)

    # Mark as CLOSED
    target_trade["status"]           = "CLOSED"
    target_trade["outcome"]          = outcome.upper()
    target_trade["exit_price"]       = exit_price
    target_trade["closed_eat"]       = now_eat_str
    target_trade["pnl_usd"]          = pnl_usd
    target_trade["post_mortem_done"] = False

    trades[target_sig_id] = target_trade
    save_active_trades(trades)

    # Run check_closed_trades to process this closure
    reports = check_closed_trades(mock_candles=mock_candles)
    return reports[0] if reports else {
        "signal_id": target_sig_id,
        "pair": target_trade.get("pair", "EUR/USD"),
        "outcome": outcome.upper(),
        "exit_price": exit_price,
        "pnl_usd": pnl_usd,
        "actionable_lesson": "Simulation completed.",
    }
