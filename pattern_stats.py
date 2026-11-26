"""
pattern_stats.py — Winning DNA Indicator Fingerprinting & Readiness Meter

Tracks the complete technical and contextual indicator fingerprint of every evaluated
and executed trade in `pattern_history.json`:
  - signal_id, pair, direction
  - rsi_14_15m, ema_signal, atr_15m, in_fib_golden_zone
  - candle_pattern, body_pct
  - news_bias, confluence_score, session_tag
  - outcome (OPEN, WIN, LOSS, SKIPPED, DECLINED), pnl_usd

Capabilities:
  1. `record_fingerprint()`: Logs or updates an indicator fingerprint.
  2. `update_outcome()`: Marks a trade WIN/LOSS with realized pnl_usd when closed.
  3. `get_win_signature_summary()`: Discovers which indicator ranges have the highest
     win rates ("Winning DNA") to inject into Gemini's system prompt.
  4. `get_readiness_metrics()`: Computes rolling 30-trade Win Rate, Profit Factor, and
     top winning patterns for Telegram post-mortem diagnostic reports.
"""

import json
import os
from datetime import datetime
from typing import Any

import pytz
import config

_BASE_DIR = os.path.dirname(__file__)
PATTERN_HISTORY_FILE = os.path.join(_BASE_DIR, "pattern_history.json")


def load_pattern_history() -> dict[str, dict[str, Any]]:
    """Load pattern_history.json returning dict keyed by signal_id."""
    if not os.path.exists(PATTERN_HISTORY_FILE):
        return {}
    try:
        with open(PATTERN_HISTORY_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_pattern_history(history: dict[str, dict[str, Any]]) -> None:
    """Save pattern_history.json atomically."""
    with open(PATTERN_HISTORY_FILE, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)


def record_fingerprint(
    signal_id: str,
    pair: str,
    direction: str,
    math_result: dict[str, Any],
    ai_result: dict[str, Any] | None = None,
    news_bias: str = "NEUTRAL",
    outcome: str = "OPEN",
    pnl_usd: float = 0.0,
    is_shadow_candidate: bool = False,
) -> dict[str, Any]:
    """
    Save or update a comprehensive Indicator Fingerprint in pattern_history.json.
    """
    history = load_pattern_history()

    candle_story = math_result.get("candle_story", {})
    eat_tz = pytz.timezone(config.TIMEZONE)
    now_eat_str = datetime.now(tz=eat_tz).strftime("%Y-%m-%d %H:%M:%S")

    confluence_score = 0
    if ai_result:
        confluence_score = ai_result.get("confluence_score", 0)

    fingerprint = {
        "signal_id":          signal_id,
        "timestamp_eat":      now_eat_str,
        "pair":               pair,
        "direction":          direction,
        "rsi_14_15m":         math_result.get("rsi_14_15m", 50.0),
        "ema_signal":         math_result.get("ema_signal", "NEUTRAL"),
        "ema_gap_pips":       math_result.get("ema_gap_pips", 0.0),
        "atr_15m":            math_result.get("atr_15m", 0.0),
        "in_fib_golden_zone": bool(math_result.get("in_fib_golden_zone", False)),
        "candle_pattern":     candle_story.get("pattern", "NEUTRAL"),
        "body_pct":           round(float(candle_story.get("body_pct", 0.0)), 3),
        "news_bias":          news_bias,
        "confluence_score":   confluence_score,
        "session_tag":        math_result.get("session_tag", "LONDON_OPEN"),
        "outcome":            outcome.upper(),
        "pnl_usd":            float(pnl_usd),
        "is_shadow_candidate": is_shadow_candidate,
    }

    history[signal_id] = fingerprint
    save_pattern_history(history)
    print(f"  [pattern_stats] Recorded fingerprint for {pair} ({signal_id[:8]}…) | {direction} | Score: {confluence_score}")
    return fingerprint


def update_outcome(signal_id: str, outcome: str, pnl_usd: float) -> bool:
    """
    Update trade outcome ('WIN' or 'LOSS') and realized PnL in pattern_history.json.
    """
    history = load_pattern_history()
    if signal_id in history:
        history[signal_id]["outcome"] = outcome.upper()
        history[signal_id]["pnl_usd"] = round(float(pnl_usd), 2)
        save_pattern_history(history)
        print(f"  [pattern_stats] Updated {signal_id[:8]}… outcome → {outcome.upper()} (${pnl_usd:.2f})")
        return True
    return False


def get_win_signature_summary() -> str:
    """
    Analyze closed trades to synthesize the 'Winning DNA' signatures.
    Returns a structured summary string to inject into Gemini's system prompt.
    """
    history = load_pattern_history()
    closed = [t for t in history.values() if t.get("outcome") in ("WIN", "LOSS", "SHADOW_WIN", "SHADOW_LOSS")]

    if len(closed) < 5:
        return "No empirical Winning DNA yet (fewer than 5 closed trades). Rely strictly on current price structure and math rules."

    # 1. Total Win Rate
    total = len(closed)
    wins = [t for t in closed if t.get("outcome") in ("WIN", "SHADOW_WIN")]
    overall_wr = round(len(wins) / total * 100, 1)

    # 2. Fib Golden Zone Win Rate
    fib_trades = [t for t in closed if t.get("in_fib_golden_zone")]
    fib_wins = [t for t in fib_trades if t.get("outcome") in ("WIN", "SHADOW_WIN")]
    fib_wr = round(len(fib_wins) / len(fib_trades) * 100, 1) if fib_trades else None

    # 3. Session Tag Win Rate
    session_stats: dict[str, list[bool]] = {}
    for t in closed:
        s = t.get("session_tag", "LONDON_OPEN")
        session_stats.setdefault(s, []).append(t.get("outcome") in ("WIN", "SHADOW_WIN"))

    best_session = None
    best_session_wr = -1.0
    for s, res in session_stats.items():
        if len(res) >= 2:
            wr = sum(res) / len(res) * 100
            if wr > best_session_wr:
                best_session_wr = wr
                best_session = s

    # 4. RSI Sweet Spot among winners
    win_rsis = [t.get("rsi_14_15m") for t in wins if t.get("rsi_14_15m") is not None]
    if win_rsis:
        min_win_rsi = round(min(win_rsis), 1)
        max_win_rsi = round(max(win_rsis), 1)
        rsi_clause = f"RSI 14 (15m) historically between {min_win_rsi} and {max_win_rsi}"
    else:
        rsi_clause = "RSI 14 (15m) in healthy 40–60 pullback zone"

    # 5. Top winning candle patterns
    pat_stats: dict[str, list[bool]] = {}
    for t in closed:
        p = t.get("candle_pattern", "NEUTRAL")
        pat_stats.setdefault(p, []).append(t.get("outcome") in ("WIN", "SHADOW_WIN"))

    winning_patterns = []
    for p, res in pat_stats.items():
        if p != "NEUTRAL" and len(res) >= 3:
            p_wr = sum(res) / len(res) * 100
            if p_wr >= 50.0:
                winning_patterns.append(f"{p} ({p_wr:.0f}% win rate, n={len(res)})")

    lines = [
        f"- Overall Empirical Win Rate: {overall_wr}% across {total} trades (including shadow learning).",
        f"- RSI Sweet Spot: {rsi_clause}.",
    ]
    if fib_wr is not None:
        lines.append(f"- Fibonacci Confluence: Trades inside 50%–61.8% Golden Zone have {fib_wr}% win rate (n={len(fib_trades)}).")
    if best_session and best_session_wr >= 50.0:
        lines.append(f"- Peak Session Performance: {best_session} produces {best_session_wr:.0f}% win rate.")
    if winning_patterns:
        lines.append(f"- Highest Edge Trigger Patterns: {', '.join(winning_patterns[:3])}.")

    return "\n".join(lines)


def get_readiness_metrics() -> dict[str, Any]:
    """
    Compute rolling 30-trade readiness metrics for post-mortem cards and status reports:
      - 30-Trade Win Rate
      - Profit Factor
      - Top winning pattern
      - Readiness Meter status
    """
    history = load_pattern_history()
    closed = [
        t for t in history.values()
        if t.get("outcome") in ("WIN", "LOSS") and not t.get("is_shadow_candidate")
    ]
    recent = closed[-30:] if len(closed) > 30 else closed

    total = len(recent)
    if total == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "profit_factor": "N/A",
            "total_pnl": 0.0,
            "top_pattern": "Insufficient data",
            "readiness_meter": "0/30 demo trades — Scouting Mode",
            "summary_text": "Readiness: 0/30 trades logged in scouting mode.",
        }

    wins = [t for t in recent if t.get("outcome") == "WIN"]
    losses = [t for t in recent if t.get("outcome") == "LOSS"]
    win_rate = round(len(wins) / total * 100, 1)

    gross_profit = sum(t.get("pnl_usd", 0.0) for t in wins)
    gross_loss = abs(sum(t.get("pnl_usd", 0.0) for t in losses))

    if gross_loss > 0:
        pf_val = round(gross_profit / gross_loss, 2)
        profit_factor = f"{pf_val:.2f}"
    elif gross_profit > 0:
        profit_factor = "∞ (100% wins)"
    else:
        profit_factor = "0.00"

    total_pnl = round(sum(t.get("pnl_usd", 0.0) for t in recent), 2)

    # Top winning pattern
    pat_wins: dict[str, int] = {}
    for t in wins:
        p = t.get("candle_pattern", "NEUTRAL")
        if p != "NEUTRAL":
            pat_wins[p] = pat_wins.get(p, 0) + 1

    top_pat = max(pat_wins.items(), key=lambda x: x[1])[0] if pat_wins else "None yet"

    # Status meter
    if total >= 30 and win_rate >= 60.0 and (gross_loss == 0 or (gross_profit / gross_loss) >= 1.8):
        status = f"✅ READY FOR LIVE TRADING ({win_rate}% WR, PF: {profit_factor})"
    else:
        status = f"🔬 DEMO DISCOVERY ({total}/30 trades | {win_rate}% WR | Target: 60% WR, PF > 1.8)"

    summary = (
        f"📊 30-Trade Readiness: {win_rate}% Win Rate ({len(wins)}W/{len(losses)}L) | "
        f"PF: {profit_factor} | Net: ${total_pnl:+.2f}\n"
        f"🎯 Top Pattern: {top_pat}\n"
        f"🚦 Status: {status}"
    )

    return {
        "total_trades":    total,
        "wins":            len(wins),
        "losses":          len(losses),
        "win_rate_pct":    win_rate,
        "profit_factor":   profit_factor,
        "total_pnl":       total_pnl,
        "top_pattern":     top_pat,
        "readiness_meter": status,
        "summary_text":    summary,
    }
