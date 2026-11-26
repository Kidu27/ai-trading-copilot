"""
main.py — AI Forex Trading Co-Pilot: Main Loop

Orchestrates all 4 gates for each pair in WATCHLIST:
  Gate 1: data_fetcher   → fetch candles
  Gate 2: market_math    → deterministic analysis (hard filter)
  Gate 3: ai_auditor     → Gemini AI risk audit (only if Gate 2 passes)
  Gate 4: notifier       → format & deliver alert
         journal         → log every evaluated setup

Usage:
  python main.py              — single scan of all pairs
  python main.py --loop 15   — scan every 15 minutes (EAT session aware)

Session guard (always active, even without --session-only):
  If the current EAT time is Saturday or Sunday, or outside
  10:30–19:30 EAT, the script exits immediately without spending
  any Twelve Data or Gemini API credits.
"""

import sys
import time
import argparse
from datetime import datetime
from typing import Any

import pytz

import config
import data_fetcher
import market_math
import ai_auditor
import notifier
import journal
import post_mortem
import pattern_stats


# ─────────────────────────────────────────────────────────────────────────────
# Session guard — called once at startup before anything else
# ─────────────────────────────────────────────────────────────────────────────

def _check_session_or_exit() -> None:
    """
    Exit immediately if:
      • The current day is Saturday (weekday 5) or Sunday (weekday 6), OR
      • The current EAT time is outside LONDON_SESSION_START–LONDON_SESSION_END.

    This runs BEFORE any API calls so zero Twelve Data or Gemini credits
    are spent outside trading hours.
    """
    eat_tz  = pytz.timezone(config.TIMEZONE)
    now_eat = datetime.now(tz=eat_tz)

    # Weekend block
    if now_eat.weekday() >= 5:   # 5 = Saturday, 6 = Sunday
        day_name = now_eat.strftime("%A")
        print(
            f"  🗓  {day_name} — market is closed. "
            f"No API calls made. Exiting."
        )
        sys.exit(0)

    # Session block
    from datetime import time as dtime
    start_h, start_m = map(int, config.LONDON_SESSION_START.split(":"))
    end_h,   end_m   = map(int, config.LONDON_SESSION_END.split(":"))
    session_start = dtime(start_h, start_m)
    session_end   = dtime(end_h,   end_m)
    current_time  = now_eat.time()

    if not (session_start <= current_time <= session_end):
        print(
            f"  ⏸  Outside trading session "
            f"({config.LONDON_SESSION_START}–{config.LONDON_SESSION_END} EAT). "
            f"Current EAT: {now_eat.strftime('%H:%M')}. "
            f"No API calls made. Exiting."
        )
        sys.exit(0)


def run_scan() -> None:
    """Run one complete scan cycle over all watchlist pairs."""
    eat_tz  = pytz.timezone(config.TIMEZONE)
    now_eat = datetime.now(tz=eat_tz)
    print(f"\n{'='*55}")
    print(f"  🕐 Scan started: {now_eat.strftime('%Y-%m-%d %H:%M:%S')} EAT")
    print(f"{'='*55}")

    # ── Daily circuit breaker check ───────────────────────────────────────────
    limit_reached, loss_count = journal.check_daily_loss_limit()
    if limit_reached:
        print(
            f"\n  🚫 DAILY LOSS LIMIT REACHED ({loss_count}/{config.MAX_DAILY_LOSSES} losses)."
            f"\n     No new trade alerts will be sent today. Rest and review your journal.\n"
        )
        return

    # ── Scan each pair ────────────────────────────────────────────────────────
    fetched_15m_by_pair: dict[str, Any] = {}
    for idx, symbol in enumerate(config.WATCHLIST):
        print(f"\n── Analysing {symbol} ─────────────────────────────────")

        # ── Pair cooldown guard ───────────────────────────────────────────────
        cooldown_hours = getattr(config, "PAIR_COOLDOWN_HOURS", 1)
        on_cooldown, hours_left = journal.check_pair_cooldown(symbol, hours=cooldown_hours)
        if on_cooldown:
            print(
                f"  ⏳ {symbol} is on cooldown — last alert was < {cooldown_hours} hours ago. "
                f"({hours_left:.1f}h remaining). Skipping."
            )
            continue

        # GATE 1: Fetch data
        try:
            df_1h, df_15m = data_fetcher.fetch_pair_data(symbol)
            fetched_15m_by_pair[symbol] = df_15m
            if df_1h is not None and not df_1h.empty:
                market_math.update_fx_rate(symbol, float(df_1h["close"].iloc[-1]))
        except RuntimeError as exc:
            print(f"  [Gate 1 FAILED] {exc}")
            continue

        # GATE 2: Deterministic math
        print(f"  [Gate 2] Running market math …")
        math_result = market_math.analyse(symbol, df_1h, df_15m)

        # Attach unique signal_id to math_result BEFORE journal.log_setup() so trades_log.csv has it
        import uuid
        signal_id = str(uuid.uuid4())
        math_result["signal_id"] = signal_id

        gate2_pass = math_result["passed_math_gate"]
        print(f"  [Gate 2] Trend: {math_result['trend_1h']} | "
              f"R:R: {math_result.get('risk_reward_ratio', 'N/A')} | "
              f"Gate: {'✅ PASS' if gate2_pass else '❌ FAIL'}")

        if not gate2_pass:
            for reason in math_result.get("fail_reasons", []):
                print(f"    ↳ {reason}")

        # GATE 3: AI audit (only if Gate 2 passes)
        if gate2_pass:
            print(f"  [Gate 3] Sending to AI auditor …")
            ai_result = ai_auditor.audit(math_result)
            decision  = ai_result.get("decision", "NO_TRADE")
            score     = ai_result.get("confluence_score", 0)
            print(f"  [Gate 3] Decision: {decision} | Score: {score}/100")
        else:
            # Build a minimal AI result so journal & notifier still work
            ai_result = ai_auditor._fail_safe(
                f"Math gate failed: {'; '.join(math_result.get('fail_reasons', []))}"
            )
            decision = "NO_TRADE"

        # GATE 4: Log + notify
        min_confluence = getattr(config, "get_min_confluence_score", lambda: config.MIN_CONFLUENCE_SCORE)()
        is_shadow = (
            gate2_pass
            and decision == "NO_TRADE"
            and (50 <= score < min_confluence)
            and not ai_result.get("_fail_safe")
        )

        if is_shadow:
            shadow_dir = "SHADOW_LONG" if "LONG" in str(math_result.get("direction", "")).upper() else "SHADOW_SHORT"
            ai_result["decision"] = shadow_dir
            decision = shadow_dir
            print(f"  [Gate 3] Tagged as shadow candidate: {decision} ({score}/100) — learning silently.")

            # Ensure trade_plan has entry, SL, TP levels from math_result so outcome_tracker can evaluate it
            tp = ai_result.setdefault("trade_plan", {})
            if not tp.get("entry"):
                tp["entry"] = math_result.get("entry_price")
            if not tp.get("stop_loss"):
                tp["stop_loss"] = math_result.get("stop_loss")
            if not tp.get("take_profit"):
                tp["take_profit"] = math_result.get("take_profit")
            if not tp.get("risk_reward"):
                tp["risk_reward"] = math_result.get("risk_reward_ratio")
            if not tp.get("risk_usd"):
                tp["risk_usd"] = math_result.get("max_risk_usd")
            if not tp.get("lots"):
                tp["lots"] = math_result.get("position_size_lots")

            journal.log_setup(
                symbol,
                math_result,
                ai_result,
                outcome="SHADOW_OPEN",
                notes=f"Shadow candidate ({score}/100)",
            )

            pattern_stats.record_fingerprint(
                signal_id=math_result["signal_id"],
                pair=symbol,
                direction=shadow_dir,
                math_result=math_result,
                ai_result=ai_result,
                news_bias=ai_result.get("news_bias", "NEUTRAL"),
                outcome="SHADOW_PENDING",
                pnl_usd=0.0,
                is_shadow_candidate=True,
            )
        else:
            journal.log_setup(symbol, math_result, ai_result)

        # Notification filter:
        # Respect config.NOTIFY_MODE:
        #   - "TRADES_ONLY": only actionable setups (ENTER_LONG/ENTER_SHORT or math-approved fallback)
        #   - "GATE2_ONLY": setups passing Gate 2 math (actionable or Gemini NO_TRADE rejections)
        #   - "ALL": all evaluated pairs
        # Always keep SHADOW_* setups completely silent.
        notify_mode = getattr(config, "NOTIFY_MODE", "TRADES_ONLY").upper()
        is_actionable = decision in ("ENTER_LONG", "ENTER_SHORT")
        is_math_fallback = (decision == "NO_TRADE" and gate2_pass and bool(ai_result.get("_fail_safe")))
        is_shadow = decision in ("SHADOW_LONG", "SHADOW_SHORT")

        if is_shadow:
            should_alert = False
        elif is_actionable or is_math_fallback:
            should_alert = True
        elif notify_mode == "ALL":
            should_alert = True
        elif notify_mode == "GATE2_ONLY":
            should_alert = gate2_pass
        elif notify_mode == "TRADES_ONLY":
            should_alert = getattr(config, "NOTIFY_NO_TRADE", False)
        else:
            should_alert = False

        if should_alert:
            notifier.send_alert(symbol, math_result, ai_result)

        # Delay between pairs to respect API credit limits
        if idx < len(config.WATCHLIST) - 1 and getattr(config, "PAIR_REQUEST_DELAY_SECONDS", 0) > 0:
            delay = config.PAIR_REQUEST_DELAY_SECONDS
            print(f"  ⏳ Pausing {delay}s before next pair to stay within API rate limit …")
            time.sleep(delay)

    # ── Check for any closed trades and run post-mortem ──────────────────────
    try:
        post_mortem.check_closed_trades()
    except Exception as exc:
        print(f"  [main] Post-mortem check error: {exc}")

    # ── Automatic Candle Outcome Tracker for Open & Missed Signals ───────────
    try:
        import outcome_tracker
        outcome_tracker.check_open_signal_outcomes(candles_by_pair=fetched_15m_by_pair)
    except Exception as exc:
        print(f"  [main] Outcome tracker error: {exc}")

    print(f"\n{'='*55}")
    print(f"  ✅ Scan complete.")
    print(f"{'='*55}\n")


def _is_in_session() -> bool:
    """Check whether the current EAT time falls in the London session window."""
    eat_tz  = pytz.timezone(config.TIMEZONE)
    now     = datetime.now(tz=eat_tz).time()

    start_h, start_m = map(int, config.LONDON_SESSION_START.split(":"))
    end_h,   end_m   = map(int, config.LONDON_SESSION_END.split(":"))

    from datetime import time as dtime
    session_start = dtime(start_h, start_m)
    session_end   = dtime(end_h,   end_m)

    return session_start <= now <= session_end


def main() -> None:
    # ── Session / weekend guard — exits immediately if outside trading hours ──
    _check_session_or_exit()

    parser = argparse.ArgumentParser(description="AI Forex Trading Co-Pilot")
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        metavar="MINUTES",
        help="Repeat scan every N minutes (0 = run once and exit).",
    )
    parser.add_argument(
        "--session-only",
        action="store_true",
        help="In loop mode, skip scans outside the London session window.",
    )
    args = parser.parse_args()

    if args.loop > 0:
        interval_seconds = args.loop * 60
        print(f"  Running in loop mode — scanning every {args.loop} minutes.")
        while True:
            if args.session_only and not _is_in_session():
                print(f"  ⏸  Outside session window ({config.LONDON_SESSION_START}–"
                      f"{config.LONDON_SESSION_END} EAT). Sleeping …")
            else:
                run_scan()
            time.sleep(interval_seconds)
    else:
        run_scan()


if __name__ == "__main__":
    main()
