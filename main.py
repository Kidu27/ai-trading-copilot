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

import pytz

import config
import data_fetcher
import market_math
import ai_auditor
import notifier
import journal
import post_mortem


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
    for idx, symbol in enumerate(config.WATCHLIST):
        print(f"\n── Analysing {symbol} ─────────────────────────────────")

        # ── Pair cooldown guard (4-hour deduplication) ────────────────────────
        on_cooldown, hours_left = journal.check_pair_cooldown(symbol, hours=4)
        if on_cooldown:
            print(
                f"  ⏳ {symbol} is on cooldown — last alert was < 4 hours ago. "
                f"({hours_left}h remaining). Skipping."
            )
            continue

        # GATE 1: Fetch data
        try:
            df_1h, df_15m = data_fetcher.fetch_pair_data(symbol)
        except RuntimeError as exc:
            print(f"  [Gate 1 FAILED] {exc}")
            continue

        # GATE 2: Deterministic math
        print(f"  [Gate 2] Running market math …")
        math_result = market_math.analyse(symbol, df_1h, df_15m)

        gate2_pass = math_result["passed_math_gate"]
        print(f"  [Gate 2] Trend: {math_result['trend_1h']} | "
              f"R:R: {math_result.get('risk_reward_ratio', 'N/A')} | "
              f"Gate: {'✅ PASS' if gate2_pass else '❌ FAIL'}")

        if not gate2_pass:
            for reason in math_result.get("fail_reasons", []):
                print(f"    ↳ {reason}")

        # GATE 3: AI audit (only if Gate 2 passes)
        if gate2_pass:
            print(f"  [Gate 3] Sending to Gemini AI auditor …")
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
        journal.log_setup(symbol, math_result, ai_result)

        # Only send alert for actionable signals OR always for NO_TRADE (transparency)
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
