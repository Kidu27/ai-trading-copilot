"""
scheduler.py — Local Autonomous Session Scheduler for macOS

Automates the entire trading lifecycle during active London / NY session hours:
  - Timezone: Africa/Addis_Ababa (EAT, UTC+3)
  - Active Window: Monday through Friday, 10:30 to 19:30 EAT
  - Sleep Cycle: Automatically sleeps outside 10:30–19:30 and on weekends
  - Session Cycle: Runs scan across all 7 pairs every 15 minutes, pauses between pairs,
    services Telegram button interactions, and inspects for closed trades.
  - Clean shutdown on KeyboardInterrupt (Ctrl+C).

Usage:
  python scheduler.py
"""

import sys
import time
from datetime import datetime, timedelta, time as dtime

import pytz

import config
import main
import telegram_listener
import post_mortem


def _get_now_eat() -> datetime:
    """Return current timezone-aware datetime in EAT."""
    eat_tz = pytz.timezone(config.TIMEZONE)
    return datetime.now(tz=eat_tz)


def _get_seconds_until_next_session(now_eat: datetime) -> tuple[int, datetime]:
    """
    Calculate seconds until the next session open (10:30 EAT on a weekday).
    Returns (seconds_to_sleep, next_session_datetime).
    """
    start_h, start_m = map(int, config.LONDON_SESSION_START.split(":"))

    candidate = now_eat.replace(hour=start_h, minute=start_m, second=0, microsecond=0)

    # If today's start is already in the past, move to tomorrow
    if now_eat >= candidate:
        candidate += timedelta(days=1)

    # If candidate falls on weekend, advance to Monday
    while candidate.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        candidate += timedelta(days=1)

    diff_seconds = max(int((candidate - now_eat).total_seconds()), 1)
    return diff_seconds, candidate


def _is_session_active(now_eat: datetime) -> bool:
    """Return True if currently between 10:30 and 19:30 EAT Monday–Friday."""
    if now_eat.weekday() >= 5:  # Saturday or Sunday
        return False

    start_h, start_m = map(int, config.LONDON_SESSION_START.split(":"))
    end_h,   end_m   = map(int, config.LONDON_SESSION_END.split(":"))

    current_t = now_eat.time()
    return dtime(start_h, start_m) <= current_t <= dtime(end_h, end_m)


def run_scheduler() -> None:
    """Main continuous scheduling loop."""
    print("=" * 65)
    print("  🤖 AI FOREX CO-PILOT — AUTONOMOUS MAC SCHEDULER")
    print(f"  📍 Timezone       : {config.TIMEZONE} (EAT)")
    print(f"  ⏰ Trading Hours  : Mon–Fri {config.LONDON_SESSION_START}–{config.LONDON_SESSION_END} EAT")
    print(f"  📊 Active Pairs   : {len(config.WATCHLIST)} ({', '.join(config.WATCHLIST)})")
    print(f"  ⏱  Scan Interval  : Every {config.SCAN_INTERVAL_MINUTES} minutes")
    print(f"  🛡️  Mode           : {'DEMO DISCOVERY' if config.DEMO_DISCOVERY_MODE else 'STANDARD'}")
    print("=" * 65)

    try:
        while True:
            now_eat = _get_now_eat()

            if not _is_session_active(now_eat):
                sleep_secs, next_start = _get_seconds_until_next_session(now_eat)
                hours_left = sleep_secs / 3600.0
                print(
                    f"\n  ⏸  Outside session window or market closed. "
                    f"Current EAT: {now_eat.strftime('%A %H:%M:%S')}"
                )
                print(
                    f"  💤 Sleeping until next session at: {next_start.strftime('%Y-%m-%d %H:%M:%S')} EAT "
                    f"(~{hours_left:.1f} hours) …"
                )

                # Sleep in increments of up to 60 seconds to remain responsive to SIGINT
                wake_deadline = time.monotonic() + sleep_secs
                while time.monotonic() < wake_deadline:
                    time.sleep(min(30.0, wake_deadline - time.monotonic()))
                continue

            # Within active session hours:
            print(f"\n{'━'*65}")
            print(f"  🚀 [SESSION ACTIVE] Starting 15m cycle at {now_eat.strftime('%Y-%m-%d %H:%M:%S')} EAT")
            print(f"{'━'*65}")

            # 1. Run full 7-pair scan
            cycle_start = time.monotonic()
            try:
                main.run_scan()
            except Exception as exc:
                print(f"  ❌ Error during scan cycle: {exc}")

            # 2. Check for closed trades & run post-mortems
            try:
                post_mortem.check_closed_trades()
            except Exception as exc:
                print(f"  ❌ Error during closed trades check: {exc}")

            # 3. Track candle outcomes for open/missed signals
            try:
                import outcome_tracker
                outcome_tracker.check_open_signal_outcomes()
            except Exception as exc:
                print(f"  ❌ Error during outcome tracker check: {exc}")

            # 4. Service any interactive Telegram taps for 45 seconds
            if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
                try:
                    telegram_listener.listen(timeout_seconds=45)
                except Exception as exc:
                    print(f"  ⚠️ Telegram listener error: {exc}")

            # 5. Wait for next 15-minute slot
            interval_secs = config.SCAN_INTERVAL_MINUTES * 60
            elapsed = time.monotonic() - cycle_start
            remaining_sleep = max(10.0, interval_secs - elapsed)

            next_cycle_time = _get_now_eat() + timedelta(seconds=remaining_sleep)
            print(f"\n  ⏳ Cycle complete. Next scan scheduled at {next_cycle_time.strftime('%H:%M:%S')} EAT (~{remaining_sleep/60:.1f} min) …")

            sleep_deadline = time.monotonic() + remaining_sleep
            while time.monotonic() < sleep_deadline:
                if not _is_session_active(_get_now_eat()):
                    break
                time.sleep(min(15.0, sleep_deadline - time.monotonic()))

    except KeyboardInterrupt:
        print("\n\n  🛑 Scheduler stopped by user (Ctrl+C). Clean shutdown complete.")
        sys.exit(0)


if __name__ == "__main__":
    run_scheduler()
