"""
outcome_tracker.py — Automated Candle Outcome Tracker for Open & Missed Signals

Scans trades_log.csv (and pending_signals.json) for any OPEN, EXECUTED, or MANUAL
signals from the last 48 hours that do not yet have a recorded exit outcome.

Using 15m candle highs and lows:
  - Detects if price touched take_profit (WIN) or stop_loss (LOSS).
  - For unentered / missed setups (e.g. today's AUD/USD 76/100 signal):
      Tags outcome as SHADOW_WIN / SHADOW_LOSS, updates pattern_history.json so the AI
      learns the winning/losing indicator fingerprint, and notifies via Telegram.
  - For active EXECUTED / MANUAL trades:
      Updates trades_log.csv, pattern_history.json, active_trades.json, sends the outcome
      alert, and triggers post_mortem.py for complete AI root-cause analysis.
"""

import csv
import json
import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytz

import config
import journal
import notifier
import pattern_stats
import post_mortem

try:
    import data_fetcher
except ImportError:
    data_fetcher = None


def check_open_signal_outcomes(candles_by_pair: dict[str, pd.DataFrame] | None = None) -> list[dict[str, Any]]:
    """
    Scan trades_log.csv for signals in the last 48h that have not yet reached an exit outcome.
    Evaluates historical 15m candle highs/lows to detect TP or SL touches.

    Args:
        candles_by_pair: Optional dict of pair -> DataFrame (used for mock tests or custom candles).

    Returns:
        List of resolved outcome report dicts.
    """
    eat_tz = pytz.timezone(config.TIMEZONE)
    now_eat = datetime.now(tz=eat_tz)
    cutoff_time = now_eat - timedelta(hours=48)

    if not os.path.exists(journal.LOG_FILE):
        return []

    # Read trades_log.csv
    try:
        with open(journal.LOG_FILE, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except Exception as exc:
        print(f"  [outcome_tracker] Error reading trades_log.csv: {exc}")
        return []

    pending_store = notifier._load_pending_signals()
    resolved_reports: list[dict[str, Any]] = []
    updated_rows_needed = False

    for idx, row in enumerate(rows):
        decision = row.get("decision", "").strip()
        outcome = row.get("outcome", "").strip()
        exit_price = row.get("exit_price", "").strip()
        pair = row.get("pair", "").strip()
        ts_str = row.get("timestamp_eat", "").strip()

        # Only evaluate actionable trade signals
        if decision not in ("ENTER_LONG", "ENTER_SHORT"):
            continue

        # Skip already completed trades
        if outcome in ("WIN", "LOSS", "SHADOW_WIN", "SHADOW_LOSS") and exit_price:
            continue

        # Parse timestamp and check 48h window
        try:
            row_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            row_dt = eat_tz.localize(row_dt)
        except Exception:
            continue

        if row_dt < cutoff_time:
            continue

        # Link signal_id if blank in CSV (e.g. historical unlinked row)
        signal_id = row.get("signal_id", "").strip()
        matched_pending = None

        if not signal_id:
            # Match by pair and timestamp
            for pid, psig in pending_store.items():
                if psig.get("pair") == pair:
                    pts = psig.get("timestamp_eat", "")
                    if pts == ts_str or abs((datetime.strptime(pts, "%Y-%m-%d %H:%M:%S") - row_dt.replace(tzinfo=None)).total_seconds()) < 60:
                        signal_id = pid
                        matched_pending = psig
                        row["signal_id"] = signal_id
                        updated_rows_needed = True
                        break
        else:
            matched_pending = pending_store.get(signal_id)

        # Extract entry, stop_loss, take_profit
        try:
            entry = float(row.get("entry_price") or (matched_pending.get("entry") if matched_pending else 0.0))
            sl = float(row.get("stop_loss") or (matched_pending.get("stop_loss") if matched_pending else 0.0))
            tp = float(row.get("take_profit") or (matched_pending.get("take_profit") if matched_pending else 0.0))
        except (ValueError, TypeError):
            continue

        if entry <= 0 or sl <= 0 or tp <= 0:
            continue

        # Obtain 15m candles covering from signal time to now
        df_15m = None
        if candles_by_pair and pair in candles_by_pair:
            df_15m = candles_by_pair[pair]
        elif data_fetcher and config.TWELVE_DATA_API_KEY:
            try:
                # Calculate required output size
                elapsed_hours = (now_eat - row_dt).total_seconds() / 3600.0
                req_candles = max(int(elapsed_hours * 4) + 10, 30)
                df_15m = data_fetcher.fetch_candles(pair, "15min", min(req_candles, 200))
            except Exception as exc:
                print(f"  [outcome_tracker] Warning: Could not fetch candles for {pair}: {exc}")
                continue

        if df_15m is None or df_15m.empty:
            continue

        # Evaluate candles chronologically
        hit_outcome = None
        hit_price = None
        hit_time_str = now_eat.strftime("%Y-%m-%d %H:%M:%S")

        # Select candles strictly after signal creation time (row_dt in Africa/Addis_Ababa)
        for candle_ts, candle in df_15m.iterrows():
            # Convert candle timestamp to EAT
            try:
                c_dt = pd.to_datetime(candle_ts)
                if c_dt.tzinfo is None:
                    c_dt_eat = pytz.utc.localize(c_dt).astimezone(eat_tz)
                else:
                    c_dt_eat = c_dt.astimezone(eat_tz)
            except Exception:
                continue

            # Skip candles at or before the signal creation timestamp
            if c_dt_eat <= row_dt:
                continue

            c_high = float(candle.get("high", 0.0))
            c_low = float(candle.get("low", 0.0))

            if decision == "ENTER_LONG":
                if c_high >= tp:
                    hit_outcome = "WIN"
                    hit_price = tp
                    hit_time_str = c_dt_eat.strftime("%Y-%m-%d %H:%M:%S")
                    break
                elif c_low <= sl:
                    hit_outcome = "LOSS"
                    hit_price = sl
                    hit_time_str = c_dt_eat.strftime("%Y-%m-%d %H:%M:%S")
                    break
            elif decision == "ENTER_SHORT":
                if c_low <= tp:
                    hit_outcome = "WIN"
                    hit_price = tp
                    hit_time_str = c_dt_eat.strftime("%Y-%m-%d %H:%M:%S")
                    break
                elif c_high >= sl:
                    hit_outcome = "LOSS"
                    hit_price = sl
                    hit_time_str = c_dt_eat.strftime("%Y-%m-%d %H:%M:%S")
                    break

        if not hit_outcome:
            continue

        # Determine if this was an executed trade or a shadow (missed/unentered) signal
        is_entered = outcome in ("EXECUTED", "MANUAL") or (
            matched_pending and matched_pending.get("status") in ("EXECUTED", "MANUAL")
        )
        final_outcome = hit_outcome if is_entered else f"SHADOW_{hit_outcome}"

        # Risk & PnL calculation
        try:
            risk_usd = float(row.get("risk_usd") or (matched_pending.get("risk_usd") if matched_pending else 10.0))
        except (ValueError, TypeError):
            risk_usd = 10.0

        try:
            rr = float(row.get("risk_reward") or (matched_pending.get("risk_reward") if matched_pending else 2.0))
        except (ValueError, TypeError):
            rr = 2.0

        if hit_outcome == "WIN":
            pnl_usd = round(risk_usd * rr, 2)
        else:
            pnl_usd = -round(risk_usd, 2)

        pnl_sign = "+" if pnl_usd > 0 else ""
        outcome_tag = "LIVE/DEMO EXECUTION" if is_entered else "SHADOW SIGNAL (MISSED/UNENTERED)"
        emoji = "🟢" if hit_outcome == "WIN" else "🔴"

        # Update row in memory
        row["outcome"] = final_outcome
        row["exit_price"] = str(hit_price)
        row["exit_time"] = hit_time_str
        row["pnl_usd"] = str(pnl_usd)
        row["notes"] = f"Automated candle detection ({outcome_tag}): {hit_outcome} hit"
        updated_rows_needed = True

        # 1. Update pattern_history.json
        if signal_id:
            try:
                pattern_stats.update_outcome(
                    signal_id=signal_id,
                    outcome=final_outcome,
                    pnl_usd=pnl_usd,
                )
            except Exception as exc:
                print(f"  [outcome_tracker] Warning: pattern_stats update error: {exc}")

            # 2. Update pending_signals.json
            notifier.update_signal_status(signal_id, f"CLOSED_{final_outcome}")

            # 3. Close in active_trades.json if present
            try:
                trades = post_mortem.load_active_trades()
                if signal_id in trades:
                    trades[signal_id]["status"] = "CLOSED"
                    trades[signal_id]["exit_price"] = hit_price
                    trades[signal_id]["closed_eat"] = hit_time_str
                    trades[signal_id]["pnl_usd"] = pnl_usd
                    trades[signal_id]["outcome"] = final_outcome
                    post_mortem.save_active_trades(trades)
            except Exception:
                pass

        # 4. Deliver Telegram summary card
        card = (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📡 AUTOMATED OUTCOME DETECTED\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 Pair     : {pair}\n"
            f"📊 Type     : {outcome_tag}\n"
            f"🎯 Signal   : {decision}\n"
            f"🏁 Result   : {emoji} {final_outcome} ({pnl_sign}${pnl_usd:.2f})\n"
            f"💵 Levels   : Entry {entry} → Exit {hit_price}\n"
            f"🛑 SL / TP  : {sl} / {tp}\n"
            f"⏱  Time     : {hit_time_str} EAT\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🧠 Signal fingerprint integrated into Winning DNA pattern engine."
        )
        notifier.send_outcome_update(card)
        print(f"  [outcome_tracker] {pair} signal {signal_id[:8]}… resolved as {final_outcome} (${pnl_usd})")

        # 5. If it was an entered trade, trigger post_mortem
        if is_entered:
            try:
                fake_trade = {
                    "signal_id": signal_id,
                    "pair": pair,
                    "entry": entry,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "direction": decision,
                }
                analysis = post_mortem.analyze_trade_with_ai(fake_trade, hit_price, hit_outcome, df_15m)
                pm_report = {
                    "signal_id": signal_id,
                    "pair": pair,
                    "outcome": hit_outcome,
                    "entry": entry,
                    "exit_price": hit_price,
                    "pnl_usd": pnl_usd,
                    "duration": hit_time_str,
                    "root_cause": analysis["root_cause"],
                    "rule_followed": analysis["rule_followed"],
                    "actionable_lesson": analysis["actionable_lesson"],
                    "timestamp_eat": hit_time_str,
                }
                post_mortem.save_lesson({
                    "timestamp_eat": hit_time_str,
                    "pair": pair,
                    "outcome": hit_outcome,
                    "actionable_lesson": analysis["actionable_lesson"],
                    "root_cause": analysis["root_cause"],
                })
                notifier.send_post_mortem(pm_report)
            except Exception as exc:
                print(f"  [outcome_tracker] Post-mortem error: {exc}")

        resolved_reports.append({
            "signal_id": signal_id,
            "pair": pair,
            "decision": decision,
            "outcome": final_outcome,
            "entry": entry,
            "exit_price": hit_price,
            "pnl_usd": pnl_usd,
            "hit_time": hit_time_str,
        })

    # Save updated CSV if any outcomes or signal_ids were resolved
    if updated_rows_needed:
        try:
            with open(journal.LOG_FILE, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=journal._CSV_HEADERS, extrasaction="ignore")
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print(f"  [outcome_tracker] Updated {len(resolved_reports)} outcome(s) in trades_log.csv.")
        except Exception as exc:
            print(f"  [outcome_tracker] Error writing updated trades_log.csv: {exc}")

    return resolved_reports
