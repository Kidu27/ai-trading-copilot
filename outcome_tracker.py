"""
outcome_tracker.py — Automated Candle Outcome Tracker for Open & Missed Signals

Scans trades_log.csv (and active_trades.json / pending_signals.json) for any OPEN,
EXECUTED, MANUAL, or SHADOW_OPEN signals from the last 48 hours that do not yet
have a recorded exit outcome.

Using 15m candle highs and lows:
  - Detects if price touched take_profit (WIN) or stop_loss (LOSS).
  - For unentered / declined / skipped setups:
      Resolves outcome as SHADOW_WIN / SHADOW_LOSS, updates pattern_history.json,
      and records loss lessons in lessons_learned.json.
  - For active EXECUTED / MANUAL MT5 trades:
      Resolves outcome to WIN / LOSS, updates trades_log.csv, removes from active_trades.json,
      updates pattern_history.json, triggers post_mortem lesson on loss, and sends an
      immediate Telegram completion alert:
        Win:  🎉 MANUAL TRADE FINISHED — WIN | {pair} ({direction}) | Exit: {exit_price} | PnL: +${pnl_usd:.2f} | AI memory updated.
        Loss: 🛑 MANUAL TRADE FINISHED — LOSS | {pair} ({direction}) | Exit: {exit_price} | PnL: -${abs(pnl_usd):.2f} | Loss lesson saved to AI memory.
"""

import csv
import json
import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytz

import config
import github_sync
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
    Scan trades_log.csv and active_trades.json for signals in the last 48h that have not yet reached an exit outcome.
    Evaluates historical 15m candle highs/lows to detect TP or SL touches.

    Args:
        candles_by_pair: Optional dict of pair -> DataFrame (used for mock tests or custom candles).

    Returns:
        List of resolved outcome report dicts.
    """
    eat_tz = pytz.timezone(config.TIMEZONE)
    now_eat = datetime.now(tz=eat_tz)
    cutoff_time = now_eat - timedelta(hours=48)

    # 1. Load active_trades.json
    active_trades = post_mortem.load_active_trades()
    active_trades_modified = False

    # 2. Load pending_signals.json
    pending_store = notifier._load_pending_signals()

    # 3. Load trades_log.csv
    rows: list[dict[str, Any]] = []
    if os.path.exists(journal.LOG_FILE):
        try:
            with open(journal.LOG_FILE, "r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
        except Exception as exc:
            print(f"  [outcome_tracker] Error reading trades_log.csv: {exc}")
            rows = []

    resolved_reports: list[dict[str, Any]] = []
    updated_rows_needed = False

    # Ensure any open trade in active_trades.json is represented in rows
    logged_signal_ids = {r.get("signal_id", "").strip() for r in rows if r.get("signal_id")}
    for sid, act in list(active_trades.items()):
        if act.get("status") == "CLOSED":
            continue
        if sid not in logged_signal_ids:
            new_row = {
                "timestamp_eat": act.get("timestamp_eat") or act.get("opened_eat") or now_eat.strftime("%Y-%m-%d %H:%M:%S"),
                "signal_id": sid,
                "pair": act.get("pair", ""),
                "decision": act.get("direction", "ENTER_LONG"),
                "confluence_score": act.get("confluence_score", 0),
                "entry_price": str(act.get("entry_price") or act.get("entry") or ""),
                "stop_loss": str(act.get("stop_loss") or ""),
                "take_profit": str(act.get("take_profit") or ""),
                "standard_lots": str(act.get("standard_lots") or 0.01),
                "risk_reward": str(act.get("risk_reward") or 2.0),
                "risk_usd": str(act.get("risk_usd") or 10.0),
                "outcome": "EXECUTED",
                "exit_price": "",
                "exit_time": "",
                "pnl_usd": "",
                "notes": "Manually entered on MT5 Mobile — auto-tracking SL/TP",
            }
            rows.append(new_row)
            logged_signal_ids.add(sid)
            updated_rows_needed = True

    # 4. Evaluate each trade in rows
    for idx, row in enumerate(rows):
        decision = row.get("decision", "").strip().upper()
        outcome = row.get("outcome", "").strip().upper()
        exit_price = row.get("exit_price", "").strip()
        pair = row.get("pair", "").strip()
        ts_str = row.get("timestamp_eat", "").strip()
        notes = row.get("notes", "").strip()
        signal_id = row.get("signal_id", "").strip()

        # Skip already completed trades that have an exit price recorded
        if outcome in ("WIN", "LOSS", "SHADOW_WIN", "SHADOW_LOSS") and exit_price:
            continue

        # Determine candidate eligibility:
        # 1. Shadow open trades (SHADOW_OPEN or SHADOW_LONG/SHADOW_SHORT)
        # 2. Executed trades (EXECUTED or MANUAL, or in active_trades.json)
        # 3. Actionable signals (ENTER_LONG/ENTER_SHORT without exit_price)
        is_shadow = (outcome == "SHADOW_OPEN" or decision.startswith("SHADOW_"))
        is_executed_or_active = (
            outcome in ("EXECUTED", "MANUAL")
            or signal_id in active_trades
            or "MANUAL" in notes.upper()
        )
        is_actionable = decision in ("ENTER_LONG", "ENTER_SHORT", "SHADOW_LONG", "SHADOW_SHORT")

        if not (is_shadow or is_executed_or_active or is_actionable):
            continue

        # Link signal_id if blank in CSV
        matched_pending = None
        if signal_id:
            matched_pending = pending_store.get(signal_id)
        else:
            for pid, psig in pending_store.items():
                if psig.get("pair") == pair:
                    pts = psig.get("timestamp_eat", "")
                    if pts == ts_str:
                        signal_id = pid
                        matched_pending = psig
                        row["signal_id"] = signal_id
                        updated_rows_needed = True
                        break

        act_trade = active_trades.get(signal_id, {})

        # Extract entry, stop_loss, take_profit
        try:
            entry = float(
                row.get("entry_price")
                or act_trade.get("entry_price")
                or act_trade.get("entry")
                or (matched_pending.get("entry") if matched_pending else 0.0)
                or 0.0
            )
            sl = float(
                row.get("stop_loss")
                or act_trade.get("stop_loss")
                or (matched_pending.get("stop_loss") if matched_pending else 0.0)
                or 0.0
            )
            tp = float(
                row.get("take_profit")
                or act_trade.get("take_profit")
                or (matched_pending.get("take_profit") if matched_pending else 0.0)
                or 0.0
            )
        except (ValueError, TypeError):
            continue

        if entry <= 0 or sl <= 0 or tp <= 0:
            continue

        # Backfill missing levels into row if empty
        if not row.get("entry_price") and entry > 0:
            row["entry_price"] = str(entry)
            updated_rows_needed = True
        if not row.get("stop_loss") and sl > 0:
            row["stop_loss"] = str(sl)
            updated_rows_needed = True
        if not row.get("take_profit") and tp > 0:
            row["take_profit"] = str(tp)
            updated_rows_needed = True

        # Direction resolution (Requirement 3: Backfill Safety for Open Manual Rows)
        # If decision was logged as NO_TRADE or empty, infer LONG/SHORT from stop_loss < entry_price
        if decision not in ("ENTER_LONG", "ENTER_SHORT", "SHADOW_LONG", "SHADOW_SHORT"):
            raw_sig_dir = act_trade.get("direction") or (matched_pending.get("direction") if matched_pending else "")
            if "LONG" in str(raw_sig_dir).upper() and "SHORT" not in str(raw_sig_dir).upper():
                decision = "SHADOW_LONG" if is_shadow else "ENTER_LONG"
            elif "SHORT" in str(raw_sig_dir).upper():
                decision = "SHADOW_SHORT" if is_shadow else "ENTER_SHORT"
            elif sl < entry:
                decision = "SHADOW_LONG" if is_shadow else "ENTER_LONG"
            else:
                decision = "SHADOW_SHORT" if is_shadow else "ENTER_SHORT"
            row["decision"] = decision
            updated_rows_needed = True

        # Parse timestamp and verify window
        row_dt = None
        if ts_str:
            try:
                parsed_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                row_dt = eat_tz.localize(parsed_dt)
            except Exception:
                row_dt = None

        if row_dt and row_dt < cutoff_time and signal_id not in active_trades:
            continue
        if not row_dt:
            row_dt = now_eat - timedelta(hours=12)

        # Fetch 15m candles
        df_15m = None
        if candles_by_pair and pair in candles_by_pair:
            df_15m = candles_by_pair[pair]
        elif data_fetcher and config.TWELVE_DATA_API_KEY:
            try:
                elapsed_hours = max((now_eat - row_dt).total_seconds() / 3600.0, 1.0)
                req_candles = max(int(elapsed_hours * 4) + 10, 30)
                df_15m = data_fetcher.fetch_candles(pair, "15min", min(req_candles, 200))
            except Exception as exc:
                print(f"  [outcome_tracker] Warning: Could not fetch candles for {pair}: {exc}")
                continue

        if df_15m is None or df_15m.empty:
            continue

        # Evaluate 15m candles chronologically
        hit_outcome = None
        hit_price = None
        hit_time_str = now_eat.strftime("%Y-%m-%d %H:%M:%S")

        for candle_ts, candle in df_15m.iterrows():
            try:
                c_dt = pd.to_datetime(candle_ts)
                if c_dt.tzinfo is None:
                    c_dt_eat = pytz.utc.localize(c_dt).astimezone(eat_tz)
                else:
                    c_dt_eat = c_dt.astimezone(eat_tz)
            except Exception:
                continue

            if c_dt_eat <= row_dt:
                continue

            c_high = float(candle.get("high", 0.0))
            c_low = float(candle.get("low", 0.0))

            if decision in ("ENTER_LONG", "SHADOW_LONG"):
                # For LONG: if candle low <= stop_loss -> LOSS; elif candle high >= take_profit -> WIN
                # If both hit on same candle, default conservatively to LOSS
                if c_low <= sl:
                    hit_outcome = "LOSS"
                    hit_price = sl
                    hit_time_str = c_dt_eat.strftime("%Y-%m-%d %H:%M:%S")
                    break
                elif c_high >= tp:
                    hit_outcome = "WIN"
                    hit_price = tp
                    hit_time_str = c_dt_eat.strftime("%Y-%m-%d %H:%M:%S")
                    break
            elif decision in ("ENTER_SHORT", "SHADOW_SHORT"):
                # For SHORT: if candle high >= stop_loss -> LOSS; elif candle low <= take_profit -> WIN
                # If both hit on same candle, default conservatively to LOSS
                if c_high >= sl:
                    hit_outcome = "LOSS"
                    hit_price = sl
                    hit_time_str = c_dt_eat.strftime("%Y-%m-%d %H:%M:%S")
                    break
                elif c_low <= tp:
                    hit_outcome = "WIN"
                    hit_price = tp
                    hit_time_str = c_dt_eat.strftime("%Y-%m-%d %H:%M:%S")
                    break

        if not hit_outcome:
            continue

        # Risk & PnL calculation
        try:
            risk_usd = float(
                row.get("risk_usd")
                or act_trade.get("risk_usd")
                or (matched_pending.get("risk_usd") if matched_pending else 10.0)
                or 10.0
            )
        except (ValueError, TypeError):
            risk_usd = 10.0

        try:
            rr = float(
                row.get("risk_reward")
                or act_trade.get("risk_reward")
                or (matched_pending.get("risk_reward") if matched_pending else 2.0)
                or 2.0
            )
        except (ValueError, TypeError):
            rr = 2.0

        if hit_outcome == "WIN":
            pnl_usd = round(risk_usd * rr, 2)
        else:
            pnl_usd = -round(risk_usd, 2)

        is_manual = (not is_shadow) and (
            outcome in ("EXECUTED", "MANUAL")
            or act_trade.get("source") == "MANUAL_MT5"
            or "MANUAL" in notes.upper()
            or signal_id in active_trades
        )

        if is_manual:
            final_outcome = hit_outcome
            row["outcome"] = final_outcome
            row["exit_price"] = str(hit_price)
            row["exit_time"] = hit_time_str
            row["pnl_usd"] = str(pnl_usd)
            row["notes"] = f"Automated candle detection (MANUAL MT5): {hit_outcome} hit"
            updated_rows_needed = True

            # 1. Remove from active_trades.json
            if signal_id in active_trades:
                del active_trades[signal_id]
                active_trades_modified = True

            # 2. Feed into pattern_stats.py
            if signal_id:
                try:
                    hist = pattern_stats.load_pattern_history()
                    if signal_id in hist:
                        pattern_stats.update_outcome(signal_id=signal_id, outcome=final_outcome, pnl_usd=pnl_usd)
                    else:
                        hist[signal_id] = {
                            "signal_id": signal_id,
                            "timestamp_eat": hit_time_str,
                            "pair": pair,
                            "direction": decision,
                            "outcome": final_outcome,
                            "pnl_usd": pnl_usd,
                            "is_shadow_candidate": False,
                        }
                        pattern_stats.save_pattern_history(hist)
                except Exception as exc:
                    print(f"  [outcome_tracker] Warning: pattern_stats manual update error: {exc}")

                notifier.update_signal_status(signal_id, f"CLOSED_{final_outcome}")

            # 3. Trigger loss-lesson recorder (lessons_learned.json) if it was a LOSS
            if hit_outcome == "LOSS":
                try:
                    fake_trade = {
                        "signal_id": signal_id,
                        "pair": pair,
                        "entry": entry,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "direction": decision,
                    }
                    analysis = post_mortem.analyze_trade_with_ai(fake_trade, hit_price, "LOSS", df_15m)
                    post_mortem.save_lesson({
                        "timestamp_eat": hit_time_str,
                        "pair": pair,
                        "outcome": "LOSS",
                        "actionable_lesson": analysis["actionable_lesson"],
                        "root_cause": analysis["root_cause"],
                    })
                except Exception as exc:
                    print(f"  [outcome_tracker] Error saving manual loss lesson: {exc}")

            # 4. Send Telegram Win/Loss notification immediately via notifier.py
            if hit_outcome == "WIN":
                alert_msg = f"🎉 MANUAL TRADE FINISHED — WIN | {pair} ({decision}) | Exit: {hit_price} | PnL: +${pnl_usd:.2f} | AI memory updated."
            else:
                alert_msg = f"🛑 MANUAL TRADE FINISHED — LOSS | {pair} ({decision}) | Exit: {hit_price} | PnL: -${abs(pnl_usd):.2f} | Loss lesson saved to AI memory."

            try:
                notifier.send_manual_outcome_alert(alert_msg)
            except Exception as exc:
                print(f"  [outcome_tracker] Error sending Telegram alert: {exc}")

        else:
            # Shadow trade
            final_outcome = f"SHADOW_{hit_outcome}"
            row["outcome"] = final_outcome
            row["exit_price"] = str(hit_price)
            row["exit_time"] = hit_time_str
            row["pnl_usd"] = str(pnl_usd)
            row["notes"] = f"Automated candle detection (SHADOW): {hit_outcome} hit"
            updated_rows_needed = True

            # 1. Update pattern_history.json
            if signal_id:
                try:
                    hist = pattern_stats.load_pattern_history()
                    if signal_id in hist:
                        pattern_stats.update_outcome(signal_id=signal_id, outcome=final_outcome, pnl_usd=pnl_usd)
                    else:
                        hist[signal_id] = {
                            "signal_id": signal_id,
                            "timestamp_eat": hit_time_str,
                            "pair": pair,
                            "direction": decision,
                            "outcome": final_outcome,
                            "pnl_usd": pnl_usd,
                            "is_shadow_candidate": True,
                        }
                        pattern_stats.save_pattern_history(hist)
                except Exception as exc:
                    print(f"  [outcome_tracker] Warning: pattern_stats shadow update error: {exc}")

                notifier.update_signal_status(signal_id, f"CLOSED_{final_outcome}")

            # 2. Update lessons_learned.json if SHADOW_LOSS
            if hit_outcome == "LOSS":
                try:
                    fake_trade = {
                        "signal_id": signal_id,
                        "pair": pair,
                        "entry": entry,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "direction": decision,
                    }
                    analysis = post_mortem.analyze_trade_with_ai(fake_trade, hit_price, "LOSS", df_15m)
                    post_mortem.save_lesson({
                        "timestamp_eat": hit_time_str,
                        "pair": pair,
                        "outcome": "SHADOW_LOSS",
                        "actionable_lesson": analysis["actionable_lesson"],
                        "root_cause": analysis["root_cause"],
                    })
                except Exception as exc:
                    print(f"  [outcome_tracker] Error saving shadow loss lesson: {exc}")

        print(f"  [outcome_tracker] {pair} signal {signal_id[:8]}… resolved as {final_outcome} (${pnl_usd:.2f})")
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

    # Save active_trades.json if any trades were removed
    if active_trades_modified:
        try:
            post_mortem.save_active_trades(active_trades)
            print("  [outcome_tracker] active_trades.json updated.")
        except Exception as exc:
            print(f"  [outcome_tracker] Error saving active_trades.json: {exc}")

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

    # Immediately push to GitHub memory branch if any trades were resolved
    if resolved_reports:
        try:
            github_sync.push_memory_files(f"Outcome tracker: {len(resolved_reports)} trade(s) resolved")
        except Exception as exc:
            print(f"  [outcome_tracker] Warning: github_sync push error: {exc}")

    return resolved_reports


if __name__ == "__main__":
    outcomes = check_open_signal_outcomes()
    print(f"Outcome tracker completed: {len(outcomes)} trade(s) resolved.")
