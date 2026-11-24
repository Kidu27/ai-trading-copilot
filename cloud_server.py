"""
cloud_server.py — FastAPI Cloud Runner & Instant Telegram Webhook for Render.com

Zero-Cost 24/5 Deployment Architecture:
  1. Render.com Web Service (Free Tier, Python 3.11, Start Command: uvicorn cloud_server:app --host 0.0.0.0 --port $PORT)
  2. cron-job.org (Free Tier, 2 recurring jobs with no credit card):
       - Job A: Every 10 mins (Mon-Fri 07:00-16:45 UTC): GET /health (prevents Render 15-min idle sleep)
       - Job B: Every 15 mins (Mon-Fri 07:30-16:30 UTC): POST /scan?token=SECRET (launches market scan in background)
       - Job C: Daily at 16:35 UTC (Mon-Fri): POST /daily-backup?token=SECRET (backs up logs & memory to Telegram)
  3. Telegram Webhook:
       - Instant button feedback on phone taps ([Accept], [Manual], [WIN], [LOSS])
       - Automatically registers setWebhook on Render startup via RENDER_EXTERNAL_URL
"""

import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import pytz
import requests
from fastapi import FastAPI, Header, HTTPException, Query, Request
import uvicorn

import config
import github_sync
import journal
import main
import notifier
import outcome_tracker
import pattern_stats
import post_mortem
import telegram_listener


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: Automatic Telegram Webhook Registration & Memory Restoration
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On Render startup:
      1. Pull persistent memory state from GitHub 'memory' branch BEFORE any scans or webhooks.
      2. Register the Telegram webhook if RENDER_EXTERNAL_URL is set.
    """
    # 1. Restore memory state from GitHub
    try:
        github_sync.pull_memory_files()
    except Exception as exc:
        print(f"  [cloud_server] ⚠️ Error pulling memory files from GitHub: {exc}")

    # 2. Register Telegram webhook
    render_url = os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
    if render_url and config.TELEGRAM_BOT_TOKEN:
        webhook_url = f"{render_url}/telegram-webhook"
        set_webhook_api = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/setWebhook"
        try:
            resp = requests.post(
                set_webhook_api,
                json={
                    "url": webhook_url,
                    "allowed_updates": ["callback_query", "message"],
                    "drop_pending_updates": False,
                },
                timeout=15,
            )
            result = resp.json()
            if result.get("ok"):
                print(f"  [cloud_server] ✅ Telegram webhook registered → {webhook_url}")
            else:
                print(f"  [cloud_server] ⚠️ Telegram setWebhook response: {result}")
        except Exception as exc:
            print(f"  [cloud_server] ⚠️ Could not register Telegram webhook: {exc}")
    else:
        print("  [cloud_server] ℹ️ RENDER_EXTERNAL_URL not set — running in local/polling mode.")

    yield
    print("  [cloud_server] 🛑 Shutting down cloud runner.")


app = FastAPI(
    title="AI Forex Co-Pilot — Cloud Runner",
    description="FastAPI cloud runner for 24/5 zero-cost market scanning and Telegram webhooks on Render.com",
    version="2.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Security Helper
# ─────────────────────────────────────────────────────────────────────────────

def _verify_cron_secret(
    token: str | None = None,
    authorization: str | None = None,
    x_cron_secret: str | None = None,
) -> bool:
    """Validate incoming cron requests against CRON_SECRET if configured."""
    expected = os.getenv("CRON_SECRET", "").strip()
    if not expected:
        return True  # If no secret is set, allow requests (e.g. testing)

    if token and token.strip() == expected:
        return True
    if x_cron_secret and x_cron_secret.strip() == expected:
        return True
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip() == expected:
            return True
        if authorization.strip() == expected:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Backup & Telegram Delivery Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _send_telegram_document(file_path: str, caption: str = "") -> bool:
    """Send a local file as an attachment to TELEGRAM_CHAT_ID."""
    if not os.path.exists(file_path) or not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendDocument"
    filename = os.path.basename(file_path)
    try:
        with open(file_path, "rb") as fh:
            files = {"document": (filename, fh)}
            data = {"chat_id": str(config.TELEGRAM_CHAT_ID), "caption": caption}
            resp = requests.post(url, data=data, files=files, timeout=30)
            return resp.status_code == 200
    except Exception as exc:
        print(f"  [cloud_server] Error sending {filename} to Telegram: {exc}")
        return False


def _send_telegram_text(text: str) -> bool:
    """Send formatted text to TELEGRAM_CHAT_ID."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": str(config.TELEGRAM_CHAT_ID),
        "text": f"<pre>{text}</pre>",
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as exc:
        print(f"  [cloud_server] Error sending text to Telegram: {exc}")
        return False


def _build_daily_report() -> str:
    """Generate the End-of-Session Readiness & Win-Signature Report."""
    eat_tz = pytz.timezone(config.TIMEZONE)
    now_eat = datetime.now(tz=eat_tz).strftime("%Y-%m-%d %H:%M:%S")

    try:
        win_dna = pattern_stats.get_win_signature_summary()
    except Exception:
        win_dna = "Empirical pattern synthesis underway."

    try:
        readiness = pattern_stats.get_readiness_metrics()
        readiness_text = readiness.get("summary_text", "")
        sample_size = readiness.get("sample_size", 0)
        win_rate = readiness.get("win_rate_pct", 0.0)
    except Exception:
        readiness_text = "Readiness: Gathering closed trade statistics."
        sample_size = 0
        win_rate = 0.0

    mode = "DEMO_DISCOVERY" if getattr(config, "DEMO_DISCOVERY_MODE", False) else ("DEMO" if config.DEMO_MODE else "LIVE")

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📦 END-OF-SESSION READINESS & BACKUP",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📅 Closed EAT : {now_eat}",
        f"⚙️ System Mode: {mode}",
        f"📊 Closed Setups: {sample_size}",
        f"🎯 Win Rate   : {win_rate:.1f}%",
        "",
        "── 30-TRADE READINESS METER ─",
        readiness_text,
        "",
        "── WINNING DNA SIGNATURE ────",
        win_dna,
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "💾 State memory files (CSV/JSON) attached below for cloud durability.",
    ]
    return "\n".join(lines)


def _execute_backup_task() -> None:
    """Background worker to send report text and all state files to Telegram."""
    report = _build_daily_report()
    _send_telegram_text(report)

    files_to_send = [
        (journal.LOG_FILE, "📊 Trades Log (CSV)"),
        (post_mortem.LESSONS_FILE, "🧠 Lessons Learned Memory (JSON)"),
        (pattern_stats.PATTERN_HISTORY_FILE, "🧬 Pattern History & Indicator Fingerprints (JSON)"),
        (notifier.PENDING_SIGNALS_FILE, "📋 Pending Signals Store (JSON)"),
    ]
    for fpath, caption in files_to_send:
        _send_telegram_document(fpath, caption=caption)
    print("  [cloud_server] ✅ End-of-session backup completed and sent to Telegram.")

    # Commit and push state files to GitHub 'memory' branch
    try:
        github_sync.push_memory_files("Daily end-of-session backup")
    except Exception as exc:
        print(f"  [cloud_server] ⚠️ github_sync.push_memory_files error: {exc}")


def _execute_scan_task() -> None:
    """Background worker running market scan and outcome tracking."""
    print("\n  [cloud_server] 🚀 Triggering market scan cycle...")
    try:
        main.run_scan()
    except Exception as exc:
        print(f"  [cloud_server] ❌ Error in run_scan: {exc}")

    try:
        outcome_tracker.check_open_signal_outcomes()
    except Exception as exc:
        print(f"  [cloud_server] ❌ Error in outcome_tracker: {exc}")

    # Commit and push updated trades_log.csv and pattern_history.json to GitHub 'memory' branch
    try:
        github_sync.push_memory_files("Scan cycle completed & outcomes tracked")
    except Exception as exc:
        print(f"  [cloud_server] ⚠️ github_sync.push_memory_files error: {exc}")

    print("  [cloud_server] ✅ Market scan cycle completed.")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/health")
def health() -> dict[str, Any]:
    """
    Instant health check (<50ms).
    Invoked by cron-job.org every 10 minutes to prevent Render 15-minute cold sleep.
    """
    mode = "DEMO_DISCOVERY" if getattr(config, "DEMO_DISCOVERY_MODE", False) else ("DEMO" if config.DEMO_MODE else "LIVE")
    return {
        "status": "ok",
        "mode": mode,
        "app": "ai-trading-copilot",
    }


@app.get("/scan")
@app.post("/scan")
def scan_endpoint(
    token: str | None = Query(None),
    authorization: str | None = Header(None, alias="Authorization"),
    x_cron_secret: str | None = Header(None, alias="X-Cron-Secret"),
) -> dict[str, Any]:
    """
    Launch 7-pair scan + outcome tracker in a background daemon thread.
    Returns immediately so cron-job.org never hits its 30s timeout.
    """
    if not _verify_cron_secret(token=token, authorization=authorization, x_cron_secret=x_cron_secret):
        raise HTTPException(status_code=403, detail="Invalid CRON_SECRET token.")

    threading.Thread(target=_execute_scan_task, daemon=True).start()
    return {
        "status": "scan_started",
        "message": "Market scan and outcome tracker launched in background thread.",
    }


@app.get("/daily-backup")
@app.post("/daily-backup")
def daily_backup_endpoint(
    token: str | None = Query(None),
    authorization: str | None = Header(None, alias="Authorization"),
    x_cron_secret: str | None = Header(None, alias="X-Cron-Secret"),
) -> dict[str, Any]:
    """
    Send daily win-signature report and memory files to Telegram.
    Returns immediately while dispatch occurs in the background.
    """
    if not _verify_cron_secret(token=token, authorization=authorization, x_cron_secret=x_cron_secret):
        raise HTTPException(status_code=403, detail="Invalid CRON_SECRET token.")

    threading.Thread(target=_execute_backup_task, daemon=True).start()
    return {
        "status": "backup_started",
        "message": "End-of-session report and backup files are being dispatched to Telegram.",
    }


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request) -> dict[str, str]:
    """
    Receive instant button interactions from Telegram without long-polling.
    Routes callback queries directly into telegram_listener._process_callback_query().
    """
    try:
        update: dict[str, Any] = await request.json()
    except Exception:
        return {"status": "ignored", "reason": "invalid_json"}

    # Process button taps
    if "callback_query" in update:
        cq = update["callback_query"]
        try:
            telegram_listener._process_callback_query(cq)

            # Persist state to GitHub 'memory' branch after user button interaction
            data_str = cq.get("data", "")
            action = data_str.split(":", 1)[0] if ":" in data_str else "action"
            threading.Thread(
                target=github_sync.push_memory_files,
                args=(f"User Telegram button: {action}",),
                daemon=True,
            ).start()
        except Exception as exc:
            print(f"  [cloud_server] Error processing callback_query: {exc}")

    # Optional handling of user text commands
    elif "message" in update:
        msg = update["message"]
        text = msg.get("text", "").strip()
        chat_id = msg.get("chat", {}).get("id")
        if str(chat_id) == str(config.TELEGRAM_CHAT_ID):
            if text in ("/status", "/health", "/ping"):
                mode = "DEMO_DISCOVERY" if getattr(config, "DEMO_DISCOVERY_MODE", False) else "DEMO"
                _send_telegram_text(f"🟢 Co-Pilot Cloud Server Active\nMode: {mode}\nHost: Render.com")
            elif text in ("/backup", "/export"):
                threading.Thread(target=_execute_backup_task, daemon=True).start()
            elif text in ("/scan", "/run"):
                threading.Thread(target=_execute_scan_task, daemon=True).start()
                _send_telegram_text("🚀 Scan triggered from Telegram command.")

    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("cloud_server:app", host="0.0.0.0", port=port, reload=False)
