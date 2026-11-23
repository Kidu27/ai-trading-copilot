"""
config.py — Centralised configuration for AI Forex Trading Co-Pilot.

All environment variables are loaded here. Every other module imports
from this file so there is a single source of truth.
"""

import os
from dotenv import load_dotenv

# Load .env file (silently OK if it doesn't exist — env vars may be set externally)
load_dotenv()

# ── API Keys ───────────────────────────────────────────────────────────────────
TWELVE_DATA_API_KEY: str = os.getenv("TWELVE_DATA_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ── Telegram (optional alerts) ─────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Account & Risk ─────────────────────────────────────────────────────────────
ACCOUNT_BALANCE: float = float(os.getenv("ACCOUNT_BALANCE", "500"))
MAX_RISK_PERCENT: float = float(os.getenv("MAX_RISK_PERCENT", "0.01"))

# ── Watchlist ──────────────────────────────────────────────────────────────────
WATCHLIST: list[str] = ["EUR/USD", "GBP/USD"]

# ── Timezone & Session Windows ─────────────────────────────────────────────────
TIMEZONE: str = "Africa/Addis_Ababa"          # EAT — UTC+3
LONDON_SESSION_START: str = "10:30"            # 07:30 UTC = 10:30 EAT
LONDON_SESSION_END: str   = "19:30"            # 16:30 UTC = 19:30 EAT

# ── Trade Quality Thresholds ───────────────────────────────────────────────────
MIN_RISK_REWARD: float = 2.0        # Minimum acceptable R:R ratio
MIN_CONFLUENCE_SCORE: int = 80      # AI auditor must score >= 80 to signal trade

# ── Safety Circuit Breaker ─────────────────────────────────────────────────────
MAX_DAILY_LOSSES: int = 2           # Stop all alerts after N losing trades today

# ── Demo / Live Mode ───────────────────────────────────────────────────────────
DEMO_MODE: bool = True              # True = paper-trading / simulation only

# ── Feature Flags ──────────────────────────────────────────────────────────────
USE_V2_INDICATORS: bool = True      # Enable EMA 50/200, Fibonacci Golden Zone, ATR

# ── AI Model ──────────────────────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-4-hydrogen"

# ── Twelve Data ───────────────────────────────────────────────────────────────
TD_BASE_URL: str = "https://api.twelvedata.com"
TD_1H_CANDLES: int = 300            # 300 candles — enough history for EMA 200 to converge
TD_15M_CANDLES: int = 50            # How many 15m candles to fetch
