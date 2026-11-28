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
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

# ── Telegram (optional alerts) ─────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Account & Risk ─────────────────────────────────────────────────────────────
ACCOUNT_BALANCE: float = float(os.getenv("ACCOUNT_BALANCE", "1000.0"))
MAX_RISK_PERCENT: float = float(os.getenv("MAX_RISK_PERCENT", "0.01"))

# ── Watchlist (7 Liquid Major & Cross Pairs) ──────────────────────────────────
# 7 pairs × 36 15m cycles/day × 2 credits = 504 credits/day (safely within 800/day limit)
PAIRS: list[str] = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "EUR/JPY",
    "GBP/JPY",
    "GBP/AUD",
    "GBP/NZD",
]
WATCHLIST: list[str] = PAIRS  # Alias for backward compatibility

# ── Timezone & Session Windows ─────────────────────────────────────────────────
TIMEZONE: str = "Africa/Addis_Ababa"          # EAT — UTC+3
LONDON_SESSION_START: str = "09:30"            # 06:30 UTC = 09:30 EAT
LONDON_SESSION_END: str   = "19:30"            # 16:30 UTC = 19:30 EAT
SCAN_INTERVAL_MINUTES: int = 15                # Scan every 15 minutes during session
PAIR_REQUEST_DELAY_SECONDS: int = 10           # 10s delay between pairs (stays under 8 req/min)

# ── Demo / Live Mode & Opportunity Discovery ───────────────────────────────────
DEMO_MODE: bool = True                         # True = paper-trading / simulation only
DEMO_DISCOVERY_MODE: bool = True               # Relax threshold in demo to study setups

# ── Trade Quality Thresholds ───────────────────────────────────────────────────
# In Demo Discovery Mode, relax thresholds to 65 confluence score and 1.4 R:R
# so Gemini acts as an active opportunity scout while hard SL and 1% risk remain strictly enforced.
def get_min_risk_reward() -> float:
    return 1.4 if (DEMO_MODE and DEMO_DISCOVERY_MODE) else 2.0

def get_min_confluence_score() -> int:
    return 65 if (DEMO_MODE and DEMO_DISCOVERY_MODE) else 80

def get_min_math_score() -> int:
    return 55 if (DEMO_MODE and DEMO_DISCOVERY_MODE) else 65

MIN_RISK_REWARD: float = get_min_risk_reward()
MIN_CONFLUENCE_SCORE: int = get_min_confluence_score()
MIN_MATH_SCORE: int = get_min_math_score()

# ── Feature Flags ──────────────────────────────────────────────────────────────
USE_V2_INDICATORS: bool = True      # Enable EMA 50/200, Fibonacci Golden Zone, ATR

# ── AI Model ──────────────────────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-3.5-flash-lite"

# ── Twelve Data ───────────────────────────────────────────────────────────────
TD_BASE_URL: str = "https://api.twelvedata.com"
TD_1H_CANDLES: int = 300            # 300 candles — enough history for EMA 200 to converge
TD_15M_CANDLES: int = 50            # How many 15m candles to fetch

# ── Signal Approval & Expiry ──────────────────────────────────────────────────
SIGNAL_EXPIRY_MINUTES: int = 15     # Telegram approval buttons expire after N minutes
NOTIFY_NO_TRADE: bool = os.getenv("NOTIFY_NO_TRADE", "false").lower() in ("true", "1")  # False = quiet mode (actionable only)
NOTIFY_MODE: str = os.getenv("NOTIFY_MODE", "TRADES_ONLY")     # Supported: "ALL", "GATE2_ONLY", "TRADES_ONLY"

# ── cTrader Open API (optional — cloud-native 1-tap order execution) ───────────
CTRADER_CLIENT_ID: str = os.getenv("CTRADER_CLIENT_ID", "").strip()
CTRADER_CLIENT_SECRET: str = os.getenv("CTRADER_CLIENT_SECRET", "").strip()
CTRADER_ACCESS_TOKEN: str = os.getenv("CTRADER_ACCESS_TOKEN", "").strip()
CTRADER_REFRESH_TOKEN: str = os.getenv("CTRADER_REFRESH_TOKEN", "").strip()
CTRADER_ACCOUNT_ID: str = os.getenv("CTRADER_ACCOUNT_ID", "").strip()

# ── Restored Compatibility Guardrails ──
MAX_DAILY_LOSSES = 3
PAIR_COOLDOWN_HOURS = 1
