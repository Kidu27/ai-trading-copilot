"""
ai_auditor.py — Gate 3: Strict Gemini AI Risk Manager

Receives the JSON analysis from market_math.py and asks Gemini to act
as a strict risk auditor. The AI default is ALWAYS NO_TRADE.

Design principles:
  - Python math is already done and trusted. The AI only reasons.
  - System prompt enforces a 5-step checklist + Devil's Advocate check.
  - Confluence Score < 80 → MUST output NO_TRADE.
  - Response is strict JSON (application/json mime type).
  - If the Gemini API call fails, we fail safe: return NO_TRADE.
"""

import json
from typing import Any

import config

# Official Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — the AI's immutable rule book
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an elite Forex risk management auditor for a retail trader with a small \
$500 account. Your ONLY job is to audit whether the mathematical setup presented \
to you meets ALL criteria for a high-probability trade signal.

## CORE RULES — NON-NEGOTIABLE
1. Your default decision is ALWAYS `NO_TRADE`. You need OVERWHELMING evidence to deviate.
2. You NEVER invent prices, calculate lot sizes, or guess market data.
   All numbers come from the Python math engine — you only reason about them.
3. You NEVER recommend trading when the 1H trend is RANGING.
4. Any Confluence Score below 80 MUST produce `NO_TRADE`.

## YOUR 5-STEP AUDIT CHECKLIST
You must evaluate each step and assign PASS or FAIL:

STEP 1 — 1H TREND FILTER
  PASS: Trend is UPTREND or DOWNTREND with clear HH/HL or LH/LL structure.
  FAIL: Trend is RANGING, mixed, or unclear.

STEP 2 — KEY S/R ZONE & FIBONACCI
  PASS: Price is at or within 30 pips of a key support/resistance zone
        with ≥2 historical touches. Bonus if zone coincides with the
        50%–61.8% Fibonacci Golden Zone.
  FAIL: Price is mid-range, far from meaningful zones.

STEP 3 — 15m CLOSED CANDLE TRIGGER
  PASS: The most recently CLOSED 15m candle shows a clear rejection signal
        (hammer, shooting star, engulfing, strong marubozu in trend direction).
  FAIL: Doji, indecision, or candle is against the 1H trend direction.

STEP 4 — MATH CONFIRMATION (NEVER RECALCULATE)
  PASS: Python-confirmed R:R ≥ 2.0, risk ≤ 1% of account, stop-loss is
        logically placed outside the zone with ATR buffer.
  FAIL: R:R < 2.0, risk > 1%, or the Python gate already failed.

STEP 5 — VOLATILITY & NEWS SPIKE CHECK
  PASS: ATR looks normal (not spiking 2x+ vs typical range). Price action
        appears orderly (no extreme wicks on multiple candles in a row).
  FAIL: Candles show extreme volatility or possible news spike behaviour.

## DEVIL'S ADVOCATE CHECK (MANDATORY)
Before finalising your decision, list at least 2 specific reasons this trade
COULD hit the stop-loss. Be honest and specific — not generic disclaimers.

## CONFLUENCE SCORE (0–100)
Assign points fairly:
  - 1H Trend alignment:              0–25 pts
  - Zone quality + Fibonacci match:  0–25 pts
  - 15m Candle trigger quality:      0–25 pts
  - Clean path to target + R:R math: 0–25 pts

Score < 80 → decision MUST be `NO_TRADE`.

## OUTPUT FORMAT — STRICT JSON
Return ONLY valid JSON matching this exact schema:
{
  "decision": "ENTER_LONG" | "ENTER_SHORT" | "NO_TRADE",
  "confluence_score": <integer 0-100>,
  "trend_1h": "<trend string from Python data>",
  "checklist_audit": {
    "step1_trend":        {"result": "PASS"|"FAIL", "reasoning": "<1 sentence>"},
    "step2_zone_fib":     {"result": "PASS"|"FAIL", "reasoning": "<1 sentence>"},
    "step3_candle":       {"result": "PASS"|"FAIL", "reasoning": "<1 sentence>"},
    "step4_math":         {"result": "PASS"|"FAIL", "reasoning": "<1 sentence>"},
    "step5_volatility":   {"result": "PASS"|"FAIL", "reasoning": "<1 sentence>"}
  },
  "devils_advocate_risks": ["<risk 1>", "<risk 2>"],
  "trade_plan": {
    "direction":    "<LONG|SHORT|NONE>",
    "entry":        <float|null>,
    "stop_loss":    <float|null>,
    "take_profit":  <float|null>,
    "risk_reward":  <float|null>,
    "risk_usd":     <float|null>,
    "lots":         <float|null>
  },
  "plain_english_summary": "<2–4 clear sentences a beginner can understand>"
}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def audit(math_result: dict[str, Any]) -> dict[str, Any]:
    """
    Send the Gate 2 math result to Gemini for risk audit.

    Args:
        math_result: The dict returned by market_math.analyse().

    Returns:
        Validated JSON dict. Falls back to safe NO_TRADE on any error.
    """
    if not _GENAI_AVAILABLE:
        return _fail_safe("google-genai package not installed. Run: pip install google-genai")

    if not config.GEMINI_API_KEY:
        return _fail_safe("GEMINI_API_KEY not set in .env — cannot call AI auditor.")

    user_prompt = _build_user_prompt(math_result)

    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)

        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.1,          # Low temperature = consistent, conservative outputs
                max_output_tokens=2048,
            ),
        )

        raw_text = response.text.strip()
        ai_json  = json.loads(raw_text)
        return _validate_response(ai_json, math_result)

    except json.JSONDecodeError as exc:
        return _fail_safe(f"Gemini returned invalid JSON: {exc}")
    except Exception as exc:  # noqa: BLE001
        return _fail_safe(f"Gemini API error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_prompt(math_result: dict[str, Any]) -> str:
    """Format the Python math dict as a structured audit request."""
    # Redact the full candle list to keep token count down
    summary = {k: v for k, v in math_result.items() if k != "last_10_candles_15m"}
    candle_story = math_result.get("candle_story", {})
    last_candles  = math_result.get("last_10_candles_15m", [])

    prompt_parts = [
        f"## Forex Setup Audit Request — {math_result.get('symbol', 'UNKNOWN')}",
        "",
        "### Python Math Engine Output (trust these numbers completely):",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "### Last 10 Closed 15m Candles (oldest → newest):",
        "```json",
        json.dumps(last_candles, indent=2),
        "```",
        "",
        "### Your Task:",
        "Run your 5-step audit checklist on this setup.",
        "Apply the Devil's Advocate check.",
        "Output ONLY the JSON response schema defined in your system prompt.",
    ]

    return "\n".join(prompt_parts)


def _validate_response(ai_json: dict, math_result: dict[str, Any]) -> dict[str, Any]:
    """
    Validate the AI response and enforce hard rules.

    Rules enforced here (belt-and-suspenders, in case AI ignores system prompt):
      - Confluence score < 80 → force NO_TRADE.
      - If Python gate failed → force NO_TRADE.
      - If decision not in allowed values → force NO_TRADE.
    """
    allowed_decisions = {"ENTER_LONG", "ENTER_SHORT", "NO_TRADE"}

    if ai_json.get("decision") not in allowed_decisions:
        ai_json["decision"] = "NO_TRADE"
        ai_json["_override"] = "Invalid decision value — forced to NO_TRADE."

    score = int(ai_json.get("confluence_score", 0))
    if score < config.MIN_CONFLUENCE_SCORE:
        ai_json["decision"] = "NO_TRADE"
        ai_json["_override"] = (
            f"Confluence score {score} < minimum {config.MIN_CONFLUENCE_SCORE} — forced to NO_TRADE."
        )

    if not math_result.get("passed_math_gate", False):
        ai_json["decision"] = "NO_TRADE"
        reasons = math_result.get("fail_reasons", ["Math gate failed."])
        ai_json["_override"] = f"Python math gate failed: {'; '.join(reasons)}"

    return ai_json


def _fail_safe(reason: str) -> dict[str, Any]:
    """Return a safe NO_TRADE response when the AI call cannot be made."""
    return {
        "decision":          "NO_TRADE",
        "confluence_score":  0,
        "trend_1h":          "UNKNOWN",
        "checklist_audit":   {},
        "devils_advocate_risks": [reason],
        "trade_plan": {
            "direction":   "NONE",
            "entry":       None,
            "stop_loss":   None,
            "take_profit": None,
            "risk_reward": None,
            "risk_usd":    None,
            "lots":        None,
        },
        "plain_english_summary": f"AI auditor unavailable — NO_TRADE by default. Reason: {reason}",
        "_fail_safe": True,
    }
