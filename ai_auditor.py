"""
ai_auditor.py — Gate 3: Strict Gemini AI Risk Manager

Receives the JSON analysis from market_math.py and asks Gemini to act
as a strict risk auditor. The AI default is ALWAYS NO_TRADE.

Design principles:
  - Python math is already done and trusted. The AI only reasons.
  - System prompt enforces a 5-step checklist + Devil's Advocate check.
  - Confluence Score < 80 → MUST output NO_TRADE.
  - Response is strict JSON enforced by response_mime_type + response_schema
    (Pydantic model passed to the google-genai SDK). JSON parsing never fails.
  - If the Gemini API call fails, we fail safe: return NO_TRADE.
"""

import json
import os
import time
from typing import Any

import requests

import config
import news_fetcher
import pattern_stats

# Official Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

# Pydantic for structured response schema
try:
    from pydantic import BaseModel
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False

# Module-level health tracking for adaptive sticky model router
_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}
_LAST_WORKING_MODEL: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic response schema — guarantees Gemini returns valid JSON
# ─────────────────────────────────────────────────────────────────────────────

if _PYDANTIC_AVAILABLE:
    class _ChecklistStep(BaseModel):
        result: str       # "PASS" or "FAIL"
        reasoning: str

    class _ChecklistAudit(BaseModel):
        step1_trend:      _ChecklistStep
        step2_zone_fib:   _ChecklistStep
        step3_candle:     _ChecklistStep
        step4_math:       _ChecklistStep
        step5_volatility: _ChecklistStep

    class _TradePlan(BaseModel):
        direction:   str = "NONE"
        entry:       float | None = None
        stop_loss:   float | None = None
        take_profit: float | None = None
        risk_reward: float | None = None
        risk_usd:    float | None = None
        lots:        float | None = None

    class AuditResponse(BaseModel):
        decision:               str                 # "ENTER_LONG" | "ENTER_SHORT" | "NO_TRADE"
        confluence_score:       int                 # 0–100
        trend_1h:               str = "UNKNOWN"
        checklist_audit:        _ChecklistAudit
        devils_advocate_risks:  list[str]
        trade_plan:             _TradePlan
        plain_english_summary:  str

        @property
        def checklist(self) -> _ChecklistAudit:
            return self.checklist_audit

        @property
        def devils_advocate(self) -> list[str]:
            return self.devils_advocate_risks

        @property
        def summary(self) -> str:
            return self.plain_english_summary


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — the AI's immutable rule book
# ─────────────────────────────────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = """
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
""".strip().replace("$500 account", f"${config.ACCOUNT_BALANCE:.0f} account")

_SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT


def _load_recent_lessons() -> list[dict[str, Any]]:
    """Load the rolling memory of lessons learned from lessons_learned.json."""
    import os
    lessons_path = os.path.join(os.path.dirname(__file__), "lessons_learned.json")
    if not os.path.exists(lessons_path):
        return []
    try:
        with open(lessons_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _build_system_prompt() -> str:
    """
    Build the full system prompt, dynamically injecting:
      1. WINNING SIGNATURES TO PRIORITIZE (Winning DNA from pattern_stats)
      2. PAST LOSS LESSONS TO AVOID (from lessons_learned.json)
    """
    min_score = getattr(config, "get_min_confluence_score", lambda: config.MIN_CONFLUENCE_SCORE)()
    prompt = _BASE_SYSTEM_PROMPT.replace(
        "$500 account",
        f"${config.ACCOUNT_BALANCE:.0f} account"
    ).replace(
        "Any Confluence Score below 80 MUST produce `NO_TRADE`.",
        f"Any Confluence Score below {min_score} MUST produce `NO_TRADE`."
    ).replace(
        "Score < 80 → decision MUST be `NO_TRADE`.",
        f"Score < {min_score} → decision MUST be `NO_TRADE`."
    )

    sections = []

    # 1. Winning DNA from pattern_stats
    try:
        win_dna = pattern_stats.get_win_signature_summary()
        if win_dna:
            sections.append(
                "\n\n### WINNING SIGNATURES TO PRIORITIZE (WINNING DNA)\n"
                "The following indicator combinations have yielded the highest win rates in our closed trades. "
                "Actively look for and prioritize setups that exhibit these high-probability characteristics:\n"
                f"{win_dna}"
            )
    except Exception as exc:
        pass

    # 2. Historical loss lessons to avoid
    lessons = _load_recent_lessons()
    if lessons:
        recent = lessons[-10:]
        loss_lines = [
            "\n\n### PAST LOSS LESSONS TO AVOID\n"
            "You have accumulated the following historical diagnostic lessons from recently closed "
            "demo trades. You MUST cross-reference the current setup against these past failure modes. "
            f"If the setup repeats a known mistake described below, strictly PENALIZE the Confluence Score below {min_score} or output `NO_TRADE`:\n"
        ]
        for i, item in enumerate(recent, 1):
            outcome = item.get("outcome", "TRADE").upper()
            pair    = item.get("pair", "UNKNOWN")
            lesson  = item.get("actionable_lesson", "").strip()
            ts      = item.get("timestamp_eat", "")
            loss_lines.append(f"{i}. [{outcome}] {pair} ({ts}): {lesson}")
        sections.append("\n".join(loss_lines))

    if sections:
        return prompt + "".join(sections)
    return prompt


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
        return _fail_safe("google-genai package not installed. Run: pip install google-genai", math_result=math_result)

    if not config.GEMINI_API_KEY:
        return _fail_safe("GEMINI_API_KEY not set in .env — cannot call AI auditor.", math_result=math_result)

    user_prompt = _build_user_prompt(math_result)

    try:
        http_opts = (
            types.HttpOptions(timeout=15000, retry_options=types.HttpRetryOptions(attempts=1))
            if hasattr(types, "HttpOptions") and hasattr(types, "HttpRetryOptions")
            else None
        )
        client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=http_opts) if http_opts else genai.Client(api_key=config.GEMINI_API_KEY)

        # Build GenerateContentConfig — strict JSON enforced by response_mime_type + response_schema
        # max_output_tokens=8192 ensures complete JSON output without truncation.
        gen_config_kwargs: dict[str, Any] = {
            "system_instruction": _build_system_prompt(),
            "response_mime_type": "application/json",
            "temperature":        0.1,          # Low temperature = consistent, conservative outputs
            "max_output_tokens": 8192,
        }
        if _PYDANTIC_AVAILABLE:
            gen_config_kwargs["response_schema"] = AuditResponse

        # Base models ranked by speed, low latency, and free-tier availability
        base_models = list(dict.fromkeys([
            config.GEMINI_MODEL,
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
            "gemma-4-31b-it",
            "gemma-4-26b-a4b-it",
            "gemini-4-hydrogen",
            "gemini-3.1-flash-lite",
            "gemini-3.7-flash",
        ]))

        # Adaptive Sticky Router: put _LAST_WORKING_MODEL first (if ready), then non-cooldown models,
        # followed by cooling models as last resort
        now_mono = time.monotonic()
        active_models = [m for m in base_models if _MODEL_COOLDOWN_UNTIL.get(m, 0.0) <= now_mono]
        cooldown_models = [m for m in base_models if _MODEL_COOLDOWN_UNTIL.get(m, 0.0) > now_mono]

        global _LAST_WORKING_MODEL
        ordered_models: list[str] = []
        if _LAST_WORKING_MODEL and _LAST_WORKING_MODEL in active_models:
            ordered_models.append(_LAST_WORKING_MODEL)
        for m in active_models:
            if m not in ordered_models:
                ordered_models.append(m)
        for m in cooldown_models:
            if m not in ordered_models:
                ordered_models.append(m)

        model_errors: list[str] = []

        for model_name in ordered_models:
            try:
                cfg_kwargs = dict(gen_config_kwargs)
                is_gemma = model_name.startswith("gemma-")
                if is_gemma:
                    cfg_kwargs.pop("thinking_config", None)
                    cfg_kwargs.pop("response_schema", None)
                else:
                    try:
                        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
                    except Exception:
                        pass

                call_contents = user_prompt
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=call_contents,
                        config=types.GenerateContentConfig(**cfg_kwargs),
                    )
                except Exception as call_err:
                    err_str = str(call_err).upper()
                    # 1. Gemma system_instruction fallback
                    if is_gemma and "400" in err_str and "system_instruction" in cfg_kwargs:
                        cfg_kwargs.pop("system_instruction", None)
                        call_contents = f"{_build_system_prompt()}\n\n{user_prompt}"
                        response = client.models.generate_content(
                            model=model_name,
                            contents=call_contents,
                            config=types.GenerateContentConfig(**cfg_kwargs),
                        )
                    # 2. Strip thinking_config and response_schema on 400 / INVALID_ARGUMENT
                    elif (
                        "400" in err_str
                        or "INVALID_ARGUMENT" in err_str
                        or "INVALIDARGUMENT" in err_str
                        or "THINKING" in err_str
                        or "SCHEMA" in err_str
                    ) and (
                        "thinking_config" in cfg_kwargs or "response_schema" in cfg_kwargs
                    ):
                        cfg_kwargs.pop("thinking_config", None)
                        cfg_kwargs.pop("response_schema", None)
                        response = client.models.generate_content(
                            model=model_name,
                            contents=call_contents,
                            config=types.GenerateContentConfig(**cfg_kwargs),
                        )
                    else:
                        raise call_err

                # Response extraction & validation
                if not response:
                    print(f"  [ai_auditor] Warning: {model_name} returned empty response. Retrying next model...")
                    model_errors.append(f"{model_name}: empty response")
                    continue

                try:
                    raw_resp = response.text
                except Exception as text_err:
                    print(f"  [ai_auditor] Warning: {model_name} response.text could not be read: {text_err}. Retrying next model...")
                    model_errors.append(f"{model_name}: text unreadable ({text_err})")
                    continue

                if not raw_resp or not raw_resp.strip():
                    print(f"  [ai_auditor] Warning: {model_name} returned empty text. Retrying next model...")
                    model_errors.append(f"{model_name}: empty text")
                    continue

                raw_text = _clean_json_text(raw_resp)
                if not raw_text:
                    print(f"  [ai_auditor] Warning: {model_name} returned empty text after stripping code fences. Retrying next model...")
                    model_errors.append(f"{model_name}: empty text after cleaning")
                    continue

                ai_json = json.loads(raw_text)
                validated = _validate_response(ai_json, math_result)

                # Sticky success: mark model as last working and clear cooldown
                _LAST_WORKING_MODEL = model_name
                _MODEL_COOLDOWN_UNTIL.pop(model_name, None)
                return validated

            except json.JSONDecodeError as decode_err:
                print(f"  [ai_auditor] Warning: {model_name} returned invalid/truncated JSON: {decode_err}. Retrying next model...")
                model_errors.append(f"{model_name}: JSONDecodeError ({decode_err})")
                continue
            except Exception as err:
                err_str = str(err).upper()
                if (
                    "503" in err_str
                    or "UNAVAILABLE" in err_str
                    or "504" in err_str
                    or "DEADLINE_EXCEEDED" in err_str
                    or "TIMEOUT" in err_str
                ):
                    _MODEL_COOLDOWN_UNTIL[model_name] = time.monotonic() + 180.0
                    print(f"  [ai_auditor] ⏱ {model_name} busy/timeout ({err}). Set on 3-min cooldown.")
                else:
                    print(f"  [ai_auditor] Warning: {model_name} call failed: {err}. Retrying next model...")
                model_errors.append(f"{model_name}: {err}")
                continue

        # Fallback to Groq API if configured and all Google models failed
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_api_key:
            try:
                print("  [ai_auditor] All Google models exhausted. Invoking Groq fallback (llama-3.3-70b-versatile)...")
                groq_resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": _build_system_prompt()},
                            {"role": "user", "content": user_prompt},
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.1,
                        "max_tokens": 4096,
                    },
                    timeout=15,
                )
                if groq_resp.status_code == 200:
                    groq_data = groq_resp.json()
                    raw_text = groq_data["choices"][0]["message"]["content"]
                    ai_json = json.loads(_clean_json_text(raw_text))
                    print("  [ai_auditor] ✅ Groq fallback succeeded.")
                    return _validate_response(ai_json, math_result)
                else:
                    model_errors.append(f"groq: HTTP {groq_resp.status_code} {groq_resp.text[:100]}")
            except Exception as groq_err:
                model_errors.append(f"groq: {groq_err}")

        errors_summary = " | ".join(model_errors) if model_errors else "No models executed"
        return _fail_safe(f"All AI models exhausted. Errors: {errors_summary}", math_result=math_result)

    except Exception as exc:  # noqa: BLE001
        return _fail_safe(f"Gemini API initialization error: {exc}", math_result=math_result)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_json_text(text: str) -> str:
    """
    Strip markdown code fences (e.g. ```json ... ```) and leading/trailing whitespace
    so json.loads() receives clean JSON.
    """
    raw = text.strip()
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        if first_newline != -1:
            raw = raw[first_newline + 1:]
        else:
            raw = raw.lstrip("`")
    if raw.endswith("```"):
        last_fence = raw.rfind("```")
        if last_fence != -1:
            raw = raw[:last_fence]
    return raw.strip()


def _build_user_prompt(math_result: dict[str, Any]) -> str:
    """Format the Python math dict and live news context as a structured audit request."""
    symbol = math_result.get("symbol", "UNKNOWN")
    summary = {k: v for k, v in math_result.items() if k not in ("last_10_candles_15m", "_news_data")}
    candle_story = math_result.get("candle_story", {})
    last_candles  = math_result.get("last_10_candles_15m", [])

    # Fetch live FX news headlines for this pair
    news_data = news_fetcher.get_news_for_pair(symbol)
    math_result["_news_data"] = news_data

    headline_bullets = [f"  • {h}" for h in news_data.get("headlines", [])] if news_data.get("headlines") else ["  • No breaking headline spikes detected; regular order flow."]

    prompt_parts = [
        f"## Forex Setup Audit Request — {symbol}",
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
        f"### Live FX News Headlines & Context ({symbol}):",
        f"- Sentiment Bias: {news_data.get('news_bias', 'NEUTRAL')}",
        f"- Context: {news_data.get('summary', '')}",
        "- Recent Headlines:",
        *headline_bullets,
        "",
        "### Your Task:",
        "Run your 5-step audit checklist on this setup.",
        "Apply the Devil's Advocate check.",
        "Cross-reference Winning DNA signatures and past loss lessons.",
        "Output ONLY the JSON response schema defined in your system prompt.",
    ]

    return "\n".join(prompt_parts)


def _validate_response(ai_json: dict, math_result: dict[str, Any]) -> dict[str, Any]:
    """
    Validate the AI response and enforce hard rules.

    Rules enforced here (belt-and-suspenders, in case AI ignores system prompt):
      - Confluence score < dynamic MIN_CONFLUENCE_SCORE (65 in Demo Discovery, 80 in Live) → force NO_TRADE.
      - If Python gate failed → force NO_TRADE.
      - If decision not in allowed values → force NO_TRADE.
    """
    # Normalize aliases so both naming conventions are always available
    if "checklist_audit" in ai_json and "checklist" not in ai_json:
        ai_json["checklist"] = ai_json["checklist_audit"]
    elif "checklist" in ai_json and "checklist_audit" not in ai_json:
        ai_json["checklist_audit"] = ai_json["checklist"]

    if "devils_advocate_risks" in ai_json and "devils_advocate" not in ai_json:
        ai_json["devils_advocate"] = ai_json["devils_advocate_risks"]
    elif "devils_advocate" in ai_json and "devils_advocate_risks" not in ai_json:
        ai_json["devils_advocate_risks"] = ai_json["devils_advocate"]

    if "plain_english_summary" in ai_json and "summary" not in ai_json:
        ai_json["summary"] = ai_json["plain_english_summary"]
    elif "summary" in ai_json and "plain_english_summary" not in ai_json:
        ai_json["plain_english_summary"] = ai_json["summary"]

    allowed_decisions = {"ENTER_LONG", "ENTER_SHORT", "NO_TRADE"}

    if ai_json.get("decision") not in allowed_decisions:
        ai_json["decision"] = "NO_TRADE"
        ai_json["_override"] = "Invalid decision value — forced to NO_TRADE."

    min_score = getattr(config, "get_min_confluence_score", lambda: config.MIN_CONFLUENCE_SCORE)()
    score = int(ai_json.get("confluence_score", 0))
    if score < min_score:
        ai_json["decision"] = "NO_TRADE"
        ai_json["_override"] = (
            f"Confluence score {score} < minimum {min_score} — forced to NO_TRADE."
        )

    if not math_result.get("passed_math_gate", False):
        ai_json["decision"] = "NO_TRADE"
        reasons = math_result.get("fail_reasons", ["Math gate failed."])
        ai_json["_override"] = f"Python math gate failed: {'; '.join(reasons)}"

    news = math_result.get("_news_data", {})
    ai_json["news_bias"] = news.get("news_bias", "NEUTRAL")
    ai_json["news_headlines"] = news.get("headlines", [])

    return ai_json


def _fail_safe(reason: str, math_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a safe NO_TRADE response when the AI call cannot be made."""
    trend_1h = "UNKNOWN"
    trade_plan = {
        "direction":   "NONE",
        "entry":       None,
        "stop_loss":   None,
        "take_profit": None,
        "risk_reward": None,
        "risk_usd":    None,
        "lots":        None,
    }

    if math_result and math_result.get("passed_math_gate"):
        trend_1h = math_result.get("trend_1h", "UNKNOWN")
        trade_plan = {
            "direction":   math_result.get("direction", "NONE"),
            "entry":       math_result.get("entry_price"),
            "stop_loss":   math_result.get("stop_loss"),
            "take_profit": math_result.get("take_profit"),
            "risk_reward": math_result.get("risk_reward_ratio"),
            "risk_usd":    math_result.get("max_risk_usd"),
            "lots":        math_result.get("position_size_lots"),
        }

    return {
        "decision":          "NO_TRADE",
        "confluence_score":  0,
        "trend_1h":          trend_1h,
        "checklist_audit":   {},
        "checklist":         {},
        "devils_advocate_risks": [reason],
        "devils_advocate":   [reason],
        "trade_plan":        trade_plan,
        "plain_english_summary": f"AI auditor unavailable — NO_TRADE by default. Reason: {reason}",
        "summary":                f"AI auditor unavailable — NO_TRADE by default. Reason: {reason}",
        "_fail_safe": True,
    }
