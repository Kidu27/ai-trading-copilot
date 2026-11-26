"""
ai_auditor.py — Gate 3: Multi-Provider AI Risk Manager (Groq -> OpenRouter -> Gemini)

Receives the JSON analysis from market_math.py and asks the AI auditor to act
as a strict risk auditor. The AI default is ALWAYS NO_TRADE.

Design principles:
  - Python math is already done and trusted. The AI only reasons.
  - System prompt enforces a 5-step checklist + Devil's Advocate check.
  - Confluence Score < 80 → MUST output NO_TRADE.
  - Multi-tier provider priority: GroqCloud (Primary) -> OpenRouter (Secondary) -> Gemini (Last Resort).
  - Clean error formatting ensures zero raw nested dict dumps into notifications.
  - If all AI calls fail, fail-safe returns NO_TRADE (enabling math-approved plan if math passed).
"""

import json
import os
import re
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

_WORD_TO_NUM: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}


def _parse_score(val: Any) -> int:
    """
    Safely parse confluence_score from int, float, numeric string,
    or English number words (e.g. 'fifty', 'sixty-five', '80%') without raising ValueError.
    """
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        val_clean = val.strip().lower()
        try:
            return int(round(float(val_clean)))
        except ValueError:
            pass

        match = re.search(r"\b(\d{1,3})\b", val_clean)
        if match:
            return int(match.group(1))

        words = re.findall(r"[a-z]+", val_clean.replace("-", " "))
        total = 0
        current = 0
        matched_any = False
        for w in words:
            if w in _WORD_TO_NUM:
                matched_any = True
                num = _WORD_TO_NUM[w]
                if num == 100:
                    current = (current if current else 1) * 100
                else:
                    current += num
            elif w == "and":
                continue
            else:
                total += current
                current = 0
        total += current
        if matched_any:
            return min(max(total, 0), 100)
    return 0


def _build_base_system_prompt() -> str:
    """Build the base system prompt with dynamic score, R:R thresholds, and account balance."""
    min_score = config.get_min_confluence_score() if hasattr(config, "get_min_confluence_score") else config.MIN_CONFLUENCE_SCORE
    min_rr = config.get_min_risk_reward() if hasattr(config, "get_min_risk_reward") else config.MIN_RISK_REWARD
    acct_balance = getattr(config, "ACCOUNT_BALANCE", 1000.0)

    return f"""
You are an elite Forex risk management auditor for a retail trader with a \
${acct_balance:.0f} account. Your ONLY job is to audit whether the mathematical setup presented \
to you meets ALL criteria for a high-probability trade signal.

## CORE RULES — NON-NEGOTIABLE
1. Your default decision is ALWAYS `NO_TRADE`. You need OVERWHELMING evidence to deviate.
2. You NEVER invent prices, calculate lot sizes, or guess market data.
   All numbers come from the Python math engine — you only reason about them.
3. You NEVER recommend trading when the 1H trend is RANGING.
4. Any Confluence Score below {min_score} MUST produce `NO_TRADE`.
5. If `confluence_score >= {min_score}` and the setup aligns with `trend_1h`, `decision` MUST be `ENTER_LONG` (for UPTREND) or `ENTER_SHORT` (for DOWNTREND).
6. If you reject a setup as `NO_TRADE`, your `confluence_score` MUST be strictly below {min_score}.

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
  PASS: Python-confirmed R:R ≥ {min_rr:.1f}, risk ≤ 1% of account, stop-loss is
        logically placed outside the zone with ATR buffer.
  FAIL: R:R < {min_rr:.1f}, risk > 1%, or the Python gate already failed.

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

CRITICAL SCORING & DECISION RULES:
- "confluence_score" MUST be an integer in digits (0–100), NEVER a word like "fifty" or "sixty".
- If confluence_score >= {min_score} and the setup aligns with trend_1h, decision MUST be `ENTER_LONG` (for UPTREND) or `ENTER_SHORT` (for DOWNTREND).
- If you reject a setup as `NO_TRADE`, your confluence_score MUST be strictly below {min_score}.
- Score < {min_score} → decision MUST be `NO_TRADE`.

## OUTPUT FORMAT — STRICT JSON
Return ONLY valid JSON matching this exact schema:
{{
  "decision": "ENTER_LONG" | "ENTER_SHORT" | "NO_TRADE",
  "confluence_score": <integer in digits 0-100, NEVER words>,
  "trend_1h": "<trend string from Python data>",
  "checklist_audit": {{
    "step1_trend":        {{"result": "PASS"|"FAIL", "reasoning": "<1 sentence>"}},
    "step2_zone_fib":     {{"result": "PASS"|"FAIL", "reasoning": "<1 sentence>"}},
    "step3_candle":       {{"result": "PASS"|"FAIL", "reasoning": "<1 sentence>"}},
    "step4_math":         {{"result": "PASS"|"FAIL", "reasoning": "<1 sentence>"}},
    "step5_volatility":   {{"result": "PASS"|"FAIL", "reasoning": "<1 sentence>"}}
  }},
  "devils_advocate_risks": ["<risk 1>", "<risk 2>"],
  "trade_plan": {{
    "direction":    "<LONG|SHORT|NONE>",
    "entry":        <float|null>,
    "stop_loss":    <float|null>,
    "take_profit":  <float|null>,
    "risk_reward":  <float|null>,
    "risk_usd":     <float|null>,
    "lots":         <float|null>
  }},
  "plain_english_summary": "<2–4 clear sentences a beginner can understand>"
}}
""".strip()

_BASE_SYSTEM_PROMPT = _build_base_system_prompt()
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
    min_score = config.get_min_confluence_score() if hasattr(config, "get_min_confluence_score") else config.MIN_CONFLUENCE_SCORE
    prompt = _build_base_system_prompt()

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

def _extract_status_code(err: Exception) -> int | None:
    """Extract HTTP status code from an exception if present."""
    if hasattr(err, "response") and getattr(err.response, "status_code", None):
        return err.response.status_code
    if hasattr(err, "status_code") and isinstance(err.status_code, int):
        return err.status_code
    if hasattr(err, "code") and isinstance(err.code, int):
        return err.code
    match = re.search(r"\b([45]\d\d)\b", str(err))
    if match:
        return int(match.group(1))
    return None


def _format_error(provider: str, model: str, status_or_err: Any) -> str:
    """Format provider errors concisely (f'{provider}/{model}: HTTP {status}')."""
    prefix = f"{provider}/{model}" if model else provider
    if isinstance(status_or_err, int):
        return f"{prefix}: HTTP {status_or_err}"

    if isinstance(status_or_err, Exception):
        status = _extract_status_code(status_or_err)
        if status:
            return f"{prefix}: HTTP {status}"
        err_str = str(status_or_err)
        if (
            isinstance(status_or_err, (requests.Timeout, TimeoutError))
            or "TIMEOUT" in err_str.upper()
            or "DEADLINE" in err_str.upper()
        ):
            return f"{prefix}: Timeout"
        if isinstance(status_or_err, (requests.ConnectionError, ConnectionError)):
            return f"{prefix}: Connection Error"
        if isinstance(status_or_err, json.JSONDecodeError):
            return f"{prefix}: Invalid JSON"
        if isinstance(status_or_err, ValueError):
            return f"{prefix}: {err_str[:60]}"
        return f"{prefix}: {type(status_or_err).__name__}"

    return f"{prefix}: {str(status_or_err)[:60]}"


def audit(math_result: dict[str, Any]) -> dict[str, Any]:
    """
    Send the Gate 2 math result to AI for risk audit.
    Provider priority: GroqCloud (Tier 1) -> OpenRouter (Tier 2) -> Gemini (Tier 3).

    Args:
        math_result: The dict returned by market_math.analyse().

    Returns:
        Validated JSON dict. Falls back to safe NO_TRADE on any error.
    """
    try:
        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(math_result)
    except Exception as exc:  # noqa: BLE001
        return _fail_safe(f"AI auditor initialization error: {exc}", math_result=math_result)

    model_errors: list[str] = []

    # ── Tier 1: Primary — GroqCloud ──────────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY", "").strip() or getattr(config, "GROQ_API_KEY", "").strip()
    if groq_key:
        groq_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
        for model_id in groq_models:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "temperature": 0.1,
                        "max_tokens": 1200,
                        "reasoning_effort": "low",
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    timeout=(5, 25),
                )
                if resp.status_code != 200:
                    err_msg = _format_error("groq", model_id, resp.status_code)
                    print(f"  [ai_auditor] Warning: Groq ({model_id}) returned HTTP {resp.status_code}")
                    model_errors.append(err_msg)
                    continue

                data = resp.json()
                choices = data.get("choices")
                if not choices:
                    raise ValueError("No choices in response")
                content = choices[0].get("message", {}).get("content", "")
                if not content or not content.strip():
                    raise ValueError("Empty content returned")

                raw_text = _clean_json_text(content)
                if not raw_text:
                    raise ValueError("Empty content returned")

                ai_json = json.loads(raw_text)
                result = _validate_response(ai_json, math_result)
                result["model_used"] = f"groq/{model_id}"
                print(f"  [ai_auditor] ✅ Audited via Groq ({model_id})")
                return result

            except Exception as err:
                err_msg = _format_error("groq", model_id, err)
                print(f"  [ai_auditor] Warning: Groq ({model_id}) failed: {err_msg}")
                model_errors.append(err_msg)
                continue
    else:
        model_errors.append("groq: GROQ_API_KEY not set")

    # ── Tier 2: Secondary — OpenRouter ───────────────────────────────────────
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip() or getattr(config, "OPENROUTER_API_KEY", "").strip()
    if or_key:
        openrouter_models = ["nvidia/nemotron-3-ultra-550b-a55b:free", "openrouter/free"]
        for model_id in openrouter_models:
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {or_key}",
                        "Content-Type": "application/json",
                        "X-Title": "VanguardFX Auditor",
                    },
                    json={
                        "model": model_id,
                        "temperature": 0.1,
                        "max_tokens": 1200,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    timeout=(5, 25),
                )
                if resp.status_code != 200:
                    err_msg = _format_error("openrouter", model_id, resp.status_code)
                    print(f"  [ai_auditor] Warning: OpenRouter ({model_id}) returned HTTP {resp.status_code}")
                    model_errors.append(err_msg)
                    continue

                data = resp.json()
                choices = data.get("choices")
                if not choices:
                    raise ValueError("No choices in response")
                content = choices[0].get("message", {}).get("content", "")
                if not content or not content.strip():
                    raise ValueError("Empty content returned")

                raw_text = _clean_json_text(content)
                if not raw_text:
                    raise ValueError("Empty content returned")

                ai_json = json.loads(raw_text)
                result = _validate_response(ai_json, math_result)
                result["model_used"] = f"openrouter/{model_id}"
                print(f"  [ai_auditor] ✅ Audited via OpenRouter ({model_id})")
                return result

            except Exception as err:
                err_msg = _format_error("openrouter", model_id, err)
                print(f"  [ai_auditor] Warning: OpenRouter ({model_id}) failed: {err_msg}")
                model_errors.append(err_msg)
                continue
    else:
        model_errors.append("openrouter: OPENROUTER_API_KEY not set")

    # ── Tier 3: Last Resort — Gemini ─────────────────────────────────────────
    gemini_model = "gemini-3.5-flash-lite"
    gemini_key = getattr(config, "GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "").strip()

    if not _GENAI_AVAILABLE:
        model_errors.append(f"gemini/{gemini_model}: google-genai not installed")
    elif not gemini_key:
        model_errors.append(f"gemini/{gemini_model}: GEMINI_API_KEY not set")
    else:
        try:
            http_opts = (
                types.HttpOptions(timeout=15000, retry_options=types.HttpRetryOptions(attempts=1))
                if hasattr(types, "HttpOptions") and hasattr(types, "HttpRetryOptions")
                else None
            )
            client = (
                genai.Client(api_key=gemini_key, http_options=http_opts)
                if http_opts
                else genai.Client(api_key=gemini_key)
            )

            gen_config_kwargs: dict[str, Any] = {
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "temperature":        0.1,
                "max_output_tokens": 8192,
            }
            if _PYDANTIC_AVAILABLE:
                gen_config_kwargs["response_schema"] = AuditResponse

            try:
                gen_config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
            except Exception:
                pass

            try:
                response = client.models.generate_content(
                    model=gemini_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(**gen_config_kwargs),
                )
            except Exception as call_err:
                err_str = str(call_err).upper()
                if (
                    "400" in err_str
                    or "INVALID_ARGUMENT" in err_str
                    or "INVALIDARGUMENT" in err_str
                    or "THINKING" in err_str
                    or "SCHEMA" in err_str
                ) and (
                    "thinking_config" in gen_config_kwargs or "response_schema" in gen_config_kwargs
                ):
                    gen_config_kwargs.pop("thinking_config", None)
                    gen_config_kwargs.pop("response_schema", None)
                    response = client.models.generate_content(
                        model=gemini_model,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(**gen_config_kwargs),
                    )
                else:
                    raise call_err

            if not response:
                raise ValueError("Empty response returned")

            try:
                raw_resp = response.text
            except Exception as text_err:
                raise ValueError(f"text unreadable ({text_err})")

            if not raw_resp or not raw_resp.strip():
                raise ValueError("Empty text returned")

            raw_text = _clean_json_text(raw_resp)
            if not raw_text:
                raise ValueError("Empty content returned")

            ai_json = json.loads(raw_text)
            result = _validate_response(ai_json, math_result)
            result["model_used"] = f"gemini/{gemini_model}"
            print(f"  [ai_auditor] ✅ Audited via Gemini ({gemini_model})")
            return result

        except Exception as err:
            err_msg = _format_error("gemini", gemini_model, err)
            print(f"  [ai_auditor] Warning: Gemini ({gemini_model}) failed: {err_msg}")
            model_errors.append(err_msg)

    # ── All Three Tiers Failed ───────────────────────────────────────────────
    errors_summary = " | ".join(model_errors) if model_errors else "All AI providers exhausted"
    return _fail_safe(f"All AI models exhausted. Errors: {errors_summary}", math_result=math_result)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_json_text(text: str) -> str:
    """
    Strip markdown code fences (e.g. ```json ... ```) and leading/trailing whitespace
    so json.loads() receives clean JSON.
    """
    raw = text.strip()
    if "```" in raw:
        start_fence = raw.find("```")
        end_fence = raw.rfind("```")
        if start_fence != -1 and end_fence != -1 and end_fence > start_fence:
            inner = raw[start_fence:end_fence]
            first_nl = inner.find("\n")
            if first_nl != -1:
                return inner[first_nl + 1:].strip()
            return inner.lstrip("`").strip()
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

    Rules enforced here:
      - Parse confluence_score safely (handle ints, floats, numeric strings, English number words).
      - Enforce bidirectional consistency:
          * If score < min_score: force decision = "NO_TRADE".
          * If score >= min_score AND math_result passed AND decision == "NO_TRADE":
            check checklist_audit — if step1_trend and step4_math are "PASS",
            automatically align decision to "ENTER_LONG" (for UPTREND) or "ENTER_SHORT" (for DOWNTREND)
            and populate trade_plan from math_result so actionable >= min_score setups
            always send Accept/Decline buttons.
          * If rejected as "NO_TRADE", ensure confluence_score is strictly below min_score.
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

    min_score = config.get_min_confluence_score() if hasattr(config, "get_min_confluence_score") else config.MIN_CONFLUENCE_SCORE
    score = _parse_score(ai_json.get("confluence_score", 0))
    ai_json["confluence_score"] = score

    math_gate_passed = bool(math_result.get("passed_math_gate", False))

    if not math_gate_passed:
        ai_json["decision"] = "NO_TRADE"
        reasons = math_result.get("fail_reasons", ["Math gate failed."])
        ai_json["_override"] = f"Python math gate failed: {'; '.join(reasons)}"
        if score >= min_score:
            score = max(0, min_score - 1)
            ai_json["confluence_score"] = score

    # Enforce bidirectional consistency
    if score < min_score:
        ai_json["decision"] = "NO_TRADE"
        ai_json["_override"] = (
            f"Confluence score {score} < minimum {min_score} — forced to NO_TRADE."
        )
    elif math_gate_passed and ai_json.get("decision") == "NO_TRADE":
        checklist = ai_json.get("checklist_audit") or ai_json.get("checklist") or {}

        def _is_step_pass(step_val: Any) -> bool:
            if isinstance(step_val, dict):
                return str(step_val.get("result", "")).strip().upper() == "PASS"
            if isinstance(step_val, str):
                return step_val.strip().upper() == "PASS"
            return False

        step1_pass = _is_step_pass(checklist.get("step1_trend"))
        step4_pass = _is_step_pass(checklist.get("step4_math"))

        trend_1h = str(math_result.get("trend_1h") or ai_json.get("trend_1h", "")).strip().upper()

        if step1_pass and step4_pass:
            if trend_1h == "UPTREND":
                ai_json["decision"] = "ENTER_LONG"
            elif trend_1h == "DOWNTREND":
                ai_json["decision"] = "ENTER_SHORT"

        if ai_json.get("decision") in ("ENTER_LONG", "ENTER_SHORT"):
            ai_json["_override"] = (
                f"Confluence score {score} >= {min_score} with PASS on trend and math — aligned to {ai_json['decision']}."
            )
            # Populate trade_plan from math_result so actionable >= min_score setups always send Accept/Decline buttons
            tp = ai_json.setdefault("trade_plan", {})
            if not isinstance(tp, dict):
                tp = {}
                ai_json["trade_plan"] = tp

            direction = "LONG" if ai_json["decision"] == "ENTER_LONG" else "SHORT"
            tp["direction"] = direction
            tp["entry"] = tp.get("entry") or math_result.get("entry_price")
            tp["stop_loss"] = tp.get("stop_loss") or math_result.get("stop_loss")
            tp["take_profit"] = tp.get("take_profit") or math_result.get("take_profit")
            tp["risk_reward"] = tp.get("risk_reward") or math_result.get("risk_reward_ratio")
            tp["risk_usd"] = tp.get("risk_usd") or math_result.get("max_risk_usd")
            tp["lots"] = tp.get("lots") or math_result.get("position_size_lots")
        else:
            # Setup was rejected as NO_TRADE: confluence_score must be strictly below min_score
            if score >= min_score:
                score = max(0, min_score - 1)
                ai_json["confluence_score"] = score

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
