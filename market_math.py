"""
market_math.py — Gate 2: Deterministic Math & Technical Indicators

ALL numbers are computed here with pure pandas/numpy.
The LLM (Gate 3) only receives the final JSON dict from this module.

Gate 2 logic:
  1. Upfront OHLC Data Validation  → _validate_candles
  2. Live FX Conversion Cache      → pair-aware pip values & risk capping
  3. 1H/4H Market Structure & BOS  → swing detection & trend classification
  4. Anchored S/R Zone Clustering  → non-chaining cluster anchors
  5. Fibonacci Golden Zone & Area  → dynamic value area (S/R, Fib, EMA50)
  6. 15m Directional Candle Story  → strict engulfing & trigger quality
  7. Risk Sizing & Safety Gates    → hard rules + weighted score (0–100)
"""

import math
import numpy as np
import pandas as pd
from typing import Any

import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. Module-Level Live FX Conversion Cache (Zero Extra API Credits)
# ─────────────────────────────────────────────────────────────────────────────

# Static reference fallbacks used only when live rates have not yet been observed in the current session
_FALLBACK_FX_RATES: dict[str, float] = {
    "USD/JPY": 159.0,
    "AUD/USD": 0.657,
    "NZD/USD": 0.550,
    "USD/CHF": 0.880,
    "EUR/USD": 1.050,
    "GBP/USD": 1.260,
}

_LATEST_FX_RATES: dict[str, float] = {}


def update_fx_rate(symbol: str, rate: float) -> None:
    """Cache live exchange rate for pip value conversions."""
    clean = symbol.split(" ")[0].upper()
    _LATEST_FX_RATES[clean] = float(rate)


def get_fx_rate(symbol: str) -> float | None:
    """Retrieve live cached rate, or None if not yet observed."""
    clean = symbol.split(" ")[0].upper()
    return _LATEST_FX_RATES.get(clean)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Upfront OHLC Data Validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_candles(df_1h: pd.DataFrame, df_15m: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate that OHLC DataFrames meet minimal row counts, required columns,
    and valid price relationship constraints without NaNs.
    """
    if not isinstance(df_1h, pd.DataFrame) or len(df_1h) < 60:
        count = len(df_1h) if isinstance(df_1h, pd.DataFrame) else 0
        return False, f"df_1h has insufficient rows ({count} < 60)."
    if not isinstance(df_15m, pd.DataFrame) or len(df_15m) < 25:
        count = len(df_15m) if isinstance(df_15m, pd.DataFrame) else 0
        return False, f"df_15m has insufficient rows ({count} < 25)."

    req_cols = ["open", "high", "low", "close"]
    for col in req_cols:
        if col not in df_1h.columns:
            return False, f"df_1h missing required column: {col}."
        if col not in df_15m.columns:
            return False, f"df_15m missing required column: {col}."

    tail_1h = df_1h[req_cols].iloc[-20:]
    if tail_1h.isna().any().any():
        return False, "df_1h contains NaN values in the last 20 rows."
    tail_15m = df_15m[req_cols].iloc[-20:]
    if tail_15m.isna().any().any():
        return False, "df_15m contains NaN values in the last 20 rows."

    for label, tail in (("df_1h", tail_1h), ("df_15m", tail_15m)):
        if not np.isfinite(tail.to_numpy(dtype=float)).all():
            return False, f"{label} contains infinite values in the last 20 rows."
        if (tail <= 0).any().any():
            return False, f"{label} contains non-positive prices in the last 20 rows."
        if not (tail["high"] >= tail["low"]).all():
            return False, f"{label} has invalid OHLC: high < low in the last 20 rows."
        if not ((tail["open"] <= tail["high"]) & (tail["open"] >= tail["low"]) &
                (tail["close"] <= tail["high"]) & (tail["close"] >= tail["low"])).all():
            return False, f"{label} has open/close outside [low, high] in the last 20 rows."

    if not df_1h.index.is_monotonic_increasing or df_1h.index.has_duplicates:
        return False, "df_1h timestamps are not strictly chronological and unique."
    if not df_15m.index.is_monotonic_increasing or df_15m.index.has_duplicates:
        return False, "df_15m timestamps are not strictly chronological and unique."

    return True, ""


def _build_empty_result(
    symbol: str,
    fail_reason: str,
    current_price: float = 0.0,
) -> dict[str, Any]:
    """
    Build a complete dict with safe default values for all 48 output keys
    when data validation fails upfront.
    """
    min_math_score = config.get_min_math_score() if hasattr(config, "get_min_math_score") else (55 if (config.DEMO_MODE and getattr(config, "DEMO_DISCOVERY_MODE", False)) else 65)
    max_risk_usd = round(config.ACCOUNT_BALANCE * config.MAX_RISK_PERCENT, 2)
    pip_val_usd = _get_pip_value_usd(symbol, current_price)

    return {
        "symbol":                       symbol,
        "passed_math_gate":             False,
        "fail_reasons":                 [fail_reason],
        "trend_1h":                     "RANGING",
        "swing_highs":                  [],
        "swing_lows":                   [],
        "current_price":                round(current_price, 5),
        "sr_zones":                     [],
        "nearest_support":              None,
        "nearest_resistance":           None,
        "price_near_key_zone":          False,
        "ema50_1h":                     0.0,
        "ema200_1h":                    0.0,
        "ema_signal":                   "NEUTRAL",
        "ema_gap_pips":                 0.0,
        "fibonacci":                    {"available": False, "reason": fail_reason},
        "in_fib_golden_zone":           False,
        "atr_15m":                      0.0,
        "atr_1h":                       0.0,
        "rsi_14_15m":                   50.0,
        "session_tag":                  _get_session_tag(),
        "candle_story":                 {"available": False, "reason": fail_reason},
        "last_10_candles_15m":          [],
        "direction":                    "NONE",
        "entry_price":                  None,
        "stop_loss":                    None,
        "take_profit":                  None,
        "risk_pips":                    None,
        "reward_pips":                  None,
        "risk_reward_ratio":            0.0,
        "position_size_lots":           None,
        "position_size_micro_lots":     None,
        "position_size_units":          None,
        "account_balance":              config.ACCOUNT_BALANCE,
        "max_risk_usd":                 max_risk_usd,
        "atr_buffer":                   0.0,
        "distance_from_zone_atr":       0.0,
        "value_area_type":              "NONE",
        "directional_candle_confirmed": False,
        "candle_quality":               "NONE",
        "pip_value_usd":                round(pip_val_usd, 4),
        "actual_risk_usd":              0.0,
        "risk_cap_exceeded":            False,
        "tp_source":                    "NONE",
        "math_setup_score":             0,
        "score_breakdown": {
            "structure":    0,
            "value_area":   0,
            "price_action": 0,
            "ema_momentum": 0,
            "fibonacci":    0,
            "risk_reward":  0,
        },
        "trend_4h":                     "UNKNOWN",
        "structure_event":              "NONE",
        "min_math_score":               min_math_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Public Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def analyse(
    symbol: str,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run all Gate 2 deterministic technical and risk checks.

    Args:
        symbol: e.g. "EUR/USD" or "EUR/USD (MOCK)"
        df_1h:  1H DataFrame (oldest → newest; >= 60 rows)
        df_15m: 15m DataFrame (oldest → newest; >= 25 rows)

    Returns:
        JSON-serialisable dict with all 48 metrics + `passed_math_gate` bool.
    """
    # ── 1. Upfront OHLC Validation ───────────────────────────────────────────
    valid, err_msg = _validate_candles(df_1h, df_15m)
    if not valid:
        curr_p = (
            float(df_1h["close"].iloc[-1])
            if (isinstance(df_1h, pd.DataFrame) and len(df_1h) > 0 and "close" in df_1h.columns)
            else 0.0
        )
        return _build_empty_result(symbol, err_msg, curr_p)

    current_price = float(df_1h["close"].iloc[-1])
    clean_sym = symbol.split(" ")[0].upper()
    _LATEST_FX_RATES[clean_sym] = current_price

    pip_size = 0.01 if "JPY" in symbol else 0.0001
    result: dict[str, Any] = {
        "symbol": symbol,
        "current_price": round(current_price, 5),
        "passed_math_gate": True,
        "fail_reasons": [],
    }

    # ── 2. Indicators & Swings Computed Strictly on Closed Candles ───────────
    df_1h_closed = df_1h.iloc[:-1]
    df_15m_closed = df_15m.iloc[:-1]

    ema50_series = _ema(df_1h_closed["close"], 50)
    ema200_series = _ema(df_1h_closed["close"], 200)
    atr_1h_series = _atr(df_1h_closed, period=14)
    atr_1h = float(atr_1h_series.iloc[-1])
    result["atr_1h"] = round(atr_1h, 5)

    atr_15m_series = _atr(df_15m_closed, period=14)
    atr_15m = float(atr_15m_series.iloc[-1])
    result["atr_15m"] = round(atr_15m, 5)

    rsi_series = _rsi(df_15m_closed["close"], period=14)
    result["rsi_14_15m"] = round(float(rsi_series.iloc[-1] if len(rsi_series) >= 1 else 50.0), 1)

    # ── 3. Market Structure & 4H Trend Context ────────────────────────────────
    swing_highs, swing_lows = _find_swings(df_1h_closed, left=3, right=3)
    result["swing_highs"] = [round(p, 5) for _, p in swing_highs[-6:]]
    result["swing_lows"]  = [round(p, 5) for _, p in swing_lows[-6:]]

    closed_1h_close = float(df_1h_closed["close"].iloc[-1])
    trend, structure_event = _classify_trend(
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        current_price=closed_1h_close,
        ema50_series=ema50_series,
        ema200_series=ema200_series,
        atr_1h=atr_1h,
    )
    result["trend_1h"] = trend
    result["structure_event"] = structure_event

    trend_4h = _compute_trend_4h(df_1h)
    result["trend_4h"] = trend_4h

    # ── 4. S/R Zone Clustering & Dynamic Tolerance ───────────────────────────
    all_pivots = [p for _, p in swing_highs] + [p for _, p in swing_lows]
    zone_tolerance = max(0.5 * atr_1h, 8 * pip_size)
    zones = _cluster_zones(all_pivots, tolerance_pips=zone_tolerance)
    result["sr_zones"] = zones

    nearest_support = _nearest_below(zones, current_price)
    nearest_resistance = _nearest_above(zones, current_price)
    result["nearest_support"] = nearest_support
    result["nearest_resistance"] = nearest_resistance

    zone_proximity = max(1.5 * atr_1h, 20 * pip_size)
    near_static_zone = _is_near_zone(
        current_price, nearest_support, nearest_resistance,
        proximity_pips=zone_proximity, trend=trend,
    )

    # ── 5. V2 Indicators & Dynamic Value Area ─────────────────────────────────
    if config.USE_V2_INDICATORS:
        ema50_val = float(ema50_series.iloc[-1])
        ema200_val = float(ema200_series.iloc[-1])
        result["ema50_1h"] = round(ema50_val, 5)
        result["ema200_1h"] = round(ema200_val, 5)
        result["ema_signal"] = "BULLISH" if ema50_val > ema200_val else "BEARISH"
        result["ema_gap_pips"] = round(abs(ema50_val - ema200_val) / pip_size, 1)

        fib = _fibonacci_zones(swing_highs, swing_lows, trend)
        result["fibonacci"] = fib
        if fib.get("available"):
            gz_min = min(fib["golden_zone_low"], fib["golden_zone_high"]) - (2 * pip_size)
            gz_max = max(fib["golden_zone_low"], fib["golden_zone_high"]) + (2 * pip_size)
            result["in_fib_golden_zone"] = bool(gz_min <= current_price <= gz_max)
        else:
            result["in_fib_golden_zone"] = False
    else:
        result["ema50_1h"] = 0.0
        result["ema200_1h"] = 0.0
        result["ema_signal"] = "NEUTRAL"
        result["ema_gap_pips"] = 0.0
        result["fibonacci"] = {"available": False, "reason": "V2 indicators disabled."}
        result["in_fib_golden_zone"] = False

    ema50_last = float(ema50_series.iloc[-1])
    near_dynamic_ema50 = (trend != "RANGING") and (abs(current_price - ema50_last) <= 1.0 * atr_1h)
    in_value_area = near_static_zone or near_dynamic_ema50 or bool(result.get("in_fib_golden_zone", False))
    result["price_near_key_zone"] = in_value_area

    if near_static_zone:
        result["value_area_type"] = "SR_ZONE"
    elif result.get("in_fib_golden_zone", False):
        result["value_area_type"] = "FIB_GOLDEN_ZONE"
    elif near_dynamic_ema50:
        result["value_area_type"] = "DYNAMIC_EMA50"
    else:
        result["value_area_type"] = "NONE"

    # ── 6. 15m Directional Candle Story & Sessions ───────────────────────────
    result["session_tag"] = _get_session_tag()

    # Always evaluate the most recently CLOSED candle (df_15m.iloc[:-1] index -1)
    df_15m_closed = df_15m.iloc[:-1]
    candle_story = _analyse_candle(df_15m_closed, candle_index=-1)
    result["candle_story"] = candle_story
    result["last_10_candles_15m"] = _last_n_candles(df_15m, n=10)

    candle_quality = _evaluate_candle_quality(candle_story, trend)
    result["candle_quality"] = candle_quality
    result["directional_candle_confirmed"] = (candle_quality == "STRONG")

    # ── 7. Risk Sizing & Hard Cap ─────────────────────────────────────────────
    risk_metrics = _calculate_risk(
        symbol=symbol,
        trend=trend,
        current_price=current_price,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        near_static_zone=near_static_zone,
        in_value_area=in_value_area,
        atr_15m=atr_15m,
        atr_1h=atr_1h,
        ema50_last=ema50_last,
        df_15m=df_15m,
    )
    result.update(risk_metrics)

    # ── Anchor Zone Distance & Trigger Candle Climax Detection ────────────────
    entry = risk_metrics.get("entry_price") or current_price
    direction = risk_metrics.get("direction", "NONE")
    va_type = result.get("value_area_type", "NONE")

    if va_type == "FIB_GOLDEN_ZONE":
        zone_distance = 0.0
    elif va_type == "DYNAMIC_EMA50":
        zone_distance = abs(entry - ema50_last)
    elif direction == "LONG" or (direction == "NONE" and trend == "UPTREND"):
        if nearest_support:
            zone_distance = abs(entry - nearest_support["price"])
        else:
            zone_distance = abs(entry - ema50_last)
    elif direction == "SHORT" or (direction == "NONE" and trend == "DOWNTREND"):
        if nearest_resistance:
            zone_distance = abs(nearest_resistance["price"] - entry)
        else:
            zone_distance = abs(entry - ema50_last)
    else:
        zone_distance = 0.0

    distance_from_zone_atr = round(zone_distance / atr_1h, 2) if atr_1h > 0 else 0.0
    result["distance_from_zone_atr"] = distance_from_zone_atr

    trigger_high = float(candle_story.get("high", 0.0))
    trigger_low = float(candle_story.get("low", 0.0))
    trigger_range = trigger_high - trigger_low
    body_pct = float(candle_story.get("body_pct", 0.0))

    is_climax_bar = bool(candle_story.get("available") and (trigger_range > 2.0 * atr_15m) and (body_pct > 0.65))
    is_overextended_move = bool(zone_distance > 0.85 * atr_1h)

    # ── 8. Gate 2 Setup Score (0–100) & Safety Rules ──────────────────────────
    score_structure = 0
    if structure_event in ("HH_HL", "LH_LL", "BOS_BULLISH", "BOS_BEARISH"):
        score_structure = 20
    elif structure_event == "EMA_TREND":
        score_structure = 15

    if (trend == "UPTREND" and trend_4h == "BULLISH") or (trend == "DOWNTREND" and trend_4h == "BEARISH"):
        score_structure += 5
    score_structure = min(score_structure, 25)

    score_value_area = 0
    if near_static_zone:
        score_value_area = 20
        aligned_zone = nearest_support if trend == "UPTREND" else nearest_resistance
        if aligned_zone and aligned_zone.get("touch_count", 0) >= 3:
            score_value_area += 5
    elif near_dynamic_ema50 or result.get("in_fib_golden_zone", False):
        score_value_area = 15
    score_value_area = min(score_value_area, 25)

    score_price_action = 0
    if candle_quality == "STRONG":
        score_price_action = 20
    elif candle_quality == "MODERATE":
        score_price_action = 10

    score_ema_momentum = 0
    if (trend == "UPTREND" and result["ema_signal"] == "BULLISH") or (trend == "DOWNTREND" and result["ema_signal"] == "BEARISH"):
        score_ema_momentum += 10
    rsi_val = result["rsi_14_15m"]
    if trend == "UPTREND" and (35.0 <= rsi_val <= 68.0):
        score_ema_momentum += 5
    elif trend == "DOWNTREND" and (32.0 <= rsi_val <= 65.0):
        score_ema_momentum += 5
    score_ema_momentum = min(score_ema_momentum, 15)

    score_fib = 10 if result.get("in_fib_golden_zone", False) else 0

    rr = risk_metrics.get("risk_reward_ratio", 0.0)
    min_rr = getattr(config, "get_min_risk_reward", lambda: config.MIN_RISK_REWARD)()
    if rr >= 2.0:
        score_rr = 5
    elif rr >= min_rr:
        score_rr = 3
    else:
        score_rr = 0

    score_breakdown = {
        "structure":    score_structure,
        "value_area":   score_value_area,
        "price_action": score_price_action,
        "ema_momentum": score_ema_momentum,
        "fibonacci":    score_fib,
        "risk_reward":  score_rr,
    }
    math_setup_score = min(sum(score_breakdown.values()), 100)
    result["math_setup_score"] = math_setup_score
    result["score_breakdown"] = score_breakdown

    min_math_score = config.get_min_math_score() if hasattr(config, "get_min_math_score") else (55 if (config.DEMO_MODE and getattr(config, "DEMO_DISCOVERY_MODE", False)) else 65)
    result["min_math_score"] = min_math_score

    # Hard Safety Rules (all must pass)
    all_hard_rules_pass = True
    if trend == "RANGING":
        all_hard_rules_pass = False
        result["fail_reasons"].append("1H trend is RANGING — no directional bias.")

    if not in_value_area:
        all_hard_rules_pass = False
        result["fail_reasons"].append("Price is not in a value area (S/R zone, Fib golden zone, or dynamic EMA50) — no edge.")

    if is_overextended_move or is_climax_bar:
        all_hard_rules_pass = False
        result["fail_reasons"].append("Entry is overextended away from value area — chasing late momentum.")

    if rr < min_rr:
        all_hard_rules_pass = False
        result["fail_reasons"].append(f"R:R ratio {rr:.2f} < minimum {min_rr}.")

    if risk_metrics.get("risk_cap_exceeded", False):
        all_hard_rules_pass = False
        actual_risk_val = risk_metrics.get("actual_risk_usd", 0.0)
        max_risk_val = risk_metrics.get("max_risk_usd", 0.0)
        result["fail_reasons"].append(
            f"Position risk cap exceeded: minimum 0.01 lot risk ${actual_risk_val:.2f} > max risk (${max_risk_val:.2f})."
        )

    if math_setup_score < min_math_score:
        result["fail_reasons"].append(f"Gate 2 setup score {math_setup_score}/100 < minimum {min_math_score}.")

    passed_math_gate = all_hard_rules_pass and (math_setup_score >= min_math_score)
    result["passed_math_gate"] = passed_math_gate

    actual_risk_usd = risk_metrics.get("actual_risk_usd", 0.0)
    print(
        f"  [market_math] {symbol} | Struct: {structure_event} | 4H: {trend_4h} | "
        f"Area: {result['value_area_type']} | Dist: {distance_from_zone_atr} ATR | Candle: {candle_quality} | "
        f"Score: {math_setup_score}/100 | Risk: ${actual_risk_usd}"
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Market Structure & 4H Trend Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_swings(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 3,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Detect swing highs and swing lows using a local pivot algorithm.

    Returns:
        (swing_highs, swing_lows) as lists of (int_idx, float_price) tuples
        preserving chronological order.
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    swing_highs: list[tuple[int, float]] = []
    swing_lows:  list[tuple[int, float]] = []

    for i in range(left, n - right):
        window_h = highs[i - left : i + right + 1]
        window_l = lows[i  - left : i + right + 1]

        if highs[i] == window_h.max():
            swing_highs.append((i, float(highs[i])))
        if lows[i] == window_l.min():
            swing_lows.append((i, float(lows[i])))

    return swing_highs, swing_lows


def _classify_trend(
    swing_highs: list[tuple[int, float]],
    swing_lows: list[tuple[int, float]],
    current_price: float,
    ema50_series: pd.Series,
    ema200_series: pd.Series,
    atr_1h: float,
) -> tuple[str, str]:
    """
    Classify trend and market structure event using:
      1. Clean Structure (HH_HL / LH_LL)
      2. Break of Structure (BOS_BULLISH / BOS_BEARISH)
      3. Strong EMA Trend Fallback (EMA_TREND)
      4. Fallback: RANGING / NONE
    """
    ema50_last = float(ema50_series.iloc[-1])
    ema200_last = float(ema200_series.iloc[-1])
    ema50_prev10 = float(ema50_series.iloc[-11]) if len(ema50_series) >= 11 else float(ema50_series.iloc[0])

    # 1. Clean Structure
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-1][1] > swing_highs[-2][1]
        hl = swing_lows[-1][1] > swing_lows[-2][1]
        lh = swing_highs[-1][1] < swing_highs[-2][1]
        ll = swing_lows[-1][1] < swing_lows[-2][1]
        if hh and hl:
            return "UPTREND", "HH_HL"
        if lh and ll:
            return "DOWNTREND", "LH_LL"

    # 2. Break of Structure (BOS) — closed 1H break past last 2 swings + directional EMA separation
    if len(swing_highs) >= 2:
        prev_high_cap = max(swing_highs[-1][1], swing_highs[-2][1])
        if (
            current_price > prev_high_cap + (0.25 * atr_1h)
            and current_price > ema50_last
            and ema50_last > ema200_last
            and (ema50_last - ema200_last) >= 0.35 * atr_1h
        ):
            return "UPTREND", "BOS_BULLISH"
    if len(swing_lows) >= 2:
        prev_low_floor = min(swing_lows[-1][1], swing_lows[-2][1])
        if (
            current_price < prev_low_floor - (0.25 * atr_1h)
            and current_price < ema50_last
            and ema50_last < ema200_last
            and (ema200_last - ema50_last) >= 0.35 * atr_1h
        ):
            return "DOWNTREND", "BOS_BEARISH"

    # 3. Strong EMA Trend Fallback
    if (
        ema50_last > ema200_last
        and (ema50_last - ema200_last) >= 0.5 * atr_1h
        and current_price >= (ema50_last - 0.5 * atr_1h)
        and ema50_last > ema50_prev10
    ):
        return "UPTREND", "EMA_TREND"

    if (
        ema50_last < ema200_last
        and (ema200_last - ema50_last) >= 0.5 * atr_1h
        and current_price <= (ema50_last + 0.5 * atr_1h)
        and ema50_last < ema50_prev10
    ):
        return "DOWNTREND", "EMA_TREND"

    return "RANGING", "NONE"


def _compute_trend_4h(df_1h: pd.DataFrame) -> str:
    """
    Resample 1H candles to 4H bars and check EMA 20 position.
    Returns: 'BULLISH', 'BEARISH', or 'UNKNOWN'.
    """
    try:
        df_copy = df_1h.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = pd.to_datetime(df_copy.index)
        df_4h = df_copy.resample("4h").agg({
            "open":  "first",
            "high":  "max",
            "low":   "min",
            "close": "last",
        }).dropna()

        df_4h_closed = df_4h.iloc[:-1]
        if len(df_4h_closed) < 20:
            return "UNKNOWN"

        ema20_4h = _ema(df_4h_closed["close"], 20)
        c_last = float(df_4h_closed["close"].iloc[-1])
        e_last = float(ema20_4h.iloc[-1])

        if c_last > e_last:
            return "BULLISH"
        elif c_last < e_last:
            return "BEARISH"
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# 5. S/R Zone Clustering & Fibonacci Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_zones(
    pivots: list[float],
    tolerance_pips: float = 0.0010,
) -> list[dict[str, Any]]:
    """
    Group pivot prices into zones anchored at the cluster start (current[0])
    so zones never chain into oversized bands. Returns zones with >= 2 touches.
    """
    if not pivots:
        return []

    sorted_pivots = sorted(pivots)
    clusters: list[list[float]] = []
    current: list[float] = [sorted_pivots[0]]

    for price in sorted_pivots[1:]:
        if price - current[0] <= tolerance_pips:
            current.append(price)
        else:
            clusters.append(current)
            current = [price]
    clusters.append(current)

    zones = []
    for cluster in clusters:
        touch_count = len(cluster)
        if touch_count >= 2:
            zone_mid = round(float(np.mean(cluster)), 5)
            zones.append({
                "price":       zone_mid,
                "touch_count": touch_count,
                "zone_low":    round(float(min(cluster)), 5),
                "zone_high":   round(float(max(cluster)), 5),
            })

    return zones


def _nearest_below(zones: list[dict], price: float) -> dict[str, Any] | None:
    """Return the nearest S/R zone strictly below `price`."""
    candidates = [z for z in zones if z["price"] < price]
    if not candidates:
        return None
    return max(candidates, key=lambda z: (z["price"], z["touch_count"]))


def _nearest_above(zones: list[dict], price: float) -> dict[str, Any] | None:
    """Return the nearest S/R zone strictly above `price`."""
    candidates = [z for z in zones if z["price"] > price]
    if not candidates:
        return None
    return min(candidates, key=lambda z: (z["price"], -z["touch_count"]))


def _is_near_zone(
    price: float,
    support: dict | None,
    resistance: dict | None,
    proximity_pips: float = 0.0030,
    trend: str = "RANGING",
) -> bool:
    """True if price is within `proximity_pips` of the trend-aligned S/R zone."""
    if trend == "UPTREND":
        return bool(support and abs(price - support["price"]) <= proximity_pips)
    if trend == "DOWNTREND":
        return bool(resistance and abs(resistance["price"] - price) <= proximity_pips)
    if support and abs(price - support["price"]) <= proximity_pips:
        return True
    if resistance and abs(resistance["price"] - price) <= proximity_pips:
        return True
    return False


def _fibonacci_zones(
    swing_highs: list[tuple[int, float]],
    swing_lows: list[tuple[int, float]],
    trend: str,
) -> dict[str, Any]:
    """
    Calculate Fibonacci retracement levels from the most recent chronological impulse leg.

    For UPTREND  : low_idx < high_idx
    For DOWNTREND: high_idx < low_idx
    Golden Zone is always strictly ordered: golden_zone_low < golden_zone_high.
    """
    if not swing_highs or not swing_lows:
        return {"available": False, "reason": "Insufficient swing data."}

    swing_start: float | None = None
    swing_end: float | None = None

    if trend == "UPTREND":
        for h_idx, h_p in reversed(swing_highs):
            valid_lows = [(l_idx, l_p) for l_idx, l_p in swing_lows if l_idx < h_idx]
            if valid_lows:
                swing_start = valid_lows[-1][1]
                swing_end = h_p
                break
    elif trend == "DOWNTREND":
        for l_idx, l_p in reversed(swing_lows):
            valid_highs = [(h_idx, h_p) for h_idx, h_p in swing_highs if h_idx < l_idx]
            if valid_highs:
                swing_start = valid_highs[-1][1]
                swing_end = l_p
                break
    else:
        return {"available": False, "reason": "RANGING — Fibonacci not applicable."}

    if swing_start is None or swing_end is None or swing_start == swing_end:
        return {"available": False, "reason": "No valid chronological impulse leg."}

    leg = swing_end - swing_start
    fib_38_2 = round(swing_end - 0.382 * leg, 5)
    fib_50_0 = round(swing_end - 0.500 * leg, 5)
    fib_61_8 = round(swing_end - 0.618 * leg, 5)

    gz_low = min(fib_50_0, fib_61_8)
    gz_high = max(fib_50_0, fib_61_8)
    if gz_low == gz_high:
        gz_high = round(gz_low + 0.00001, 5)

    return {
        "available":        True,
        "swing_start":      round(swing_start, 5),
        "swing_end":        round(swing_end,   5),
        "fib_38_2":         fib_38_2,
        "fib_50_0":         fib_50_0,
        "fib_61_8":         fib_61_8,
        "golden_zone_low":  gz_low,
        "golden_zone_high": gz_high,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Technical Indicators & Candle Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average — pure pandas."""
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing) — pure pandas."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _analyse_candle(df: pd.DataFrame, candle_index: int = -1) -> dict[str, Any]:
    """
    Analyse a single closed 15m candle for anatomical ratios and candlestick patterns.
    Enforces strict opposite-color and real body engulfing detection.
    """
    if len(df) < abs(candle_index):
        return {"available": False, "reason": "Not enough candles."}

    candle = df.iloc[candle_index]
    o = float(candle["open"])
    h = float(candle["high"])
    l = float(candle["low"])
    c = float(candle["close"])
    full_range = h - l if h != l else 1e-10

    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    is_bullish = c > o
    body_pct = body / full_range
    upper_pct = upper_wick / full_range
    lower_pct = lower_wick / full_range

    # Pattern detection
    pattern = "NEUTRAL"
    if body_pct < 0.1:
        pattern = "DOJI"
    elif lower_pct >= 0.6 and is_bullish:
        pattern = "BULLISH_HAMMER"
    elif lower_pct >= 0.6 and not is_bullish:
        pattern = "BEARISH_HAMMER_ON_RESISTANCE"
    elif upper_pct >= 0.6 and not is_bullish:
        pattern = "BEARISH_SHOOTING_STAR"
    elif upper_pct >= 0.6 and is_bullish:
        pattern = "BULLISH_WITH_UPPER_REJECTION"
    elif body_pct >= 0.7 and is_bullish:
        pattern = "STRONG_BULLISH_MARUBOZU"
    elif body_pct >= 0.7 and not is_bullish:
        pattern = "STRONG_BEARISH_MARUBOZU"

    # Strict Engulfing: opposite candle colors AND real open/close engulfment
    engulfing = "NONE"
    if len(df) >= abs(candle_index) + 1:
        prev = df.iloc[candle_index - 1]
        prev_o = float(prev["open"])
        prev_c = float(prev["close"])
        if prev_c < prev_o and c > o and o <= prev_c and c >= prev_o:
            engulfing = "BULLISH_ENGULFING"
        elif prev_c > prev_o and c < o and o >= prev_c and c <= prev_o:
            engulfing = "BEARISH_ENGULFING"

    return {
        "available":      True,
        "timestamp":      str(df.index[candle_index]),
        "open":           round(o, 5),
        "high":           round(h, 5),
        "low":            round(l, 5),
        "close":          round(c, 5),
        "is_bullish":     is_bullish,
        "body_pct":       round(body_pct,  3),
        "upper_wick_pct": round(upper_pct, 3),
        "lower_wick_pct": round(lower_pct, 3),
        "pattern":        pattern,
        "engulfing":      engulfing,
    }


def _evaluate_candle_quality(candle_story: dict[str, Any], trend: str) -> str:
    """
    Compute directional candle quality: 'STRONG', 'MODERATE', or 'NONE'.
    """
    if trend == "RANGING" or not candle_story.get("available", False):
        return "NONE"

    is_bullish = candle_story.get("is_bullish", False)
    body_pct = float(candle_story.get("body_pct", 0.0))
    upper_wick_pct = float(candle_story.get("upper_wick_pct", 0.0))
    lower_wick_pct = float(candle_story.get("lower_wick_pct", 0.0))
    engulfing = candle_story.get("engulfing", "NONE")

    if trend == "UPTREND":
        if is_bullish and (body_pct >= 0.45 or lower_wick_pct >= 0.55 or engulfing == "BULLISH_ENGULFING"):
            return "STRONG"
        elif is_bullish and (body_pct >= 0.25 or lower_wick_pct >= 0.40):
            return "MODERATE"
        return "NONE"
    elif trend == "DOWNTREND":
        if (not is_bullish) and (body_pct >= 0.45 or upper_wick_pct >= 0.55 or engulfing == "BEARISH_ENGULFING"):
            return "STRONG"
        elif (not is_bullish) and (body_pct >= 0.25 or upper_wick_pct >= 0.40):
            return "MODERATE"
        return "NONE"

    return "NONE"


def _last_n_candles(df: pd.DataFrame, n: int = 10) -> list[dict]:
    """Return the last N closed candles as a list of dicts for the AI prompt."""
    tail = df.iloc[-(n + 1):-1]  # exclude the live forming candle
    records = []
    for ts, row in tail.iterrows():
        records.append({
            "time":    str(ts),
            "open":    round(float(row["open"]),  5),
            "high":    round(float(row["high"]),  5),
            "low":     round(float(row["low"]),   5),
            "close":   round(float(row["close"]), 5),
            "bullish": bool(row["close"] > row["open"]),
        })
    return records


def _get_session_tag() -> str:
    """
    Categorise current market session using real Europe/London and America/New_York
    timezones (automatically DST-aware):
      - LONDON_NY_OVERLAP: London 07:00–17:00 and NY 08:00–17:00
      - LONDON_OPEN:       London 07:00–17:00 before NY open (NY < 08:00)
      - NY_AFTERNOON:      NY 08:00–16:00 after London close (London >= 17:00)
      - OFF_HOURS:         otherwise (including on exception)
    """
    try:
        import pytz
        from datetime import datetime, time as dtime
        now_utc = datetime.now(pytz.utc)

        london_tz = pytz.timezone("Europe/London")
        ny_tz = pytz.timezone("America/New_York")

        london_time = now_utc.astimezone(london_tz).time()
        ny_time = now_utc.astimezone(ny_tz).time()

        in_london = dtime(7, 0) <= london_time < dtime(17, 0)
        in_ny_overlap = dtime(8, 0) <= ny_time < dtime(17, 0)
        in_ny_afternoon = dtime(8, 0) <= ny_time < dtime(16, 0)

        if in_london and in_ny_overlap:
            return "LONDON_NY_OVERLAP"
        elif in_london and ny_time < dtime(8, 0):
            return "LONDON_OPEN"
        elif (not in_london or london_time >= dtime(17, 0)) and in_ny_afternoon:
            return "NY_AFTERNOON"
        return "OFF_HOURS"
    except Exception:
        return "OFF_HOURS"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Position Sizing, Pair-Aware Pip Value & Risk Calculation
# ─────────────────────────────────────────────────────────────────────────────

def _get_pip_value_usd(symbol: str, current_price: float) -> float:
    """
    Return exact live USD value of 1 pip for 1 standard lot (100,000 units)
    using live cross-rate algebra from scanned majors (zero extra API calls).
    """
    clean_sym = symbol.split(" ")[0].upper()
    parts = clean_sym.split("/")
    if len(parts) != 2:
        return 10.0

    base, quote = parts[0], parts[1]
    pip_mult = 1000.0 if quote == "JPY" else 10.0

    if quote == "USD":
        return 10.0
    if base == "USD" and current_price > 0:
        return pip_mult / current_price

    # Live cross derivation: e.g. GBP/AUD -> 10 * (GBP/USD / GBP/AUD); EUR/JPY -> 1000 * (EUR/USD / EUR/JPY)
    base_usd_key = f"{base}/USD"
    if base_usd_key in _LATEST_FX_RATES and current_price > 0:
        return pip_mult * (_LATEST_FX_RATES[base_usd_key] / current_price)
    elif base_usd_key in _FALLBACK_FX_RATES and current_price > 0:
        fallback_rate = _FALLBACK_FX_RATES[base_usd_key]
        print(f"  [market_math] ⚠️ Fallback FX rate used for {base_usd_key} ({fallback_rate}).")
        return pip_mult * (fallback_rate / current_price)

    if quote == "JPY":
        usd_jpy = _LATEST_FX_RATES.get("USD/JPY")
        if usd_jpy is None:
            usd_jpy = _FALLBACK_FX_RATES.get("USD/JPY", 159.0)
            print(f"  [market_math] ⚠️ Fallback FX rate used for USD/JPY ({usd_jpy}).")
        return 1000.0 / usd_jpy if usd_jpy > 0 else 10.0

    if quote == "AUD":
        aud_usd = _LATEST_FX_RATES.get("AUD/USD")
        if aud_usd is None:
            aud_usd = _FALLBACK_FX_RATES.get("AUD/USD", 0.657)
            print(f"  [market_math] ⚠️ Fallback FX rate used for AUD/USD ({aud_usd}).")
        return 10.0 * aud_usd

    if quote == "NZD":
        nzd_usd = _LATEST_FX_RATES.get("NZD/USD")
        if nzd_usd is None:
            nzd_usd = _FALLBACK_FX_RATES.get("NZD/USD", 0.550)
            print(f"  [market_math] ⚠️ Fallback FX rate used for NZD/USD ({nzd_usd}).")
        return 10.0 * nzd_usd

    if quote == "CHF":
        usd_chf = _LATEST_FX_RATES.get("USD/CHF")
        if usd_chf is None:
            usd_chf = _FALLBACK_FX_RATES.get("USD/CHF", 0.880)
            print(f"  [market_math] ⚠️ Fallback FX rate used for USD/CHF ({usd_chf}).")
        return 10.0 / usd_chf if usd_chf > 0 else 10.0

    return 10.0


def _calculate_risk(
    symbol: str,
    trend: str,
    current_price: float,
    nearest_support: dict | None,
    nearest_resistance: dict | None,
    near_static_zone: bool,
    in_value_area: bool,
    atr_15m: float,
    atr_1h: float,
    ema50_last: float,
    df_15m: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate stop-loss, take-profit, R:R ratio, and position size.
    Enforces pair-aware pip values, structural SL/TP, and a strict 1.0x risk cap.
    """
    pip_size = 0.01 if "JPY" in symbol else 0.0001
    max_risk_usd = round(config.ACCOUNT_BALANCE * config.MAX_RISK_PERCENT, 2)
    atr_buffer = max(0.75 * atr_1h, 1.5 * atr_15m) + (2 * pip_size)
    pip_value_usd = _get_pip_value_usd(symbol, current_price)

    # Enforce pair-aware minimum SL distance floor
    clean_sym = symbol.split(" ")[0].upper()
    if clean_sym in ("EUR/JPY", "GBP/JPY", "GBP/AUD", "GBP/NZD"):
        min_sl_pips = 28.0
    elif clean_sym in ("EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD"):
        min_sl_pips = 18.0
    else:
        min_sl_pips = 18.0

    min_sl_distance = min_sl_pips * pip_size

    closed_15m = df_15m.iloc[:-1]
    recent_slice = closed_15m.iloc[-6:] if len(closed_15m) >= 6 else closed_15m
    recent_high = float(recent_slice["high"].max())
    recent_low = float(recent_slice["low"].min())

    if trend == "UPTREND" and in_value_area:
        direction = "LONG"
        entry = current_price

        # SL Anchor
        if near_static_zone and nearest_support:
            stop_loss = nearest_support["zone_low"] - atr_buffer
        else:
            stop_loss = min(recent_low, ema50_last) - atr_buffer

        risk_distance = entry - stop_loss
        if risk_distance < min_sl_distance:
            stop_loss = entry - min_sl_distance
            risk_distance = min_sl_distance

        # TP Placement
        if nearest_resistance and (nearest_resistance["price"] - entry) >= 1.0 * atr_1h:
            take_profit = nearest_resistance["price"] - (2 * pip_size)
            tp_source = "SR_ZONE"
        else:
            fallback_dist = min(2.2 * risk_distance, 3.5 * atr_1h)
            take_profit = entry + fallback_dist
            tp_source = "ATR_STRUCTURAL"

    elif trend == "DOWNTREND" and in_value_area:
        direction = "SHORT"
        entry = current_price

        # SL Anchor
        if near_static_zone and nearest_resistance:
            stop_loss = nearest_resistance["zone_high"] + atr_buffer
        else:
            stop_loss = max(recent_high, ema50_last) + atr_buffer

        risk_distance = stop_loss - entry
        if risk_distance < min_sl_distance:
            stop_loss = entry + min_sl_distance
            risk_distance = min_sl_distance

        # TP Placement
        if nearest_support and (entry - nearest_support["price"]) >= 1.0 * atr_1h:
            take_profit = nearest_support["price"] + (2 * pip_size)
            tp_source = "SR_ZONE"
        else:
            fallback_dist = min(2.2 * risk_distance, 3.5 * atr_1h)
            take_profit = entry - fallback_dist
            tp_source = "ATR_STRUCTURAL"

    else:
        # No valid trade setup
        return {
            "account_balance":          config.ACCOUNT_BALANCE,
            "max_risk_usd":             max_risk_usd,
            "atr_buffer":               round(atr_buffer, 5),
            "direction":                "NONE",
            "entry_price":              None,
            "stop_loss":                None,
            "take_profit":              None,
            "risk_pips":                None,
            "reward_pips":              None,
            "risk_reward_ratio":        0.0,
            "position_size_lots":       None,
            "position_size_micro_lots": None,
            "position_size_units":      None,
            "pip_value_usd":            round(pip_value_usd, 4),
            "actual_risk_usd":          0.0,
            "risk_cap_exceeded":        False,
            "tp_source":                "NONE",
        }

    # Reward & Sizing Calculations
    if direction == "LONG":
        reward_distance = take_profit - entry
    else:
        reward_distance = entry - take_profit

    reward_distance = max(reward_distance, 0.0)
    rr_ratio = round(reward_distance / max(risk_distance, 1e-5), 2)

    pips_at_risk = risk_distance / pip_size
    raw_lots = max_risk_usd / (pips_at_risk * pip_value_usd)
    floored_lots = math.floor(raw_lots * 100.0) / 100.0
    std_lots = max(floored_lots, 0.01)
    position_size_units = int(round(std_lots * 100_000))
    micro_lots = round(std_lots * 100.0, 2)

    actual_risk_usd = round(std_lots * pips_at_risk * pip_value_usd, 2)
    risk_cap_exceeded = bool(actual_risk_usd > max_risk_usd)

    return {
        "account_balance":          config.ACCOUNT_BALANCE,
        "max_risk_usd":             max_risk_usd,
        "atr_buffer":               round(atr_buffer, 5),
        "direction":                direction,
        "entry_price":              round(entry,       5),
        "stop_loss":                round(stop_loss,   5),
        "take_profit":              round(take_profit, 5),
        "risk_pips":                round(risk_distance / pip_size,   1),
        "reward_pips":              round(reward_distance / pip_size, 1),
        "risk_reward_ratio":        rr_ratio,
        "position_size_lots":       std_lots,
        "position_size_micro_lots": micro_lots,
        "position_size_units":      int(position_size_units),
        "pip_value_usd":            round(pip_value_usd, 4),
        "actual_risk_usd":          actual_risk_usd,
        "risk_cap_exceeded":        risk_cap_exceeded,
        "tp_source":                tp_source,
    }
