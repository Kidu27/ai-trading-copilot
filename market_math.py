"""
market_math.py — Gate 2: Deterministic Math & Technical Indicators

ALL numbers are computed here with pure pandas/numpy.
The LLM (Gate 3) only receives the final JSON dict from this module.

Gate 2 logic:
  1. 1H Market Structure  → trend classification
  2. S/R Zone Clustering  → support, resistance, role-reversal zones
  3. 15m Candle Story     → most recently CLOSED candle analysis
  4. V2 Indicators        → EMA 50/200, Fibonacci Golden Zone, ATR(14)
  5. Risk & Position Size → hard R:R gate, lot sizing

If any hard rule fails, `passed_math_gate` is set to False.
"""

import numpy as np
import pandas as pd
from typing import Any

import config


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyse(
    symbol: str,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run all Gate 2 deterministic checks.

    Args:
        symbol: e.g. "EUR/USD"
        df_1h:  100-candle 1H DataFrame (oldest → newest)
        df_15m: 50-candle 15m DataFrame (oldest → newest)

    Returns:
        JSON-serialisable dict with all metrics + `passed_math_gate` bool.
    """
    result: dict[str, Any] = {"symbol": symbol, "passed_math_gate": True, "fail_reasons": []}

    # ── 1. 1H Market Structure ────────────────────────────────────────────────
    swing_highs, swing_lows = _find_swings(df_1h)
    trend      = _classify_trend(swing_highs, swing_lows)
    result["trend_1h"]    = trend
    result["swing_highs"] = [round(p, 5) for p in swing_highs[-6:]]
    result["swing_lows"]  = [round(p, 5) for p in swing_lows[-6:]]

    if trend == "RANGING":
        result["passed_math_gate"] = False
        result["fail_reasons"].append("1H trend is RANGING — no directional bias.")

    # ── 2. S/R Zone Clustering ────────────────────────────────────────────────
    current_price = float(df_1h["close"].iloc[-1])
    result["current_price"] = round(current_price, 5)

    all_pivots = list(swing_highs) + list(swing_lows)
    zones      = _cluster_zones(all_pivots, tolerance_pips=0.0010)
    result["sr_zones"] = zones

    nearest_support    = _nearest_below(zones, current_price)
    nearest_resistance = _nearest_above(zones, current_price)
    result["nearest_support"]    = nearest_support
    result["nearest_resistance"] = nearest_resistance

    near_zone = _is_near_zone(current_price, nearest_support, nearest_resistance)
    result["price_near_key_zone"] = near_zone
    if not near_zone:
        result["passed_math_gate"] = False
        result["fail_reasons"].append("Price is not near a key S/R zone — no edge.")

    # ── 3. V2 Indicators (EMA, Fibonacci, ATR) ────────────────────────────────
    if config.USE_V2_INDICATORS:
        ema50  = _ema(df_1h["close"], 50)
        ema200 = _ema(df_1h["close"], 200)
        result["ema50_1h"]  = round(float(ema50.iloc[-1]),  5)
        result["ema200_1h"] = round(float(ema200.iloc[-1]), 5)
        result["ema_signal"] = (
            "BULLISH" if ema50.iloc[-1] > ema200.iloc[-1] else "BEARISH"
        )

        fib = _fibonacci_zones(swing_highs, swing_lows, trend)
        result["fibonacci"] = fib

        atr_15m = _atr(df_15m, period=14)
        result["atr_15m"] = round(float(atr_15m.iloc[-1]), 5)
    else:
        result["atr_15m"] = round(float(_atr(df_15m, period=14).iloc[-1]), 5)

    # ── 4. 15m Candle Story ───────────────────────────────────────────────────
    # Use -2 index: last CLOSED candle (index -1 may still be forming)
    candle_story = _analyse_candle(df_15m, candle_index=-2)
    result["candle_story"] = candle_story

    last_10_candles = _last_n_candles(df_15m, n=10)
    result["last_10_candles_15m"] = last_10_candles

    # ── 5. Risk & Position Sizing (Hard Gate) ────────────────────────────────
    risk_metrics = _calculate_risk(
        symbol=symbol,
        trend=trend,
        current_price=current_price,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        atr=result["atr_15m"],
    )
    result.update(risk_metrics)

    if risk_metrics.get("risk_reward_ratio", 0) < config.MIN_RISK_REWARD:
        result["passed_math_gate"] = False
        result["fail_reasons"].append(
            f"R:R ratio {risk_metrics.get('risk_reward_ratio', 0):.2f} < "
            f"minimum {config.MIN_RISK_REWARD}."
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1. Market Structure
# ─────────────────────────────────────────────────────────────────────────────

def _find_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect swing highs and swing lows using a local pivot algorithm.

    A swing high at index i: df['high'][i] is the max over [i-left .. i+right].
    A swing low  at index i: df['low'][i]  is the min over [i-left .. i+right].
    """
    highs  = df["high"].values
    lows   = df["low"].values
    n      = len(df)

    swing_high_prices: list[float] = []
    swing_low_prices:  list[float] = []

    for i in range(left, n - right):
        window_h = highs[i - left : i + right + 1]
        window_l = lows[i  - left : i + right + 1]

        if highs[i] == window_h.max():
            swing_high_prices.append(highs[i])
        if lows[i] == window_l.min():
            swing_low_prices.append(lows[i])

    return np.array(swing_high_prices), np.array(swing_low_prices)


def _classify_trend(swing_highs: np.ndarray, swing_lows: np.ndarray) -> str:
    """
    Classify trend using Higher Highs / Higher Lows (HH/HL) or LH/LL logic.

    Rules:
      UPTREND   — last 2 swing highs are rising AND last 2 swing lows are rising.
      DOWNTREND — last 2 swing highs are falling AND last 2 swing lows are falling.
      RANGING   — anything else.
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "RANGING"

    hh = swing_highs[-1] > swing_highs[-2]   # Higher High
    hl = swing_lows[-1]  > swing_lows[-2]    # Higher Low
    lh = swing_highs[-1] < swing_highs[-2]   # Lower High
    ll = swing_lows[-1]  < swing_lows[-2]    # Lower Low

    if hh and hl:
        return "UPTREND"
    if lh and ll:
        return "DOWNTREND"
    return "RANGING"


# ─────────────────────────────────────────────────────────────────────────────
# 2. S/R Zone Clustering
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_zones(
    pivots: list[float],
    tolerance_pips: float = 0.0010,
) -> list[dict[str, Any]]:
    """
    Group pivot prices that are within `tolerance_pips` of each other
    into zones. Returns zones with >= 2 touches (meaningful S/R).
    """
    if not pivots:
        return []

    sorted_pivots = sorted(pivots)
    clusters: list[list[float]] = []
    current: list[float] = [sorted_pivots[0]]

    for price in sorted_pivots[1:]:
        if price - current[-1] <= tolerance_pips:
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
    """Return the strongest zone strictly below `price`."""
    candidates = [z for z in zones if z["price"] < price]
    if not candidates:
        return None
    return max(candidates, key=lambda z: (z["touch_count"], z["price"]))


def _nearest_above(zones: list[dict], price: float) -> dict[str, Any] | None:
    """Return the strongest zone strictly above `price`."""
    candidates = [z for z in zones if z["price"] > price]
    if not candidates:
        return None
    return max(candidates, key=lambda z: (z["touch_count"], -z["price"]))


def _is_near_zone(
    price: float,
    support: dict | None,
    resistance: dict | None,
    proximity_pips: float = 0.0030,
) -> bool:
    """True if price is within `proximity_pips` of either zone."""
    if support and abs(price - support["price"]) <= proximity_pips:
        return True
    if resistance and abs(resistance["price"] - price) <= proximity_pips:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 3. V2 Indicators
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average — pure pandas."""
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (Wilder smoothing).

    TR = max(H-L, |H-prev_C|, |L-prev_C|)
    ATR = Wilder smoothed TR (equivalent to EMA with alpha=1/period)
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Wilder smoothing = EMA with span = 2*period - 1
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _fibonacci_zones(
    swing_highs: np.ndarray,
    swing_lows: np.ndarray,
    trend: str,
) -> dict[str, Any]:
    """
    Calculate Fibonacci retracement levels from the most recent swing leg.

    For UPTREND  : leg = most recent swing low  → swing high
    For DOWNTREND: leg = most recent swing high → swing low

    Returns the 38.2%, 50%, 61.8% (Golden Zone) retracement prices.
    """
    if len(swing_highs) == 0 or len(swing_lows) == 0:
        return {"available": False, "reason": "Insufficient swing data."}

    if trend == "UPTREND":
        swing_start = float(swing_lows[-1])
        swing_end   = float(swing_highs[-1])
    elif trend == "DOWNTREND":
        swing_start = float(swing_highs[-1])
        swing_end   = float(swing_lows[-1])
    else:
        return {"available": False, "reason": "RANGING — Fibonacci not applicable."}

    leg = swing_end - swing_start

    return {
        "available":    True,
        "swing_start":  round(swing_start, 5),
        "swing_end":    round(swing_end,   5),
        "fib_38_2":     round(swing_end - 0.382 * leg, 5),
        "fib_50_0":     round(swing_end - 0.500 * leg, 5),
        "fib_61_8":     round(swing_end - 0.618 * leg, 5),
        "golden_zone_low":  round(swing_end - 0.618 * leg, 5),
        "golden_zone_high": round(swing_end - 0.500 * leg, 5),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. 15m Candle Story
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_candle(df: pd.DataFrame, candle_index: int = -2) -> dict[str, Any]:
    """
    Analyse a single closed 15m candle for significant patterns.

    Candle anatomy:
      body      = abs(close - open)
      upper_wick = high - max(open, close)
      lower_wick = min(open, close) - low
      full_range = high - low

    Returns metrics and human-readable pattern label.
    """
    if len(df) < abs(candle_index):
        return {"available": False, "reason": "Not enough candles."}

    candle   = df.iloc[candle_index]
    o, h, l, c = float(candle["open"]), float(candle["high"]), float(candle["low"]), float(candle["close"])
    full_range  = h - l if h != l else 1e-10

    body        = abs(c - o)
    upper_wick  = h - max(o, c)
    lower_wick  = min(o, c) - l
    is_bullish  = c > o
    body_pct    = body / full_range
    upper_pct   = upper_wick / full_range
    lower_pct   = lower_wick / full_range

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

    # Check for engulfing vs prior candle
    engulfing = "NONE"
    if len(df) > abs(candle_index) + 1:
        prev        = df.iloc[candle_index - 1]
        prev_body   = abs(float(prev["close"]) - float(prev["open"]))
        if body > prev_body:
            engulfing = "BULLISH_ENGULFING" if is_bullish else "BEARISH_ENGULFING"

    return {
        "available":    True,
        "timestamp":    str(df.index[candle_index]),
        "open":         round(o, 5),
        "high":         round(h, 5),
        "low":          round(l, 5),
        "close":        round(c, 5),
        "is_bullish":   is_bullish,
        "body_pct":     round(body_pct,   3),
        "upper_wick_pct": round(upper_pct, 3),
        "lower_wick_pct": round(lower_pct, 3),
        "pattern":      pattern,
        "engulfing":    engulfing,
    }


def _last_n_candles(df: pd.DataFrame, n: int = 10) -> list[dict]:
    """Return the last N closed candles as a list of dicts for the AI prompt."""
    tail = df.iloc[-(n + 1):-1]  # exclude the live forming candle
    records = []
    for ts, row in tail.iterrows():
        records.append({
            "time":  str(ts),
            "open":  round(float(row["open"]),  5),
            "high":  round(float(row["high"]),  5),
            "low":   round(float(row["low"]),   5),
            "close": round(float(row["close"]), 5),
            "bullish": bool(row["close"] > row["open"]),
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 5. Risk & Position Sizing
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_risk(
    symbol: str,
    trend: str,
    current_price: float,
    nearest_support: dict | None,
    nearest_resistance: dict | None,
    atr: float,
) -> dict[str, Any]:
    """
    Calculate stop-loss, take-profit, R:R ratio, and position size.

    Stop-Loss:  placed just outside the S/R zone + 0.5 * ATR buffer.
    Take-Profit: placed just before the opposing S/R zone.

    Position size in micro-lots (1 micro-lot = 1,000 units for most pairs).
    For EUR/USD: 1 pip ≈ $0.10 per micro-lot.

    Returns dict with all risk metrics.
    """
    max_risk_usd = round(config.ACCOUNT_BALANCE * config.MAX_RISK_PERCENT, 2)
    atr_buffer   = 0.5 * atr

    out: dict[str, Any] = {
        "account_balance": config.ACCOUNT_BALANCE,
        "max_risk_usd":    max_risk_usd,
        "atr_buffer":      round(atr_buffer, 5),
    }

    if trend == "UPTREND" and nearest_support:
        # Long setup: buy near support, stop below it
        entry       = current_price
        stop_loss   = nearest_support["zone_low"] - atr_buffer
        take_profit = (
            nearest_resistance["price"] * 0.9998  # just below resistance
            if nearest_resistance
            else entry + 3 * (entry - stop_loss)  # fallback: 1:3 R:R
        )
        direction = "LONG"

    elif trend == "DOWNTREND" and nearest_resistance:
        # Short setup: sell near resistance, stop above it
        entry       = current_price
        stop_loss   = nearest_resistance["zone_high"] + atr_buffer
        take_profit = (
            nearest_support["price"] * 1.0002  # just above support
            if nearest_support
            else entry - 3 * (stop_loss - entry)  # fallback: 1:3 R:R
        )
        direction = "SHORT"

    else:
        # No valid setup
        out.update({
            "direction":          "NONE",
            "entry_price":        None,
            "stop_loss":          None,
            "take_profit":        None,
            "risk_pips":          None,
            "reward_pips":        None,
            "risk_reward_ratio":  0.0,
            "position_size_lots": None,
        })
        return out

    # Pip values (absolute distance)
    if direction == "LONG":
        risk_pips   = entry - stop_loss
        reward_pips = take_profit - entry
    else:
        risk_pips   = stop_loss - entry
        reward_pips = entry - take_profit

    risk_pips   = max(risk_pips, 1e-5)
    reward_pips = max(reward_pips, 0)
    rr_ratio    = reward_pips / risk_pips

    # Position size (micro-lots)
    # Assumption: 1 pip = 0.0001 for major pairs; $0.10 per micro-lot per pip
    pip_value_per_microlot = 0.10
    pips_at_risk   = risk_pips / 0.0001          # convert price units → pips
    micro_lots     = max_risk_usd / (pips_at_risk * pip_value_per_microlot)

    out.update({
        "direction":          direction,
        "entry_price":        round(entry,       5),
        "stop_loss":          round(stop_loss,   5),
        "take_profit":        round(take_profit, 5),
        "risk_pips":          round(risk_pips / 0.0001,   1),
        "reward_pips":        round(reward_pips / 0.0001, 1),
        "risk_reward_ratio":  round(rr_ratio, 2),
        "position_size_lots": round(micro_lots, 2),
        "position_size_units": int(micro_lots * 1000),
    })

    return out
