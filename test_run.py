"""
test_run.py — Offline & live verification CLI

Usage:
  python test_run.py --mock   Run end-to-end tests with synthetic data.
                               Zero API keys needed. Verifies math logic.

  python test_run.py --live   Fetch real EUR/USD data from Twelve Data and
                               run full Gate 2 + Gate 3 so you can compare
                               output against TradingView.

  python test_run.py --mock --regime uptrend   Test a specific price regime.
  python test_run.py --mock --regime downtrend
  python test_run.py --mock --regime ranging
"""

import argparse
import json
import sys

import config


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'━'*55}")
    print(f"  {title}")
    print(f"{'━'*55}")


def _ok(label: str, value: str = "") -> None:
    print(f"  ✅ {label} {value}")


def _fail(label: str, detail: str = "") -> None:
    print(f"  ❌ FAIL: {label} {detail}")
    sys.exit(1)


def _check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        _ok(label, detail)
    else:
        _fail(label, detail)


# ─────────────────────────────────────────────────────────────────────────────
# --mock mode
# ─────────────────────────────────────────────────────────────────────────────

def run_mock_tests(regime: str = "uptrend") -> None:
    """Full end-to-end test with synthetic data. No API keys required."""
    _header(f"🧪 MOCK TEST RUN — regime: {regime.upper()}")

    # ── Test imports ──────────────────────────────────────────────────────────
    print("\n[1/6] Testing imports …")
    try:
        import data_fetcher
        import market_math
        import ai_auditor
        import notifier
        import journal
        _ok("All modules imported successfully.")
    except ImportError as exc:
        _fail("Import error", str(exc))

    # ── Test mock data generation ─────────────────────────────────────────────
    print("\n[2/6] Generating mock data …")
    df_1h, df_15m = data_fetcher.generate_mock_data(regime=regime, seed=42)
    _check(len(df_1h) == 100,  "df_1h has 100 rows",  str(len(df_1h)))
    _check(len(df_15m) == 50,  "df_15m has 50 rows",  str(len(df_15m)))
    _check(all(c in df_1h.columns for c in ["open", "high", "low", "close"]),
           "df_1h has correct columns")
    _check(df_1h["high"].ge(df_1h["low"]).all(),
           "All 1H highs >= lows (valid OHLC)")
    _check(df_15m["high"].ge(df_15m["low"]).all(),
           "All 15m highs >= lows (valid OHLC)")
    print(f"     Price range 1H: {df_1h['close'].min():.5f} → {df_1h['close'].max():.5f}")

    # ── Test market math ──────────────────────────────────────────────────────
    print("\n[3/6] Running market_math.analyse() …")
    result = market_math.analyse("EUR/USD (MOCK)", df_1h, df_15m)

    _check("trend_1h" in result,          "trend_1h key present", f"→ {result['trend_1h']}")
    _check("sr_zones" in result,          "sr_zones computed")
    _check("candle_story" in result,      "candle_story computed")
    _check("atr_15m" in result,           "atr_15m computed",     f"→ {result.get('atr_15m')}")
    _check("current_price" in result,     "current_price present")
    _check("risk_reward_ratio" in result, "risk_reward_ratio computed",
           f"→ {result.get('risk_reward_ratio')}")
    _check("passed_math_gate" in result,  "passed_math_gate key present",
           f"→ {result['passed_math_gate']}")
    _check("last_10_candles_15m" in result, "last_10_candles_15m present",
           f"→ {len(result['last_10_candles_15m'])} candles")

    # Ranging regime should always fail gate
    if regime == "ranging":
        _check(
            not result["passed_math_gate"],
            "RANGING regime correctly fails math gate",
        )
    
    # Validate S/R zones structure
    for zone in result["sr_zones"]:
        _check(
            zone["touch_count"] >= 2,
            f"Zone at {zone['price']} has ≥2 touches",
            f"({zone['touch_count']})",
        )

    # Validate Fibonacci output (if applicable)
    if config.USE_V2_INDICATORS:
        fib = result.get("fibonacci", {})
        _check("available" in fib, "Fibonacci field present")
        if fib.get("available"):
            _check(
                fib["golden_zone_low"] < fib["golden_zone_high"],
                "Fibonacci golden zone is well-ordered",
            )

    # ── Test candle analysis ───────────────────────────────────────────────────
    print("\n[4/6] Verifying 15m candle story …")
    story = result["candle_story"]
    _check(story.get("available", False),  "Candle story is available")
    _check("pattern" in story,             "Pattern detected",  f"→ {story.get('pattern')}")
    _check("body_pct" in story,            "body_pct computed", f"→ {story.get('body_pct')}")

    # ── Test CSV logging ───────────────────────────────────────────────────────
    print("\n[5/6] Testing journal.log_setup() …")
    import os
    import journal as _journal

    # Temporarily redirect log file to a temp path
    original_log = _journal.LOG_FILE
    _journal.LOG_FILE = "/tmp/test_trades_log.csv"

    fake_ai = {
        "decision":          "NO_TRADE",
        "confluence_score":  55,
        "trend_1h":          result["trend_1h"],
        "plain_english_summary": "Test run only.",
        "trade_plan":        {"entry": None, "stop_loss": None, "take_profit": None,
                              "risk_reward": None, "risk_usd": None, "lots": None},
        "devils_advocate_risks": ["Test risk 1", "Test risk 2"],
        "checklist_audit":   {},
    }

    _journal.log_setup("EUR/USD", result, fake_ai, outcome="SKIPPED", notes="mock test")
    _check(os.path.exists(_journal.LOG_FILE), "trades_log.csv created")

    limit_reached, loss_count = _journal.check_daily_loss_limit()
    _check(isinstance(limit_reached, bool), "check_daily_loss_limit() returns bool",
           f"→ limit={limit_reached}, losses={loss_count}")

    # Cleanup temp log
    os.remove(_journal.LOG_FILE)
    _journal.LOG_FILE = original_log

    # ── Test notifier (console mode) ───────────────────────────────────────────
    print("\n[6/6] Testing notifier (console print) …")
    import notifier as _notifier
    # Temporarily clear Telegram credentials so it falls back to console
    orig_token = config.TELEGRAM_BOT_TOKEN
    orig_chat  = config.TELEGRAM_CHAT_ID
    config.TELEGRAM_BOT_TOKEN = ""
    config.TELEGRAM_CHAT_ID   = ""

    _notifier.send_alert("EUR/USD (MOCK)", result, fake_ai)

    config.TELEGRAM_BOT_TOKEN = orig_token
    config.TELEGRAM_CHAT_ID   = orig_chat
    _ok("Notifier printed card to console without errors.")

    # ── Summary ───────────────────────────────────────────────────────────────
    _header("✅ ALL MOCK TESTS PASSED")
    print(f"\n  Regime tested : {regime.upper()}")
    print(f"  Trend found   : {result['trend_1h']}")
    print(f"  S/R zones     : {len(result['sr_zones'])} cluster(s)")
    print(f"  Math gate     : {'PASS' if result['passed_math_gate'] else 'FAIL (expected for RANGING)'}")
    print(f"  R:R ratio     : {result.get('risk_reward_ratio', 'N/A')}")
    print(f"\n  Full Gate 2 JSON output:")
    # Print a readable subset (omit last_10_candles for brevity)
    display = {k: v for k, v in result.items() if k != "last_10_candles_15m"}
    print(json.dumps(display, indent=2, default=str))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# --live mode
# ─────────────────────────────────────────────────────────────────────────────

def run_live_test() -> None:
    """
    Fetch real EUR/USD data from Twelve Data and run Gate 2 + Gate 3.
    Requires TWELVE_DATA_API_KEY and GEMINI_API_KEY in .env.
    """
    _header("🌐 LIVE TEST RUN — EUR/USD")

    if not config.TWELVE_DATA_API_KEY:
        _fail("TWELVE_DATA_API_KEY not set",
              "Add it to .env or set as an environment variable.")

    # Gate 1
    print("\n[Gate 1] Fetching EUR/USD candles from Twelve Data …")
    import data_fetcher
    df_1h, df_15m = data_fetcher.fetch_pair_data("EUR/USD")
    _ok("1H candles",  f"→ {len(df_1h)} rows, latest: {df_1h.index[-1]}")
    _ok("15m candles", f"→ {len(df_15m)} rows, latest: {df_15m.index[-1]}")

    # Gate 2
    print("\n[Gate 2] Running market_math.analyse() …")
    import market_math
    result = market_math.analyse("EUR/USD", df_1h, df_15m)

    print(f"\n  Trend        : {result['trend_1h']}")
    print(f"  Current Price: {result['current_price']}")
    print(f"  EMA50 / EMA200: {result.get('ema50_1h')} / {result.get('ema200_1h')}")
    print(f"  EMA Signal    : {result.get('ema_signal')}")
    print(f"  ATR (15m)     : {result.get('atr_15m')}")
    print(f"  S/R Zones     : {len(result['sr_zones'])} cluster(s)")
    print(f"  Near Zone?    : {result.get('price_near_key_zone')}")
    print(f"  Candle Pattern: {result['candle_story'].get('pattern')}")
    print(f"  R:R Ratio     : {result.get('risk_reward_ratio')}")
    print(f"  Math Gate     : {'✅ PASS' if result['passed_math_gate'] else '❌ FAIL'}")

    for reason in result.get("fail_reasons", []):
        print(f"    ↳ {reason}")

    print(f"\n  Full Gate 2 JSON (compare with TradingView):")
    display = {k: v for k, v in result.items() if k != "last_10_candles_15m"}
    print(json.dumps(display, indent=2, default=str))

    # Gate 3 (only if Gate 2 passes, or user wants to see AI output anyway)
    if result["passed_math_gate"]:
        print(f"\n[Gate 3] Sending to Gemini AI auditor …")
        import ai_auditor
        ai_result = ai_auditor.audit(result)
        print(f"\n  AI Decision   : {ai_result.get('decision')}")
        print(f"  Confluence    : {ai_result.get('confluence_score')}/100")
        print(f"\n  Full Gate 3 JSON:")
        print(json.dumps(ai_result, indent=2, default=str))
    else:
        print(f"\n[Gate 3] Skipped — Gate 2 did not pass.")

    _header("✅ LIVE TEST COMPLETE")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Forex Co-Pilot — Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="Run offline mock tests.")
    group.add_argument("--live", action="store_true", help="Run live EUR/USD test.")
    parser.add_argument(
        "--regime",
        choices=["uptrend", "downtrend", "ranging"],
        default="uptrend",
        help="Price regime for --mock mode (default: uptrend).",
    )
    args = parser.parse_args()

    if args.mock:
        run_mock_tests(regime=args.regime)
    elif args.live:
        run_live_test()


if __name__ == "__main__":
    main()
