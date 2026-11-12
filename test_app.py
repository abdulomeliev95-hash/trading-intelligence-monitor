"""Unit tests for Trading Intelligence Monitor.   Run from the project folder:  pytest -v

All market data below is SYNTHETIC and used only for tests - never for real alerts.
Yahoo Finance, Telegram and the clock are replaced with fakes, so no internet or token is needed.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import pytest
import requests

import app


# =============================================================================
# Fixtures and synthetic data builders
# =============================================================================
@pytest.fixture(autouse=True)
def isolated_app(tmp_path, monkeypatch):
    """Every test gets its own empty database, settings file, charts folder and no Telegram keys."""
    charts = tmp_path / "charts"
    charts.mkdir()
    monkeypatch.setattr(app, "RT", app.Runtime())
    monkeypatch.setattr(app, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(app, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(app, "CHARTS_DIR", charts)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    app.init_db()


def set_telegram(monkeypatch, token="123456:ABC-secret", chat="555"):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", chat)


def synth(closes, volumes=None, freq="1h", start="2024-01-01 00:00", spread=0.5):
    """Build candles from closing prices.
    open = midpoint of previous close and close (so neighbouring candles never have equal highs);
    high/low = open/close extremes +- spread."""
    closes = np.asarray(closes, dtype=float)
    prev = np.concatenate([[closes[0]], closes[:-1]])
    opens = (prev + closes) / 2
    vols = np.full(len(closes), 1000.0) if volumes is None else np.asarray(volumes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq=freq)
    return pd.DataFrame({"Open": opens,
                         "High": np.maximum(opens, closes) + spread,
                         "Low": np.minimum(opens, closes) - spread,
                         "Close": closes, "Volume": vols}, index=idx)


def ranging_closes(n=100):
    """Price oscillating between ~95 and ~105 (period 20 candles): a clear range.
    Swing highs sit near 105.48 and swing lows near 94.52."""
    t = np.arange(n)
    return 100 + 5 * np.sin(2 * np.pi * (t + 0.3) / 20)


def breakout_df(last_close, last_volume):
    closes = np.append(ranging_closes(100), last_close)
    vols = np.append(np.full(100, 1000.0), last_volume)
    return app.add_indicators(synth(closes, vols))


def fixed_now(monkeypatch, ts):
    monkeypatch.setattr(app, "now_naive_utc", lambda: pd.Timestamp(ts))


# =============================================================================
# Indicators: EMA, RSI, MACD, ATR
# =============================================================================
def test_ema_matches_hand_calculation():
    # EMA20:  Step 1: alpha = 2/(20+1) = 2/21
    #         Step 2: EMA_0 = first close = 10
    #         Step 3: EMA_1 = alpha*20 + (1-alpha)*10 = 10 + 10*(2/21) = 10.952381
    ind = app.add_indicators(synth([10, 20]))
    assert ind["EMA20"].iloc[0] == pytest.approx(10.0)
    assert ind["EMA20"].iloc[1] == pytest.approx(10 + 10 * 2 / 21)
    # EMA50: alpha = 2/51, so EMA_1 = 10 + 10*(2/51)
    assert ind["EMA50"].iloc[1] == pytest.approx(10 + 10 * 2 / 51)


def test_ema_of_constant_price_is_that_price():
    ind = app.add_indicators(synth([50.0] * 80))
    assert ind["EMA20"].iloc[-1] == pytest.approx(50.0)
    assert ind["EMA50"].iloc[-1] == pytest.approx(50.0)


def test_rsi_extremes():
    # Only gains: avg_loss = 0, so RS is infinite and RSI = 100 - 100/(1+inf) = 100
    up = app.add_indicators(synth(np.arange(100, 160, dtype=float)))
    # Only losses: avg_gain = 0, so RS = 0 and RSI = 100 - 100/(1+0) = 0
    down = app.add_indicators(synth(np.arange(160, 100, -1, dtype=float)))
    assert up["RSI14"].iloc[-1] == pytest.approx(100.0)
    assert down["RSI14"].iloc[-1] == pytest.approx(0.0)


def test_rsi_alternating_prices_is_close_to_52():
    # Prices alternate +1, -1 (last step is up). With a = 1/14 (Wilder):
    #   Step 1: avg_gain after an up step   x = 1/(2-a) = 0.5185
    #   Step 2: avg_loss after an up step   (1-a)*x      = 0.4815
    #   Step 3: RS  = 0.5185/0.4815 = 1.0769
    #   Step 4: RSI = 100 - 100/(1+1.0769) = 51.85
    closes = [100.0 if i % 2 == 0 else 101.0 for i in range(80)]
    ind = app.add_indicators(synth(closes))
    assert ind["RSI14"].iloc[-1] == pytest.approx(51.85, abs=0.5)


def test_rsi_stays_between_0_and_100():
    rng = np.random.default_rng(42)
    closes = 100 + np.cumsum(rng.normal(0, 1, 300))
    rsi = app.add_indicators(synth(closes))["RSI14"].dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_macd_constant_price_is_zero():
    ind = app.add_indicators(synth([100.0] * 100))
    assert ind["MACD"].abs().max() == pytest.approx(0.0, abs=1e-9)
    assert ind["MACD_HIST"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_macd_on_straight_line_settles_at_7():
    # For a line with slope s, an EMA with span N lags the price by (N-1)/2 * s.
    #   Step 1: lag EMA12 = 11/2 = 5.5   (slope 1)
    #   Step 2: lag EMA26 = 25/2 = 12.5
    #   Step 3: MACD = EMA12 - EMA26 = (p - 5.5) - (p - 12.5) = 7
    #   Step 4: signal (EMA9 of a constant 7) = 7, so histogram = 0
    ind = app.add_indicators(synth(np.arange(300, dtype=float)))
    assert ind["MACD"].iloc[-1] == pytest.approx(7.0, abs=0.01)
    assert ind["MACD_SIGNAL"].iloc[-1] == pytest.approx(7.0, abs=0.01)
    assert ind["MACD_HIST"].iloc[-1] == pytest.approx(0.0, abs=0.01)


def test_macd_histogram_is_macd_minus_signal():
    rng = np.random.default_rng(1)
    ind = app.add_indicators(synth(100 + np.cumsum(rng.normal(0, 1, 150))))
    assert np.allclose(ind["MACD_HIST"], ind["MACD"] - ind["MACD_SIGNAL"])


def test_atr_constant_true_range():
    # Flat price 100, spread 0.5: high = 100.5, low = 99.5
    #   Step 1: high - low = 1.0;  |high - prev close| = 0.5;  |low - prev close| = 0.5
    #   Step 2: TR = max(1.0, 0.5, 0.5) = 1.0 for every candle
    #   Step 3: ATR14 (average of a constant 1.0) = 1.0
    ind = app.add_indicators(synth([100.0] * 60))
    assert ind["ATR14"].iloc[-1] == pytest.approx(1.0)


def test_add_indicators_does_not_modify_input():
    df = synth(ranging_closes(80))
    before = df.copy()
    app.add_indicators(df)
    pd.testing.assert_frame_equal(df, before)


# =============================================================================
# Price change calculations
# =============================================================================
def test_price_change_5_minutes():
    # closes 100, 100, 100, 102 (5-minute candles)
    #   Step 1: target = last time - 5 min -> the candle before the last one
    #   Step 2: reference close = 100
    #   Step 3: change = (102/100 - 1) * 100 = 2.0 %
    df = synth([100, 100, 100, 102], freq="5min")
    pct, ref = app.price_change(df, 5)
    assert ref == pytest.approx(100.0)
    assert pct == pytest.approx(2.0)


def test_price_change_15_minutes():
    # closes 100..104, 5-minute candles; last = 104
    #   Step 1: target = last time - 15 min = 3 candles back -> close 101
    #   Step 2: change = (104/101 - 1) * 100 = 2.9703 %
    df = synth([100, 101, 102, 103, 104], freq="5min")
    assert app.pct_change_over(df, 15) == pytest.approx((104 / 101 - 1) * 100)


def test_price_change_returns_none_when_history_too_short():
    df = synth([100, 101, 102], freq="5min")
    assert app.price_change(df, 60) == (None, None)


def test_price_change_returns_none_across_a_data_gap():
    # Candles at 00:00, 00:05 and then 10 hours later (e.g. market closed overnight).
    # A "5 minute change" would really be the overnight jump, so it must NOT be reported.
    df = synth([100, 100, 150], freq="5min")
    df.index = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 10:05"])
    assert app.price_change(df, 5) == (None, None)


def test_closed_candles_excludes_the_forming_candle(monkeypatch):
    # 1h candles starting 10:00, 11:00, 12:00 and "now" = 12:30
    #   10:00 + 1h = 11:00 <= 12:30 -> closed;  11:00 + 1h = 12:00 <= 12:30 -> closed
    #   12:00 + 1h = 13:00 >  12:30 -> still forming
    df = synth([1, 2, 3], start="2024-01-01 10:00")
    fixed_now(monkeypatch, "2024-01-01 12:30")
    assert len(app.closed_candles(df, "1h")) == 2


def test_is_crypto():
    assert app.is_crypto("BTC-USD") and app.is_crypto("sol-usd")
    assert not app.is_crypto("AAPL") and not app.is_crypto("BRK-B")


# =============================================================================
# Data fetching (yfinance replaced by a fake)
# =============================================================================
def raw_history(tz="America/New_York", freq="1h", start="2024-01-02 09:30"):
    idx = pd.date_range(start, periods=4, freq=freq, tz=tz)
    return pd.DataFrame({
        "Open": [10, 11, 12, 13.0], "High": [11, 12, 13, 14.0], "Low": [9, 10, 11, 12.0],
        "Close": [10.5, 11.5, np.nan, 13.5], "Adj Close": [10.5, 11.5, 12.5, 13.5],
        "Volume": [100, 200, 300, np.nan]}, index=idx)


def fake_ticker(monkeypatch, result=None, exc=None, counter=None):
    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            if counter is not None:
                counter["n"] += 1
            if exc:
                raise exc
            return result
    monkeypatch.setattr(app.yf, "Ticker", FakeTicker)


def test_download_cleans_data_and_converts_intraday_to_utc(monkeypatch):
    fake_ticker(monkeypatch, raw_history())
    res = app._download("AAPL", "1h")
    assert res.error is None
    df = res.df
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]   # 'Adj Close' removed
    assert len(df) == 3                                                    # row with NaN Close dropped
    assert df["Volume"].iloc[-1] == 0                                      # NaN volume -> 0
    assert df.index.tz is None
    # 09:30 New York in January is UTC-5 -> 14:30 UTC
    assert df.index[0] == pd.Timestamp("2024-01-02 14:30")


def test_download_daily_keeps_the_exchange_date(monkeypatch):
    fake_ticker(monkeypatch, raw_history(freq="1D", start="2024-01-02"))
    df = app._download("AAPL", "1d").df
    assert df.index[0] == pd.Timestamp("2024-01-02 00:00")


def test_download_reports_empty_data_instead_of_inventing_values(monkeypatch):
    fake_ticker(monkeypatch, pd.DataFrame())
    res = app._download("NOPE", "5m")
    assert res.df is None and "No 5m data" in res.error


def test_download_reports_provider_errors(monkeypatch):
    fake_ticker(monkeypatch, exc=RuntimeError("rate limited"))
    res = app._download("AAPL", "5m")
    assert res.df is None and "request failed" in res.error and "rate limited" in res.error


def test_fetch_candles_uses_cache(monkeypatch):
    counter = {"n": 0}
    fake_ticker(monkeypatch, raw_history(), counter=counter)
    first = app.fetch_candles("AAPL", "1h")
    second = app.fetch_candles("AAPL", "1h")
    assert counter["n"] == 1 and first is second


def test_fetch_candles_caches_failures_briefly(monkeypatch):
    counter = {"n": 0}
    fake_ticker(monkeypatch, exc=RuntimeError("boom"), counter=counter)
    app.fetch_candles("AAPL", "1h")
    app.fetch_candles("AAPL", "1h")
    assert counter["n"] == 1   # no hammering of Yahoo while it is failing


def test_get_snapshot_values(monkeypatch):
    closes = np.linspace(100, 110, 600)   # 600 five-minute candles = 50 hours
    df = synth(closes, freq="5min")
    monkeypatch.setattr(app, "fetch_candles", lambda s, tf: app.FetchResult(df, None, app.utc_now()))
    fixed_now(monkeypatch, df.index[-1] + pd.Timedelta(minutes=3))
    snap = app.get_snapshot("BTC-USD")
    assert snap["error"] is None and snap["type"] == "Crypto"
    assert snap["price"] == pytest.approx(110.0)
    # 5 min = 1 candle back, 1 hour = 12 candles back, 24 hours = 288 candles back
    assert snap["chg_5m"] == pytest.approx((closes[-1] / closes[-2] - 1) * 100)
    assert snap["chg_1h"] == pytest.approx((closes[-1] / closes[-13] - 1) * 100)
    assert snap["chg_24h"] == pytest.approx((closes[-1] / closes[-289] - 1) * 100)
    assert snap["age"] == pytest.approx(3.0) and snap["stale"] is False


def test_get_snapshot_flags_stale_stock_data(monkeypatch):
    df = synth(np.linspace(100, 110, 50), freq="5min")
    monkeypatch.setattr(app, "fetch_candles", lambda s, tf: app.FetchResult(df, None, app.utc_now()))
    fixed_now(monkeypatch, df.index[-1] + pd.Timedelta(minutes=120))   # 120 min > 45 min limit
    assert app.get_snapshot("AAPL")["stale"] is True


def test_get_snapshot_reports_error(monkeypatch):
    monkeypatch.setattr(app, "fetch_candles", lambda s, tf: app.FetchResult(None, "boom", app.utc_now()))
    assert app.get_snapshot("AAPL")["error"] == "boom"


# =============================================================================
# Price monitoring: rapid moves and volume spikes
# =============================================================================
def five_min_df(last_close, last_volume=1000.0, n=40):
    closes = [100.0] * (n - 1) + [last_close]
    vols = [1000.0] * (n - 1) + [last_volume]
    return app.add_indicators(synth(closes, vols, freq="5min"))


def test_crypto_rapid_increase_and_volume_spike(monkeypatch):
    # Crypto rule: 2 % in 5 min.   Step 1: reference close = 100, last close = 102.5
    #                              Step 2: change = (102.5/100 - 1) * 100 = 2.5 % >= 2.0 %
    # Volume rule: 2x the average of previous 20 candles.
    #                              Step 3: average = 1000, last volume = 2500 -> ratio 2.5 >= 2.0
    ind = five_min_df(102.5, 2500)
    fixed_now(monkeypatch, ind.index[-1] + pd.Timedelta(minutes=10))
    events = app.detect_price_events("BTC-USD", ind, app.load_settings())
    by_type = {e["alert_type"]: e for e in events}
    assert set(by_type) == {"Rapid Price Increase", "Volume Spike"}
    assert by_type["Rapid Price Increase"]["change_pct"] == pytest.approx(2.5)
    assert by_type["Volume Spike"]["direction"] == "bull"
    assert all(e["notify"] for e in events)


def test_crypto_move_below_threshold_gives_no_alert(monkeypatch):
    ind = five_min_df(101.0)     # +1 % is below the 2 % crypto threshold
    fixed_now(monkeypatch, ind.index[-1] + pd.Timedelta(minutes=10))
    assert app.detect_price_events("BTC-USD", ind, app.load_settings()) == []


def test_stock_uses_15_minute_window_and_1_percent(monkeypatch):
    # Step 1: 15 min = 3 candles back, reference close = 100
    # Step 2: change = (98.5/100 - 1) * 100 = -1.5 %, |-1.5| >= 1.0
    ind = five_min_df(98.5)
    fixed_now(monkeypatch, ind.index[-1] + pd.Timedelta(minutes=10))
    events = app.detect_price_events("AAPL", ind, app.load_settings())
    assert [e["alert_type"] for e in events] == ["Rapid Price Decrease"]
    assert events[0]["direction"] == "bear"


def test_user_changed_thresholds_are_respected(monkeypatch):
    ind = five_min_df(101.0)
    fixed_now(monkeypatch, ind.index[-1] + pd.Timedelta(minutes=10))
    settings = app.load_settings()
    settings["crypto_pct"] = 0.5       # now +1 % is enough
    events = app.detect_price_events("BTC-USD", ind, settings)
    assert [e["alert_type"] for e in events] == ["Rapid Price Increase"]


def test_event_freshness(monkeypatch):
    fixed_now(monkeypatch, "2024-01-10 12:00")
    recent = {"event_time": pd.Timestamp("2024-01-10 10:20")}   # candle closed 40 min ago
    old = {"event_time": pd.Timestamp("2024-01-10 07:00")}      # candle closed 4 h ago
    # limit for 1h candles = 2*60 + 45 = 165 minutes
    assert app.event_is_fresh(recent, "1h") is True
    assert app.event_is_fresh(old, "1h") is False


# =============================================================================
# Pattern detection: support/resistance, breakout, breakdown, reversal
# =============================================================================
def test_support_and_resistance_levels_are_found():
    levels = app.current_levels(app.add_indicators(synth(ranging_closes(100))))
    assert any(abs(lv - 105.48) < 0.5 for lv in levels)   # resistance
    assert any(abs(lv - 94.52) < 0.5 for lv in levels)    # support


def test_confirmed_bullish_breakout():
    # Close 112 vs resistance ~105.48 (prev close ~98.9 was below it), volume 3000 vs average 1000
    #   Step 1: close beyond level by more than 0.25 ATR            -> true
    #   Step 2: volume ratio = 3000/1000 = 3.0 >= 1.5               -> confirmed
    ind = breakout_df(112.0, 3000)
    events = [e for e in app.detect_breakouts(ind, "TEST", "1h") if e["alert_type"] == "Bullish Breakout"]
    assert len(events) == 1
    ev = events[0]
    assert ev["status"] == "Confirmed" and ev["direction"] == "bull" and ev["notify"] is True
    assert ev["level"] == pytest.approx(105.48, abs=0.5)
    assert "3.0x" in ev["explanation"]


def test_breakout_without_volume_is_only_potential():
    ind = breakout_df(112.0, 1000)     # volume ratio 1.0 < 1.5
    events = [e for e in app.detect_breakouts(ind, "TEST", "1h") if e["alert_type"] == "Bullish Breakout"]
    assert len(events) == 1
    assert events[0]["status"] == "Potential" and events[0]["notify"] is False
    assert "NOT confirmed" in events[0]["explanation"]


def test_confirmed_bearish_breakdown():
    ind = breakout_df(88.0, 3000)      # support ~94.52, prev close ~98.9
    events = [e for e in app.detect_breakouts(ind, "TEST", "1h") if e["alert_type"] == "Bearish Breakdown"]
    assert len(events) == 1
    assert events[0]["status"] == "Confirmed" and events[0]["direction"] == "bear"
    assert events[0]["level"] == pytest.approx(94.52, abs=0.5)


def test_wick_above_resistance_without_a_close_is_not_a_breakout():
    # The candle spikes to 108 (above resistance) but CLOSES at 104 (below it): not a breakout.
    ind = breakout_df(104.0, 3000)
    ind.iloc[-1, ind.columns.get_loc("High")] = 108.0
    assert app.detect_breakouts(ind, "TEST", "1h") == []


def test_forming_candle_is_ignored_until_it_closes(monkeypatch):
    ind = breakout_df(112.0, 3000)
    last_ts = ind.index[-1]
    # 10 minutes after the candle started: the 1h candle is still forming -> excluded
    fixed_now(monkeypatch, last_ts + pd.Timedelta(minutes=10))
    closed = app.closed_candles(ind, "1h")
    assert len(closed) == 100
    assert not [e for e in app.pattern_events(closed, "TEST", "1h") if e["alert_type"] == "Bullish Breakout"]
    # 61 minutes later it has closed -> the breakout is now visible
    fixed_now(monkeypatch, last_ts + pd.Timedelta(minutes=61))
    closed = app.closed_candles(ind, "1h")
    assert len(closed) == 101
    assert [e for e in app.pattern_events(closed, "TEST", "1h") if e["alert_type"] == "Bullish Breakout"]


def test_no_breakout_signals_inside_a_quiet_range():
    ind = app.add_indicators(synth(ranging_closes(100)))
    assert app.detect_breakouts(ind, "TEST", "1h") == []


def test_too_little_history_gives_no_patterns():
    ind = app.add_indicators(synth(ranging_closes(30)))
    assert app.pattern_events(ind, "TEST", "1h") == []


def test_potential_bullish_reversal_after_a_long_decline():
    # 80 candles falling 1 per step (RSI near 0, EMA20 below EMA50), then a pause and a rebound
    closes = list(200 - np.arange(80)) + [121.0] * 3 + list(121 + 1.5 * np.arange(1, 21))
    ind = app.add_indicators(synth(closes))
    turn = ind.index[80]
    events = app.detect_reversals(ind, "TEST", "1h", only_last=False)
    bullish = [e for e in events if e["direction"] == "bull" and e["event_time"] >= turn]
    assert bullish
    assert all(e["status"] == "Potential" and e["notify"] is False for e in events)
    assert bullish[0]["alert_type"] == "Potential Trend Reversal"


def test_potential_bearish_reversal_after_a_long_rise():
    closes = list(100 + np.arange(80)) + [179.0] * 3 + list(179 - 1.5 * np.arange(1, 21))
    ind = app.add_indicators(synth(closes))
    turn = ind.index[80]
    events = app.detect_reversals(ind, "TEST", "1h", only_last=False)
    assert any(e["direction"] == "bear" and e["event_time"] >= turn for e in events)


# =============================================================================
# SQLite persistence and alert cooldown
# =============================================================================
def sample_event(asset="BTC-USD", alert_type="Rapid Price Increase", status="Info",
                 ts="2024-01-01 10:00", tf="5m"):
    return app.make_event(asset, tf, alert_type, status, "bull", pd.Timestamp(ts), 65000.0, 2.5,
                          "last 5 min", "Test explanation", True)


def test_alert_is_stored_with_all_fields():
    alert_id = app.try_register_alert(sample_event(), 30)
    assert isinstance(alert_id, int)
    df = app.get_alerts()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["asset"] == "BTC-USD" and row["alert_type"] == "Rapid Price Increase"
    assert row["timeframe"] == "5m" and row["status"] == "Info"
    assert row["price"] == pytest.approx(65000.0) and row["change_pct"] == pytest.approx(2.5)
    assert row["description"] == "Test explanation"
    assert len(row["created_at"]) == 19    # "YYYY-MM-DD HH:MM:SS"


def test_same_candle_cannot_alert_twice():
    assert app.try_register_alert(sample_event(), 30) is not None
    assert app.try_register_alert(sample_event(), 0) is None   # duplicate even with cooldown 0


def test_cooldown_blocks_repeat_of_same_type_and_asset():
    assert app.try_register_alert(sample_event(ts="2024-01-01 10:00"), 30) is not None
    # a later candle, same asset/type/timeframe/status, created a moment later -> inside 30 minutes
    assert app.try_register_alert(sample_event(ts="2024-01-01 10:05"), 30) is None
    assert len(app.get_alerts()) == 1


def test_cooldown_does_not_block_other_assets_or_types():
    assert app.try_register_alert(sample_event(), 30) is not None
    assert app.try_register_alert(sample_event(asset="ETH-USD", ts="2024-01-01 10:05"), 30) is not None
    assert app.try_register_alert(sample_event(alert_type="Volume Spike", ts="2024-01-01 10:05"), 30) is not None
    assert app.try_register_alert(sample_event(tf="1h", ts="2024-01-01 10:05"), 30) is not None
    assert len(app.get_alerts()) == 4


def test_potential_and_confirmed_have_separate_cooldowns():
    pot = sample_event(alert_type="Bullish Breakout", status="Potential", tf="1h")
    conf = sample_event(alert_type="Bullish Breakout", status="Confirmed", tf="1h", ts="2024-01-01 11:00")
    assert app.try_register_alert(pot, 30) is not None
    assert app.try_register_alert(conf, 30) is not None   # the upgrade must not be suppressed


def test_cooldown_expires_after_30_minutes():
    first_id = app.try_register_alert(sample_event(ts="2024-01-01 10:00"), 30)
    old = (app.utc_now() - timedelta(minutes=31)).strftime("%Y-%m-%d %H:%M:%S")
    app.db_execute("UPDATE alerts SET created_at=? WHERE id=?", (old, first_id))
    assert app.try_register_alert(sample_event(ts="2024-01-01 10:05"), 30) is not None


def test_history_survives_a_restart():
    app.try_register_alert(sample_event(), 30)
    app.try_register_alert(sample_event(asset="ETH-USD"), 30)
    app.RT = app.Runtime()      # "restart": all in-memory state is gone, only the database file remains
    app.init_db()               # must not wipe existing data
    assert len(app.get_alerts()) == 2
    # the duplicate guard also survives the restart
    assert app.try_register_alert(sample_event(), 30) is None


def test_get_alerts_filters_and_limit():
    app.try_register_alert(sample_event(asset="BTC-USD"), 30)
    app.try_register_alert(sample_event(asset="ETH-USD"), 30)
    app.try_register_alert(sample_event(asset="ETH-USD", alert_type="Volume Spike"), 30)
    assert len(app.get_alerts(asset="ETH-USD")) == 2
    assert len(app.get_alerts(alert_type="Volume Spike")) == 1
    assert len(app.get_alerts(limit=1)) == 1
    # newest first
    assert app.get_alerts().iloc[0]["alert_type"] == "Volume Spike"


def test_get_alerts_since_minutes():
    old_id = app.try_register_alert(sample_event(asset="BTC-USD"), 30)
    app.try_register_alert(sample_event(asset="ETH-USD"), 30)
    old = (app.utc_now() - timedelta(minutes=90)).strftime("%Y-%m-%d %H:%M:%S")
    app.db_execute("UPDATE alerts SET created_at=? WHERE id=?", (old, old_id))
    recent = app.get_alerts(since_minutes=60)
    assert list(recent["asset"]) == ["ETH-USD"]


def test_process_event_stores_once_and_notifies_once(monkeypatch):
    set_telegram(monkeypatch)
    sent = []
    monkeypatch.setattr(app, "send_alert_notification", lambda ev, ind, tf, chat_id=None: sent.append(ev) or (True, "ok"))
    ind = breakout_df(112.0, 3000)
    ev = [e for e in app.detect_breakouts(ind, "TEST", "1h") if e["alert_type"] == "Bullish Breakout"][0]
    settings = app.load_settings()
    app.process_event(dict(ev), ind, "1h", settings)
    app.process_event(dict(ev), ind, "1h", settings)       # second run = same candle
    assert len(app.get_alerts()) == 1 and len(sent) == 1
    assert "RSI14" in sent[0]["indicators"]                 # indicators attached before sending


def test_potential_alert_is_stored_but_not_sent_to_telegram(monkeypatch):
    set_telegram(monkeypatch)
    sent = []
    monkeypatch.setattr(app, "send_alert_notification", lambda *a, **k: sent.append(1) or (True, "ok"))
    ind = breakout_df(112.0, 1000)       # no volume confirmation -> Potential
    ev = [e for e in app.detect_breakouts(ind, "TEST", "1h") if e["alert_type"] == "Bullish Breakout"][0]
    app.process_event(ev, ind, "1h", app.load_settings())
    assert len(app.get_alerts()) == 1 and sent == []


def test_breakout_event_reports_distance_in_atr():
    ind = breakout_df(112.0, 3000)
    ev = [e for e in app.detect_breakouts(ind, "TEST", "1h")
          if e["alert_type"] == "Bullish Breakout"][0]
    expected = abs(ev["price"] - ev["level"]) / ind.loc[ev["event_time"], "ATR14"]
    assert ev["break_distance_atr"] == pytest.approx(expected)
    assert ev["break_distance_atr"] > app.DEFAULT_SETTINGS["min_breakout_atr"]


def test_weak_structural_break_is_stored_but_not_sent_to_telegram(monkeypatch):
    set_telegram(monkeypatch)
    sent = []
    monkeypatch.setattr(app, "send_alert_notification",
                        lambda ev, ind, tf, chat_id=None: sent.append(ev) or (True, "ok"))
    ind = breakout_df(112.0, 3000)
    ev = [e for e in app.detect_breakouts(ind, "TEST", "1h")
          if e["alert_type"] == "Bullish Breakout"][0]
    ev["break_distance_atr"] = 0.50
    settings = app.load_settings()
    settings["min_breakout_atr"] = 0.75

    app.process_event(dict(ev), ind, "1h", settings)

    assert len(app.get_alerts()) == 1  # keep it in history
    assert sent == []                 # but do not send a weak break to Telegram


# =============================================================================
# Settings file
# =============================================================================
def test_settings_defaults_roundtrip_and_corrupt_file():
    assert app.load_settings() == app.DEFAULT_SETTINGS           # no file yet
    s = app.load_settings()
    s["watchlist"].append("AMZN")
    s["crypto_pct"] = 3.5
    app.save_settings(s)
    again = app.load_settings()
    assert "AMZN" in again["watchlist"] and again["crypto_pct"] == 3.5
    app.SETTINGS_PATH.write_text("{ not valid json", encoding="utf-8")
    assert app.load_settings() == app.DEFAULT_SETTINGS           # corrupt file -> safe defaults


def test_default_watchlist_matches_the_specification():
    assert app.DEFAULT_SETTINGS["watchlist"] == [
        "AAPL", "MSFT", "NVDA", "TSLA", "SPY", "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "^GSPC", "^OEX", "AMD", "RKLB"]
    s = app.DEFAULT_SETTINGS
    assert (s["crypto_pct"], s["crypto_minutes"]) == (2.0, 5)
    assert (s["stock_pct"], s["stock_minutes"]) == (1.0, 15)
    assert (s["volume_mult"], s["cooldown_minutes"]) == (2.0, 30)


# =============================================================================
# Chart generation
# =============================================================================
def test_chart_contains_requested_elements():
    ind = app.add_indicators(synth(ranging_closes(100)))
    ev_bull = app.make_event("T", "1h", "Bullish Breakout", "Confirmed", "bull", ind.index[-5], 100, 1, "x", "e", True)
    ev_pot = app.make_event("T", "1h", "Bearish Breakdown", "Potential", "bear", ind.index[-8], 100, -1, "x", "e", False)
    fig = app.build_chart(ind, "T", "1h", [95.0, 105.0], [ev_bull, ev_pot], candles=100, with_osc=True)
    names = {tr.name for tr in fig.data}
    assert {"Price", "EMA 20", "EMA 50", "Volume", "RSI 14", "MACD", "Signal", "ATR 14"} <= names
    symbols = [tr.marker.symbol for tr in fig.data if getattr(tr, "mode", None) == "markers"]
    assert "triangle-up" in symbols            # confirmed = filled
    assert "triangle-down-open" in symbols     # potential = hollow
    assert "T" in fig.layout.title.text and "1h" in fig.layout.title.text


def test_chart_draws_support_and_resistance_lines():
    ind = app.add_indicators(synth(ranging_closes(100)))      # last close ~98.9
    fig = app.build_chart(ind, "T", "1h", [95.0, 105.0], [], candles=100, with_osc=False)
    assert len(fig.layout.shapes) == 2         # one support (95) and one resistance (105)


def test_chart_png_is_created():
    pytest.importorskip("kaleido")
    ind = app.add_indicators(synth(ranging_closes(100)))
    path = app.render_chart_png(ind, "TEST", "1h")
    assert path.exists() and path.parent == app.CHARTS_DIR
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 5000


# =============================================================================
# Telegram (HTTP is replaced by fakes)
# =============================================================================
class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload, self.status_code = payload, status_code

    def json(self):
        return self._payload


def notification_event():
    ev = app.make_event("BTC-USD", "5m", "Rapid Price Increase", "Info", "bull", pd.Timestamp("2024-01-01 10:00"),
                        65432.10, 2.35, "last 5 min",
                        "Price moved +2.35% in about 5 minutes (alert threshold 2%).", True)
    ev["indicators"] = "RSI14 62.1 | MACD hist +12.3 | ATR14 410.2 | EMA20 > EMA50"
    return ev


def test_telegram_configured_requires_both_values(monkeypatch):
    assert app.telegram_configured() is False
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    assert app.telegram_configured() is False
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "555")
    assert app.telegram_configured() is True


def test_notification_text_has_all_required_items():
    text = app.build_message(notification_event())
    for part in ["BTC-USD",                       # 1 asset name
                 "65,432.10",                     # 2 current price
                 "+2.35%",                        # 3 percentage change
                 "5m",                            # 4 timeframe
                 "RSI14 62.1", "EMA20 > EMA50",   # 5 indicators
                 "Price moved +2.35%",            # 6 explanation
                 "Not financial advice"]:
        assert part in text


def test_confirmed_prediction_notification_shows_long_or_short_and_horizon():
    ev = app.make_event("BTC-USD", "1h", "Bullish Breakout", "Confirmed", "bull",
                        pd.Timestamp("2024-01-01 10:00"), 65000, 1.2, "1h candle", "breakout", True)
    ev["prediction_horizon_minutes"] = 240
    message = app.build_message(ev)
    assert "Prediction: LONG" in message
    assert "Result check after: 240 min" in message

    ev["direction"] = "bear"
    message = app.build_message(ev)
    assert "Prediction: SHORT" in message


def test_notification_text_fits_telegram_caption_limit():
    ev = notification_event()
    ev["explanation"] = "very long explanation " * 100
    assert len(app.build_message(ev)) <= 1024


def test_alert_is_sent_as_photo_with_caption(monkeypatch, tmp_path):
    set_telegram(monkeypatch)
    png = tmp_path / "fake.png"
    png.write_bytes(b"\x89PNG fake")
    monkeypatch.setattr(app, "render_chart_png", lambda ind, symbol, tf, ev=None: png)
    calls = []
    monkeypatch.setattr(app, "tg_send_photo", lambda path, caption, chat_id=None: calls.append((path, caption)) or (True, {}))
    monkeypatch.setattr(app, "tg_send_message", lambda *a, **k: pytest.fail("text fallback must not be used"))
    ok, info = app.send_alert_notification(notification_event(), None, "5m")
    assert ok and "chart" in info
    assert len(calls) == 1 and "BTC-USD" in calls[0][1]


def test_text_is_sent_if_chart_generation_fails(monkeypatch):
    set_telegram(monkeypatch)

    def broken_render(*args, **kwargs):
        raise RuntimeError("kaleido missing")
    monkeypatch.setattr(app, "render_chart_png", broken_render)
    monkeypatch.setattr(app, "tg_send_photo", lambda *a, **k: pytest.fail("no photo expected"))
    texts = []
    monkeypatch.setattr(app, "tg_send_message", lambda text, chat_id=None: texts.append(text) or (True, {}))
    ok, info = app.send_alert_notification(notification_event(), None, "5m")
    assert ok and "text only" in info and "chart failed" in info
    assert len(texts) == 1 and "BTC-USD" in texts[0]


def test_text_is_sent_if_sendphoto_fails(monkeypatch, tmp_path):
    set_telegram(monkeypatch)
    png = tmp_path / "fake.png"
    png.write_bytes(b"x")
    monkeypatch.setattr(app, "render_chart_png", lambda *a, **k: png)
    monkeypatch.setattr(app, "tg_send_photo", lambda *a, **k: (False, "Bad Request"))
    texts = []
    monkeypatch.setattr(app, "tg_send_message", lambda text, chat_id=None: texts.append(text) or (True, {}))
    ok, info = app.send_alert_notification(notification_event(), None, "5m")
    assert ok and "sendPhoto failed" in info and len(texts) == 1


def test_no_notification_without_configuration():
    ok, info = app.send_alert_notification(notification_event(), None, "5m")
    assert ok is False and "not configured" in info


def test_tg_send_message_builds_the_right_request(monkeypatch):
    set_telegram(monkeypatch, token="123:ABC", chat="555")
    seen = {}

    def fake_post(url, data=None, files=None, timeout=None):
        seen.update(url=url, data=data)
        return FakeResponse({"ok": True, "result": {"message_id": 1}})
    monkeypatch.setattr(app.requests, "post", fake_post)
    ok, _ = app.tg_send_message("hello")
    assert ok
    assert seen["url"] == "https://api.telegram.org/bot123:ABC/sendMessage"
    assert seen["data"]["chat_id"] == "555" and seen["data"]["text"] == "hello"


def test_tg_send_photo_uses_sendphoto_with_file(monkeypatch, tmp_path):
    set_telegram(monkeypatch, token="123:ABC", chat="555")
    png = tmp_path / "chart.png"
    png.write_bytes(b"\x89PNG data")
    seen = {}

    def fake_post(url, data=None, files=None, timeout=None):
        seen.update(url=url, data=data, has_photo="photo" in files)
        return FakeResponse({"ok": True, "result": {}})
    monkeypatch.setattr(app.requests, "post", fake_post)
    ok, _ = app.tg_send_photo(png, "caption text")
    assert ok and seen["url"].endswith("/sendPhoto") and seen["has_photo"]
    assert seen["data"]["caption"] == "caption text"


def test_telegram_api_error_is_returned_not_raised(monkeypatch):
    set_telegram(monkeypatch)
    monkeypatch.setattr(app.requests, "post",
                        lambda *a, **k: FakeResponse({"ok": False, "description": "chat not found"}, 400))
    assert app.tg_send_message("hi") == (False, "chat not found")


def test_bot_token_never_appears_in_errors(monkeypatch):
    set_telegram(monkeypatch, token="123456:ABC-secret")

    def failing_post(url, **kwargs):
        raise requests.exceptions.ConnectionError(f"Max retries exceeded with url: {url}")
    monkeypatch.setattr(app.requests, "post", failing_post)
    ok, msg = app.tg_send_message("hi")
    assert ok is False and "ABC-secret" not in msg and "***" in msg


def run_command(monkeypatch, text, chat_id="555"):
    replies = []
    def fake_send(t, chat_id=None, reply_markup=None):
        replies.append((chat_id, t))
        run_command.last_reply_markup = reply_markup
        return True, {}
    run_command.last_reply_markup = None
    monkeypatch.setattr(app, "tg_send_message", fake_send)
    app.handle_update({"update_id": 1, "message": {"text": text, "chat": {"id": int(chat_id)}}})
    return replies


def test_help_and_start_show_chat_id_and_commands(monkeypatch):
    set_telegram(monkeypatch, chat="555")
    for cmd in ("/help", "/start", "/help@MyTimBot"):
        (chat, text), = run_command(monkeypatch, cmd)
        assert chat == "555" and "Your chat ID: 555" in text
        for name in ("/watchlist", "/alerts", "/status", "/test"):
            assert name in text


def test_start_from_unknown_chat_explains_how_to_authorise(monkeypatch):
    set_telegram(monkeypatch, chat="555")
    (_, text), = run_command(monkeypatch, "/start", chat_id="999")
    assert "Your chat ID: 999" in text and "TELEGRAM_CHAT_ID" in text


def test_commands_are_refused_for_unknown_chats(monkeypatch):
    set_telegram(monkeypatch, chat="555")
    (_, text), = run_command(monkeypatch, "/watchlist", chat_id="999")
    assert "not authorised" in text and "999" in text


def test_watchlist_command(monkeypatch):
    set_telegram(monkeypatch)
    monkeypatch.setattr(app, "get_snapshot", lambda s: {"error": None, "price": 100.0, "chg_24h": 1.5, "age": 3.0})
    (_, text), = run_command(monkeypatch, "/watchlist")
    assert "AAPL: 100.00 (24h +1.50%" in text and "BTC-USD" in text


def test_alerts_command_empty_and_filled(monkeypatch):
    set_telegram(monkeypatch)
    (_, text), = run_command(monkeypatch, "/alerts")
    assert "No alerts stored yet" in text
    app.try_register_alert(sample_event(), 30)
    (_, text), = run_command(monkeypatch, "/alerts")
    assert "BTC-USD" in text and "Rapid Price Increase" in text


def test_status_command(monkeypatch):
    set_telegram(monkeypatch)
    (_, text), = run_command(monkeypatch, "/status")
    assert "TIM status" in text and "Assets watched: 13" in text and "cooldown: 30 min" in text


def test_test_command_sends_a_notification_instead_of_a_text_reply(monkeypatch):
    set_telegram(monkeypatch)
    called = []
    monkeypatch.setattr(app, "send_test_notification", lambda chat_id=None: called.append(chat_id) or (True, "ok"))
    assert run_command(monkeypatch, "/test") == []
    assert called == ["555"]


def test_unknown_command(monkeypatch):
    set_telegram(monkeypatch)
    (_, text), = run_command(monkeypatch, "/dance")
    assert "Unknown command" in text


def test_plain_text_messages_are_ignored(monkeypatch):
    set_telegram(monkeypatch)
    assert run_command(monkeypatch, "hello bot") == []


# =============================================================================
# Chart patterns: Double Top/Bottom, Head and Shoulders (synthetic data only)
# =============================================================================
def double_top_closes():
    # Tops near 120.5 and 120.0 (index 30 and 60), pullback to ~104.5 at index 45, then a fall.
    # The close first drops below the neckline (104.5 minus 0.25 ATR) at index 76.
    return np.interp(np.arange(91), [0, 30, 45, 60, 90], [100, 120, 105, 119.5, 90])


def head_shoulders_closes():
    # Shoulders ~115.5 (index 20 and 80), head ~125.5 (index 50), flat neckline ~104.5.
    # The fall after the right shoulder is steep: the first close below the neckline is index 85,
    # but the right shoulder is only "known" 6 candles after index 80, so the event is at index 86.
    return np.interp(np.arange(96), [0, 20, 30, 50, 65, 80, 95], [100, 115, 105, 125, 105, 115, 80])


def pattern_df(closes, spike_at=None, volume=True):
    vols = np.full(len(closes), 1000.0 if volume else 0.0)
    if spike_at is not None and volume:
        vols[spike_at] = 3000.0      # 3x the 20-candle average
    return app.add_indicators(synth(closes, vols))


def chart_pattern(ind, name, only_last=False):
    return [e for e in app.detect_chart_patterns(ind, "TEST", "1h", only_last) if e["alert_type"] == name]


def test_confirmed_double_top():
    ind = pattern_df(double_top_closes(), spike_at=76)
    events = chart_pattern(ind, "Double Top")
    assert len(events) == 1
    ev = events[0]
    assert ev["status"] == "Confirmed" and ev["direction"] == "bear" and ev["notify"] is True
    assert ev["event_time"] == ind.index[76]
    assert ev["neckline"] == pytest.approx(104.5, abs=0.1)
    # measured move: Step 1 height = mean(120.5, 120.0) - 104.5 = 15.75
    #                Step 2 reference = 104.5 - 15.75 = 88.75
    assert ev["target"] == pytest.approx(88.75, abs=0.1)
    assert "Textbook" in ev["explanation"] and "not a forecast" in ev["explanation"]


def test_double_top_without_volume_confirmation_is_potential():
    ev = chart_pattern(pattern_df(double_top_closes()), "Double Top")[0]
    assert ev["status"] == "Potential" and ev["notify"] is False and "NOT confirmed" in ev["explanation"]


def test_double_top_on_instrument_without_volume_data_stays_potential():
    ev = chart_pattern(pattern_df(double_top_closes(), volume=False), "Double Top")[0]
    assert ev["status"] == "Potential" and ev["notify"] is False
    assert "not available" in ev["explanation"]


def test_double_top_is_reported_only_after_the_neckline_candle_closes():
    ind = pattern_df(double_top_closes(), spike_at=76)
    assert chart_pattern(ind.iloc[:76], "Double Top") == []                   # candle 76 not closed yet
    assert len(chart_pattern(ind.iloc[:77], "Double Top", only_last=True)) == 1


def test_confirmed_double_bottom():
    ind = pattern_df(200 - double_top_closes(), spike_at=76)      # mirror image of the double top
    events = chart_pattern(ind, "Double Bottom")
    assert len(events) == 1
    ev = events[0]
    assert ev["status"] == "Confirmed" and ev["direction"] == "bull" and ev["event_time"] == ind.index[76]
    assert ev["neckline"] == pytest.approx(95.5, abs=0.1)


def test_confirmed_head_and_shoulders():
    ind = pattern_df(head_shoulders_closes(), spike_at=85)
    all_events = app.detect_chart_patterns(ind, "TEST", "1h", only_last=False)
    assert [e["alert_type"] for e in all_events] == ["Head and Shoulders"]
    ev = all_events[0]
    assert ev["status"] == "Confirmed" and ev["direction"] == "bear" and ev["notify"] is True
    assert ev["event_time"] == ind.index[86]
    assert ev["neckline"] == pytest.approx(104.5, abs=0.1)
    assert "shoulder, head, shoulder" in ev["explanation"]


def test_confirmed_inverse_head_and_shoulders():
    ind = pattern_df(200 - head_shoulders_closes(), spike_at=85)
    all_events = app.detect_chart_patterns(ind, "TEST", "1h", only_last=False)
    assert [e["alert_type"] for e in all_events] == ["Inverse Head and Shoulders"]
    ev = all_events[0]
    assert ev["status"] == "Confirmed" and ev["direction"] == "bull" and ev["event_time"] == ind.index[86]
    assert ev["neckline"] == pytest.approx(95.5, abs=0.1)


def test_double_top_fails_if_price_closes_above_the_tops():
    # after the second top the price rallies to 123 (above both tops): the pattern is invalid
    closes = np.interp(np.arange(91), [0, 30, 45, 60, 66, 75, 90], [100, 120, 105, 119.5, 116, 123, 90])
    assert chart_pattern(pattern_df(closes, spike_at=80), "Double Top") == []


def test_no_chart_patterns_in_a_quiet_range():
    ind = app.add_indicators(synth(ranging_closes(100)))
    assert app.detect_chart_patterns(ind, "TEST", "1h", only_last=False) == []


def test_pattern_events_includes_chart_patterns():
    ind = pattern_df(double_top_closes(), spike_at=76)
    types = {e["alert_type"] for e in app.pattern_events(ind, "TEST", "1h", only_last=False)}
    assert "Double Top" in types


def test_breakout_on_instrument_without_volume_stays_potential():
    closes = np.append(ranging_closes(100), 112.0)
    ind = app.add_indicators(synth(closes, np.zeros(len(closes))))
    events = [e for e in app.detect_breakouts(ind, "TEST", "1h") if e["alert_type"] == "Bullish Breakout"]
    assert len(events) == 1
    assert events[0]["status"] == "Potential" and events[0]["notify"] is False
    assert "not available" in events[0]["explanation"]


def test_index_labels_and_type(monkeypatch):
    assert app.asset_label("^GSPC") == "^GSPC (S&P 500 Index)"
    assert app.asset_label("^OEX") == "^OEX (S&P 100 Index)"
    assert app.asset_label("AAPL") == "AAPL"
    df = synth(np.linspace(100, 110, 50), freq="5min")
    monkeypatch.setattr(app, "fetch_candles", lambda s, tf: app.FetchResult(df, None, app.utc_now()))
    assert app.get_snapshot("^GSPC")["type"] == "Index"

def test_pending_symbol_is_added_when_it_starts_trading(monkeypatch):
    set_telegram(monkeypatch)
    df = synth([100, 101, 102], freq="1D", start="2026-11-16")
    monkeypatch.setattr(app, "fetch_candles", lambda s, tf: app.FetchResult(df, None, app.utc_now()))
    fixed_now(monkeypatch, "2026-11-18 12:00")
    sent = []
    monkeypatch.setattr(app, "tg_send_message", lambda text, chat_id=None: sent.append(text) or (True, {}))
    app.check_pending_symbols(app.load_settings())
    cfg = app.load_settings()
    assert "ANT" in cfg["watchlist"] and "ANT" not in cfg["pending_symbols"]
    assert len(sent) == 1 and "ANT (Anthropic)" in sent[0]


def test_pending_symbol_stays_pending_without_data(monkeypatch):
    set_telegram(monkeypatch)
    monkeypatch.setattr(app, "fetch_candles", lambda s, tf: app.FetchResult(None, "no data", app.utc_now()))
    sent = []
    monkeypatch.setattr(app, "tg_send_message", lambda *a, **k: sent.append(1) or (True, {}))
    app.check_pending_symbols(app.load_settings())
    cfg = app.load_settings()
    assert "ANT" not in cfg["watchlist"] and "ANT" in cfg["pending_symbols"] and sent == []


def test_old_data_of_a_previous_security_does_not_activate_the_symbol(monkeypatch):
    df = synth([10, 11, 12], freq="1D", start="2006-01-01")      # ancient candles
    monkeypatch.setattr(app, "fetch_candles", lambda s, tf: app.FetchResult(df, None, app.utc_now()))
    fixed_now(monkeypatch, "2026-11-18 12:00")
    app.check_pending_symbols(app.load_settings())
    assert "ANT" in app.load_settings()["pending_symbols"]

# =============================================================================
# 4h candles, Change of Character, flip zones, context, Telegram routing
# =============================================================================
def test_resample_4h_builds_blocks_of_four_hourly_candles():
    df = synth(np.arange(10, 18, dtype=float), start="2024-01-01 00:00")      # 8 hourly candles
    out = app.resample_4h(df)
    assert list(out.index) == [pd.Timestamp("2024-01-01 00:00"), pd.Timestamp("2024-01-01 04:00")]
    first = df.iloc[:4]
    assert out["Open"].iloc[0] == first["Open"].iloc[0]
    assert out["High"].iloc[0] == first["High"].max() and out["Low"].iloc[0] == first["Low"].min()
    assert out["Close"].iloc[0] == first["Close"].iloc[-1] and out["Volume"].iloc[0] == 4000


def test_resample_4h_restarts_blocks_every_day():
    df = synth(np.arange(12, dtype=float) + 10)
    df.index = pd.DatetimeIndex(list(pd.date_range("2024-01-01 14:00", periods=6, freq="1h")) +
                                list(pd.date_range("2024-01-02 14:00", periods=6, freq="1h")))
    out = app.resample_4h(df)
    assert list(out.index) == [pd.Timestamp("2024-01-01 14:00"), pd.Timestamp("2024-01-01 18:00"),
                               pd.Timestamp("2024-01-02 14:00"), pd.Timestamp("2024-01-02 18:00")]


def test_download_builds_4h_candles_from_hourly_data(monkeypatch):
    fake_ticker(monkeypatch, raw_history())
    res = app._download("AAPL", "4h")
    assert res.error is None and len(res.df) == 1
    assert res.df["Open"].iloc[0] == 10 and res.df["Volume"].iloc[0] == 300


def choch_df(kind="break", mirror=False):
    # Uptrend: swing highs at 20/40/60 (112, 122, 130), swing lows at 30/50 (106, 114 = last higher low).
    # From 60 the price falls 2.5 per candle: first close below the last higher low is candle 67 (112.5).
    if kind == "win":      # keeps falling: the 1R target (about 94) is reached around candle 76
        xs, ys = [0, 10, 20, 30, 40, 50, 60, 70, 80], [95, 100, 112, 106, 122, 114, 130, 105, 85]
    elif kind == "loss":   # rebounds above the stop (about 131) instead
        xs, ys = [0, 10, 20, 30, 40, 50, 60, 67, 80], [95, 100, 112, 106, 122, 114, 130, 112.5, 140]
    else:
        xs, ys = [0, 10, 20, 30, 40, 50, 60, 70], [95, 100, 112, 106, 122, 114, 130, 105]
    closes = np.interp(np.arange(xs[-1] + 1), xs, ys)
    if mirror:
        closes = 230 - closes
    return app.add_indicators(synth(closes))


def test_change_of_character_bearish_on_4h():
    ind = choch_df()
    events = app.detect_choch(ind, "TEST", "4h", only_last=False)
    assert len(events) == 1
    ev = events[0]
    assert ev["alert_type"] == "Change of Character" and ev["direction"] == "bear"
    assert ev["status"] == "Confirmed" and ev["notify"] is True
    assert ev["event_time"] == ind.index[67]
    plan = ev["plan"]
    assert plan["entry"] == pytest.approx(ind["Close"].iloc[67])
    # stop = last swing high + 0.25 ATR; target = entry - risk (1R)
    assert plan["stop"] == pytest.approx(ind["High"].iloc[60] + 0.25 * ind["ATR14"].iloc[67])
    assert plan["target"] == pytest.approx(2 * plan["entry"] - plan["stop"])
    assert "first close against the trend" in ev["explanation"] and "not advice" in ev["explanation"]


def test_change_of_character_bullish_mirror():
    ind = choch_df(mirror=True)
    events = app.detect_choch(ind, "TEST", "4h", only_last=False)
    assert len(events) == 1
    ev = events[0]
    assert ev["direction"] == "bull" and ev["event_time"] == ind.index[67]
    assert ev["plan"]["stop"] == pytest.approx(ind["Low"].iloc[60] - 0.25 * ind["ATR14"].iloc[67])
    assert ev["plan"]["target"] == pytest.approx(2 * ev["plan"]["entry"] - ev["plan"]["stop"])


def test_change_of_character_below_4h_is_only_potential():
    ev = app.detect_choch(choch_df(), "TEST", "1h", only_last=False)[0]
    assert ev["status"] == "Potential" and ev["notify"] is False and "below 4h" in ev["explanation"]


def test_change_of_character_waits_for_the_candle_to_close():
    ind = choch_df()
    assert app.detect_choch(ind.iloc[:67], "TEST", "4h", only_last=True) == []
    assert len(app.detect_choch(ind.iloc[:68], "TEST", "4h", only_last=True)) == 1


def test_no_change_of_character_in_a_quiet_range():
    ind = app.add_indicators(synth(ranging_closes(100)))
    assert app.detect_choch(ind, "TEST", "4h", only_last=False) == []


def test_pattern_events_includes_change_of_character():
    types = {e["alert_type"] for e in app.pattern_events(choch_df(), "TEST", "4h", only_last=False)}
    assert "Change of Character" in types


def test_plan_reaches_the_1r_target():
    ind = choch_df("win")
    signals = app.detect_choch(ind, "TEST", "4h", only_last=False)
    assert len(signals) == 1
    trade, = app.simulate_plan(ind, signals, cost_pct=0.0)
    assert trade["outcome"] == "win" and trade["r"] == pytest.approx(1.0)


def test_plan_hits_the_stop():
    ind = choch_df("loss")
    signals = app.detect_choch(ind, "TEST", "4h", only_last=False)
    assert len(signals) == 1
    trade, = app.simulate_plan(ind, signals, cost_pct=0.0)
    assert trade["outcome"] == "loss" and trade["r"] == pytest.approx(-1.0)


def plan_df_and_event(high1, low1):
    idx = pd.date_range("2024-01-01", periods=3, freq="1h")
    df = pd.DataFrame({"Open": [100.0] * 3, "High": [100.5, high1, 101.0], "Low": [99.5, low1, 99.0],
                       "Close": [100.0] * 3, "Volume": [1.0] * 3}, index=idx)
    ev = {"event_time": idx[0], "direction": "bull",
          "plan": {"entry": 100.0, "stop": 95.0, "target": 105.0, "risk_pct": 5.0}}
    return df, ev


def test_candle_touching_stop_and_target_counts_as_a_loss():
    df, ev = plan_df_and_event(high1=106.0, low1=94.0)     # one candle reaches BOTH levels
    trade, = app.simulate_plan(df, [ev], cost_pct=0.0)
    assert trade["outcome"] == "loss" and trade["r"] == -1.0


def test_plan_costs_and_open_trades():
    df, ev = plan_df_and_event(high1=106.0, low1=99.0)     # reaches only the target
    trade, = app.simulate_plan(df, [ev], cost_pct=0.5)
    # cost in R = 0.5 % / 5 % risk = 0.1 R, so a win is 1 - 0.1 = 0.9
    assert trade["outcome"] == "win" and trade["r"] == pytest.approx(0.9)
    df2, ev2 = plan_df_and_event(high1=101.0, low1=99.0)   # reaches neither level
    trade2, = app.simulate_plan(df2, [ev2], cost_pct=0.5)
    assert trade2["outcome"] == "open" and trade2["r"] == 0.0


def flip_df(mirror=False, bad_retest=False):
    # candle 100 = breakout (volume 3x); candle 103 = the first retest of the broken level
    closes = np.append(ranging_closes(100), [112.0, 110.0, 108.0, 107.2, 108.5, 109.0])
    ohlc = [106.0, 106.2, 104.8, 105.0] if bad_retest else [106.8, 107.6, 105.7, 107.2]   # O, H, L, C
    if mirror:
        closes = 200 - closes
        o, h, l, c = ohlc
        ohlc = [200 - o, 200 - l, 200 - h, 200 - c]
    vols = np.full(len(closes), 1000.0)
    vols[100] = 3000.0
    df = synth(closes, vols)
    df.loc[df.index[103], ["Open", "High", "Low", "Close"]] = ohlc
    return app.add_indicators(df)


def test_flip_zone_retest_after_a_breakout():
    ind = flip_df()
    events = app.detect_flip_retests(ind, "TEST", "1h", only_last=False)
    assert len(events) == 1
    ev = events[0]
    assert ev["alert_type"] == "Flip Zone Retest" and ev["direction"] == "bull"
    assert ev["status"] == "Confirmed" and ev["notify"] is True and ev["event_time"] == ind.index[103]
    assert ev["level"] == pytest.approx(105.48, abs=0.5)
    assert "flip zone" in ev["explanation"]


def test_failed_first_retest_gives_no_signal():
    # the retest candle touches the zone but closes BELOW it: the breakout failed
    assert app.detect_flip_retests(flip_df(bad_retest=True), "TEST", "1h", only_last=False) == []


def test_flip_zone_retest_after_a_breakdown():
    ind = flip_df(mirror=True)
    events = app.detect_flip_retests(ind, "TEST", "1h", only_last=False)
    assert len(events) == 1
    ev = events[0]
    assert ev["direction"] == "bear" and ev["event_time"] == ind.index[103]
    assert ev["level"] == pytest.approx(94.52, abs=0.5)


def test_run_context_flags_chasing():
    ind = app.add_indicators(synth(np.linspace(100, 130, 120)))        # 5 days of a steady rise
    last = ind.index[-1]
    up = app.make_event("T", "1h", "Bullish Breakout", "Confirmed", "bull", last, 130.0, 1.0, "x", "base.", True)
    app.add_run_context(ind, up)
    ref = ind["Close"].iloc[-73]                                       # close exactly 72 hours earlier
    assert up["run_pct"] == pytest.approx((ind["Close"].iloc[-1] / ref - 1) * 100)
    assert up["chasing"] is True and "chasing" in up["explanation"]
    down = app.make_event("T", "1h", "Bearish Breakdown", "Confirmed", "bear", last, 130.0, 1.0, "x", "base.", True)
    app.add_run_context(ind, down)
    assert down["chasing"] is False and "chasing" not in down["explanation"]


def test_five_minute_alerts_can_be_kept_out_of_telegram(monkeypatch):
    set_telegram(monkeypatch)
    sent = []
    monkeypatch.setattr(app, "send_alert_notification", lambda ev, ind, tf, chat_id=None: sent.append(tf) or (True, "ok"))
    ind = five_min_df(102.5, 2500)
    fixed_now(monkeypatch, ind.index[-1] + pd.Timedelta(minutes=10))
    settings = app.load_settings()
    settings["telegram_timeframes"] = ["15m", "1h", "4h", "1d"]
    for ev in app.detect_price_events("BTC-USD", ind, settings):
        app.process_event(ev, ind, "5m", settings)
    assert len(app.get_alerts()) == 2 and sent == []         # stored and shown, but not sent

def test_five_minute_alerts_are_not_sent(monkeypatch):
    set_telegram(monkeypatch)
    sent = []
    monkeypatch.setattr(app, "send_alert_notification", lambda ev, ind, tf, chat_id=None: sent.append(tf) or (True, "ok"))
    ind = five_min_df(102.5, 2500)
    fixed_now(monkeypatch, ind.index[-1] + pd.Timedelta(minutes=10))
    settings = app.load_settings()
    for ev in app.detect_price_events("BTC-USD", ind, settings):
        app.process_event(ev, ind, "5m", settings)
    assert sent == []


def test_long_explanations_keep_the_disclaimer():
    ev = notification_event()
    ev["explanation"] = "x" * 3000
    text = app.build_message(ev)
    assert len(text) <= 1024 and text.endswith(app.DISCLAIMER) and "..." in text

# =============================================================================
# Telegram routing by timeframe, necklines on the chart, forming patterns
# =============================================================================
def test_old_telegram_5m_setting_is_migrated():
    app.SETTINGS_PATH.write_text('{"telegram_5m": false}', encoding="utf-8")
    tfs = app.load_settings()["telegram_timeframes"]
    assert "5m" not in tfs and "1h" in tfs and "1d" in tfs


def test_telegram_routing_by_timeframe(monkeypatch):
    set_telegram(monkeypatch)
    sent = []
    monkeypatch.setattr(app, "send_alert_notification", lambda ev, ind, tf, chat_id=None: sent.append(tf) or (True, "ok"))
    settings = app.load_settings()
    settings["telegram_timeframes"] = ["1h", "1d"]
    ind = five_min_df(102.5, 2500)
    for i, tf in enumerate(["5m", "15m", "1h", "1d"]):
        app.process_event(sample_event(asset=f"A{i}", tf=tf), ind, tf, settings)
    assert sent == ["1h", "1d"]
    assert len(app.get_alerts()) == 4          # every alert is still stored in the history


def test_status_command_shows_telegram_routing(monkeypatch):
    set_telegram(monkeypatch)
    (_, text), = run_command(monkeypatch, "/status")
    assert "Sent to Telegram: 15m, 1h, 4h, 1d" in text


def test_double_top_event_carries_its_neckline():
    ind = pattern_df(double_top_closes(), spike_at=76)
    ev = chart_pattern(ind, "Double Top")[0]
    (i0, y0), (i1, y1) = ev["neck_line"]
    assert (i0, i1) == (30, 76)                # from the first top to the detection candle
    assert y0 == pytest.approx(104.5, abs=0.1) and y1 == pytest.approx(104.5, abs=0.1)


def test_head_and_shoulders_neckline_runs_from_the_first_trough():
    ind = pattern_df(head_shoulders_closes(), spike_at=85)
    ev = chart_pattern(ind, "Head and Shoulders")[0]
    (i0, y0), (i1, y1) = ev["neck_line"]
    assert (i0, i1) == (30, 86)
    assert y0 == pytest.approx(104.5, abs=0.1) and y1 == pytest.approx(104.5, abs=0.1)
    inverse = chart_pattern(pattern_df(200 - head_shoulders_closes(), spike_at=85), "Inverse Head and Shoulders")[0]
    assert inverse["neck_line"][0][1] == pytest.approx(95.5, abs=0.1)


def test_chart_draws_the_neckline_of_a_pattern():
    ind = pattern_df(double_top_closes(), spike_at=76)
    ev = chart_pattern(ind, "Double Top")[0]
    fig = app.build_chart(ind, "T", "1h", [], [ev], candles=100, with_osc=False)
    lines = [t for t in fig.data if t.name == "Neckline"]
    assert len(lines) == 1 and list(lines[0].y) == pytest.approx([104.5, 104.5], abs=0.1)
    # only the last 40 candles are shown: the line is cut at the left edge of the chart
    fig2 = app.build_chart(ind, "T", "1h", [], [ev], candles=40, with_osc=False)
    line2 = [t for t in fig2.data if t.name == "Neckline"][0]
    assert line2.x[0] == ind.index[51].strftime("%Y-%m-%d %H:%M")
    assert line2.x[1] == ind.index[76].strftime("%Y-%m-%d %H:%M")


def test_forming_double_top_is_reported_before_the_break():
    ind = pattern_df(double_top_closes(), spike_at=76)
    cut = ind.iloc[:70]                        # candle 69: the second top is known, the neckline is not broken
    assert chart_pattern(cut, "Double Top") == []
    forming = app.detect_forming_patterns(cut, "TEST", "1h")
    assert len(forming) == 1
    ev = forming[0]
    assert ev["alert_type"] == "Double Top" and ev["status"] == "Forming" and ev["direction"] == "bear"
    assert ev["notify"] is False and ev["event_time"] == cut.index[-1]
    assert ev["neckline"] == pytest.approx(104.5, abs=0.1) and ev["limit"] > 120.5
    assert ev["neck_line"][0][0] == 30 and ev["neck_line"][1][0] == 69
    assert "NOT a signal yet" in ev["explanation"]
    # once the neckline is broken it is no longer "forming": it becomes a real event
    assert app.detect_forming_patterns(ind.iloc[:77], "TEST", "1h") == []
    assert len(chart_pattern(ind.iloc[:77], "Double Top", only_last=True)) == 1


def test_forming_pattern_expires_if_no_break_comes():
    closes = np.interp(np.arange(101), [0, 30, 45, 60, 100], [100, 120, 105, 119.5, 110])   # slow decline
    ind = pattern_df(closes)
    assert len(app.detect_forming_patterns(ind.iloc[:80], "TEST", "1h")) == 1
    assert app.detect_forming_patterns(ind, "TEST", "1h") == []       # 40 candles later: the window is over


def test_forming_patterns_are_not_part_of_the_alert_events():
    cut = pattern_df(double_top_closes(), spike_at=76).iloc[:70]
    assert all(e["status"] != "Forming" for e in app.pattern_events(cut, "TEST", "1h", only_last=False))


def test_chart_draws_forming_pattern_as_dotted_neckline():
    ind = pattern_df(double_top_closes()).iloc[:70]
    forming = app.detect_forming_patterns(ind, "TEST", "1h")
    fig = app.build_chart(ind, "T", "1h", [], forming, candles=100, with_osc=False)
    line = [t for t in fig.data if t.name == "Neckline"][0]
    assert line.line.dash == "dot" and "forming" in line.text[1]
    assert not [t for t in fig.data if getattr(t, "mode", None) == "markers"]    # no signal marker

# =============================================================================
# Prediction tracking and statistics
# =============================================================================
def confirmed_prediction_event(tf="1h", direction="bull", ts="2024-01-01 10:00"):
    return app.make_event("BTC-USD", tf, "Bullish Breakout" if direction == "bull" else "Bearish Breakdown",
                          "Confirmed", direction, pd.Timestamp(ts), 100.0, 0.8,
                          f"{tf} candle", "Synthetic confirmed test signal", True)


def test_only_confirmed_directional_events_are_predictions():
    alert_id = app.try_register_alert(confirmed_prediction_event(), 0)
    assert alert_id is not None
    pred_id = app.record_prediction(alert_id, confirmed_prediction_event(), "1h", app.load_settings())
    assert pred_id is not None
    rows = app.db_query("SELECT direction, timeframe, horizon_minutes, entry_price FROM predictions")
    assert rows == [("bull", "1h", 240, 100.0)]


def test_prediction_horizon_starts_after_signal_candle_closes():
    ev = confirmed_prediction_event(ts="2024-01-01 10:00")
    alert_id = app.try_register_alert(ev, 0)
    app.record_prediction(alert_id, ev, "1h", app.load_settings())
    row = app.db_query("SELECT signal_time, due_at FROM predictions")[0]
    assert row == ("2024-01-01 11:00:00", "2024-01-01 15:00:00")


def test_process_event_stores_prediction_and_marks_telegram_delivery(monkeypatch):
    set_telegram(monkeypatch)
    sent = []
    monkeypatch.setattr(app, "send_alert_notification", lambda ev, ind, tf, chat_id=None: sent.append(ev) or (True, "ok"))
    ev = confirmed_prediction_event()
    ind = add_test_indicators_for_prediction()
    settings = app.load_settings()
    app.process_event(ev, ind, "1h", settings)
    row = app.db_query("SELECT telegram_sent, horizon_minutes FROM predictions")[0]
    assert row == (1, 240)
    assert sent and sent[0]["prediction_horizon_minutes"] == 240


def add_test_indicators_for_prediction():
    idx = pd.date_range("2024-01-01 00:00", periods=80, freq="1h")
    df = pd.DataFrame({"Open": np.linspace(90, 100, 80), "High": np.linspace(91, 101, 80),
                       "Low": np.linspace(89, 99, 80), "Close": np.linspace(90, 100, 80),
                       "Volume": np.full(80, 1000.0)}, index=idx)
    return app.add_indicators(df)


def test_potential_and_info_events_do_not_create_predictions():
    ev = app.make_event("BTC-USD", "1h", "Bullish Breakout", "Potential", "bull",
                        pd.Timestamp("2024-01-01 10:00"), 100, 0.5, "1h candle", "potential", False)
    alert_id = app.try_register_alert(ev, 0)
    assert app.record_prediction(alert_id, ev, "1h", app.load_settings()) is None
    assert app.db_query("SELECT COUNT(*) FROM predictions")[0][0] == 0


def test_evaluator_marks_long_success_and_notifies_only_once(monkeypatch):
    set_telegram(monkeypatch)
    ev = confirmed_prediction_event(ts="2024-01-01 10:00")
    alert_id = app.try_register_alert(ev, 0)
    pred_id = app.record_prediction(alert_id, ev, "1h", app.load_settings())
    app.mark_prediction_telegram_sent(pred_id)
    idx = pd.date_range("2024-01-01 10:00", periods=6, freq="1h")
    prices = [100.0, 100.1, 100.2, 100.3, 101.0, 101.1]
    df = pd.DataFrame({"Open": prices, "High": np.array(prices) + 0.2,
                       "Low": np.array(prices) - 0.2, "Close": prices,
                       "Volume": np.full(6, 1000.0)}, index=idx)
    monkeypatch.setattr(app, "fetch_candles", lambda symbol, tf: app.FetchResult(df, None, app.utc_now()))
    fixed_now(monkeypatch, "2024-01-01 16:00")
    messages = []
    monkeypatch.setattr(app, "tg_send_message", lambda text, chat_id=None: messages.append(text) or (True, {}))
    assert app.evaluate_due_predictions() == 1
    row = app.db_query("SELECT outcome, return_pct, result_notified FROM predictions")[0]
    assert row[0] == "SUCCESS" and row[1] == pytest.approx(1.0) and row[2] == 1
    assert len(messages) == 1 and "Prediction result: SUCCESS" in messages[0]
    assert app.evaluate_due_predictions() == 0
    assert len(messages) == 1


def test_short_outcome_and_neutral_threshold():
    assert app._prediction_outcome("bear", -0.5, 0.25) == "SUCCESS"
    assert app._prediction_outcome("bear", 0.5, 0.25) == "FAILED"
    assert app._prediction_outcome("bull", 0.1, 0.25) == "NEUTRAL"


def test_stats_command_groups_by_pattern_and_timeframe(monkeypatch):
    set_telegram(monkeypatch)
    for tf, side, pattern, outcome in [("1h", "bull", "Bullish Breakout", "SUCCESS"),
                                      ("1h", "bear", "Bearish Breakdown", "FAILED"),
                                      ("4h", "bull", "Bullish Breakout", "SUCCESS")]:
        ev = app.make_event("BTC-USD", tf, pattern, "Confirmed", side, pd.Timestamp("2024-01-01 10:00"),
                            100.0, 0.5, f"{tf} candle", "test", True)
        alert_id = app.try_register_alert(ev, 0)
        app.record_prediction(alert_id, ev, tf, app.load_settings())
        app.db_execute("UPDATE predictions SET outcome=?, evaluated_at=? WHERE alert_id=?",
                       (outcome, "2024-01-02 00:00:00", alert_id))
    assert "66.7%" in app.cmd_stats()
    assert "Bullish Breakout" in app.cmd_stats(mode="patterns")
    assert "1h" in app.cmd_stats(mode="timeframes")
    assert "1h / Bullish Breakout" in app.cmd_stats(mode="detailed")


def test_stats_telegram_command(monkeypatch):
    set_telegram(monkeypatch)
    (_, reply), = run_command(monkeypatch, "/stats patterns")
    assert "Prediction statistics" in reply and "No evaluated predictions yet" in reply


def test_streamlit_secrets_are_read_when_env_is_empty(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    fake_streamlit = type("FakeStreamlit", (), {"secrets": {"TELEGRAM_BOT_TOKEN": "123:ABC", "TELEGRAM_CHAT_ID": "555"}})()
    monkeypatch.setattr(app, "st", fake_streamlit)
    assert app.tg_token() == "123:ABC"
    assert app.tg_chat_id() == "555"
    assert app.telegram_configured()


def test_prediction_keeps_neutral_threshold_from_signal_time():
    ev = confirmed_prediction_event()
    alert_id = app.try_register_alert(ev, 0)
    app.record_prediction(alert_id, ev, "1h", {**app.load_settings(), "prediction_neutral_pct": 0.25})
    row = app.db_query("SELECT neutral_pct FROM predictions WHERE alert_id=?", (alert_id,))[0]
    assert row[0] == pytest.approx(0.25)
    # Changing the dashboard setting later does not rewrite this prediction's evaluation rule.
    assert app._prediction_outcome("bull", 0.5, row[0]) == "SUCCESS"


def test_prediction_pnl_is_direction_aware():
    assert app.prediction_pnl_pct("bull", 1.25) == pytest.approx(1.25)
    assert app.prediction_pnl_pct("bear", -1.25) == pytest.approx(1.25)
    assert app.prediction_pnl_pct("bear", 1.25) == pytest.approx(-1.25)
    assert app.prediction_pnl_pct("bull", None) is None


def test_prediction_history_includes_pending_and_evaluated_rows():
    bull = confirmed_prediction_event(direction="bull", ts="2024-01-01 10:00")
    bear = confirmed_prediction_event(direction="bear", ts="2024-01-01 11:00")
    bear["asset"] = "AAPL"
    bull_id = app.try_register_alert(bull, 0)
    bear_id = app.try_register_alert(bear, 0)
    app.record_prediction(bull_id, bull, "1h", app.load_settings())
    app.record_prediction(bear_id, bear, "1h", app.load_settings())
    app.db_execute("UPDATE predictions SET outcome=?, return_pct=?, exit_price=? WHERE alert_id=?",
                   ("SUCCESS", -1.5, 98.5, bear_id))

    rows = app.get_predictions(limit=10)
    assert len(rows) == 2
    short_row = rows[rows["asset"] == "AAPL"].iloc[0]
    assert short_row["outcome"] == "SUCCESS"
    assert app.prediction_pnl_pct(short_row["direction"], short_row["return_pct"]) == pytest.approx(1.5)
    assert rows[rows["asset"] == "BTC-USD"].iloc[0]["outcome"] is None
    assert "Synthetic confirmed test signal" in short_row["description"]


def test_prediction_history_filters_and_sorts_by_directional_pnl():
    cases = [
        ("AAPL", "bull", "Bullish Breakout", 0.6),
        ("MSFT", "bear", "Bearish Breakdown", -2.1),
        ("NVDA", "bull", "Bullish Breakout", -0.4),
    ]
    for i, (asset, side, pattern, raw_change) in enumerate(cases):
        ev = app.make_event(asset, "1h", pattern, "Confirmed", side,
                            pd.Timestamp("2024-01-01 10:00") + pd.Timedelta(hours=i),
                            100.0, 0.5, "1h candle", "test", True)
        alert_id = app.try_register_alert(ev, 0)
        app.record_prediction(alert_id, ev, "1h", app.load_settings())
        app.db_execute("UPDATE predictions SET outcome=?, return_pct=? WHERE alert_id=?",
                       ("SUCCESS", raw_change, alert_id))

    best = app.get_predictions(limit=10, sort_by="best")
    worst = app.get_predictions(limit=10, sort_by="worst")
    bullish = app.get_predictions(limit=10, pattern="Bullish Breakout")
    assert best.iloc[0]["asset"] == "MSFT"  # a 2.1% fall benefits the SHORT prediction
    assert worst.iloc[0]["asset"] == "NVDA"
    assert set(bullish["asset"]) == {"AAPL", "NVDA"}


def test_prediction_history_command_shows_pending_and_directional_result():
    ev = confirmed_prediction_event(direction="bear")
    ev["asset"] = "AAPL"
    alert_id = app.try_register_alert(ev, 0)
    app.record_prediction(alert_id, ev, "1h", app.load_settings())
    app.db_execute("UPDATE predictions SET outcome=?, return_pct=? WHERE alert_id=?",
                   ("SUCCESS", -0.8, alert_id))
    reply = app.cmd_prediction_history()
    assert "AAPL" in reply and "SHORT" in reply and "+0.80%" in reply and "SUCCESS" in reply


def test_underscore_stats_commands_are_supported(monkeypatch):
    set_telegram(monkeypatch)
    for tf, pattern, outcome in [("1h", "Bullish Breakout", "SUCCESS"),
                                 ("4h", "Bearish Breakdown", "FAILED")]:
        ev = app.make_event("BTC-USD", tf, pattern, "Confirmed", "bull", pd.Timestamp("2024-01-01 10:00"),
                            100.0, 0.5, f"{tf} candle", "test", True)
        alert_id = app.try_register_alert(ev, 0)
        app.record_prediction(alert_id, ev, tf, app.load_settings())
        app.db_execute("UPDATE predictions SET outcome=? WHERE alert_id=?", (outcome, alert_id))
    (_, patterns_text), = run_command(monkeypatch, "/stats_patterns")
    (_, timeframe_text), = run_command(monkeypatch, "/stats_1h")
    (_, pattern_text), = run_command(monkeypatch, "/stats_pattern_bullish_breakout")
    assert "Bullish Breakout" in patterns_text
    assert "1h" in timeframe_text
    assert "Bullish Breakout" in pattern_text


def test_start_includes_mobile_inline_menu(monkeypatch):
    set_telegram(monkeypatch)
    run_command(monkeypatch, "/start")
    keyboard = run_command.last_reply_markup
    assert keyboard and keyboard["inline_keyboard"]
    labels = [button["text"] for row in keyboard["inline_keyboard"] for button in row]
    assert "📊 Statistics" in labels and "📜 Prediction history" in labels


def test_telegram_callback_updates_menu_message(monkeypatch):
    set_telegram(monkeypatch)
    edited, answered = [], []
    monkeypatch.setattr(app, "tg_answer_callback_query", lambda callback_id, text=None: answered.append(callback_id) or (True, {}))
    monkeypatch.setattr(app, "tg_edit_message_text",
                        lambda text, chat_id, message_id, reply_markup=None:
                        edited.append((text, chat_id, message_id, reply_markup)) or (True, {}))
    app.handle_update({"update_id": 5, "callback_query": {
        "id": "callback-1", "data": "ui:history",
        "message": {"message_id": 7, "chat": {"id": 555}}
    }})
    assert answered == ["callback-1"]
    assert edited and "Prediction history" in edited[0][0]
    assert edited[0][1:3] == ("555", 7)


# =============================================================================
# v3 architecture: shared market cache, snapshots, and preserved user settings
# =============================================================================
def test_neutral_threshold_default_is_not_changed_by_the_upgrade():
    assert app.DEFAULT_SETTINGS["prediction_neutral_pct"] == pytest.approx(0.25)
    settings = app.load_settings()
    assert settings["prediction_neutral_pct"] == pytest.approx(0.25)


def test_fetch_candles_single_flight_shares_one_provider_request(monkeypatch):
    calls = {"n": 0}
    source = app.FetchResult(raw_history(), None, app.utc_now())
    lock = threading.Lock()

    def slow_download(symbol, tf):
        with lock:
            calls["n"] += 1
        time.sleep(0.15)
        return source

    monkeypatch.setattr(app, "_download", slow_download)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: app.fetch_candles("AAPL", "1h"), range(6)))
    assert calls["n"] == 1
    assert all(result is source for result in results)


def test_indicator_cache_reuses_only_the_same_source_dataframe():
    first_source = synth(np.linspace(100, 110, 90))
    first_result = app.FetchResult(first_source, None, app.utc_now())
    ind1 = app.get_indicators("AAPL", "1h", first_result)
    ind2 = app.get_indicators("AAPL", "1h", first_result)
    assert ind1 is ind2

    # A fresh Yahoo response must force a full recalculation, even if the shape is unchanged.
    second_source = first_source.copy()
    second_source.iloc[-1, second_source.columns.get_loc("Close")] += 1.0
    second_result = app.FetchResult(second_source, None, app.utc_now())
    ind3 = app.get_indicators("AAPL", "1h", second_result)
    assert ind3 is not ind1
    assert ind3["Close"].iloc[-1] == pytest.approx(second_source["Close"].iloc[-1])


def test_market_snapshot_roundtrip_uses_shared_database():
    snap = {
        "asset": "AAPL", "type": "Stock", "error": None, "price": 225.5,
        "chg_5m": 0.12, "chg_1h": -0.35, "chg_24h": 1.1,
        "last_candle": pd.Timestamp("2026-11-11 20:00:00"), "age": 3.0,
        "stale": False, "fetched_at": app.utc_now(),
    }
    app.save_market_snapshot(snap)
    rows = app.get_market_snapshots(["AAPL"])
    assert rows["AAPL"]["price"] == pytest.approx(225.5)
    assert rows["AAPL"]["chg_1h"] == pytest.approx(-0.35)
    assert rows["AAPL"]["stale"] is False
    assert "MSFT" not in rows


def test_worker_status_roundtrip_is_visible_to_web_runtime():
    app.RT.status.update({
        "worker_state": "running", "last_cycle": app.utc_now(), "cycles": 7,
        "cycle_seconds": 12.4, "symbols_scanned": 13,
    })
    app.publish_monitor_status()
    status = app.get_monitor_status()
    assert status["worker_state"] == "running"
    assert status["cycles"] == 7
    assert status["cycle_seconds"] == pytest.approx(12.4)
    assert status["symbols_scanned"] == 13


def test_postgres_queries_use_pool_without_global_db_lock(monkeypatch):
    class FakeCursor:
        def __init__(self, rows=None, one=None):
            self.rows = rows or []
            self.one = one
        def fetchall(self):
            return self.rows
        def fetchone(self):
            return self.one

    class FakeConnection:
        def __init__(self):
            self.sql = []
        def execute(self, sql, params=()):
            self.sql.append((sql, params))
            if sql.startswith("SELECT value"):
                return FakeCursor(rows=[("pooled",)])
            if "RETURNING id" in sql:
                return FakeCursor(one=(42,))
            return FakeCursor()

    class FakeConnectionContext:
        def __init__(self, connection):
            self.connection_obj = connection
        def __enter__(self):
            return self.connection_obj
        def __exit__(self, exc_type, exc, tb):
            return False

    class FakePool:
        def __init__(self):
            self.connection_obj = FakeConnection()
        def connection(self, timeout=None):
            return FakeConnectionContext(self.connection_obj)

    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/test")
    app.RT.db_pool = FakePool()
    app.RT.db_pool_url = app.database_url()

    class ForbiddenLock:
        def __enter__(self):
            raise AssertionError("PostgreSQL queries must not use the global SQLite lock")
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(app.RT, "db_lock", ForbiddenLock())
    assert app.db_query("SELECT value FROM demo") == [("pooled",)]
    assert app.db_execute("INSERT INTO demo (value) VALUES (?)", ("x",)) == 42
    sent_sql = [sql for sql, _ in app.RT.db_pool.connection_obj.sql]
    assert any("%s" in sql for sql in sent_sql)


def test_get_runtime_does_not_start_background_threads():
    # Streamlit owns only the UI process; worker.py starts scanner and Telegram polling.
    before = {thread.name for thread in threading.enumerate()}
    runtime = app.get_runtime()
    after = {thread.name for thread in threading.enumerate()}
    assert isinstance(runtime, app.Runtime)
    assert not ({"tim-monitor", "tim-telegram"} & (after - before))


def test_local_worker_lock_prevents_two_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    app.RT = app.Runtime()
    assert app.acquire_worker_lock() is True
    first_runtime = app.RT

    app.RT = app.Runtime()
    assert app.acquire_worker_lock() is False

    app.RT = first_runtime
    app.release_worker_lock()
    app.RT = app.Runtime()
    assert app.acquire_worker_lock() is True
    app.release_worker_lock()


def test_existing_custom_neutral_threshold_is_preserved():
    app.SETTINGS_PATH.write_text('{"prediction_neutral_pct": 0.8}', encoding="utf-8")
    settings = app.load_settings()
    assert settings["prediction_neutral_pct"] == pytest.approx(0.8)
