# =============================================================================
# Trading Intelligence Monitor (TIM) - MVP                 Run: streamlit run app.py
# Informational tool only. It does NOT execute trades or manage money.
#
# Sections: 1 Config | 2 Helpers | 3 Market data | 4 Indicators | 5 Patterns |
#           6 Database | 7 Charts | 8 Telegram | 9 Background monitor | 10 Web UI
# =============================================================================
import inspect
import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from plotly.subplots import make_subplots

# =============================================================================
# 1. CONFIG
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHARTS_DIR = BASE_DIR / "charts"
DB_PATH = DATA_DIR / "tim.db"
SETTINGS_PATH = DATA_DIR / "settings.json"
DATA_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")  # reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tim")

DEFAULT_SETTINGS = {
    "watchlist": ["AAPL", "MSFT", "NVDA", "TSLA", "SPY",
                  "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "^GSPC", "^OEX", "AMD", "RKLB"],
    "pending_symbols": ["ANT"],
    "telegram_timeframes": ["5m", "15m", "1h", "4h", "1d", "1w"],   # these go to Telegram; others: history only
    "crypto_pct": 2.0,       # % change ...
    "crypto_minutes": 5,     # ... within this many minutes
    "stock_pct": 1.0,
    "stock_minutes": 15,
    "volume_mult": 2.0,      # volume spike = N x average of previous 20 candles
    "min_breakout_atr": 0.75, # minimum close distance beyond a broken level for Telegram
    "cooldown_minutes": 30,  # no repeated alert of the same kind within this time
}

# interval/period are what Yahoo Finance understands; ttl = seconds we reuse cached data
TIMEFRAMES = {
    "5m":  {"interval": "5m",  "period": "5d",  "ttl": 50},
    "15m": {"interval": "15m", "period": "30d", "ttl": 120},
    "1h":  {"interval": "1h",  "period": "60d", "ttl": 300},
    "4h":  {"interval": "1h",  "period": "6mo", "ttl": 600},   # built from 1-hour candles (resample_4h)
    "1d":  {"interval": "1d",  "period": "5y",  "ttl": 900},   # 5 years: more signals for the plan test
    "1w":  {"interval": "1wk", "period": "5y",  "ttl": 1800},
}
INTERVAL_MIN = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}
INTRADAY = {"5m", "15m", "1h", "4h"}
PATTERN_SCAN_TFS = ["15m", "1h", "4h", "1d"]  # timeframes the background monitor scans for patterns
ALERT_TYPES = ["Double Top", "Double Bottom", "Head and Shoulders", "Inverse Head and Shoulders",
               "Change of Character", "Flip Zone Retest"]

# Pattern parameters
PIVOT_K = 5                 # a swing high/low must be the extreme of +-5 candles
LOOKBACK = 150              # candles used to find support/resistance
MIN_TOUCHES = 2             # a level needs at least 2 swing points
LEVEL_TOL_ATR = 0.5         # swing points within 0.5 ATR belong to the same level
BREAKOUT_BUFFER_ATR = 0.25  # a close must be 0.25 ATR beyond the level
BREAKOUT_VOL_MULT = 1.5     # volume confirmation for breakouts
MIN_CANDLES = 60            # minimum history for indicators and patterns
HISTORY_SCAN = 150          # how many recent candles are scanned for chart markers
# Chart pattern parameters (Double Top/Bottom, Head and Shoulders)
MIN_PATTERN_SEP = 6          # min candles between the two tops/bottoms of a double top/bottom
MAX_PATTERN_SPAN = 120       # max candles from the first to the last peak of a pattern
MAX_BREAK_DELAY = 30         # the neckline must break within 30 candles after the last peak
DOUBLE_TOL_ATR = 1.0         # two tops (bottoms) must be within 1 ATR of each other
PATTERN_MIN_DEPTH_ATR = 2.0  # pullback / head height must be at least 2 ATR (filters out noise)
HEAD_MIN_ATR = 1.0           # the head must stick out above both shoulders by at least 1 ATR
SHOULDER_TOL_ATR = 1.5       # the two shoulders must be within 1.5 ATR of each other
NECKLINE_SLOPE_MAX = 0.5     # neckline may rise/fall by at most 50% of the head height

# Trend-structure setups (ideas from a published price-action test; see README)
STRUCT_K = 3               # swing points for trend structure need 3 candles on each side
MAX_STRUCT_AGE = 60        # the last swing must be at most 60 candles old
HTF_MIN_MINUTES = 240      # 4h and above count as "higher timeframe" (more reliable in that test)
RETEST_MAX_DELAY = 30      # the first retest of a broken level must come within 30 candles
RUN_DAYS = 3               # "recent run" window
RUN_MIN_ATR = 2.0          # the run must be at least 2 ATR to count as a run

MONITOR_INTERVAL_SEC = 60
REFRESH_SECONDS = 60
STALE_MIN = {"crypto": 30, "stock": 45}  # data older than this is flagged as stale

UP, DOWN, NEUTRAL = "#26a69a", "#ef5350", "#ffca28"
DISCLAIMER = "Informational only. Not financial advice. Patterns do not predict future prices."

RT = None  # shared Runtime object, set in main() (see section 10)


# =============================================================================
# 2. HELPERS
# =============================================================================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def now_naive_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def fmt_price(v) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    v = float(v)
    return f"{v:,.2f}" if abs(v) >= 1 else f"{v:.6f}"


def fmt_pct(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:+.2f}%"


def fmt_ts(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def is_crypto(symbol: str) -> bool:
    parts = symbol.upper().split("-")
    return len(parts) == 2 and parts[1] in {"USD", "USDT", "EUR", "GBP", "BTC", "ETH"}

ASSET_NAMES = {"^GSPC": "S&P 500 Index", "^OEX": "S&P 100 Index",
               "SPY": "S&P 500 ETF", "OEF": "S&P 100 ETF", "AMD": "Advanced Micro Devices", "RKLB": "Rocket Lab", "ANT": "Anthropic"}


def asset_label(symbol: str) -> str:
    name = ASSET_NAMES.get(symbol)
    return f"{symbol} ({name})" if name else symbol



class Runtime:
    """Objects that must survive Streamlit re-runs (Streamlit re-executes this whole
    file on every click, so plain global variables would be recreated each time)."""

    def __init__(self):
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.db_lock = threading.Lock()
        self.settings_lock = threading.Lock()
        self.status = {"started_at": utc_now(), "last_cycle": None, "cycles": 0,
                       "last_error": None, "telegram_error": None}


def load_settings() -> dict:
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))  # deep copy
    with RT.settings_lock:
        try:
            if SETTINGS_PATH.exists():
                saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                for key in DEFAULT_SETTINGS:
                    if key in saved:
                        settings[key] = saved[key]
                # older versions had a single "telegram_5m" switch: keep the user's choice
                if "telegram_timeframes" not in saved and saved.get("telegram_5m") is False:
                    settings["telegram_timeframes"] = [t for t in settings["telegram_timeframes"] if t != "5m"]
        except Exception as exc:
            log.warning("Could not read settings.json, using defaults: %s", exc)
    settings["watchlist"] = [s for s in settings["watchlist"] if isinstance(s, str) and s.strip()] \
        or list(DEFAULT_SETTINGS["watchlist"])
    return settings

def save_settings(settings: dict) -> None:
    with RT.settings_lock:
        tmp = SETTINGS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        os.replace(tmp, SETTINGS_PATH)


# =============================================================================
# 3. MARKET DATA (yfinance)
# =============================================================================
@dataclass
class FetchResult:
    df: Optional[pd.DataFrame]
    error: Optional[str]
    fetched_at: datetime

def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Builds 4-hour candles from 1-hour candles: blocks of 4 consecutive 1h candles inside each UTC day
    (crypto: 00-04, 04-08 ...; US stocks: 9:30-13:30 and the rest of the session)."""
    day = df.index.normalize()
    block = df.groupby(day).cumcount() // 4
    g = df.groupby([day, block])
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                        "Close": g["Close"].last(), "Volume": g["Volume"].sum()})
    starts = pd.Series(df.index, index=df.index).groupby([day, block]).first()
    out.index = pd.DatetimeIndex(starts.to_numpy())
    return out


def _download(symbol: str, tf: str) -> FetchResult:
    cfg = TIMEFRAMES[tf]
    fetched_at = utc_now()
    try:
        raw = yf.Ticker(symbol).history(period=cfg["period"], interval=cfg["interval"],
                                        auto_adjust=False, actions=False)
    except Exception as exc:
        return FetchResult(None, f"Yahoo Finance request failed for {symbol} ({tf}): {exc}", fetched_at)
    if raw is None or raw.empty:
        return FetchResult(None, f"No {tf} data from Yahoo Finance for {symbol} "
                                 f"(unknown symbol, market closed, or provider limit).", fetched_at)
    try:
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    except KeyError:
        return FetchResult(None, f"Unexpected data format received for {symbol}.", fetched_at)
    # Step 1: clean the raw candles FIRST (rows with missing prices must never reach resample_4h)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None) if tf in INTRADAY else idx.tz_localize(None)
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Step 2: only now build 4-hour candles from the clean 1-hour candles
    if tf == "4h":
        df = resample_4h(df)
    if df.empty:
        return FetchResult(None, f"No usable candles for {symbol} ({tf}).", fetched_at)
    return FetchResult(df, None, fetched_at)

def fetch_candles(symbol: str, tf: str) -> FetchResult:
    """Cached download (so the dashboard and the background monitor don't hammer Yahoo)."""
    key = (symbol, tf)
    now = time.time()
    with RT.cache_lock:
        hit = RT.cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    result = _download(symbol, tf)
    ttl = TIMEFRAMES[tf]["ttl"] if result.df is not None else 20
    with RT.cache_lock:
        RT.cache[key] = (now + ttl, result)
    return result


def closed_candles(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Only candles that have finished (start + interval <= now). Pattern confirmation uses these."""
    delta = pd.Timedelta(minutes=INTERVAL_MIN[tf])
    return df[df.index + delta <= now_naive_utc()]


def data_age_minutes(df: pd.DataFrame) -> float:
    return (now_naive_utc() - df.index[-1]).total_seconds() / 60


def price_change(df: pd.DataFrame, minutes: int):
    """Close-to-close % change over `minutes`, using candle timestamps.
    Step 1: target time  = last candle time - minutes
    Step 2: reference    = close of the last candle at or before the target time
    Step 3: change %     = (last close / reference close - 1) * 100
    Returns (None, None) if the history has a gap or doesn't reach far enough back."""
    if df is None or len(df) < 2:
        return None, None
    last_ts = df.index[-1]
    target = last_ts - pd.Timedelta(minutes=minutes)
    older = df.loc[df.index <= target, "Close"]
    if older.empty:
        return None, None
    ref_ts, ref_price = older.index[-1], float(older.iloc[-1])
    max_gap = pd.Timedelta(minutes=max(10, int(minutes * 0.05)))
    if target - ref_ts > max_gap or ref_price <= 0:
        return None, None
    return (float(df["Close"].iloc[-1]) / ref_price - 1) * 100, ref_price


def pct_change_over(df, minutes):
    return price_change(df, minutes)[0]


def get_snapshot(symbol: str) -> dict:
    res = fetch_candles(symbol, "5m")
    crypto = is_crypto(symbol)
    base = {"asset": symbol, "type": "Crypto" if crypto else "Index" if symbol.startswith("^") else "Stock"}
    if res.df is None or res.df.empty:
        return {**base, "error": res.error or "No data"}
    df = res.df
    nan = float("nan")

    def pc(minutes):
        v = pct_change_over(df, minutes)
        return nan if v is None else v

    age = data_age_minutes(df)
    return {**base, "error": None, "price": float(df["Close"].iloc[-1]),
            "chg_5m": pc(5), "chg_1h": pc(60), "chg_24h": pc(1440),
            "last_candle": df.index[-1], "age": age,
            "stale": age > STALE_MIN["crypto" if crypto else "stock"],
            "fetched_at": res.fetched_at}


# =============================================================================
# 4. INDICATORS
# =============================================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]

    # EMA with span N:  alpha = 2/(N+1);  EMA_t = alpha*close_t + (1-alpha)*EMA_{t-1}
    out["EMA20"] = close.ewm(span=20, adjust=False).mean()
    out["EMA50"] = close.ewm(span=50, adjust=False).mean()

    # RSI 14 (Wilder smoothing)
    #   Step 1: delta_t = close_t - close_{t-1}
    #   Step 2: gain_t = max(delta_t, 0);  loss_t = max(-delta_t, 0)
    #   Step 3: avg_gain, avg_loss = Wilder average (alpha = 1/14) of gain, loss
    #   Step 4: RS = avg_gain / avg_loss
    #   Step 5: RSI = 100 - 100 / (1 + RS)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["RSI14"] = 100 - 100 / (1 + avg_gain / avg_loss)

    # MACD (12, 26, 9)
    #   Step 1: MACD = EMA12(close) - EMA26(close)
    #   Step 2: Signal = EMA9(MACD)
    #   Step 3: Histogram = MACD - Signal
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    out["MACD"], out["MACD_SIGNAL"], out["MACD_HIST"] = macd, signal, macd - signal

    # ATR 14
    #   Step 1: TR_t = max(high-low, |high - prev close|, |low - prev close|)
    #   Step 2: ATR = Wilder average (alpha = 1/14) of TR
    prev_close = close.shift(1)
    tr = pd.concat([out["High"] - out["Low"],
                    (out["High"] - prev_close).abs(),
                    (out["Low"] - prev_close).abs()], axis=1).max(axis=1)
    out["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return out


def indicator_summary(row) -> str:
    def num(v, f):
        return "n/a" if pd.isna(v) else format(v, f)

    trend = "n/a"
    if not (pd.isna(row["EMA20"]) or pd.isna(row["EMA50"])):
        trend = "EMA20 > EMA50" if row["EMA20"] > row["EMA50"] else "EMA20 < EMA50"
    return (f"RSI14 {num(row['RSI14'], '.1f')} | MACD hist {num(row['MACD_HIST'], '+.4g')} | "
            f"ATR14 {num(row['ATR14'], '.4g')} | {trend}")


# =============================================================================
# 5. EVENTS AND PATTERNS
# =============================================================================
def make_event(asset, tf, alert_type, status, direction, ts, price, change_pct,
               change_label, explanation, notify, **extra) -> dict:
    ev = {"asset": asset, "timeframe": tf, "alert_type": alert_type, "status": status,
          "direction": direction, "event_time": pd.Timestamp(ts), "price": float(price),
          "change_pct": None if change_pct is None else float(change_pct),
          "change_label": change_label, "explanation": explanation,
          "notify": bool(notify), "indicators": ""}
    ev.update(extra)
    return ev


def detect_price_events(symbol: str, ind: pd.DataFrame, settings: dict) -> list:
    """Rapid price moves and volume spikes, from 5-minute candles."""
    events = []
    crypto = is_crypto(symbol)
    thr = float(settings["crypto_pct"] if crypto else settings["stock_pct"])
    window = int(settings["crypto_minutes"] if crypto else settings["stock_minutes"])

    pct, ref = price_change(ind, window)
    last_ts, last_price = ind.index[-1], float(ind["Close"].iloc[-1])
    if pct is not None and abs(pct) >= thr:
        up = pct > 0
        events.append(make_event(
            symbol, "5m", "Rapid Price Increase" if up else "Rapid Price Decrease", "Info",
            "bull" if up else "bear", last_ts, last_price, pct, f"last {window} min",
            f"Price moved {pct:+.2f}% in about {window} minutes (alert threshold {thr:g}%), "
            f"from {fmt_price(ref)} to {fmt_price(last_price)}. Based on 5-minute Yahoo Finance "
            f"candles, which can be delayed.", True))

    closed = closed_candles(ind, "5m")
    if len(closed) >= 22:
        vols = closed["Volume"].to_numpy()
        avg = vols[-21:-1].mean()  # previous 20 closed candles
        mult = float(settings["volume_mult"])
        # Volume ratio = volume of last closed candle / average volume of previous 20 candles
        if avg > 0 and vols[-1] >= mult * avg:
            c = closed.iloc[-1]
            chg = (c["Close"] / closed["Close"].iloc[-2] - 1) * 100
            direction = "bull" if c["Close"] > c["Open"] else "bear" if c["Close"] < c["Open"] else "neutral"
            events.append(make_event(
                symbol, "5m", "Volume Spike", "Info", direction, closed.index[-1],
                c["Close"], chg, "5m candle",
                f"Volume of the last closed 5m candle was {vols[-1] / avg:.1f}x the average of the "
                f"previous 20 candles (threshold {mult:g}x). A spike shows unusual activity, "
                f"not the direction of the next move.", True))
    return events


def _pivot_points(highs, lows, k):
    """Swing highs/lows: a candle whose high (low) is the extreme of the k candles on each side."""
    n, ph, pl = len(highs), [], []
    for j in range(k, n - k):
        if highs[j] == highs[j - k:j + k + 1].max():
            ph.append((j, float(highs[j])))
        if lows[j] == lows[j - k:j + k + 1].min():
            pl.append((j, float(lows[j])))
    return ph, pl


def _cluster_levels(prices, tol, min_touches):
    clusters = []
    for p in sorted(prices):
        if clusters and p - np.mean(clusters[-1]) <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [float(np.mean(c)) for c in clusters if len(c) >= min_touches]


def levels_at(ph, pl, i, tol):
    """Support/resistance levels as known at candle i. A swing point at j is only known
    PIVOT_K candles later, so we only use j <= i - PIVOT_K - 1 (no peeking into the future)."""
    lo, hi = i - LOOKBACK, i - PIVOT_K - 1
    prices = [p for j, p in ph + pl if lo <= j <= hi]
    return _cluster_levels(prices, tol, MIN_TOUCHES)


def current_levels(closed: pd.DataFrame) -> list:
    n = len(closed)
    if n < MIN_CANDLES:
        return []
    atr, price = closed["ATR14"].iloc[-1], closed["Close"].iloc[-1]
    tol = LEVEL_TOL_ATR * atr if np.isfinite(atr) and atr > 0 else 0.005 * price
    ph, pl = _pivot_points(closed["High"].to_numpy(), closed["Low"].to_numpy(), PIVOT_K)
    return levels_at(ph, pl, n, tol)


def split_levels(levels, price, per_side=3):
    supports = sorted([x for x in levels if x <= price], reverse=True)[:per_side]
    resistances = sorted([x for x in levels if x > price])[:per_side]
    return supports, resistances


def _volume_ratio(vol, i):
    """Volume of candle i divided by the average of the 20 candles before it.
    Returns None when there is no volume data (average is 0), e.g. for many indexes."""
    if i < 20:
        return None
    avg = float(vol[i - 20:i].mean())
    return float(vol[i]) / avg if avg > 0 else None


def _volume_text(ratio):
    """Returns (confirmed, text for the explanation)."""
    if ratio is None:
        return False, ("volume data is not available for this instrument (common for indexes), "
                       "so the move can NOT be volume-confirmed")
    if ratio >= BREAKOUT_VOL_MULT:
        return True, (f"volume was {ratio:.1f}x the 20-candle average (>= {BREAKOUT_VOL_MULT:g}x needed), "
                      f"which supports the move")
    return False, (f"but volume was only {ratio:.1f}x the 20-candle average (>= {BREAKOUT_VOL_MULT:g}x "
                   f"needed), so it is NOT confirmed and may be a false breakout")


def detect_breakouts(ind, asset, tf, only_last=True) -> list:
    """Bullish Breakout / Bearish Breakdown on CLOSED candles only.
    Confirmed = the close is beyond the level by 0.25 ATR AND volume >= 1.5x the 20-candle average.
    Potential = the close is beyond the level but volume does not confirm it (or there is no volume data)."""
    events, n = [], len(ind)
    if n < MIN_CANDLES:
        return events
    high, low = ind["High"].to_numpy(), ind["Low"].to_numpy()
    close, vol, atr = ind["Close"].to_numpy(), ind["Volume"].to_numpy(), ind["ATR14"].to_numpy()
    ph, pl = _pivot_points(high, low, PIVOT_K)
    first = n - 1 if only_last else max(MIN_CANDLES - 1, n - HISTORY_SCAN)

    for i in range(first, n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0 or i < 21:
            continue
        levels = levels_at(ph, pl, i, LEVEL_TOL_ATR * a)
        if not levels:
            continue
        buf = BREAKOUT_BUFFER_ATR * a
        confirmed, vol_txt = _volume_text(_volume_ratio(vol, i))
        chg = (close[i] / close[i - 1] - 1) * 100
        status = "Confirmed" if confirmed else "Potential"

        up = [lv for lv in levels if close[i - 1] <= lv and close[i] > lv + buf]
        down = [lv for lv in levels if close[i - 1] >= lv and close[i] < lv - buf]
        if up:
            lv = max(up)
            events.append(make_event(
                asset, tf, "Bullish Breakout", status, "bull", ind.index[i], close[i], chg,
                f"{tf} candle",
                f"The closed {tf} candle ({fmt_ts(ind.index[i])}) finished at {fmt_price(close[i])}, above "
                f"the resistance level near {fmt_price(lv)} by more than {BREAKOUT_BUFFER_ATR:g} ATR; {vol_txt}.",
                confirmed, level=lv, break_distance_atr=float(abs(close[i] - lv) / a)))
        if down:
            lv = min(down)
            events.append(make_event(
                asset, tf, "Bearish Breakdown", status, "bear", ind.index[i], close[i], chg,
                f"{tf} candle",
                f"The closed {tf} candle ({fmt_ts(ind.index[i])}) finished at {fmt_price(close[i])}, below "
                f"the support level near {fmt_price(lv)} by more than {BREAKOUT_BUFFER_ATR:g} ATR; {vol_txt}.",
                confirmed, level=lv, break_distance_atr=float(abs(close[i] - lv) / a)))
    return events

def detect_reversals(ind, asset, tf, only_last=True) -> list:
    """Potential Trend Reversal (always only 'Potential'):
    bullish = MACD crosses above its signal line while EMA20 < EMA50 and RSI was <= 35 recently;
    bearish = MACD crosses below its signal line while EMA20 > EMA50 and RSI was >= 65 recently."""
    events, n = [], len(ind)
    if n < MIN_CANDLES:
        return events
    close = ind["Close"].to_numpy()
    ema20, ema50 = ind["EMA20"].to_numpy(), ind["EMA50"].to_numpy()
    macd, sig, rsi = ind["MACD"].to_numpy(), ind["MACD_SIGNAL"].to_numpy(), ind["RSI14"].to_numpy()
    first = n - 1 if only_last else max(MIN_CANDLES - 1, n - HISTORY_SCAN)

    for i in range(first, n):
        window = rsi[i - 10:i + 1]
        if np.isnan(window).all():
            continue
        chg = (close[i] / close[i - 1] - 1) * 100
        cross_up = macd[i - 1] <= sig[i - 1] and macd[i] > sig[i]
        cross_dn = macd[i - 1] >= sig[i - 1] and macd[i] < sig[i]
        if cross_up and ema20[i] < ema50[i] and np.nanmin(window) <= 35:
            events.append(make_event(
                asset, tf, "Potential Trend Reversal", "Potential", "bull", ind.index[i], close[i], chg,
                f"{tf} candle",
                f"On the closed {tf} candle ({fmt_ts(ind.index[i])}) MACD crossed above its signal line "
                f"while EMA20 is below EMA50 (downtrend) and RSI14 recently dropped to "
                f"{np.nanmin(window):.0f}. Selling pressure may be fading, but this is only a "
                f"potential reversal.", False))
        if cross_dn and ema20[i] > ema50[i] and np.nanmax(window) >= 65:
            events.append(make_event(
                asset, tf, "Potential Trend Reversal", "Potential", "bear", ind.index[i], close[i], chg,
                f"{tf} candle",
                f"On the closed {tf} candle ({fmt_ts(ind.index[i])}) MACD crossed below its signal line "
                f"while EMA20 is above EMA50 (uptrend) and RSI14 recently reached "
                f"{np.nanmax(window):.0f}. Buying pressure may be fading, but this is only a "
                f"potential reversal.", False))
    return events


def _find_top_structures(high, low, close, atr, ph):
    """Looks for BEARISH structures ('tops'): Double Top and Head and Shoulders.
    The bullish versions (Double Bottom, Inverse H&S) are found by calling this function with the chart
    flipped upside down (all prices negated), so there is only one copy of the logic.
    Nothing here looks into the future: a swing point is used only after PIVOT_K more candles have
    closed, and the neckline break must be a CLOSED candle.
    Returns (found, forming):
      found   = structures whose neckline was already broken by a closed candle
      forming = complete structures still waiting for the neckline break (display only, never an alert)
    'line' = ((start candle position, neckline price), (end candle position, neckline price))."""
    n = len(close)
    found, forming = [], []

    def finish(kind, start, j_last, peak_limit, neck_at, buf, peaks, target_fn):
        i_break = None
        for i in range(j_last + 1, min(n, j_last + MAX_BREAK_DELAY + 1)):
            if close[i] > peak_limit:
                return                      # price closed above the tops: the pattern failed
            if close[i] < neck_at(i) - buf:
                i_break = i
                break
        if i_break is None:
            # No break yet. 'Forming' = the last peak is already known and the break window is still open.
            if j_last + PIVOT_K + 1 <= n - 1 <= j_last + MAX_BREAK_DELAY:
                forming.append({"kind": kind, "peaks": peaks, "neck_now": neck_at(n - 1),
                                "limit": peak_limit, "target": target_fn(n - 1),
                                "line": ((start, neck_at(start)), (n - 1, neck_at(n - 1)))})
            return
        i_det = max(i_break, j_last + PIVOT_K + 1)   # the last peak is only known PIVOT_K candles later
        if i_det >= n:
            return
        if close[j_last + 1:i_det + 1].max() > peak_limit or close[i_det] >= neck_at(i_det):
            return                          # failed, or price already recovered above the neckline
        found.append({"kind": kind, "i": i_det, "i_break": i_break, "neck": neck_at(i_break),
                      "peaks": peaks, "target": target_fn(i_break),
                      "line": ((start, neck_at(start)), (i_det, neck_at(i_det)))})

    # ---- Double Top: two similar swing highs with a pullback between them ----
    for (j1, h1), (j2, h2) in zip(ph[:-1], ph[1:]):
        a = atr[j2]
        if not np.isfinite(a) or a <= 0:
            continue
        if not (MIN_PATTERN_SEP <= j2 - j1 <= MAX_PATTERN_SPAN) or abs(h1 - h2) > DOUBLE_TOL_ATR * a:
            continue
        neck = float(low[j1:j2 + 1].min())            # neckline = lowest low between the two tops
        if min(h1, h2) - neck < PATTERN_MIN_DEPTH_ATR * a:
            continue
        buf = BREAKOUT_BUFFER_ATR * a
        finish("double", j1, j2, max(h1, h2) + buf, lambda i, v=neck: v, buf, (h1, h2),
               lambda ib, v=neck, m=(h1 + h2) / 2: v - (m - v))   # measured move = pattern height

    # ---- Head and Shoulders: three swing highs, the middle one clearly the highest ----
    for (j1, h1), (j2, h2), (j3, h3) in zip(ph[:-2], ph[1:-1], ph[2:]):
        a = atr[j3]
        if not np.isfinite(a) or a <= 0 or j3 - j1 > MAX_PATTERN_SPAN:
            continue
        if min(h2 - h1, h2 - h3) < HEAD_MIN_ATR * a or abs(h1 - h3) > SHOULDER_TOL_ATR * a:
            continue
        t1 = j1 + int(np.argmin(low[j1:j2 + 1]))      # trough between left shoulder and head
        t2 = j2 + int(np.argmin(low[j2:j3 + 1]))      # trough between head and right shoulder
        if t2 <= t1:
            continue
        v1, v2 = float(low[t1]), float(low[t2])
        slope = (v2 - v1) / (t2 - t1)
        neck_at = lambda i, v1=v1, t1=t1, s=slope: v1 + s * (i - t1)   # neckline through both troughs
        head_height = h2 - neck_at(j2)
        if head_height < PATTERN_MIN_DEPTH_ATR * a or abs(v2 - v1) > NECKLINE_SLOPE_MAX * head_height:
            continue
        buf = BREAKOUT_BUFFER_ATR * a
        finish("hs", t1, j3, h2 + buf, neck_at, buf, (h1, h2, h3),
               lambda ib, f=neck_at, hh=head_height: f(ib) - hh)
    return found, forming

def detect_chart_patterns(ind, asset, tf, only_last=True) -> list:
    """Double Top, Double Bottom, Head and Shoulders, Inverse Head and Shoulders (CLOSED candles only).
    Confirmed = neckline broken by a closed candle (0.25 ATR beyond it) AND the break candle's volume
    is >= 1.5x the 20-candle average. Potential = neckline broken but volume does not confirm."""
    events, n = [], len(ind)
    if n < MIN_CANDLES:
        return events
    high, low, close = ind["High"].to_numpy(), ind["Low"].to_numpy(), ind["Close"].to_numpy()
    vol, atr = ind["Volume"].to_numpy(), ind["ATR14"].to_numpy()
    first = n - 1 if only_last else max(MIN_CANDLES - 1, n - HISTORY_SCAN)
    names = {(1, "double"): "Double Top", (1, "hs"): "Head and Shoulders",
             (-1, "double"): "Double Bottom", (-1, "hs"): "Inverse Head and Shoulders"}

    for sign in (1, -1):    # 1 = tops (bearish); -1 = bottoms (bullish), found on the flipped chart
        h, l, c = (high, low, close) if sign == 1 else (-low, -high, -close)
        ph, _ = _pivot_points(h, l, PIVOT_K)
        found, _ = _find_top_structures(h, l, c, atr, ph)
        for s in found:
            i, ib = s["i"], s["i_break"]
            if i < first:
                continue
            confirmed, vol_txt = _volume_text(_volume_ratio(vol, ib))
            name = names[(sign, s["kind"])]
            bearish = sign == 1
            neckline, target = sign * s["neck"], sign * s["target"]
            (i0, y0), (i1, y1) = s["line"]
            peaks = ", ".join(fmt_price(sign * p) for p in s["peaks"])
            swing = "highs" if bearish else "lows"
            side = "below" if bearish else "above"
            if s["kind"] == "double":
                shape = f"Two swing {swing} near {peaks} (within {DOUBLE_TOL_ATR:g} ATR of each other)"
            else:
                shape = f"Three swing {swing} ({peaks}: shoulder, head, shoulder)"
            explanation = (
                f"{shape} form a {name} with the neckline near {fmt_price(neckline)}. The closed {tf} candle "
                f"({fmt_ts(ind.index[ib])}) finished at {fmt_price(close[ib])}, {side} the neckline by more "
                f"than {BREAKOUT_BUFFER_ATR:g} ATR; {vol_txt}. Textbook measured-move reference: "
                f"{fmt_price(target)} (a reference level, not a forecast).")
            chg = (close[i] / close[i - 1] - 1) * 100
            events.append(make_event(
                asset, tf, name, "Confirmed" if confirmed else "Potential", "bear" if bearish else "bull",
                ind.index[i], close[i], chg, f"{tf} candle", explanation, confirmed,
                neckline=neckline, target=target,
                break_distance_atr=(float(abs(close[ib] - neckline) / atr[ib])
                                    if np.isfinite(atr[ib]) and atr[ib] > 0 else None),
                neck_line=((i0, sign * y0), (i1, sign * y1))))   # candle positions + prices, for the chart
    return events


def detect_forming_patterns(ind, asset, tf) -> list:
    """Complete patterns that are still waiting for the neckline break. For the chart only: these are
    never stored as alerts and never sent to Telegram."""
    events, n = [], len(ind)
    if n < MIN_CANDLES:
        return events
    high, low, close = ind["High"].to_numpy(), ind["Low"].to_numpy(), ind["Close"].to_numpy()
    atr = ind["ATR14"].to_numpy()
    names = {(1, "double"): "Double Top", (1, "hs"): "Head and Shoulders",
             (-1, "double"): "Double Bottom", (-1, "hs"): "Inverse Head and Shoulders"}

    for sign in (1, -1):
        h, l, c = (high, low, close) if sign == 1 else (-low, -high, -close)
        ph, _ = _pivot_points(h, l, PIVOT_K)
        _, forming = _find_top_structures(h, l, c, atr, ph)
        for s in forming:
            bearish = sign == 1
            name = names[(sign, s["kind"])]
            neck_now, limit, target = sign * s["neck_now"], sign * s["limit"], sign * s["target"]
            (i0, y0), (i1, y1) = s["line"]
            dist = abs(close[-1] / neck_now - 1) * 100
            peaks = ", ".join(fmt_price(sign * p) for p in s["peaks"])
            swing = "highs" if bearish else "lows"
            explanation = (
                f"{name} is FORMING on the {tf} chart: swing {swing} near {peaks}, neckline now near "
                f"{fmt_price(neck_now)}; the last close is {dist:.1f}% {'above' if bearish else 'below'} it. "
                f"It is NOT a signal yet: it needs a closed {tf} candle {'below' if bearish else 'above'} the "
                f"neckline by more than {BREAKOUT_BUFFER_ATR:g} ATR (volume >= {BREAKOUT_VOL_MULT:g}x average "
                f"to confirm). It fails if price closes {'above' if bearish else 'below'} {fmt_price(limit)}. "
                f"Many forming patterns never complete. Textbook reference after a break: "
                f"{fmt_price(target)} (not a forecast).")
            events.append(make_event(
                asset, tf, name, "Forming", "bear" if bearish else "bull", ind.index[-1], close[-1],
                (close[-1] / close[-2] - 1) * 100, f"{tf} candle", explanation, False,
                neckline=neck_now, limit=limit, neck_line=((i0, sign * y0), (i1, sign * y1))))
    return events

def detect_choch(ind, asset, tf, only_last=True, history=HISTORY_SCAN) -> list:
    """Change of Character (CHoCH): an established trend (higher highs + higher lows, or lower highs +
    lower lows) is broken for the FIRST time by a closed candle that closes beyond the last swing low
    (swing high). Each event carries a reference trade plan: entry at the close, hard stop beyond the
    last swing (+0.25 ATR), target = 1R. Below 4h the signal is only 'Potential' (noise-prone)."""
    events, n = [], len(ind)
    if n < MIN_CANDLES:
        return events
    high, low = ind["High"].to_numpy(), ind["Low"].to_numpy()
    close, atr = ind["Close"].to_numpy(), ind["ATR14"].to_numpy()
    ph, pl = _pivot_points(high, low, STRUCT_K)
    htf = INTERVAL_MIN[tf] >= HTF_MIN_MINUTES
    first = n - 1 if only_last else max(MIN_CANDLES - 1, n - history)

    for i in range(first, n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        known = i - STRUCT_K - 1                       # a swing point is only known STRUCT_K candles later
        highs = [(j, p) for j, p in ph if j <= known][-2:]
        lows = [(j, p) for j, p in pl if j <= known][-2:]
        if len(highs) < 2 or len(lows) < 2:
            continue
        (_, h_old), (jh, h_new) = highs
        (_, l_old), (jl, l_new) = lows
        buf = BREAKOUT_BUFFER_ATR * a
        if h_new > h_old and l_new > l_old and i - jl <= MAX_STRUCT_AGE and close[i - 1] >= l_new > close[i]:
            bearish, stop = True, float(h_new + buf)    # uptrend broken: stop above the last swing high
        elif h_new < h_old and l_new < l_old and i - jh <= MAX_STRUCT_AGE and close[i - 1] <= h_new < close[i]:
            bearish, stop = False, float(l_new - buf)   # downtrend broken: stop below the last swing low
        else:
            continue
        entry = float(close[i])
        if (bearish and stop <= entry) or (not bearish and stop >= entry):
            continue
        risk = abs(stop - entry)
        target = entry - risk if bearish else entry + risk          # 1R
        risk_pct = risk / entry * 100
        if bearish:
            what = (f"Uptrend (swing highs {fmt_price(h_old)} -> {fmt_price(h_new)}, swing lows "
                    f"{fmt_price(l_old)} -> {fmt_price(l_new)}) broke: the closed {tf} candle "
                    f"({fmt_ts(ind.index[i])}) finished at {fmt_price(entry)}, below the last higher low "
                    f"{fmt_price(l_new)} - the first close against the trend.")
        else:
            what = (f"Downtrend (swing highs {fmt_price(h_old)} -> {fmt_price(h_new)}, swing lows "
                    f"{fmt_price(l_old)} -> {fmt_price(l_new)}) broke: the closed {tf} candle "
                    f"({fmt_ts(ind.index[i])}) finished at {fmt_price(entry)}, above the last lower high "
                    f"{fmt_price(h_new)} - the first close against the trend.")
        plan_txt = (f" Reference plan (from a published futures test, not advice): entry {fmt_price(entry)}, "
                    f"hard stop {fmt_price(stop)} beyond the last swing, 1R target {fmt_price(target)} "
                    f"(risk {risk_pct:.1f}% = {risk / a:.1f} ATR).")
        note = "" if htf else (f" The {tf} timeframe is below 4h: in that test lower timeframes were much "
                               f"less reliable (costs and noise).")
        events.append(make_event(
            asset, tf, "Change of Character", "Confirmed" if htf else "Potential",
            "bear" if bearish else "bull", ind.index[i], entry, (close[i] / close[i - 1] - 1) * 100,
            f"{tf} candle", what + plan_txt + note, htf,
            break_distance_atr=(float(abs(entry - (l_new if bearish else h_new)) / a)),
            plan={"entry": entry, "stop": stop, "target": float(target), "risk_pct": risk_pct}))
    return events


def detect_flip_retests(ind, asset, tf, only_last=True) -> list:
    """Flip zone: after a level is broken, the FIRST retest of it. If the retest candle touches the zone
    and closes back on the breakout side with a candle in that direction, the old resistance is acting
    as support (or old support as resistance). Only the first retest counts."""
    events, n = [], len(ind)
    if n < MIN_CANDLES:
        return events
    open_, high, low = ind["Open"].to_numpy(), ind["High"].to_numpy(), ind["Low"].to_numpy()
    close, atr = ind["Close"].to_numpy(), ind["ATR14"].to_numpy()
    first = n - 1 if only_last else max(MIN_CANDLES - 1, n - HISTORY_SCAN)

    for b in detect_breakouts(ind, asset, tf, only_last=False):
        ib = ind.index.get_loc(b["event_time"])
        level, bull = b["level"], b["direction"] == "bull"
        for j in range(ib + 1, min(n, ib + RETEST_MAX_DELAY + 1)):
            a = atr[j]
            if not np.isfinite(a) or a <= 0:
                break
            tol = LEVEL_TOL_ATR * a
            touched = low[j] <= level + tol if bull else high[j] >= level - tol
            if not touched:
                continue
            held = (close[j] > level and close[j] > open_[j]) if bull else (close[j] < level and close[j] < open_[j])
            if held and j >= first:
                status = "Confirmed" if b["status"] == "Confirmed" else "Potential"
                if bull:
                    text = (f"The resistance near {fmt_price(level)} was broken upward on {fmt_ts(ind.index[ib])}. "
                            f"On the first retest the closed {tf} candle ({fmt_ts(ind.index[j])}) touched the zone "
                            f"and closed back above it with a bullish candle: the old resistance is acting as "
                            f"support (a 'flip zone').")
                else:
                    text = (f"The support near {fmt_price(level)} was broken downward on {fmt_ts(ind.index[ib])}. "
                            f"On the first retest the closed {tf} candle ({fmt_ts(ind.index[j])}) touched the zone "
                            f"and closed back below it with a bearish candle: the old support is acting as "
                            f"resistance (a 'flip zone').")
                text += f" The original break was {b['status'].lower()}. A held first retest is a popular setup, not a guarantee."
                events.append(make_event(
                    asset, tf, "Flip Zone Retest", status, "bull" if bull else "bear", ind.index[j],
                    close[j], (close[j] / close[j - 1] - 1) * 100, f"{tf} candle", text,
                    status == "Confirmed", level=level,
                    break_distance_atr=b.get("break_distance_atr")))
            break                       # only the FIRST retest counts, held or not
    return events


def add_run_context(ind, ev: dict):
    """Adds a note about the last 3 days. A published futures test found that skipping trades that go the
    SAME way as a 3-day run helped, so such signals are flagged as 'chasing' (information only)."""
    if ev["event_time"] not in ind.index:
        return
    i = ind.index.get_loc(ev["event_time"])
    pos = ind.index.searchsorted(ev["event_time"] - pd.Timedelta(days=RUN_DAYS), side="right") - 1
    if pos < 0 or pos >= i:
        return
    move = float(ind["Close"].iloc[i] - ind["Close"].iloc[pos])
    a = ind["ATR14"].iloc[i]
    ev["run_pct"] = move / float(ind["Close"].iloc[pos]) * 100
    same_way = (move > 0 and ev["direction"] == "bull") or (move < 0 and ev["direction"] == "bear")
    ev["chasing"] = bool(same_way and np.isfinite(a) and a > 0 and abs(move) >= RUN_MIN_ATR * a)
    ev["explanation"] += f" Context: price moved {ev['run_pct']:+.1f}% over the last {RUN_DAYS} days"
    ev["explanation"] += " - this signal goes the same way (chasing)." if ev["chasing"] else "."


def simulate_plan(ind, events, cost_pct=0.0) -> list:
    """Replays events that carry a reference plan. Entry = the signal candle's close. A candle that touches
    the stop counts as a LOSS even if it also reached the target (conservative). Result in R (1R = the
    initial risk). cost_pct = round-trip cost as % of the entry price, converted to R."""
    high, low = ind["High"].to_numpy(), ind["Low"].to_numpy()
    out = []
    for ev in events:
        plan = ev.get("plan")
        if not plan or ev["event_time"] not in ind.index:
            continue
        i = ind.index.get_loc(ev["event_time"])
        bull = ev["direction"] == "bull"
        outcome, r = "open", 0.0
        for j in range(i + 1, len(ind)):
            hit_stop = low[j] <= plan["stop"] if bull else high[j] >= plan["stop"]
            hit_target = high[j] >= plan["target"] if bull else low[j] <= plan["target"]
            if hit_stop:
                outcome, r = "loss", -1.0
                break
            if hit_target:
                outcome, r = "win", 1.0
                break
        if outcome != "open":
            r -= cost_pct / plan["risk_pct"]
        out.append({"time": ev["event_time"], "direction": ev["direction"], "entry": plan["entry"],
                    "stop": plan["stop"], "target": plan["target"], "outcome": outcome, "r": r})
    return out


def pattern_events(closed, asset, tf, only_last=True) -> list:
    if len(closed) < MIN_CANDLES:
        return []
    events = (detect_breakouts(closed, asset, tf, only_last)
              + detect_reversals(closed, asset, tf, only_last)
              + detect_chart_patterns(closed, asset, tf, only_last)
              + detect_choch(closed, asset, tf, only_last)
              + detect_flip_retests(closed, asset, tf, only_last))
    for ev in events:
        add_run_context(closed, ev)
    return sorted(events, key=lambda e: e["event_time"])
# =============================================================================
# 6. DATABASE (SQLite): one table, alert history survives restarts
# =============================================================================
def db_query(sql, params=()):
    with RT.db_lock:
        con = sqlite3.connect(DB_PATH, timeout=10)
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()


def db_execute(sql, params=()):
    with RT.db_lock:
        con = sqlite3.connect(DB_PATH, timeout=10)
        try:
            with con:
                cur = con.execute(sql, params)
            return cur.lastrowid
        finally:
            con.close()


def init_db():
    db_execute("""CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        asset TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        status TEXT NOT NULL,
        price REAL,
        change_pct REAL,
        description TEXT NOT NULL,
        event_time TEXT NOT NULL)""")
    db_execute("CREATE INDEX IF NOT EXISTS idx_alerts_lookup "
               "ON alerts (asset, alert_type, timeframe, status, created_at)")


def try_register_alert(ev: dict, cooldown_min: int) -> Optional[int]:
    """Store the alert unless it is a duplicate or inside the cooldown. Returns the new id or None.
    - duplicate = same asset, type, timeframe, status and candle time already stored
    - cooldown  = same asset, type, timeframe and status already alerted < cooldown_min ago"""
    event_time = fmt_ts(ev["event_time"])
    key = (ev["asset"], ev["alert_type"], ev["timeframe"], ev["status"])
    if db_query("SELECT 1 FROM alerts WHERE asset=? AND alert_type=? AND timeframe=? AND status=? "
                "AND event_time=? LIMIT 1", key + (event_time,)):
        return None
    last = db_query("SELECT MAX(created_at) FROM alerts WHERE asset=? AND alert_type=? "
                    "AND timeframe=? AND status=?", key)[0][0]
    if last:
        last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if utc_now() - last_dt < timedelta(minutes=cooldown_min):
            return None
    return db_execute(
        "INSERT INTO alerts (created_at, asset, alert_type, timeframe, status, price, change_pct, "
        "description, event_time) VALUES (?,?,?,?,?,?,?,?,?)",
        (utc_now().strftime("%Y-%m-%d %H:%M:%S"), ev["asset"], ev["alert_type"], ev["timeframe"],
         ev["status"], ev["price"], ev["change_pct"], ev["explanation"], event_time))


def get_alerts(limit=200, asset=None, alert_type=None, since_minutes=None) -> pd.DataFrame:
    sql = ("SELECT created_at, asset, alert_type, status, timeframe, price, change_pct, description "
           "FROM alerts WHERE 1=1")
    params = []
    if asset:
        sql += " AND asset=?"
        params.append(asset)
    if alert_type:
        sql += " AND alert_type=?"
        params.append(alert_type)
    if since_minutes:
        sql += " AND created_at>=?"
        params.append((utc_now() - timedelta(minutes=since_minutes)).strftime("%Y-%m-%d %H:%M:%S"))
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    cols = ["created_at", "asset", "alert_type", "status", "timeframe", "price", "change_pct", "description"]
    return pd.DataFrame(db_query(sql, tuple(params)), columns=cols)


# =============================================================================
# 7. CHARTS (Plotly): used both in the web page and for Telegram PNG images
# =============================================================================
def build_chart(ind, symbol, tf, levels, markers, candles=150, with_osc=True) -> go.Figure:
    d = ind.tail(candles)
    fmt = "%Y-%m-%d %H:%M" if tf in INTRADAY else "%Y-%m-%d"
    x = d.index.strftime(fmt).tolist()  # text labels -> no gaps for weekends / market closes
    pos = {lab: i for i, lab in enumerate(x)}
    offset = len(ind) - len(d)     # candle position of the first candle shown

    rows = 5 if with_osc else 2
    heights = [0.46, 0.14, 0.14, 0.14, 0.12] if with_osc else [0.75, 0.25]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=heights)

    fig.add_trace(go.Candlestick(
        x=x, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Price",
        increasing=dict(line=dict(color=UP), fillcolor=UP),
        decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["EMA20"], name="EMA 20", mode="lines",
                             line=dict(color="#ffa726", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["EMA50"], name="EMA 50", mode="lines",
                             line=dict(color="#42a5f5", width=1.4)), row=1, col=1)

    # Support / resistance: the 3 nearest levels on each side of the last price
    last_price = float(d["Close"].iloc[-1])
    supports, resistances = split_levels(levels, last_price)
    for lv in supports:
        fig.add_hline(y=lv, row=1, col=1, line_dash="dot", line_color=UP, line_width=1,
                      annotation_text=f"S {fmt_price(lv)}", annotation_position="top left")
    for lv in resistances:
        fig.add_hline(y=lv, row=1, col=1, line_dash="dot", line_color=DOWN, line_width=1,
                      annotation_text=f"R {fmt_price(lv)}", annotation_position="top left")

    # Pattern necklines (dashed = pattern already broken, dotted = still forming)
    for m in markers:
        ln = m.get("neck_line")
        if not ln:
            continue
        (i0, y0), (i1, y1) = ln
        if i1 < offset or i1 - offset >= len(x):
            continue                                   # the line is outside the candles shown
        if i0 < offset and i1 > i0:                    # the start is off-screen: cut at the left edge
            y0 = y0 + (y1 - y0) * (offset - i0) / (i1 - i0)
            i0 = offset
        is_forming = m["status"] == "Forming"
        note = f"{m['alert_type']} forming" if is_forming else ""
        fig.add_trace(go.Scatter(
            x=[x[i0 - offset], x[i1 - offset]], y=[y0, y1], mode="lines+text", name="Neckline",
            showlegend=False, hoverinfo="skip", text=["", note], textposition="top left",
            textfont=dict(color=NEUTRAL, size=11),
            line=dict(color=NEUTRAL, width=1.6, dash="dot" if is_forming else "dash")), row=1, col=1)

    # Pattern markers (filled = confirmed/info, hollow = potential)
    for m in markers:
        if m["status"] == "Forming":
            continue
        label = pd.Timestamp(m["event_time"]).strftime(fmt)
        if label not in pos:
            continue
        row = d.iloc[pos[label]]
        atr = row["ATR14"] if pd.notna(row["ATR14"]) else (row["High"] - row["Low"])
        if m["direction"] == "bull":
            y, sym, color = row["Low"] - 0.6 * atr, "triangle-up", UP
        elif m["direction"] == "bear":
            y, sym, color = row["High"] + 0.6 * atr, "triangle-down", DOWN
        else:
            y, sym, color = row["High"] + 0.6 * atr, "diamond", NEUTRAL
        if m["status"] == "Potential":
            sym += "-open"
        fig.add_trace(go.Scatter(
            x=[label], y=[y], mode="markers", showlegend=False, hoverinfo="text",
            hovertext=f"{m['alert_type']} ({m['status']})",
            marker=dict(symbol=sym, size=13, color=color, line=dict(width=2, color=color))), row=1, col=1)

    vol_colors = [UP if c >= o else DOWN for o, c in zip(d["Open"], d["Close"])]
    fig.add_trace(go.Bar(x=x, y=d["Volume"], marker_color=vol_colors, name="Volume",
                         showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    if with_osc:
        fig.add_trace(go.Scatter(x=x, y=d["RSI14"], name="RSI 14", mode="lines",
                                 line=dict(color="#ab47bc", width=1.3)), row=3, col=1)
        fig.add_hline(y=70, row=3, col=1, line_dash="dot", line_color=DOWN, line_width=1)
        fig.add_hline(y=30, row=3, col=1, line_dash="dot", line_color=UP, line_width=1)
        fig.update_yaxes(title_text="RSI 14", range=[0, 100], row=3, col=1)

        hist_colors = [UP if (pd.notna(v) and v >= 0) else DOWN for v in d["MACD_HIST"]]
        fig.add_trace(go.Bar(x=x, y=d["MACD_HIST"], marker_color=hist_colors, name="MACD hist",
                             showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=x, y=d["MACD"], name="MACD", mode="lines",
                                 line=dict(color="#42a5f5", width=1.2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=x, y=d["MACD_SIGNAL"], name="Signal", mode="lines",
                                 line=dict(color="#ffa726", width=1.2)), row=4, col=1)
        fig.update_yaxes(title_text="MACD", row=4, col=1)

        fig.add_trace(go.Scatter(x=x, y=d["ATR14"], name="ATR 14", mode="lines",
                                 line=dict(color="#26c6da", width=1.3)), row=5, col=1)
        fig.update_yaxes(title_text="ATR 14", row=5, col=1)

    step = max(1, len(x) // 8)
    fig.update_xaxes(type="category", tickmode="array", tickvals=x[::step], ticktext=x[::step],
                     rangeslider_visible=False, showgrid=False)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=950 if with_osc else 700, margin=dict(l=50, r=30, t=60, b=30),
        title=f"{symbol} - {tf} (last {len(d)} candles)",
        legend=dict(orientation="h", y=1.03, x=0), hovermode="x unified")
    return fig


def cleanup_charts(max_age_days=3):
    cutoff = time.time() - max_age_days * 86400
    for p in CHARTS_DIR.glob("*.png"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def render_chart_png(ind, symbol, tf, ev=None) -> Path:
    """Creates the PNG for Telegram (needs the 'kaleido' package)."""
    closed = closed_candles(ind, tf)
    levels = current_levels(closed)
    markers = pattern_events(closed, symbol, tf, only_last=False)
    if ev is not None and ev["status"] != "Test":
        markers = markers + [ev]
    unique = {(m["event_time"], m["alert_type"]): m for m in markers}
    fig = build_chart(ind, symbol, tf, levels, list(unique.values()), candles=120, with_osc=False)
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", symbol)
    path = CHARTS_DIR / f"{safe}_{tf}_{utc_now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.write_image(str(path), format="png", width=1200, height=700, scale=1)
    cleanup_charts()
    return path


# =============================================================================
# 8. TELEGRAM (plain HTTP requests to the official Bot API)
# =============================================================================
def tg_token() -> str:
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def tg_chat_id() -> str:
    return os.getenv("TELEGRAM_CHAT_ID", "").strip()


def telegram_configured() -> bool:
    return bool(tg_token() and tg_chat_id())


def redact(text) -> str:
    """Never show the bot token in logs or on screen (request errors can contain it)."""
    token = tg_token()
    return str(text).replace(token, "***") if token else str(text)


def tg_request(method, *, data=None, files=None, params=None, timeout=20, http="post"):
    url = f"https://api.telegram.org/bot{tg_token()}/{method}"
    try:
        if http == "get":
            r = requests.get(url, params=params, timeout=timeout)
        else:
            r = requests.post(url, data=data, files=files, timeout=timeout)
        payload = r.json()
    except Exception as exc:
        return False, redact(f"Network error: {exc}")
    if not payload.get("ok"):
        return False, redact(payload.get("description", f"HTTP {r.status_code}"))
    return True, payload.get("result")


def tg_send_message(text, chat_id=None):
    return tg_request("sendMessage", data={"chat_id": chat_id or tg_chat_id(), "text": text[:4096],
                                           "disable_web_page_preview": "true"})


def tg_send_photo(path, caption, chat_id=None):
    with open(path, "rb") as fh:
        return tg_request("sendPhoto", data={"chat_id": chat_id or tg_chat_id(), "caption": caption[:1024]},
                          files={"photo": fh}, timeout=60)


def build_message(ev: dict) -> str:
    icon = {"bull": "🟢", "bear": "🔴"}.get(ev["direction"], "🟡")
    title = ev["alert_type"] if ev["status"] in ("Info", "Test") else f"{ev['alert_type']} ({ev['status']})"
    if ev["status"] == "Test":
        title = "🧪 " + title
    head = "\n".join([
        f"{icon} {title}",
        f"Asset: {asset_label(ev['asset'])}",
        f"Price: {fmt_price(ev['price'])}",
        f"Change: {fmt_pct(ev['change_pct'])} ({ev['change_label']})",
        f"Timeframe: {ev['timeframe']}",
        f"Indicators: {ev['indicators'] or 'n/a'}"])
    tail = f"Candle time: {fmt_ts(ev['event_time'])} UTC\n{DISCLAIMER}"
    room = 1024 - len(head) - len(tail) - len("Why: ") - 2       # 2 line breaks
    why = ev["explanation"]
    if len(why) > room:
        why = why[:max(0, room - 3)] + "..."
    return f"{head}\nWhy: {why}\n{tail}"

def send_alert_notification(ev, ind, tf, chat_id=None):
    """Chart + text via sendPhoto. If the chart fails, the text goes out alone."""
    if not (tg_token() and (chat_id or tg_chat_id())):
        return False, "Telegram is not configured."
    text = build_message(ev)
    note, png = "", None
    try:
        png = render_chart_png(ind, ev["asset"], tf, ev)
    except Exception as exc:
        note = f"chart failed: {redact(exc)}"
        log.warning("Chart generation failed for %s: %s", ev["asset"], redact(exc))
    if png is not None:
        ok, info = tg_send_photo(png, text, chat_id)
        try:
            png.unlink()
        except OSError:
            pass
        if ok:
            return True, "Notification sent with chart image."
        note = f"sendPhoto failed: {info}"
    ok, info = tg_send_message(text + "\n\n(Chart not available.)", chat_id)
    return (True, f"Sent as text only ({note}).") if ok else (False, f"Telegram error: {info}")


def send_test_notification(chat_id=None):
    settings = load_settings()
    symbol = settings["watchlist"][0]
    res = fetch_candles(symbol, "1h")
    if res.df is None or len(res.df) < 3:
        ok, info = tg_send_message(f"🧪 TIM test: Telegram works, but there is no market data for "
                                   f"{symbol} to draw a chart ({res.error}).", chat_id)
        return (True, "Text-only test sent (no chart data).") if ok else (False, f"Telegram error: {info}")
    ind = add_indicators(res.df)
    last = ind.iloc[-1]
    chg = (last["Close"] / ind["Close"].iloc[-2] - 1) * 100
    ev = make_event(symbol, "1h", "Test Notification", "Test", "neutral", ind.index[-1], last["Close"],
                    chg, "last 1h candle",
                    "This is a test message to check Telegram and chart delivery. It is not a market signal.",
                    True)
    ev["indicators"] = indicator_summary(last)
    return send_alert_notification(ev, ind, "1h", chat_id)


def cmd_watchlist() -> str:
    lines = ["📋 Watchlist"]
    for sym in load_settings()["watchlist"]:
        s = get_snapshot(sym)
        if s["error"]:
            lines.append(f"• {sym}: no data")
        else:
            lines.append(f"• {sym}: {fmt_price(s['price'])} (24h {fmt_pct(s['chg_24h'])}, "
                         f"data age {s['age']:.0f} min)")
    lines.append("Source: Yahoo Finance via yfinance (may be delayed).")
    return "\n".join(lines)


def cmd_alerts() -> str:
    df = get_alerts(limit=5)
    if df.empty:
        return "No alerts stored yet."
    lines = ["🔔 Last alerts (UTC)"]
    for r in df.itertuples():
        lines.append(f"• {r.created_at[5:16]} {r.asset} - {r.alert_type} ({r.status}, {r.timeframe})")
    return "\n".join(lines)


def cmd_status() -> str:
    s, cfg = RT.status, load_settings()
    last = s["last_cycle"].strftime("%Y-%m-%d %H:%M:%S") if s["last_cycle"] else "not finished yet"
    return "\n".join([
        "ℹ️ TIM status",
        f"Monitor last cycle (UTC): {last} (cycles: {s['cycles']})",
        f"Assets watched: {len(cfg['watchlist'])}",
        f"Crypto threshold: {cfg['crypto_pct']:g}% / {cfg['crypto_minutes']} min",
        f"Stock threshold: {cfg['stock_pct']:g}% / {cfg['stock_minutes']} min",
        f"Volume spike: {cfg['volume_mult']:g}x | cooldown: {cfg['cooldown_minutes']} min",
        f"Minimum structural break for Telegram: {cfg.get('min_breakout_atr', DEFAULT_SETTINGS['min_breakout_atr']):g} ATR",
        f"Sent to Telegram: {', '.join(cfg['telegram_timeframes']) or 'nothing'} (other timeframes: history only)",
        f"Last monitor error: {s['last_error'] or 'none'}",
        "Data: Yahoo Finance via yfinance (not guaranteed real-time)."])

def help_text(chat_id: str, allowed: bool) -> str:
    lines = ["📈 Trading Intelligence Monitor", f"Your chat ID: {chat_id}"]
    if not allowed:
        lines.append("To receive alerts here, put this number in TELEGRAM_CHAT_ID in your .env file "
                     "and restart the app.")
    lines += ["", "Commands:", "/watchlist - assets and last prices", "/alerts - last 5 alerts",
              "/status - monitor status", "/test - send a test notification with a chart",
              "/help - this message", "", DISCLAIMER]
    return "\n".join(lines)


def handle_update(update: dict):
    msg = update.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat_id = str((msg.get("chat") or {}).get("id", ""))
    if not text.startswith("/") or not chat_id:
        return
    command = text.split()[0].split("@")[0].lower()
    allowed = chat_id == tg_chat_id()
    if command in ("/start", "/help"):
        reply = help_text(chat_id, allowed)
    elif not allowed:
        reply = ("This chat is not authorised. Put this chat ID in TELEGRAM_CHAT_ID in your .env file "
                 f"and restart the app: {chat_id}")
    elif command == "/watchlist":
        reply = cmd_watchlist()
    elif command == "/alerts":
        reply = cmd_alerts()
    elif command == "/status":
        reply = cmd_status()
    elif command == "/test":
        ok, info = send_test_notification(chat_id)
        if ok:
            return
        reply = f"Test failed: {info}"
    else:
        reply = "Unknown command. Send /help."
    tg_send_message(reply, chat_id)


def telegram_poll_loop():
    """Background thread: listens for /commands using long polling (getUpdates)."""
    offset = None
    time.sleep(3)
    while True:
        if not tg_token():
            time.sleep(30)
            continue
        if offset is None:  # skip old messages that arrived while the app was off
            ok, result = tg_request("getUpdates", params={"timeout": 0}, http="get")
            if ok:
                offset = result[-1]["update_id"] + 1 if result else 0
        params = {"timeout": 25, "allowed_updates": json.dumps(["message"])}
        if offset:
            params["offset"] = offset
        ok, result = tg_request("getUpdates", params=params, timeout=40, http="get")
        if not ok:
            RT.status["telegram_error"] = result
            time.sleep(15)
            continue
        RT.status["telegram_error"] = None
        for update in result:
            offset = update["update_id"] + 1
            try:
                handle_update(update)
            except Exception as exc:
                log.warning("Telegram command failed: %s", redact(exc))


# =============================================================================
# 9. BACKGROUND MONITOR
# =============================================================================
def event_is_fresh(ev: dict, tf: str) -> bool:
    """Skip events built from old candles (e.g. market closed over the weekend)."""
    closes_at = ev["event_time"] + pd.Timedelta(minutes=INTERVAL_MIN[tf])
    age_min = (now_naive_utc() - closes_at).total_seconds() / 60
    return age_min <= 2 * INTERVAL_MIN[tf] + 45


def passes_notification_quality(ev: dict, settings: dict) -> bool:
    """Filter weak structural breaks from Telegram without deleting them from alert history.

    The detector's existing 0.25 ATR threshold still discovers a possible break. This
    separate threshold controls whether a confirmed structural event is strong enough
    to notify. Price-move and volume-spike alerts do not carry break_distance_atr and
    therefore keep their existing settings.
    """
    distance = ev.get("break_distance_atr")
    if distance is None:
        return True
    try:
        distance = float(distance)
        minimum = float(settings.get("min_breakout_atr", DEFAULT_SETTINGS["min_breakout_atr"]))
    except (TypeError, ValueError):
        return False
    if not np.isfinite(distance):
        return False
    return distance >= minimum


def process_event(ev: dict, ind: pd.DataFrame, tf: str, settings: dict):
    if ev["event_time"] in ind.index:
        ev["indicators"] = indicator_summary(ind.loc[ev["event_time"]])
    alert_id = try_register_alert(ev, int(settings["cooldown_minutes"]))
    if alert_id is None:
        return  # duplicate or inside cooldown
    log.info("ALERT #%s %s %s (%s, %s)", alert_id, ev["asset"], ev["alert_type"], ev["status"], tf)
    routed = tf in settings.get("telegram_timeframes", list(TIMEFRAMES))   # timeframe routing
    quality_ok = passes_notification_quality(ev, settings)
    if ev["notify"] and quality_ok and routed and telegram_configured():
        ok, info = send_alert_notification(ev, ind, tf)
        if not ok:
            log.warning("Telegram notification failed: %s", info)
    elif ev["notify"] and not quality_ok:
        distance = ev.get("break_distance_atr")
        try:
            distance_text = f"{float(distance):.2f}"
        except (TypeError, ValueError):
            distance_text = "unknown"
        minimum = settings.get("min_breakout_atr", DEFAULT_SETTINGS["min_breakout_atr"])
        log.info("Telegram alert filtered: %s %s %s (%s ATR; minimum %s ATR)",
                 ev["asset"], ev["alert_type"], tf, distance_text, minimum)

def scan_symbol(symbol: str, settings: dict):
    jobs = []
    res5 = fetch_candles(symbol, "5m")
    if res5.df is not None and len(res5.df) >= 2:
        ind5 = add_indicators(res5.df)
        jobs.append(("5m", ind5, detect_price_events(symbol, ind5, settings)))
    for tf in PATTERN_SCAN_TFS:
        res = fetch_candles(symbol, tf)
        if res.df is None:
            continue
        ind = add_indicators(res.df)
        jobs.append((tf, ind, pattern_events(closed_candles(ind, tf), symbol, tf, only_last=True)))
    for tf, ind, events in jobs:
        for ev in events:
            if event_is_fresh(ev, tf):
                process_event(ev, ind, tf, settings)



def check_pending_symbols(settings: dict):
    """Symbols that are not trading yet (for example a company that has announced an IPO).
    As soon as Yahoo Finance returns RECENT daily candles for one of them, it is moved to the
    watchlist and you get a Telegram message."""
    for sym in list(settings.get("pending_symbols", [])):
        res = fetch_candles(sym, "1d")
        if res.df is None or (now_naive_utc() - res.df.index[-1]).days > 3:
            continue          # still no data, or only OLD data of some earlier security with this ticker
        fresh = load_settings()
        if sym not in fresh["watchlist"]:
            fresh["watchlist"].append(sym)
        fresh["pending_symbols"] = [s for s in fresh["pending_symbols"] if s != sym]
        save_settings(fresh)
        log.info("Pending symbol %s started trading; added to the watchlist.", sym)
        if telegram_configured():
            tg_send_message(f"🆕 {asset_label(sym)} now returns market data and was added to your watchlist.\n"
                            "Please check on Yahoo Finance that this symbol is the company you expect. "
                            "New listings are very volatile and have almost no history, so indicators "
                            "and patterns appear only after enough candles.\n" + DISCLAIMER)

def monitor_loop():
    time.sleep(5)
    log.info("Background monitor started (every %s s).", MONITOR_INTERVAL_SEC)
    while True:
        started, errors = time.time(), []
        try:
            settings = load_settings()
            for symbol in settings["watchlist"]:
                try:
                    scan_symbol(symbol, settings)
                except Exception as exc:
                    errors.append(f"{symbol}: {redact(exc)}")
                    log.warning("Scan failed for %s: %s", symbol, redact(exc))
                time.sleep(0.3)
            try:
                check_pending_symbols(settings)
            except Exception as exc:
                errors.append(f"pending symbols: {redact(exc)}")
            RT.status["last_cycle"] = utc_now()
            RT.status["cycles"] += 1
        except Exception as exc:
            errors.append(redact(exc))
            log.warning("Monitor cycle failed: %s", redact(exc))
        RT.status["last_error"] = "; ".join(errors[:3]) if errors else None
        time.sleep(max(5.0, MONITOR_INTERVAL_SEC - (time.time() - started)))

@st.cache_resource(show_spinner=False)
def get_runtime() -> Runtime:
    """Runs once per server process: creates shared state and starts the 2 background threads.
    NOTE: after you edit app.py, stop (Ctrl+C) and start Streamlit again so the threads reload."""
    rt = Runtime()
    names = {t.name for t in threading.enumerate()}
    if "tim-monitor" not in names:
        threading.Thread(target=monitor_loop, name="tim-monitor", daemon=True).start()
    if "tim-telegram" not in names:
        threading.Thread(target=telegram_poll_loop, name="tim-telegram", daemon=True).start()
    return rt


# =============================================================================
# 10. WEB INTERFACE (Streamlit)
# =============================================================================
_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show_plotly(fig, key=None):
    if _PLOTLY_HAS_WIDTH:  # newer Streamlit
        st.plotly_chart(fig, width="stretch", key=key)
    else:                  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, key=key)


def _color_pct(v):
    if pd.isna(v):
        return ""
    return f"color: {UP}" if v > 0 else f"color: {DOWN}" if v < 0 else ""


def add_asset(symbol: str, settings: dict):
    if not re.fullmatch(r"[A-Z0-9.=^-]{1,15}", symbol):
        st.sidebar.error("Enter a valid Yahoo Finance symbol, e.g. AMZN or DOGE-USD.")
        return
    if symbol in settings["watchlist"]:
        st.sidebar.info(f"{symbol} is already in the watchlist.")
        return
    res = fetch_candles(symbol, "1d")
    if res.df is None:
        st.sidebar.error(f"Cannot add {symbol}: {res.error}")
        return
    settings["watchlist"].append(symbol)
    save_settings(settings)
    st.sidebar.success(f"Added {symbol}.")


def render_sidebar():
    sb = st.sidebar
    settings = load_settings()
    sb.title("⚙️ Control panel")

    sb.subheader("Watchlist")
    new_symbol = sb.text_input("Add asset", placeholder="AMZN or DOGE-USD", key="new_symbol").strip().upper()
    if sb.button("➕ Add asset"):
        add_asset(new_symbol, settings)
    remove_symbol = sb.selectbox("Remove asset", settings["watchlist"], key="remove_symbol")
    if sb.button("🗑 Remove asset"):
        if len(settings["watchlist"]) <= 1:
            sb.error("Keep at least one asset in the watchlist.")
        else:
            settings["watchlist"].remove(remove_symbol)
            save_settings(settings)
            sb.success(f"Removed {remove_symbol}. Refreshing the list on the next update.")

    sb.subheader("Alert thresholds")
    with sb.form("thresholds"):
        c_pct = st.number_input("Crypto: price change (%)", 0.1, 50.0, float(settings["crypto_pct"]), 0.1)
        c_min = st.number_input("Crypto: within (minutes)", 5, 120, int(settings["crypto_minutes"]), 5)
        s_pct = st.number_input("Stocks: price change (%)", 0.1, 50.0, float(settings["stock_pct"]), 0.1)
        s_min = st.number_input("Stocks: within (minutes)", 5, 120, int(settings["stock_minutes"]), 5)
        v_mult = st.number_input("Volume spike (x average of 20 candles)", 1.0, 20.0,
                                 float(settings["volume_mult"]), 0.5)
        min_breakout_atr = st.number_input(
            "Minimum structural break for Telegram (ATR)", 0.25, 3.0,
            float(settings.get("min_breakout_atr", DEFAULT_SETTINGS["min_breakout_atr"])), 0.05,
            help=("A confirmed breakout must close at least this far beyond its level. "
                  "Weaker structural signals remain in alert history but are not sent to Telegram."))
        cooldown = st.number_input("Cooldown between repeated alerts (minutes)", 1, 1440,
                                   int(settings["cooldown_minutes"]), 5)
        tg_tfs = st.multiselect(
            "Send these timeframes to Telegram",
            list(TIMEFRAMES),
            default=[t for t in settings.get("telegram_timeframes", list(TIMEFRAMES)) if t in TIMEFRAMES],
            help="Alerts of the other timeframes are still saved in the history and shown in the app.")
        if st.form_submit_button("💾 Save thresholds"):
            settings.update({"crypto_pct": c_pct, "crypto_minutes": int(c_min), "stock_pct": s_pct,
                             "stock_minutes": int(s_min), "volume_mult": v_mult,
                             "min_breakout_atr": min_breakout_atr,
                             "cooldown_minutes": int(cooldown),
                             "telegram_timeframes": [t for t in TIMEFRAMES if t in tg_tfs]})
            save_settings(settings)
            st.success("Saved. The background monitor uses the new values on its next cycle.")

    with sb.expander("📨 Telegram notifications"):
        if telegram_configured():
            st.success("Telegram is configured.")
            if st.button("Send test notification"):
                with st.spinner("Sending..."):
                    ok, info = send_test_notification()
                (st.success if ok else st.error)(info)
            if RT.status["telegram_error"]:
                st.warning(f"Command listener problem: {RT.status['telegram_error']}")
        else:
            st.warning("Telegram is not configured. The app works without it, but no notifications "
                       "will be sent.")
            st.markdown(
                "1. Create a bot with **@BotFather** and copy the token.\n"
                "2. Copy `.env.example` to `.env` and fill in `TELEGRAM_BOT_TOKEN`.\n"
                "3. Restart the app, send `/start` to your bot - it replies with your chat ID.\n"
                "4. Put that number in `TELEGRAM_CHAT_ID` in `.env` and restart again.\n\n"
                "Full guide: README.md")

    s = RT.status
    last = s["last_cycle"].strftime("%H:%M:%S UTC") if s["last_cycle"] else "first cycle running..."
    sb.caption(f"Background monitor: last cycle {last} | cycles: {s['cycles']}")
    if s["last_error"]:
        sb.caption(f"Last monitor error: {s['last_error']}")

@st.fragment(run_every=REFRESH_SECONDS)
def dashboard_fragment():
    settings = load_settings()
    rows, problems = [], []
    for sym in settings["watchlist"]:
        snap = get_snapshot(sym)
        if snap["error"]:
            problems.append(f"{sym}: {snap['error']}")
            rows.append({"Asset": asset_label(sym), "Type": snap["type"], "Price": np.nan, "5m %": np.nan,
                         "1h %": np.nan, "24h %": np.nan, "Last candle (UTC)": "n/a",
                         "Age (min)": np.nan, "Status": "No data"})
        else:
            rows.append({"Asset": asset_label(sym), "Type": snap["type"], "Price": snap["price"],
                         "5m %": snap["chg_5m"], "1h %": snap["chg_1h"], "24h %": snap["chg_24h"],
                         "Last candle (UTC)": fmt_ts(snap["last_candle"]), "Age (min)": snap["age"],
                         "Status": "Stale (market closed or delayed)" if snap["stale"] else "OK"})

    recent = get_alerts(since_minutes=60, limit=50)
    c1, c2, c3 = st.columns(3)
    c1.metric("Assets watched", len(settings["watchlist"]))
    c2.metric("Active alerts (last 60 min)", len(recent))
    c3.metric("Dashboard refreshed (UTC)", utc_now().strftime("%H:%M:%S"))
    st.caption("Data source: Yahoo Finance via yfinance (unofficial feed). Quotes can be delayed or "
               "incomplete and are NOT guaranteed real-time. 'n/a' means the history does not cover "
               f"that period. Auto-refresh every {REFRESH_SECONDS} s.")
    pending = settings.get("pending_symbols", [])
    if pending:
        st.info("Waiting for the first trade (will be added automatically): "
                + ", ".join(asset_label(s) for s in pending))
    df = pd.DataFrame(rows)
    styler = (df.style.map(_color_pct, subset=["5m %", "1h %", "24h %"])
              .format({"Price": fmt_price, "5m %": fmt_pct, "1h %": fmt_pct, "24h %": fmt_pct,
                       "Age (min)": lambda v: "n/a" if pd.isna(v) else f"{v:.0f}"}))
    st.subheader("Watchlist")
    st.dataframe(styler, hide_index=True)
    for p in problems:
        st.warning(p)

    st.subheader("Active alerts (last 60 minutes)")
    if recent.empty:
        st.info("No alerts in the last 60 minutes.")
    else:
        show = recent[["created_at", "asset", "alert_type", "status", "timeframe", "change_pct"]]
        show = show.rename(columns={"created_at": "Detected (UTC)", "asset": "Asset", "alert_type": "Type",
                                    "status": "Status", "timeframe": "Timeframe", "change_pct": "Change %"})
        st.dataframe(show.style.map(_color_pct, subset=["Change %"]).format({"Change %": fmt_pct}),
                     hide_index=True)

def show_backtest(trades: list):
    done = [t for t in trades if t["outcome"] != "open"]
    if not done:
        st.info("No finished trades in this history (no signals, or the last signal is still open).")
        return
    wins = sum(1 for t in done if t["outcome"] == "win")
    total_r = sum(t["r"] for t in done)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Finished trades", len(done))
    c2.metric("Win rate", f"{wins / len(done) * 100:.0f}%")
    c3.metric("Average R (after cost)", f"{total_r / len(done):+.2f}")
    c4.metric("Total R (after cost)", f"{total_r:+.1f}")
    if len(done) < 30:
        st.warning(f"Only {len(done)} finished trades: with so few, luck dominates. Do not draw conclusions "
                   "from this alone (30+ trades, ideally on data the idea was not tuned on).")
    table = pd.DataFrame([{"Signal (UTC)": fmt_ts(t["time"]), "Side": "Long" if t["direction"] == "bull" else "Short",
                           "Entry": t["entry"], "Stop": t["stop"], "Target": t["target"],
                           "Result": t["outcome"], "R (after cost)": t["r"]} for t in trades])
    st.dataframe(table.style.format({"Entry": fmt_price, "Stop": fmt_price, "Target": fmt_price,
                                     "R (after cost)": "{:+.2f}"}), hide_index=True)


def chart_analysis():
    settings = load_settings()
    c1, c2, c3 = st.columns([2, 3, 2])
    symbol = c1.selectbox("Asset", settings["watchlist"], key="chart_symbol")
    tf = c2.radio("Timeframe", list(TIMEFRAMES), index=2, horizontal=True, key="chart_tf")
    candles = c3.slider("Candles shown", 50, 400, 150, 10, key="chart_candles")

    res = fetch_candles(symbol, tf)
    if res.df is None:
        st.warning(res.error)
        return
    ind = add_indicators(res.df)
    if len(ind) < MIN_CANDLES:
        st.warning(f"Only {len(ind)} candles available for {symbol} ({tf}); at least {MIN_CANDLES} are "
                   f"needed for indicators and pattern detection.")
        return

    closed = closed_candles(ind, tf)
    levels = current_levels(closed)
    events = pattern_events(closed, symbol, tf, only_last=False)
    forming = detect_forming_patterns(closed, symbol, tf)
    show_plotly(build_chart(ind, symbol, tf, levels, events + forming, candles=candles, with_osc=True),
                key="main_chart")

    last = ind.iloc[-1]
    st.caption(f"Data source: Yahoo Finance via yfinance. Downloaded {res.fetched_at:%Y-%m-%d %H:%M:%S} UTC | "
               f"last candle {fmt_ts(ind.index[-1])} UTC (may still be forming; patterns use closed "
               f"candles only). Not guaranteed real-time.")
    st.markdown(f"**Indicators (last candle):** {indicator_summary(last)}")
    supports, resistances = split_levels(levels, float(ind["Close"].iloc[-1]))
    st.markdown("**Support:** " + (", ".join(fmt_price(v) for v in supports) or "none found") +
                "  |  **Resistance:** " + (", ".join(fmt_price(v) for v in resistances) or "none found"))
    st.caption("Levels are estimated from past swing highs/lows. Filled chart markers = confirmed, "
               "hollow markers = potential. " + DISCLAIMER)

    st.subheader("Detected events on this chart")
    st.subheader("Patterns forming now")
    if not forming:
        st.info("No complete pattern is waiting for a neckline break right now.")
    for f in forming:
        st.warning(f["explanation"])
    with st.expander("🧪 Test the 'Change of Character' plan on this chart's history"):
        st.caption("Replays every Change of Character signal found in the downloaded history: entry at the "
                   "signal candle's close, hard stop beyond the last swing, target = 1R. If one candle touches "
                   "both stop and target it counts as a LOSS (conservative). Yahoo's history is short, so the "
                   "number of trades is usually small. This tests an idea, it is not a forecast.")
        cost = st.number_input("Round-trip cost per trade (% of price)", 0.0, 2.0, 0.10, 0.05, key="bt_cost")
        if st.button("Run test", key="bt_run"):
            signals = detect_choch(closed, symbol, tf, only_last=False, history=len(closed))
            show_backtest(simulate_plan(closed, signals, cost))
    if not events:
        st.info("No breakouts, breakdowns or reversal signals in the recent candles.")
    else:
        table = pd.DataFrame([{"Candle (UTC)": fmt_ts(e["event_time"]), "Type": e["alert_type"],
                               "Status": e["status"], "Explanation": e["explanation"]}
                              for e in reversed(events)])
        st.dataframe(table, hide_index=True)


@st.fragment(run_every=REFRESH_SECONDS)
def alerts_fragment():
    settings = load_settings()
    st.subheader("Alert history")
    st.caption("Alerts are created by the background monitor (checks every 60 s while the app is running) "
               "and stored in SQLite. 'Potential' = not confirmed yet; 'Confirmed' = closed candle beyond "
               "the level with volume confirmation. Only confirmed patterns, rapid moves and volume spikes "
               "are sent to Telegram.")
    f1, f2, f3 = st.columns(3)
    asset = f1.selectbox("Asset", ["All"] + settings["watchlist"], key="alerts_asset")
    atype = f2.selectbox("Type", ["All"] + ALERT_TYPES, key="alerts_type")
    limit = f3.selectbox("Show last", [50, 100, 200, 500], index=1, key="alerts_limit")
    df = get_alerts(limit=limit, asset=None if asset == "All" else asset,
                    alert_type=None if atype == "All" else atype)
    if df.empty:
        st.info("No alerts yet. They appear here when a price move, volume spike or pattern is detected.")
        return
    df = df.rename(columns={"created_at": "Detected (UTC)", "asset": "Asset", "alert_type": "Type",
                            "status": "Status", "timeframe": "Timeframe", "price": "Price",
                            "change_pct": "Change %", "description": "Explanation"})
    st.dataframe(df.style.map(_color_pct, subset=["Change %"]).format({"Price": fmt_price, "Change %": fmt_pct}),
                 hide_index=True)


def main():
    st.set_page_config(page_title="Trading Intelligence Monitor", page_icon="📈", layout="wide")
    global RT
    RT = get_runtime()
    init_db()

    render_sidebar()
    st.title("📈 Trading Intelligence Monitor")
    st.caption("Tracks stocks and crypto, detects price moves and basic patterns, and alerts you. "
               "It does NOT trade or manage money. " + DISCLAIMER)

    tab_dash, tab_chart, tab_alerts = st.tabs(["📊 Dashboard", "📈 Chart Analysis", "🔔 Alerts & Patterns"])
    with tab_dash:
        dashboard_fragment()
    with tab_chart:
        chart_analysis()
    with tab_alerts:
        alerts_fragment()


if __name__ == "__main__":
    main()