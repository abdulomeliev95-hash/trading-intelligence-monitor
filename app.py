# =============================================================================
# Trading Intelligence Monitor (TIM) - MVP                 Run: streamlit run app.py
# Informational tool only. It does NOT execute trades or manage money.
#
# Sections: 1 Config | 2 Helpers | 3 Market data | 4 Indicators | 5 Patterns |
#           6 Database | 7 Charts | 8 Telegram | 9 Background monitor | 10 Web UI
# =============================================================================
import inspect
from concurrent.futures import ThreadPoolExecutor
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
    "polymarket_perps": [],
    "pending_symbols": ["ANT"],
    "telegram_timeframes": ["15m", "1h", "4h", "1d"],
    "crypto_pct": 2.0,       # % change ...
    "crypto_minutes": 5,     # ... within this many minutes
    "stock_pct": 1.0,
    "stock_minutes": 15,
    "volume_mult": 2.0,      # volume spike = N x average of previous 20 candles
    "min_breakout_atr": 0.75, # minimum close distance beyond a broken level for Telegram
    "cooldown_minutes": 30,  # no repeated alert of the same kind within this time
    "paper_starting_balance": 10000.0,
}

# interval/period are what Yahoo Finance understands; ttl = seconds we reuse cached data
TIMEFRAMES = {
    "5m":  {"interval": "5m",  "period": "5d",  "ttl": 50},
    "15m": {"interval": "15m", "period": "30d", "ttl": 120},
    "1h":  {"interval": "1h",  "period": "60d", "ttl": 300},
    "4h":  {"interval": "1h",  "period": "6mo", "ttl": 600},   # built from 1-hour candles (resample_4h)
    "1d":  {"interval": "1d",  "period": "5y",  "ttl": 900},   # 5 years: more signals for the plan test
}
INTERVAL_MIN = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
INTRADAY = {"5m", "15m", "1h", "4h"}
PATTERN_SCAN_TFS = ["15m", "1h", "4h", "1d"]  # strategy timeframes; 1m and 5m excluded
PAPER_RISK_PCT = 2.0
PAPER_REWARD_R = 3.0
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
SCANNER_MAX_WORKERS = 3        # deliberately bounded to avoid burst traffic to Yahoo Finance
DB_POOL_MIN_SIZE = 1
DB_POOL_MAX_SIZE = 3             # per process; web + worker can use at most six connections combined
DB_POOL_TIMEOUT_SEC = 10
WORKER_ADVISORY_LOCK_KEY = 7312042601
STALE_MIN = {"crypto": 30, "stock": 45}  # data older than this is flagged as stale
POLYMARKET_PERPS_API = "https://api.perpetuals.polymarket.com"
POLYMARKET_PERPS_REFRESH_SEC = 3600
YAHOO_TICKER_ALIASES = {
    "SKHYNIX": "SKHY",  # Yahoo's USD Nasdaq listing for SK hynix
    "SAMSUNG": "SMSN.IL",  # Yahoo's USD London listing for Samsung Electronics
}
_YAHOO_INIT_LOCK = threading.Lock()
_YAHOO_CACHE_PRIMED = False

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
        self.fetch_inflight = {}
        self.indicator_cache = {}
        self.polymarket_instruments = {}
        self.polymarket_instruments_fetched_at = 0.0
        self.db_lock = threading.Lock()          # SQLite only; PostgreSQL uses its connection pool.
        self.db_pool_lock = threading.Lock()
        self.db_pool = None
        self.db_pool_url = None
        self.worker_lock_connection = None
        self.worker_lock_file = None
        self.settings_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        self.metrics = {
            "market_requests": 0, "market_errors": 0, "cache_hits": 0,
            "cache_misses": 0, "fetch_wait_timeouts": 0,
            "market_latency_ms_total": 0.0, "db_queries": 0,
            "db_latency_ms_total": 0.0,
        }
        self.status = {"started_at": utc_now(), "last_cycle": None, "cycles": 0,
                       "last_error": None, "telegram_error": None,
                       "cycle_seconds": None, "symbols_scanned": 0}
        self.stop_event = threading.Event()


def _read_settings_file() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        value = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        log.warning("Could not read settings.json: %s", redact(exc))
        return {}


def _merge_settings(saved: dict) -> dict:
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    for key in DEFAULT_SETTINGS:
        if key in saved:
            settings[key] = saved[key]
    try:
        settings["paper_starting_balance"] = max(1.0, float(settings.get("paper_starting_balance", 10000.0)))
    except (TypeError, ValueError):
        settings["paper_starting_balance"] = 10000.0
    raw_telegram_timeframes = settings.get("telegram_timeframes", [])
    if not isinstance(raw_telegram_timeframes, list):
        raw_telegram_timeframes = []
    settings["telegram_timeframes"] = [
        tf for tf in raw_telegram_timeframes if tf in PATTERN_SCAN_TFS
    ]
    raw_watchlist = settings.get("watchlist", [])
    if not isinstance(raw_watchlist, list):
        raw_watchlist = []
    settings["watchlist"] = [s.strip().upper() for s in raw_watchlist
                              if isinstance(s, str) and s.strip()] or list(DEFAULT_SETTINGS["watchlist"])
    raw_pending = settings.get("pending_symbols", [])
    if not isinstance(raw_pending, list):
        raw_pending = []
    settings["pending_symbols"] = [s.strip().upper() for s in raw_pending
                                    if isinstance(s, str) and s.strip()]
    raw_perps = settings.get("polymarket_perps", [])
    if not isinstance(raw_perps, list):
        raw_perps = []
    settings["polymarket_perps"] = [
        {"instrument_id": int(item["instrument_id"]), "symbol": str(item["symbol"]).strip().upper(),
         "data_source": "yahoo" if item.get("data_source") == "yahoo" else "polymarket",
         "yahoo_symbol": str(item.get("yahoo_symbol") or item["symbol"]).strip().upper()}
        for item in raw_perps if isinstance(item, dict)
        and str(item.get("instrument_id", "")).isdigit() and item.get("symbol")
    ]
    if RT is not None and settings["polymarket_perps"]:
        with RT.cache_lock:
            RT.polymarket_instruments.update({
                f"{item['symbol']}-PERP": item for item in settings["polymarket_perps"]
            })
    # Do not rewrite the user's neutral movement threshold or reclassify old predictions on startup.
    return settings


def load_settings() -> dict:
    """Load shared PostgreSQL settings, or the legacy local JSON file for SQLite installs."""
    if database_url():
        rows = db_query("SELECT settings_json FROM app_settings WHERE id=1")
        if rows:
            try:
                saved = json.loads(rows[0][0])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Shared app_settings row contains invalid JSON: {exc}") from exc
        else:
            # On first PostgreSQL startup, seed from an existing local settings.json if present.
            seed = _read_settings_file() or DEFAULT_SETTINGS
            seed_text = json.dumps(seed, separators=(",", ":"))
            db_execute(
                "INSERT INTO app_settings (id, settings_json, updated_at) VALUES (1, ?, ?) "
                "ON CONFLICT (id) DO NOTHING",
                (seed_text, utc_now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            rows = db_query("SELECT settings_json FROM app_settings WHERE id=1")
            if not rows:
                raise RuntimeError("Could not initialize shared settings in PostgreSQL.")
            saved = json.loads(rows[0][0])
        return _merge_settings(saved if isinstance(saved, dict) else {})

    with RT.settings_lock:
        saved = _read_settings_file()
    return _merge_settings(saved)


def save_settings(settings: dict) -> None:
    clean = _merge_settings(settings)
    if database_url():
        db_execute(
            "INSERT INTO app_settings (id, settings_json, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET settings_json=excluded.settings_json, updated_at=excluded.updated_at",
            (json.dumps(clean, separators=(",", ":")), utc_now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        return
    with RT.settings_lock:
        tmp = SETTINGS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(clean, indent=2), encoding="utf-8")
        os.replace(tmp, SETTINGS_PATH)


# =============================================================================
# 3. MARKET DATA (yfinance)
# =============================================================================
@dataclass
class FetchResult:
    df: Optional[pd.DataFrame]
    error: Optional[str]
    fetched_at: datetime


def _yahoo_history(symbol: str, **kwargs) -> pd.DataFrame:
    """Fetch Yahoo candles, including equity extended hours for intraday data."""
    global _YAHOO_CACHE_PRIMED
    interval = kwargs.get("interval")
    if (interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
            and not symbol.startswith("^") and not is_crypto(symbol)):
        kwargs.setdefault("prepost", True)
    if _YAHOO_CACHE_PRIMED:
        return yf.Ticker(symbol).history(**kwargs)
    with _YAHOO_INIT_LOCK:
        if not _YAHOO_CACHE_PRIMED:
            try:
                return yf.Ticker(symbol).history(**kwargs)
            finally:
                _YAHOO_CACHE_PRIMED = True
    return yf.Ticker(symbol).history(**kwargs)


def watched_symbols(settings: dict) -> list:
    """Prefer Yahoo symbols; use a Polymarket alias only when Yahoo has no recent data."""
    perps = {item["symbol"]: item for item in settings.get("polymarket_perps", [])}

    def resolved_symbol(ticker):
        item = perps.get(ticker)
        if item and item.get("data_source") == "polymarket":
            return f"{ticker}-PERP"
        return item.get("yahoo_symbol", ticker) if item else ticker

    symbols = [resolved_symbol(symbol) for symbol in settings.get("watchlist", [])]
    symbols.extend(resolved_symbol(item["symbol"]) for item in perps.values())
    return list(dict.fromkeys(symbols))


def _yahoo_has_recent_market_data(symbol: str) -> Optional[bool]:
    """Return whether Yahoo has recent 15m candles; None means the availability check failed."""
    try:
        raw = _yahoo_history(symbol, period="5d", interval="15m", auto_adjust=False,
                             actions=False, timeout=12)
        if raw is None or raw.empty:
            return False
        latest = pd.Timestamp(raw.index[-1])
        if latest.tzinfo is not None:
            latest = latest.tz_convert("UTC").tz_localize(None)
        return (now_naive_utc().normalize() - latest.normalize()).days <= 7
    except Exception as exc:
        log.warning("Could not check Yahoo availability for %s: %s", symbol, redact(exc))
        return None


def refresh_polymarket_perps(settings: dict, force=False) -> dict:
    """Discover equity perps and route each ticker to Yahoo when it has recent candles."""
    if RT is None:
        return settings
    now = time.time()
    with RT.cache_lock:
        if not force and RT.polymarket_instruments and now - RT.polymarket_instruments_fetched_at < POLYMARKET_PERPS_REFRESH_SEC:
            settings["polymarket_perps"] = list(RT.polymarket_instruments.values())
            return settings
    response = requests.get(f"{POLYMARKET_PERPS_API}/v1/info/instruments", timeout=15)
    response.raise_for_status()
    payload = response.json()
    rows = payload if isinstance(payload, list) else payload.get("instruments", payload.get("data", []))
    instruments = {}
    for row in rows:
        if not isinstance(row, dict) or str(row.get("category", "")).lower() != "equity":
            continue
        raw_symbol = str(row.get("symbol", row.get("ticker", ""))).upper()
        ticker = raw_symbol.split("-", 1)[0].strip()
        instrument_id = row.get("instrument_id", row.get("id"))
        if ticker and str(instrument_id).isdigit():
            instruments[f"{ticker}-PERP"] = {
                "instrument_id": int(instrument_id), "symbol": ticker,
            }
    if not instruments:
        raise ValueError("Polymarket Perps API returned no equity instruments; keeping the last known list.")

    old_sources = {
        item["symbol"]: {"data_source": item.get("data_source", "polymarket"),
                         "yahoo_symbol": item.get("yahoo_symbol", item["symbol"])}
        for item in settings.get("polymarket_perps", [])
    }

    def resolve_source(item):
        ticker = item["symbol"]
        yahoo_symbol = YAHOO_TICKER_ALIASES.get(ticker, ticker)
        available = _yahoo_has_recent_market_data(yahoo_symbol)
        source = ("yahoo" if available else "polymarket") if available is not None else old_sources.get(ticker, {}).get("data_source", "polymarket")
        if available is None:
            yahoo_symbol = old_sources.get(ticker, {}).get("yahoo_symbol", yahoo_symbol)
        return ticker, source, yahoo_symbol

    with ThreadPoolExecutor(max_workers=SCANNER_MAX_WORKERS, thread_name_prefix="tim-source-check") as pool:
        sources = {ticker: (source, yahoo_symbol)
                   for ticker, source, yahoo_symbol in pool.map(resolve_source, instruments.values())}
    for alias, item in instruments.items():
        item["data_source"], item["yahoo_symbol"] = sources[item["symbol"]]

    with RT.cache_lock:
        RT.polymarket_instruments = instruments
        RT.polymarket_instruments_fetched_at = now
    settings["polymarket_perps"] = list(instruments.values())
    return settings


def _download_polymarket_perp(symbol: str, tf: str, instrument: dict) -> FetchResult:
    """Load Polymarket perpetual candles in the requested native interval."""
    fetched_at = utc_now()
    supported = {"5m", "15m", "1h", "4h", "1d"}
    if tf not in supported:
        return FetchResult(None, f"Unsupported Polymarket Perps interval: {tf}", fetched_at)
    days = {"5m": 5, "15m": 30, "1h": 60, "4h": 180, "1d": 1825}[tf]
    params = {
        "instrument_id": instrument["instrument_id"],
        "interval": tf,
        "start_timestamp": int((time.time() - days * 86400) * 1000),
        "limit": 5000,
    }
    try:
        response = requests.get(f"{POLYMARKET_PERPS_API}/v1/info/klines", params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        candles = payload.get("klines", payload.get("data", [])) if isinstance(payload, dict) else payload
        rows = []
        for candle in candles:
            if isinstance(candle, dict):
                ts = candle.get("timestamp", candle.get("t"))
                values = [candle.get(k) for k in ("open", "high", "low", "close", "volume")]
            else:
                ts, *values = candle[:6]
            if ts is None or len(values) < 5:
                continue
            rows.append([pd.to_datetime(int(ts), unit="ms", utc=True).tz_localize(None),
                         *[float(v) for v in values[:5]]])
        if not rows:
            return FetchResult(None, f"No Polymarket Perps candles for {symbol} ({tf}).", fetched_at)
        df = pd.DataFrame(rows, columns=["Time", "Open", "High", "Low", "Close", "Volume"]).set_index("Time")
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return FetchResult(df, None, fetched_at)
    except Exception as exc:
        return FetchResult(None, f"Polymarket Perps request failed for {symbol} ({tf}): {redact(exc)}", fetched_at)

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
    instrument = None
    if RT is not None:
        instrument = RT.polymarket_instruments.get(symbol)
    if instrument is None and symbol.endswith("-PERP"):
        return FetchResult(None, f"Polymarket Perps instrument not loaded for {symbol}.", utc_now())
    if instrument is not None:
        return _download_polymarket_perp(symbol, tf, instrument)
    cfg = TIMEFRAMES[tf]
    fetched_at = utc_now()
    try:
        raw = _yahoo_history(symbol, period=cfg["period"], interval=cfg["interval"],
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
    """TTL cache with single-flight downloads: concurrent callers share one Yahoo request.

    This does not serve expired/stale candles as fresh data and does not alter OHLCV values.
    Failed requests are cached only briefly to avoid repeatedly hitting a failing provider.
    """
    key = (symbol, tf)
    now = time.time()
    owner = False
    with RT.cache_lock:
        hit = RT.cache.get(key)
        if hit and hit[0] > now:
            with RT.metrics_lock:
                RT.metrics["cache_hits"] += 1
            return hit[1]
        with RT.metrics_lock:
            RT.metrics["cache_misses"] += 1
        event = RT.fetch_inflight.get(key)
        if event is None:
            event = threading.Event()
            RT.fetch_inflight[key] = event
            owner = True

    if not owner:
        # Wait for the in-flight request instead of making a duplicate provider call.
        if not event.wait(timeout=65):
            with RT.metrics_lock:
                RT.metrics["fetch_wait_timeouts"] += 1
            return FetchResult(None, f"Timed out waiting for the existing Yahoo Finance request for {symbol} ({tf}).",
                               utc_now())
        with RT.cache_lock:
            hit = RT.cache.get(key)
            if hit:
                return hit[1]
        return FetchResult(None, f"The shared Yahoo Finance request for {symbol} ({tf}) did not return data.",
                           utc_now())

    started = time.perf_counter()
    with RT.metrics_lock:
        RT.metrics["market_requests"] += 1
    try:
        result = _download(symbol, tf)
    except Exception as exc:
        result = FetchResult(None, f"Yahoo Finance request failed for {symbol} ({tf}): {redact(exc)}", utc_now())
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    with RT.metrics_lock:
        RT.metrics["market_latency_ms_total"] += elapsed_ms
        if result.df is None:
            RT.metrics["market_errors"] += 1
    if result.df is not None:
        ttl = TIMEFRAMES[tf]["ttl"]
    elif result.error and result.error.startswith("No "):
        ttl = 300  # missing/delisted symbols should not be requested again every monitor cycle
    else:
        ttl = 20
    with RT.cache_lock:
        RT.cache[key] = (time.time() + ttl, result)
        completed = RT.fetch_inflight.pop(key, None)
        if completed is not None:
            completed.set()
    return result


def get_indicators(symbol: str, tf: str, result: FetchResult) -> pd.DataFrame:
    """Reuse indicators only while the exact same raw candle DataFrame is cached.

    Using object identity is intentional: if Yahoo returns a refreshed/corrected history,
    every indicator is recomputed from that new history rather than reusing an old result.
    """
    if result.df is None:
        raise ValueError(f"No candle data available for {symbol} ({tf}).")
    key = (symbol, tf)
    with RT.cache_lock:
        cached = RT.indicator_cache.get(key)
        if cached is not None and cached[0] is result.df:
            return cached[1]
    indicators = add_indicators(result.df)
    with RT.cache_lock:
        RT.indicator_cache[key] = (result.df, indicators)
        # Keep this cache bounded if symbols/timeframes are changed repeatedly.
        while len(RT.indicator_cache) > 200:
            RT.indicator_cache.pop(next(iter(RT.indicator_cache)))
    return indicators


def closed_candles(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Only candles that have finished (start + interval <= now). Pattern confirmation uses these."""
    delta = pd.Timedelta(minutes=INTERVAL_MIN[tf])
    return df[df.index + delta <= now_naive_utc()]


def data_age_minutes(df: pd.DataFrame, tf: str) -> float:
    """Age from the most recent candle's close, not its opening timestamp."""
    candle_close = df.index[-1] + pd.Timedelta(minutes=INTERVAL_MIN[tf])
    return max(0.0, (now_naive_utc() - candle_close).total_seconds() / 60)


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


def get_snapshot_from_result(symbol: str, res: FetchResult, tf: Optional[str] = None) -> dict:
    crypto = is_crypto(symbol)
    base = {"asset": symbol, "type": "Crypto" if crypto else "Index" if symbol.startswith("^") else "Stock"}
    if res.df is None or res.df.empty:
        return {**base, "error": res.error or "No data", "stale": True,
                "fetched_at": res.fetched_at}
    df = res.df
    nan = float("nan")

    def pc(minutes):
        v = pct_change_over(df, minutes)
        return nan if v is None else v

    tf = tf or ("15m" if symbol.endswith("-PERP") else "5m")
    age = data_age_minutes(df, tf)
    return {**base, "error": None, "price": float(df["Close"].iloc[-1]),
            "chg_5m": pc(5), "chg_1h": pc(60), "chg_24h": pc(1440),
            "last_candle": df.index[-1], "age": age,
            "stale": age > STALE_MIN["crypto" if crypto else "stock"],
            "fetched_at": res.fetched_at}


def get_snapshot(symbol: str) -> dict:
    tf = "15m" if symbol.endswith("-PERP") else "5m"
    return get_snapshot_from_result(symbol, fetch_candles(symbol, tf), tf)


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
    last swing (+0.25 ATR), target = 3R. Signals below 4h are noisier and must be validated in replay."""
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
        target = entry - 3 * risk if bearish else entry + 3 * risk  # 3R target
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
        plan_txt = (f" Paper plan: entry {fmt_price(entry)}, hard stop {fmt_price(stop)} beyond the last swing, "
                    f"3R target {fmt_price(target)} "
                    f"(risk {risk_pct:.1f}% = {risk / a:.1f} ATR).")
        note = "" if htf else f" {tf} is below 4h and has higher noise; validate in replay."
        events.append(make_event(
            asset, tf, "Change of Character", "Confirmed",
            "bear" if bearish else "bull", ind.index[i], entry, (close[i] / close[i - 1] - 1) * 100,
            f"{tf} candle", what + plan_txt + note, True,
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


def simulate_plan(ind, events, cost_pct=0.0, risk_budget_pct=2.0) -> list:
    """Replays events that carry a reference plan. Entry = the signal candle's close. A candle that touches
    the stop counts as a LOSS even if it also reached the target (conservative). Result in R (1R = the
    initial risk). cost_pct = round-trip cost as % of the entry price, converted to R.
    Position sizing risks 2% of starting equity per signal. This is a replay only."""
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
                outcome = "win"
                r = abs(float(plan["target"]) - float(plan["entry"])) / abs(
                    float(plan["stop"]) - float(plan["entry"]))
                break
        if outcome != "open":
            r -= cost_pct / plan["risk_pct"]
        out.append({"time": ev["event_time"], "direction": ev["direction"], "entry": plan["entry"],
                    "stop": plan["stop"], "target": plan["target"], "outcome": outcome, "r": r,
                    "equity_return_pct": r * float(risk_budget_pct)})
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
def setting_value(name: str, default="") -> str:
    """Read a setting from environment/.env first, then Streamlit Cloud Secrets."""
    value = os.getenv(name, "")
    if value:
        return str(value).strip()
    try:
        value = st.secrets.get(name, default)
    except Exception:  # StreamlitSecretNotFoundError when no secrets file exists
        value = default
    return str(value or "").strip()


def database_url() -> str:
    return setting_value("DATABASE_URL")


def _get_postgres_pool():
    """Create/reuse one bounded psycopg pool per process."""
    url = database_url()
    if not url:
        return None
    if RT is None:
        raise RuntimeError("Runtime is not initialized before using the database.")
    with RT.db_pool_lock:
        if RT.db_pool is not None and RT.db_pool_url == url:
            return RT.db_pool
        if RT.db_pool is not None:
            try:
                RT.db_pool.close()
            except Exception:
                log.warning("Could not close the previous PostgreSQL pool.")
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise RuntimeError(
                "DATABASE_URL is set, but psycopg_pool is missing. Install psycopg[pool,binary]>=3.2,<4."
            ) from exc
        pool = ConnectionPool(
            conninfo=url,
            min_size=DB_POOL_MIN_SIZE,
            max_size=DB_POOL_MAX_SIZE,
            timeout=DB_POOL_TIMEOUT_SEC,
            open=True,
            name=f"tim-{os.getpid()}",
        )
        try:
            pool.wait(timeout=DB_POOL_TIMEOUT_SEC)
        except Exception:
            pool.close()
            raise
        RT.db_pool = pool
        RT.db_pool_url = url
        return pool


def close_database_pool() -> None:
    if RT is None:
        return
    with RT.db_pool_lock:
        if RT.db_pool is not None:
            RT.db_pool.close()
            RT.db_pool = None
            RT.db_pool_url = None


def _db_sql(sql: str, is_postgres: bool) -> str:
    return sql.replace("?", "%s") if is_postgres else sql


def _record_db_metric(started: float) -> None:
    if RT is None:
        return
    with RT.metrics_lock:
        RT.metrics["db_queries"] += 1
        RT.metrics["db_latency_ms_total"] += (time.perf_counter() - started) * 1000.0


def db_query(sql, params=()):
    started = time.perf_counter()
    if database_url():
        pool = _get_postgres_pool()
        try:
            with pool.connection(timeout=DB_POOL_TIMEOUT_SEC) as con:
                cur = con.execute(_db_sql(sql, True), params)
                return [tuple(row) for row in cur.fetchall()]
        finally:
            _record_db_metric(started)
    with RT.db_lock:
        con = sqlite3.connect(DB_PATH, timeout=10)
        try:
            cur = con.execute(sql, params)
            return [tuple(row) for row in cur.fetchall()]
        finally:
            con.close()
            _record_db_metric(started)


def db_execute(sql, params=()):
    started = time.perf_counter()
    if database_url():
        pool = _get_postgres_pool()
        try:
            with pool.connection(timeout=DB_POOL_TIMEOUT_SEC) as con:
                query = _db_sql(sql, True)
                if query.lstrip().upper().startswith("INSERT") and "RETURNING" not in query.upper():
                    target = re.search(r"INSERT\s+INTO\s+([A-Za-z_][A-Za-z0-9_]*)", query, re.IGNORECASE)
                    table_name = target.group(1).lower() if target else ""
                    if table_name != "market_snapshots":
                        cur = con.execute(query.rstrip().rstrip(";") + " RETURNING id", params)
                        row = cur.fetchone()
                        return row[0] if row else None
                con.execute(query, params)
                return None
        finally:
            _record_db_metric(started)
    with RT.db_lock:
        con = sqlite3.connect(DB_PATH, timeout=10)
        try:
            cur = con.execute(sql, params)
            con.commit()
            return cur.lastrowid
        finally:
            con.close()
            _record_db_metric(started)


def acquire_worker_lock() -> bool:
    """Ensure only one worker instance scans and long-polls Telegram at a time."""
    if database_url():
        pool = _get_postgres_pool()
        con = pool.getconn(timeout=DB_POOL_TIMEOUT_SEC)
        try:
            acquired = bool(con.execute("SELECT pg_try_advisory_lock(%s)", (WORKER_ADVISORY_LOCK_KEY,)).fetchone()[0])
            con.commit()
        except Exception:
            pool.putconn(con)
            raise
        if acquired:
            RT.worker_lock_connection = con  # hold this session until graceful shutdown
            return True
        pool.putconn(con)
        return False

    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("The local SQLite worker lock currently requires a POSIX system (macOS/Linux).") from exc
    lock_file = (DATA_DIR / "worker.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return False
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    RT.worker_lock_file = lock_file
    return True


def release_worker_lock() -> None:
    if RT is None:
        return
    if RT.worker_lock_connection is not None:
        con = RT.worker_lock_connection
        RT.worker_lock_connection = None
        try:
            con.execute("SELECT pg_advisory_unlock(%s)", (WORKER_ADVISORY_LOCK_KEY,))
            con.commit()
        finally:
            if RT.db_pool is not None:
                RT.db_pool.putconn(con)
    if RT.worker_lock_file is not None:
        try:
            import fcntl
            fcntl.flock(RT.worker_lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            RT.worker_lock_file.close()
            RT.worker_lock_file = None


def init_db():
    if database_url():
        alerts_sql = """CREATE TABLE IF NOT EXISTS alerts (
            id BIGSERIAL PRIMARY KEY,
            created_at TEXT NOT NULL,
            asset TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            status TEXT NOT NULL,
            price DOUBLE PRECISION,
            change_pct DOUBLE PRECISION,
            description TEXT NOT NULL,
            event_time TEXT NOT NULL)"""
        predictions_sql = """CREATE TABLE IF NOT EXISTS predictions (
            id BIGSERIAL PRIMARY KEY,
            alert_id BIGINT NOT NULL UNIQUE REFERENCES alerts(id),
            asset TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            pattern TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price DOUBLE PRECISION NOT NULL,
            signal_time TEXT NOT NULL,
            horizon_minutes INTEGER NOT NULL,
            neutral_pct DOUBLE PRECISION NOT NULL DEFAULT 0.25,
            due_at TEXT NOT NULL,
            telegram_sent INTEGER NOT NULL DEFAULT 0,
            evaluated_at TEXT,
            exit_candle_time TEXT,
            exit_price DOUBLE PRECISION,
            return_pct DOUBLE PRECISION,
            outcome TEXT,
            result_notified INTEGER NOT NULL DEFAULT 0)"""
    else:
        alerts_sql = """CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            asset TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            status TEXT NOT NULL,
            price REAL,
            change_pct REAL,
            description TEXT NOT NULL,
            event_time TEXT NOT NULL)"""
        predictions_sql = """CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id INTEGER NOT NULL UNIQUE REFERENCES alerts(id),
            asset TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            pattern TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            signal_time TEXT NOT NULL,
            horizon_minutes INTEGER NOT NULL,
            neutral_pct REAL NOT NULL DEFAULT 0.25,
            due_at TEXT NOT NULL,
            telegram_sent INTEGER NOT NULL DEFAULT 0,
            evaluated_at TEXT,
            exit_candle_time TEXT,
            exit_price REAL,
            return_pct REAL,
            outcome TEXT,
            result_notified INTEGER NOT NULL DEFAULT 0)"""
    db_execute(alerts_sql)
    db_execute("CREATE INDEX IF NOT EXISTS idx_alerts_lookup ON alerts (asset, alert_type, timeframe, status, created_at)")
    db_execute(predictions_sql)
    trade_columns = {
        "stop_price": "DOUBLE PRECISION" if database_url() else "REAL",
        "target_price": "DOUBLE PRECISION" if database_url() else "REAL",
        "position_units": "DOUBLE PRECISION" if database_url() else "REAL",
        "risk_amount": "DOUBLE PRECISION" if database_url() else "REAL",
        "equity_at_entry": "DOUBLE PRECISION" if database_url() else "REAL",
        "pnl_amount": "DOUBLE PRECISION" if database_url() else "REAL",
        "pnl_equity_pct": "DOUBLE PRECISION" if database_url() else "REAL",
        "pnl_r": "DOUBLE PRECISION" if database_url() else "REAL",
        "mark_price": "DOUBLE PRECISION" if database_url() else "REAL",
        "unrealized_pnl": "DOUBLE PRECISION" if database_url() else "REAL",
        "unrealized_pnl_pct": "DOUBLE PRECISION" if database_url() else "REAL",
        "mark_time": "TEXT",
        "exit_reason": "TEXT",
    }
    if database_url():
        for column, sql_type in trade_columns.items():
            db_execute(f"ALTER TABLE predictions ADD COLUMN IF NOT EXISTS {column} {sql_type}")
    else:
        existing_columns = {row[1] for row in db_query("PRAGMA table_info(predictions)")}
        for column, sql_type in trade_columns.items():
            if column not in existing_columns:
                db_execute(f"ALTER TABLE predictions ADD COLUMN {column} {sql_type}")
    db_execute("CREATE INDEX IF NOT EXISTS idx_predictions_open ON predictions (outcome, signal_time)")
    db_execute("CREATE INDEX IF NOT EXISTS idx_predictions_stats ON predictions (timeframe, pattern, outcome)")
    if database_url():
        db_execute("CREATE TABLE IF NOT EXISTS app_settings (id INTEGER PRIMARY KEY, settings_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
        db_execute("CREATE TABLE IF NOT EXISTS monitor_status (id INTEGER PRIMARY KEY, status_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
        db_execute("""CREATE TABLE IF NOT EXISTS market_snapshots (
            asset TEXT PRIMARY KEY, asset_type TEXT NOT NULL, price DOUBLE PRECISION,
            chg_5m DOUBLE PRECISION, chg_1h DOUBLE PRECISION, chg_24h DOUBLE PRECISION,
            last_candle TEXT, age_minutes DOUBLE PRECISION, stale INTEGER NOT NULL DEFAULT 1,
            error TEXT, fetched_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    else:
        db_execute("CREATE TABLE IF NOT EXISTS app_settings (id INTEGER PRIMARY KEY, settings_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
        db_execute("CREATE TABLE IF NOT EXISTS monitor_status (id INTEGER PRIMARY KEY, status_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
        db_execute("""CREATE TABLE IF NOT EXISTS market_snapshots (
            asset TEXT PRIMARY KEY, asset_type TEXT NOT NULL, price REAL,
            chg_5m REAL, chg_1h REAL, chg_24h REAL,
            last_candle TEXT, age_minutes REAL, stale INTEGER NOT NULL DEFAULT 1,
            error TEXT, fetched_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    db_execute("CREATE INDEX IF NOT EXISTS idx_market_snapshots_updated ON market_snapshots (updated_at)")


def _json_safe_status() -> dict:
    payload = dict(RT.status)
    payload["started_at"] = payload["started_at"].isoformat() if isinstance(payload.get("started_at"), datetime) else payload.get("started_at")
    payload["last_cycle"] = payload["last_cycle"].isoformat() if isinstance(payload.get("last_cycle"), datetime) else payload.get("last_cycle")
    payload["metrics"] = dict(RT.metrics)
    payload["pid"] = os.getpid()
    return payload


def publish_monitor_status() -> None:
    """Publish worker heartbeat and performance counters for the separate web process."""
    payload = json.dumps(_json_safe_status(), separators=(",", ":"))
    db_execute(
        "INSERT INTO monitor_status (id, status_json, updated_at) VALUES (1, ?, ?) "
        "ON CONFLICT (id) DO UPDATE SET status_json=excluded.status_json, updated_at=excluded.updated_at",
        (payload, utc_now().strftime("%Y-%m-%d %H:%M:%S")),
    )


def get_monitor_status() -> dict:
    fallback = _json_safe_status() if RT is not None else {}
    try:
        rows = db_query("SELECT status_json, updated_at FROM monitor_status WHERE id=1")
        if rows:
            payload = json.loads(rows[0][0])
            payload["updated_at"] = rows[0][1]
            return payload
    except Exception as exc:
        log.warning("Could not read monitor status: %s", redact(exc))
    return fallback


def save_market_snapshot(snapshot: dict) -> None:
    """Persist the exact 5-minute Yahoo snapshot so the web process need not re-fetch it."""
    values = (
        snapshot["asset"], snapshot["type"], snapshot.get("price"), snapshot.get("chg_5m"),
        snapshot.get("chg_1h"), snapshot.get("chg_24h"),
        pd.Timestamp(snapshot["last_candle"]).strftime("%Y-%m-%d %H:%M:%S") if snapshot.get("last_candle") is not None else None,
        snapshot.get("age"), 1 if snapshot.get("stale", True) else 0, snapshot.get("error"),
        snapshot.get("fetched_at").strftime("%Y-%m-%d %H:%M:%S") if snapshot.get("fetched_at") else utc_now().strftime("%Y-%m-%d %H:%M:%S"),
        utc_now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    db_execute(
        "INSERT INTO market_snapshots (asset, asset_type, price, chg_5m, chg_1h, chg_24h, last_candle, age_minutes, stale, error, fetched_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (asset) DO UPDATE SET asset_type=excluded.asset_type, price=excluded.price, "
        "chg_5m=excluded.chg_5m, chg_1h=excluded.chg_1h, chg_24h=excluded.chg_24h, "
        "last_candle=excluded.last_candle, age_minutes=excluded.age_minutes, stale=excluded.stale, "
        "error=excluded.error, fetched_at=excluded.fetched_at, updated_at=excluded.updated_at",
        values,
    )


def get_market_snapshots(symbols: list[str]) -> dict:
    if not symbols:
        return {}
    placeholders = ",".join("?" for _ in symbols)
    rows = db_query(
        "SELECT asset, asset_type, price, chg_5m, chg_1h, chg_24h, last_candle, age_minutes, stale, error, fetched_at, updated_at "
        f"FROM market_snapshots WHERE asset IN ({placeholders})", tuple(symbols))
    result = {}
    for row in rows:
        (asset, asset_type, price, chg_5m, chg_1h, chg_24h, last_candle, age, stale,
         error, fetched_at, updated_at) = row
        result[asset] = {
            "asset": asset, "type": asset_type, "price": price, "chg_5m": chg_5m,
            "chg_1h": chg_1h, "chg_24h": chg_24h,
            "last_candle": pd.Timestamp(last_candle) if last_candle else None,
            "age": age, "stale": bool(stale), "error": error,
            "fetched_at": pd.Timestamp(fetched_at) if fetched_at else None,
            "updated_at": pd.Timestamp(updated_at) if updated_at else None,
        }
    return result


def is_prediction_event(ev: dict) -> bool:
    """Only confirmed directional signals create simulated positions."""
    return (ev.get("status") == "Confirmed"
            and ev.get("direction") in ("bull", "bear")
            and bool(ev.get("notify")))


def _paper_trade_parameters(ev: dict, ind: pd.DataFrame, settings: dict) -> Optional[dict]:
    """Build a stop-based paper position: 2% of equity at risk and a 3R target."""
    event_time = pd.Timestamp(ev["event_time"])
    if ind is None or event_time not in ind.index:
        return None
    row = ind.loc[event_time]
    entry = float(ev["price"])
    atr = float(row.get("ATR14", np.nan))
    if entry <= 0 or not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
        return None
    side = 1.0 if ev["direction"] == "bull" else -1.0
    stop = (ev.get("plan") or {}).get("stop")
    if stop is None:
        level = ev.get("level")
        stop = (float(level) - side * 0.25 * atr
                if level is not None and np.isfinite(float(level))
                else entry - side * atr)
    stop = float(stop)
    if (side > 0 and stop >= entry) or (side < 0 and stop <= entry):
        stop = entry - side * atr
    risk_per_unit = abs(entry - stop)
    if not np.isfinite(risk_per_unit) or risk_per_unit <= 0:
        return None
    target = entry + side * PAPER_REWARD_R * risk_per_unit
    starting_balance = float(settings.get("paper_starting_balance", 10000.0))
    realized = db_query("SELECT COALESCE(SUM(pnl_amount), 0) FROM predictions WHERE outcome IS NOT NULL")[0][0]
    equity = max(0.01, starting_balance + float(realized or 0.0))
    risk_amount = equity * PAPER_RISK_PCT / 100.0
    units = risk_amount / risk_per_unit
    signal_time = event_time + pd.Timedelta(minutes=INTERVAL_MIN[ev["timeframe"]])
    return {"entry": entry, "stop": stop, "target": target, "equity": equity,
            "risk_amount": risk_amount, "units": units, "signal_time": signal_time}


def record_prediction(alert_id: int, ev: dict, tf: str, settings: dict, ind=None):
    """Record a paper trade for a confirmed signal; no time-based close is used."""
    if not is_prediction_event(ev):
        return None
    existing = db_query("SELECT id FROM predictions WHERE alert_id=?", (alert_id,))
    if existing:
        return int(existing[0][0])
    plan = _paper_trade_parameters(ev, ind, settings)
    if plan is None:
        log.info("Paper trade skipped for %s %s: no valid ATR stop.", ev["asset"], tf)
        return None
    ev["paper_trade"] = plan
    signal_time_text = plan["signal_time"].strftime("%Y-%m-%d %H:%M:%S")
    return db_execute(
        "INSERT INTO predictions (alert_id, asset, timeframe, pattern, direction, entry_price, signal_time, "
        "horizon_minutes, neutral_pct, due_at, stop_price, target_price, position_units, risk_amount, "
        "equity_at_entry, mark_price, unrealized_pnl, unrealized_pnl_pct, mark_time) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (alert_id, ev["asset"], tf, ev["alert_type"], ev["direction"], float(ev["price"]),
         signal_time_text, 0, 0.0, signal_time_text, plan["stop"], plan["target"], plan["units"],
         plan["risk_amount"], plan["equity"], plan["entry"], 0.0, 0.0, signal_time_text))


def mark_prediction_telegram_sent(prediction_id: int):
    db_execute("UPDATE predictions SET telegram_sent=1 WHERE id=?", (prediction_id,))


def evaluate_due_predictions(settings=None) -> int:
    """Mark open paper trades and close only when stop or target is reached."""
    settings = settings or load_settings()
    now_text = now_naive_utc().strftime("%Y-%m-%d %H:%M:%S")
    due = db_query(
        "SELECT id, asset, timeframe, direction, entry_price, stop_price, target_price, signal_time, "
        "exit_candle_time, risk_amount, position_units, equity_at_entry FROM predictions "
        "WHERE outcome IS NULL ORDER BY signal_time LIMIT 500")
    evaluated = 0
    for (pred_id, asset, tf, direction, entry, stop, target, signal_time, last_checked,
         risk_amount, units, equity_at_entry) in due:
        try:
            # Keep open-trade marks and stop/target checks current on a 15m
            # candle cadence, regardless of the signal's strategy timeframe.
            # The signal timeframe remains metadata; there is no time-based exit.
            evaluation_tf = "15m"
            res = fetch_candles(asset, evaluation_tf)
            if res.df is None or res.df.empty:
                continue  # data provider may be temporarily unavailable; retry next monitor cycle
            closed = closed_candles(res.df, evaluation_tf)
            if closed.empty:
                continue
            candle_closes = closed.index + pd.Timedelta(minutes=INTERVAL_MIN[evaluation_tf])
            start_after = pd.Timestamp(last_checked or signal_time)
            eligible = np.flatnonzero(candle_closes > start_after)
            if len(eligible) == 0:
                continue
            side = 1.0 if direction == "bull" else -1.0
            closed_trade = None
            for pos in eligible:
                bar = closed.iloc[int(pos)]
                open_price = float(bar["Open"])
                hit_stop = float(bar["Low"]) <= float(stop) if side > 0 else float(bar["High"]) >= float(stop)
                hit_target = float(bar["High"]) >= float(target) if side > 0 else float(bar["Low"]) <= float(target)
                if hit_stop:
                    # A gap through the stop gets the worse candle-open fill.
                    exit_price = min(float(stop), open_price) if side > 0 else max(float(stop), open_price)
                    outcome, reason = "FAILED", "STOP"
                elif hit_target:
                    exit_price, outcome, reason = float(target), "SUCCESS", "TARGET"
                else:
                    continue
                exit_time = pd.Timestamp(candle_closes[int(pos)]).strftime("%Y-%m-%d %H:%M:%S")
                directional_return = side * (exit_price / float(entry) - 1.0) * 100.0
                pnl_amount = side * (exit_price - float(entry)) * float(units)
                pnl_equity_pct = pnl_amount / float(equity_at_entry) * 100.0
                pnl_r = pnl_amount / float(risk_amount)
                closed_trade = (exit_price, outcome, reason, exit_time, directional_return,
                                pnl_amount, pnl_equity_pct, pnl_r)
                break

            if closed_trade:
                (exit_price, outcome, reason, exit_time, directional_return, pnl_amount,
                 pnl_equity_pct, pnl_r) = closed_trade
                db_execute(
                    "UPDATE predictions SET evaluated_at=?, exit_candle_time=?, exit_price=?, return_pct=?, "
                    "outcome=?, exit_reason=?, pnl_amount=?, pnl_equity_pct=?, pnl_r=?, mark_price=?, "
                    "unrealized_pnl=0, unrealized_pnl_pct=0 WHERE id=? AND outcome IS NULL",
                    (now_text, exit_time, exit_price, directional_return, outcome, reason, pnl_amount,
                     pnl_equity_pct, pnl_r, exit_price, pred_id))
                evaluated += 1
            else:
                latest = int(eligible[-1])
                mark_price = float(closed["Close"].iloc[latest])
                mark_time = pd.Timestamp(candle_closes[latest]).strftime("%Y-%m-%d %H:%M:%S")
                unrealized = side * (mark_price - float(entry)) * float(units)
                unrealized_pct = unrealized / float(equity_at_entry) * 100.0
                db_execute(
                    "UPDATE predictions SET exit_candle_time=?, mark_price=?, unrealized_pnl=?, "
                    "unrealized_pnl_pct=?, mark_time=? WHERE id=? AND outcome IS NULL",
                    (mark_time, mark_price, unrealized, unrealized_pct, mark_time, pred_id))
        except Exception as exc:
            log.warning("Paper trade evaluation failed for %s %s: %s", asset, tf, redact(exc))

    # Report closure P&L only for signals that were sent to Telegram.
    if telegram_configured():
        unsent = db_query(
            "SELECT id, asset, timeframe, pattern, direction, entry_price, stop_price, target_price, "
            "exit_price, pnl_amount, pnl_equity_pct, pnl_r, outcome, exit_reason, exit_candle_time "
            "FROM predictions "
            "WHERE outcome IS NOT NULL AND telegram_sent=1 AND result_notified=0 ORDER BY evaluated_at LIMIT 50")
        for row in unsent:
            (pred_id, asset, tf, pattern, direction, entry, stop, target, exit_price,
             pnl_amount, pnl_pct, pnl_r, outcome, reason, exit_time) = row
            side = "LONG" if direction == "bull" else "SHORT"
            icon = "✅" if outcome == "SUCCESS" else "❌"
            message = (f"{icon} Paper trade closed: {reason}\n{asset} | {side} | {pattern} | {tf}\n"
                       f"Entry: {fmt_price(entry)} | Stop: {fmt_price(stop)} | Target: {fmt_price(target)}\n"
                       f"Exit: {fmt_price(exit_price)}\nP&L: {pnl_amount:+.2f} "
                       f"({pnl_pct:+.2f}% equity, {pnl_r:+.2f}R)\nClosed candle: {exit_time} UTC")
            ok, _ = tg_send_message(message)
            if ok:
                db_execute("UPDATE predictions SET result_notified=1 WHERE id=?", (pred_id,))
    return evaluated


def _prediction_filters(timeframe=None, pattern=None):
    where, params = ["outcome IS NOT NULL"], []
    if timeframe:
        where.append("timeframe=?")
        params.append(timeframe)
    if pattern:
        where.append("LOWER(pattern) LIKE LOWER(?)")
        params.append(f"%{pattern}%")
    return " AND ".join(where), params


def cmd_stats(mode="summary", timeframe=None, pattern=None) -> str:
    where, params = _prediction_filters(timeframe, pattern)
    totals = db_query(
        f"SELECT COUNT(*), SUM(CASE WHEN outcome='SUCCESS' THEN 1 ELSE 0 END), "
        f"SUM(CASE WHEN outcome='FAILED' THEN 1 ELSE 0 END) FROM predictions WHERE {where}", tuple(params))[0]
    total, wins, losses = (int(v or 0) for v in totals)
    decided = wins + losses
    win_rate = wins / decided * 100 if decided else 0.0
    pnl_total = db_query(f"SELECT COALESCE(SUM(pnl_amount), 0) FROM predictions WHERE {where}", tuple(params))[0][0]
    total_r = db_query(f"SELECT COALESCE(SUM(pnl_r), 0) FROM predictions WHERE {where}", tuple(params))[0][0]
    title = "📊 Paper trading statistics"
    if timeframe:
        title += f" — {timeframe}"
    if pattern:
        title += f" — {pattern}"
    lines = [title, f"Closed trades: {total} | ✅ {wins} | ❌ {losses}",
             f"Win rate: {win_rate:.1f}% ({wins}/{decided})",
             f"Realized P&L: {float(pnl_total or 0):+.2f} ({float(total_r or 0):+.2f}R), before costs"]
    if total == 0:
        lines.append("No paper trades have reached a stop or target yet.")
        return "\n".join(lines)
    if mode in ("patterns", "timeframes", "detailed"):
        group_cols = {"patterns": "pattern", "timeframes": "timeframe",
                      "detailed": "timeframe, pattern"}[mode]
        rows = db_query(
            f"SELECT {group_cols}, COUNT(*), SUM(CASE WHEN outcome='SUCCESS' THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN outcome='FAILED' THEN 1 ELSE 0 END) "
            f"FROM predictions WHERE {where} GROUP BY {group_cols} ORDER BY COUNT(*) DESC LIMIT 25",
            tuple(params))
        lines.extend(["", f"By {mode}:"])
        for row in rows:
            if mode == "detailed":
                label = f"{row[0]} / {row[1]}"
            else:
                label = str(row[0])
            count, w, l = (int(v or 0) for v in row[-3:])
            rate = w / (w + l) * 100 if w + l else 0.0
            lines.append(f"• {label}: {rate:.1f}% | {count} closed (✅{w} ❌{l})")
    return "\n".join(lines)


def cmd_prediction_history(limit=8) -> str:
    """Compact Telegram view of latest open and closed paper trades."""
    df = get_predictions(limit=limit)
    if df.empty:
        return "📜 Paper trade history\nNo paper trades stored yet. Only confirmed directional signals are tracked."
    lines = ["📜 Recent paper trades", "Gross P&L before fees, spread and slippage."]
    for row in df.itertuples(index=False):
        side = "LONG" if row.direction == "bull" else "SHORT"
        if row.outcome is None or pd.isna(row.outcome):
            result = "⏳ OPEN"
            pnl_text = f"unrealized {float(row.unrealized_pnl or 0):+.2f} ({float(row.unrealized_pnl_pct or 0):+.2f}% equity)"
        else:
            icon = {"SUCCESS": "✅", "FAILED": "❌"}.get(row.outcome, "•")
            result = f"{icon} {row.exit_reason or row.outcome}"
            pnl_text = f"{float(row.pnl_amount or 0):+.2f} ({float(row.pnl_equity_pct or 0):+.2f}% equity, {float(row.pnl_r or 0):+.2f}R)"
        lines.append(f"\n{row.asset} | {side} | {row.timeframe}")
        lines.append(f"{row.pattern} — {result} | P&L {pnl_text}")
        lines.append(f"Entry {fmt_price(row.entry_price)} | Stop {fmt_price(row.stop_price)} | Target {fmt_price(row.target_price)}")
    return "\n".join(lines)[:3900]


def _pattern_command_slug(pattern: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(pattern).lower()).strip("_")


def _pattern_from_command_slug(slug: str) -> str:
    rows = db_query("SELECT DISTINCT pattern FROM predictions ORDER BY pattern")
    for row in rows:
        if _pattern_command_slug(row[0]) == slug:
            return row[0]
    return slug.replace("_", " ")

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


PREDICTION_COLUMNS = [
    "id", "alert_id", "asset", "timeframe", "pattern", "direction", "entry_price",
    "signal_time", "telegram_sent",
    "evaluated_at", "exit_candle_time", "exit_price", "return_pct", "outcome",
    "result_notified", "description", "stop_price", "target_price", "position_units",
    "risk_amount", "equity_at_entry", "pnl_amount", "pnl_equity_pct", "pnl_r",
    "mark_price", "unrealized_pnl", "unrealized_pnl_pct", "mark_time", "exit_reason",
]


def get_prediction_filter_options() -> dict:
    """Fetch unique filter values using SQL, without loading the full prediction history."""
    rows = db_query(
        "SELECT 'asset' AS kind, asset AS value FROM predictions GROUP BY asset "
        "UNION ALL SELECT 'timeframe', timeframe FROM predictions GROUP BY timeframe "
        "UNION ALL SELECT 'pattern', pattern FROM predictions GROUP BY pattern ORDER BY kind, value"
    )
    options = {"asset": [], "timeframe": [], "pattern": []}
    for kind, value in rows:
        if kind in options and value is not None:
            options[kind].append(str(value))
    return options


def get_predictions(limit=500, asset=None, timeframe=None, pattern=None, outcome=None,
                    sort_by="newest") -> pd.DataFrame:
    """Read prediction history, including pending rows, joined to each alert explanation."""
    where, params = [], []
    if asset:
        where.append("p.asset=?")
        params.append(asset)
    if timeframe:
        where.append("p.timeframe=?")
        params.append(timeframe)
    if pattern:
        where.append("p.pattern=?")
        params.append(pattern)
    if outcome == "PENDING":
        where.append("p.outcome IS NULL")
    elif outcome:
        where.append("p.outcome=?")
        params.append(outcome)

    sort_sql = {
        "newest": "p.signal_time DESC, p.id DESC",
        "oldest": "p.signal_time ASC, p.id ASC",
        "asset": "p.asset ASC, p.signal_time DESC",
        "pattern": "p.pattern ASC, p.signal_time DESC",
        "best": "CASE WHEN p.outcome IS NULL THEN p.unrealized_pnl ELSE p.pnl_amount END DESC",
        "worst": "CASE WHEN p.outcome IS NULL THEN p.unrealized_pnl ELSE p.pnl_amount END ASC",
    }.get(sort_by, "p.signal_time DESC, p.id DESC")
    sql = (
        "SELECT p.id, p.alert_id, p.asset, p.timeframe, p.pattern, p.direction, p.entry_price, "
        "p.signal_time, p.telegram_sent, p.evaluated_at, p.exit_candle_time, p.exit_price, "
        "p.return_pct, p.outcome, "
        "p.result_notified, COALESCE(a.description, ''), p.stop_price, p.target_price, p.position_units, "
        "p.risk_amount, p.equity_at_entry, p.pnl_amount, p.pnl_equity_pct, p.pnl_r, p.mark_price, "
        "p.unrealized_pnl, p.unrealized_pnl_pct, p.mark_time, p.exit_reason "
        "FROM predictions p LEFT JOIN alerts a ON a.id=p.alert_id"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY {sort_sql} LIMIT ?"
    params.append(max(1, min(int(limit), 5000)))
    return pd.DataFrame(db_query(sql, tuple(params)), columns=PREDICTION_COLUMNS)


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
    return setting_value("TELEGRAM_BOT_TOKEN")


def tg_chat_id() -> str:
    return setting_value("TELEGRAM_CHAT_ID")


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


def tg_send_message(text, chat_id=None, reply_markup=None):
    data = {"chat_id": chat_id or tg_chat_id(), "text": text[:4096],
            "disable_web_page_preview": "true"}
    if reply_markup:
        data["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)
    return tg_request("sendMessage", data=data)


def tg_edit_message_text(text, chat_id, message_id, reply_markup=None):
    data = {"chat_id": chat_id, "message_id": message_id, "text": text[:4096],
            "disable_web_page_preview": "true"}
    if reply_markup:
        data["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)
    return tg_request("editMessageText", data=data)


def tg_answer_callback_query(callback_query_id, text=None):
    data = {"callback_query_id": callback_query_id}
    if text:
        data["text"] = str(text)[:180]
    return tg_request("answerCallbackQuery", data=data)


def tg_send_photo(path, caption, chat_id=None):
    with open(path, "rb") as fh:
        return tg_request("sendPhoto", data={"chat_id": chat_id or tg_chat_id(), "caption": caption[:1024]},
                          files={"photo": fh}, timeout=60)


def build_message(ev: dict) -> str:
    icon = {"bull": "🟢", "bear": "🔴"}.get(ev["direction"], "🟡")
    title = ev["alert_type"] if ev["status"] in ("Info", "Test") else f"{ev['alert_type']} ({ev['status']})"
    if ev["status"] == "Test":
        title = "🧪 " + title
    paper = ev.get("paper_trade")
    paper_lines = []
    if paper:
        paper_lines = [
            f"Paper trade only: risk {PAPER_RISK_PCT:g}% ({chr(36)}{paper['risk_amount']:.2f}) | target {PAPER_REWARD_R:g}R",
            f"Entry {fmt_price(paper['entry'])} | Stop {fmt_price(paper['stop'])} | Target {fmt_price(paper['target'])}",
            f"Position size: {paper['units']:.6f} units | account equity: {chr(36)}{paper['equity']:.2f}",
        ]
    head = "\n".join([
        f"{icon} {title}",
        f"Asset: {asset_label(ev['asset'])}",
        f"Price: {fmt_price(ev['price'])}",
        f"Change: {fmt_pct(ev['change_pct'])} ({ev['change_label']})",
        f"Timeframe: {ev['timeframe']}",
        *paper_lines,
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
    symbols = watched_symbols(load_settings())
    lines = ["📋 Watchlist"]
    if symbols:
        with ThreadPoolExecutor(max_workers=min(SCANNER_MAX_WORKERS, len(symbols)), thread_name_prefix="tim-watchlist") as pool:
            snapshots = list(pool.map(get_snapshot, symbols))
        for sym, s in zip(symbols, snapshots):
            if s["error"]:
                lines.append(f"• {sym}: no data")
            else:
                lines.append(f"• {sym}: {fmt_price(s['price'])} (24h {fmt_pct(s['chg_24h'])}, "
                             f"data age {s['age']:.0f} min)")
    lines.append("Sources: Yahoo Finance for configured assets; Polymarket Perps API for -PERP instruments.")
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
    s, cfg = get_monitor_status(), load_settings()
    last_value = s.get("last_cycle")
    last = pd.Timestamp(last_value).strftime("%Y-%m-%d %H:%M:%S") if last_value else "not finished yet"
    metrics = s.get("metrics", {})
    market_requests = int(metrics.get("market_requests", 0) or 0)
    db_queries = int(metrics.get("db_queries", 0) or 0)
    market_avg = (float(metrics.get("market_latency_ms_total", 0.0)) / market_requests) if market_requests else 0.0
    db_avg = (float(metrics.get("db_latency_ms_total", 0.0)) / db_queries) if db_queries else 0.0
    cache_lookups = int(metrics.get("cache_hits", 0) or 0) + int(metrics.get("cache_misses", 0) or 0)
    cache_hit_pct = (100.0 * int(metrics.get("cache_hits", 0) or 0) / cache_lookups) if cache_lookups else 0.0
    return "\n".join([
        "ℹ️ TIM status",
        f"Worker state: {s.get('worker_state', 'unknown')}",
        f"Monitor last cycle (UTC): {last} (cycles: {s.get('cycles', 0)})",
        f"Last scan: {s.get('cycle_seconds', 'n/a')} sec | assets: {s.get('symbols_scanned', 'n/a')}",
        f"Market data requests: {market_requests} | avg latency: {market_avg:.0f} ms | cache hit: {cache_hit_pct:.1f}%",
        f"Database queries: {db_queries} | avg latency: {db_avg:.1f} ms",
        f"Assets watched: {len(watched_symbols(cfg))}",
        f"Cooldown between repeated alerts: {cfg['cooldown_minutes']} min",
        f"Minimum structural break for Telegram: {cfg.get('min_breakout_atr', DEFAULT_SETTINGS['min_breakout_atr']):g} ATR",
        f"Sent to Telegram: {', '.join(cfg['telegram_timeframes']) or 'nothing'} (other timeframes: history only)",
        f"Paper trading: {PAPER_RISK_PCT:g}% risk / {PAPER_REWARD_R:g}R target | starting balance: ${cfg.get('paper_starting_balance', 10000.0):,.2f}",
        f"Database: {'PostgreSQL' if database_url() else 'local SQLite (not durable on ephemeral hosting)'}",
        f"Last monitor error: {s.get('last_error') or 'none'}",
        "Data: Yahoo Finance for configured assets; Polymarket Perps API for -PERP instruments."])

def telegram_main_keyboard():
    """Small phone-friendly home menu; no command typing is required for common tasks."""
    return {"inline_keyboard": [
        [{"text": "📊 Paper P&L", "callback_data": "ui:stats"},
         {"text": "📜 Paper trades", "callback_data": "ui:history"}],
        [{"text": "📋 Watchlist", "callback_data": "ui:watchlist"},
         {"text": "🔔 Latest alerts", "callback_data": "ui:alerts"}],
        [{"text": "⚙️ Monitor status", "callback_data": "ui:status"},
         {"text": "❔ Help", "callback_data": "ui:help"}],
    ]}


def telegram_back_keyboard(target="ui:home"):
    return {"inline_keyboard": [[{"text": "↩️ Back to menu", "callback_data": target}]]}


def telegram_stats_keyboard():
    return {"inline_keyboard": [
        [{"text": "📊 Overview", "callback_data": "ui:stats"}],
        [{"text": "Patterns", "callback_data": "ui:stats_patterns"},
         {"text": "Timeframes", "callback_data": "ui:stats_timeframes"}],
        [{"text": "Detailed breakdown", "callback_data": "ui:stats_detailed"}],
        [{"text": "🏠 Main menu", "callback_data": "ui:home"}],
    ]}


def _button_keyboard(buttons, callback_prefix, footer=True):
    rows, row = [], []
    for label, value in buttons:
        row.append({"text": str(label)[:55], "callback_data": f"{callback_prefix}:{value}"[:64]})
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    if footer:
        rows.append([{"text": "🏠 Main menu", "callback_data": "ui:home"}])
    return {"inline_keyboard": rows}


def help_text(chat_id: str, allowed: bool) -> str:
    lines = ["📈 Trading Intelligence Monitor", f"Your chat ID: {chat_id}"]
    if not allowed:
        lines.append("To use the bot, add this chat ID as TELEGRAM_CHAT_ID in Streamlit Cloud → Settings → Secrets "
                     "(or to your local .env file), then restart the app.")
    lines += ["", "Quick menu: /menu", "Commands:",
              "/watchlist - assets and last prices", "/alerts - last 5 alerts",
              "/status - monitor status", "/prediction_history - recent predictions",
              "/stats - overall prediction statistics", "/stats_patterns - results by pattern",
              "/stats_timeframes - results by timeframe", "/stats_detailed - pattern + timeframe",
              "/stats_1h - one timeframe", "/stats_pattern_bullish_breakout - one pattern",
              "/test - send a test notification with a chart", "/help - this message", "", DISCLAIMER]
    return "\n".join(lines)


def telegram_callback_view(action: str):
    """Return text + keyboard for an inline-menu action."""
    if action == "home":
        return "📈 Trading Intelligence Monitor\nChoose what you want to view:", telegram_main_keyboard()
    if action == "stats":
        return cmd_stats(), telegram_stats_keyboard()
    if action == "stats_detailed":
        return cmd_stats(mode="detailed"), telegram_stats_keyboard()
    if action == "stats_patterns":
        patterns = [r[0] for r in db_query("SELECT DISTINCT pattern FROM predictions ORDER BY pattern")]
        if not patterns:
            return "No prediction patterns yet.", telegram_back_keyboard("ui:stats")
        return "Choose a pattern to see its completed prediction statistics:", _button_keyboard(
            [(p, p) for p in patterns[:20]], "pattern")
    if action == "stats_timeframes":
        timeframes = [r[0] for r in db_query("SELECT DISTINCT timeframe FROM predictions ORDER BY timeframe")]
        if not timeframes:
            return "No prediction timeframes yet.", telegram_back_keyboard("ui:stats")
        return "Choose a timeframe:", _button_keyboard([(tf, tf) for tf in timeframes], "timeframe")
    if action == "history":
        keyboard = {"inline_keyboard": [
            [{"text": "🔄 Refresh", "callback_data": "ui:history"}],
            [{"text": "📊 Statistics", "callback_data": "ui:stats"},
             {"text": "🏠 Main menu", "callback_data": "ui:home"}],
        ]}
        return cmd_prediction_history(8), keyboard
    if action == "watchlist":
        return cmd_watchlist(), telegram_back_keyboard()
    if action == "alerts":
        return cmd_alerts(), telegram_back_keyboard()
    if action == "status":
        return cmd_status(), telegram_back_keyboard()
    if action == "help":
        return help_text(tg_chat_id(), True), telegram_back_keyboard()
    if action.startswith("pattern:"):
        return cmd_stats(pattern=action.split(":", 1)[1]), telegram_back_keyboard("ui:stats_patterns")
    if action.startswith("timeframe:"):
        return cmd_stats(timeframe=action.split(":", 1)[1]), telegram_back_keyboard("ui:stats_timeframes")
    return "Unknown menu action. Send /menu to reopen the menu.", telegram_back_keyboard()


def handle_callback_query(callback: dict):
    callback_id = callback.get("id")
    message = callback.get("message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    message_id = message.get("message_id")
    action = callback.get("data", "")
    if not chat_id or not callback_id:
        return
    if chat_id != tg_chat_id():
        tg_answer_callback_query(callback_id, "This bot is not authorised for this chat.")
        return
    tg_answer_callback_query(callback_id)
    if action.startswith("ui:"):
        action = action.split(":", 1)[1]
    text, keyboard = telegram_callback_view(action)
    ok, _ = tg_edit_message_text(text, chat_id, message_id, keyboard) if message_id else (False, None)
    if not ok:
        tg_send_message(text, chat_id, reply_markup=keyboard)


def _stats_reply_for_command(command: str, text: str) -> str:
    """Accept underscore aliases while retaining the original space-separated syntax."""
    if command == "/stats_patterns":
        return cmd_stats(mode="patterns")
    if command == "/stats_timeframes":
        return cmd_stats(mode="timeframes")
    if command == "/stats_detailed":
        return cmd_stats(mode="detailed")
    if command.startswith("/stats_pattern_"):
        slug = command[len("/stats_pattern_"):]
        return cmd_stats(pattern=_pattern_from_command_slug(slug))
    if command.startswith("/stats_") and command[len("/stats_"):] in TIMEFRAMES:
        return cmd_stats(timeframe=command[len("/stats_"):])
    args = text.split()[1:]
    if not args:
        return cmd_stats()
    if args[0].lower() in ("patterns", "timeframes", "detailed"):
        return cmd_stats(mode=args[0].lower())
    if args[0].lower() in TIMEFRAMES:
        return cmd_stats(timeframe=args[0].lower())
    if args[0].lower() == "pattern" and len(args) > 1:
        return cmd_stats(pattern=" ".join(args[1:]))
    return ("Usage: /stats | /stats_patterns | /stats_timeframes | /stats_detailed | "
            "/stats_1h | /stats_pattern_bullish_breakout")


def handle_update(update: dict):
    callback = update.get("callback_query")
    if callback:
        handle_callback_query(callback)
        return
    msg = update.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat_id = str((msg.get("chat") or {}).get("id", ""))
    if not text.startswith("/") or not chat_id:
        return
    command = text.split()[0].split("@")[0].lower()
    allowed = chat_id == tg_chat_id()
    reply_markup = None
    if command in ("/start", "/help", "/menu"):
        reply = help_text(chat_id, allowed)
        if allowed:
            reply_markup = telegram_main_keyboard()
    elif not allowed:
        reply = ("This chat is not authorised. Put this chat ID in TELEGRAM_CHAT_ID in your .env file "
                 f"and restart the app: {chat_id}")
    elif command == "/watchlist":
        reply = cmd_watchlist()
    elif command == "/alerts":
        reply = cmd_alerts()
    elif command == "/status":
        reply = cmd_status()
    elif command in ("/stats", "/stats_patterns", "/stats_timeframes", "/stats_detailed") \
            or command.startswith("/stats_"):
        reply = _stats_reply_for_command(command, text)
    elif command in ("/prediction_history", "/predictions"):
        reply = cmd_prediction_history()
    elif command == "/test":
        ok, info = send_test_notification(chat_id)
        if ok:
            return
        reply = f"Test failed: {info}"
    else:
        reply = "Unknown command. Send /menu or /help."
    tg_send_message(reply, chat_id, reply_markup=reply_markup)


def telegram_poll_loop(stop_event: Optional[threading.Event] = None):
    """Worker thread: listens for /commands using long polling (getUpdates)."""
    offset = None
    commands_registered = False
    if _wait_or_stop(stop_event, 3):
        return
    while stop_event is None or not stop_event.is_set():
        if not tg_token():
            if _wait_or_stop(stop_event, 30):
                return
            continue
        if not commands_registered:
            commands = [
                {"command": "menu", "description": "Open the mobile menu"},
                {"command": "prediction_history", "description": "Recent predictions and P/L"},
                {"command": "stats", "description": "Overall prediction statistics"},
                {"command": "stats_patterns", "description": "Statistics by pattern"},
                {"command": "stats_timeframes", "description": "Statistics by timeframe"},
                {"command": "stats_detailed", "description": "Pattern and timeframe breakdown"},
                {"command": "stats_15m", "description": "Statistics for 15m"},
                {"command": "stats_1h", "description": "Statistics for 1h"},
                {"command": "stats_4h", "description": "Statistics for 4h"},
                {"command": "stats_1d", "description": "Statistics for 1d"},
                {"command": "stats_pattern_bullish_breakout", "description": "Stats for bullish breakouts"},
                {"command": "watchlist", "description": "Assets and latest prices"},
                {"command": "alerts", "description": "Latest alerts"},
                {"command": "status", "description": "Monitor status"},
                {"command": "help", "description": "Show help"},
            ]
            ok_commands, _ = tg_request("setMyCommands", data={"commands": json.dumps(commands)})
            commands_registered = bool(ok_commands)
        if offset is None:  # skip old messages that arrived while the app was off
            ok, result = tg_request("getUpdates", params={"timeout": 0}, http="get")
            if ok:
                offset = result[-1]["update_id"] + 1 if result else 0
        params = {"timeout": 25, "allowed_updates": json.dumps(["message", "callback_query"])}
        if offset:
            params["offset"] = offset
        ok, result = tg_request("getUpdates", params=params, timeout=40, http="get")
        if not ok:
            RT.status["telegram_error"] = result
            if _wait_or_stop(stop_event, 15):
                return
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
    routed = tf in PATTERN_SCAN_TFS and tf in settings.get("telegram_timeframes", PATTERN_SCAN_TFS)
    quality_ok = passes_notification_quality(ev, settings)
    prediction_id = None
    # Notification quality is a Telegram noise filter. Keep every confirmed
    # directional signal in the paper ledger so the strategy can be evaluated
    # independently of the alert-delivery threshold.
    if is_prediction_event(ev) and tf in PATTERN_SCAN_TFS:
        prediction_id = record_prediction(alert_id, ev, tf, settings, ind)
    if ev["notify"] and quality_ok and routed and telegram_configured():
        ok, info = send_alert_notification(ev, ind, tf)
        if ok and prediction_id is not None:
            mark_prediction_telegram_sent(prediction_id)
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
    is_perps = symbol.endswith("-PERP")
    snapshot_tf = "15m" if is_perps else "5m"
    res5 = fetch_candles(symbol, snapshot_tf)
    try:
        if res5.df is not None and not res5.df.empty:
            save_market_snapshot(get_snapshot_from_result(symbol, res5, snapshot_tf))
        else:
            save_market_snapshot({"asset": symbol,
                                  "type": "Crypto" if is_crypto(symbol) else "Index" if symbol.startswith("^") else "Stock",
                                  "error": res5.error or "No market data", "stale": True})
            if symbol.endswith("-PERP"):
                log.warning("No dashboard candles for %s from Polymarket Perps: %s",
                            symbol, res5.error or "empty response")
    except Exception as exc:
        log.warning("Could not publish dashboard snapshot for %s: %s", symbol, redact(exc))
    for tf in PATTERN_SCAN_TFS:
        res = fetch_candles(symbol, tf)
        if res.df is None:
            continue
        ind = get_indicators(symbol, tf, res)
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

def _wait_or_stop(stop_event: Optional[threading.Event], seconds: float) -> bool:
    if stop_event is None:
        time.sleep(seconds)
        return False
    return stop_event.wait(seconds)


def monitor_loop(stop_event: Optional[threading.Event] = None):
    """Run scanning in the dedicated worker process, never inside the Streamlit server."""
    stop_event = stop_event or RT.stop_event
    _wait_or_stop(stop_event, 5)
    if stop_event.is_set():
        return
    RT.status["worker_state"] = "running"
    publish_monitor_status()
    log.info("Background worker monitor started (every %s s).", MONITOR_INTERVAL_SEC)
    with ThreadPoolExecutor(max_workers=SCANNER_MAX_WORKERS, thread_name_prefix="tim-scan") as pool:
        while not stop_event.is_set():
            started = time.time()
            errors = []
            symbols = []
            try:
                settings = load_settings()
                old_perps = settings.get("polymarket_perps", [])
                try:
                    settings = refresh_polymarket_perps(settings)
                    if settings.get("polymarket_perps") != old_perps:
                        save_settings(settings)
                except Exception as exc:
                    log.warning("Could not refresh Polymarket stock perps: %s", redact(exc))
                symbols = watched_symbols(settings)

                def scan_one(symbol):
                    try:
                        scan_symbol(symbol, settings)
                        return None
                    except Exception as exc:
                        log.warning("Scan failed for %s: %s", symbol, redact(exc))
                        return f"{symbol}: {redact(exc)}"

                for error in pool.map(scan_one, symbols):
                    if error:
                        errors.append(error)
                try:
                    check_pending_symbols(settings)
                except Exception as exc:
                    errors.append(f"pending symbols: {redact(exc)}")
                try:
                    evaluate_due_predictions(settings)
                except Exception as exc:
                    errors.append(f"prediction evaluator: {redact(exc)}")
                RT.status["last_cycle"] = utc_now()
                RT.status["cycles"] += 1
                RT.status["symbols_scanned"] = len(symbols)
            except Exception as exc:
                errors.append(redact(exc))
                log.warning("Monitor cycle failed: %s", redact(exc))
            RT.status["last_error"] = "; ".join(errors[:3]) if errors else None
            RT.status["cycle_seconds"] = round(time.time() - started, 2)
            try:
                publish_monitor_status()
            except Exception as exc:
                log.warning("Could not publish worker heartbeat: %s", redact(exc))
            remaining = max(5.0, MONITOR_INTERVAL_SEC - (time.time() - started))
            if _wait_or_stop(stop_event, remaining):
                break


@st.cache_resource(show_spinner=False)
def get_runtime() -> Runtime:
    """Create web-process state only. The worker is started separately with `python worker.py`."""
    return Runtime()


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
            "Send these strategy timeframes to Telegram",
            PATTERN_SCAN_TFS,
            default=[t for t in settings.get("telegram_timeframes", PATTERN_SCAN_TFS)
                     if t in PATTERN_SCAN_TFS],
            help="1m and 5m are excluded. Strategy signals and predictions start at 15m; other detected alerts remain in history.")
        if st.form_submit_button("💾 Save thresholds"):
            settings.update({"crypto_pct": c_pct, "crypto_minutes": int(c_min), "stock_pct": s_pct,
                             "stock_minutes": int(s_min), "volume_mult": v_mult,
                             "min_breakout_atr": min_breakout_atr,
                             "cooldown_minutes": int(cooldown),
                             "telegram_timeframes": [t for t in PATTERN_SCAN_TFS if t in tg_tfs]})
            save_settings(settings)
            st.success("Saved. The background monitor uses the new values on its next cycle.")

    sb.subheader("Paper trading")
    with sb.form("paper_trade_settings"):
        paper_balance = st.number_input(
            "Starting virtual account balance ($)", min_value=100.0, max_value=10000000.0,
            value=float(settings.get("paper_starting_balance", 10000.0)), step=100.0)
        st.caption("Fixed rules: 2% account risk per signal, 3R target. A paper position stays open until its stop or target is reached; no time-based exit and no real orders.")
        if st.form_submit_button("💾 Save paper trading settings"):
            settings["paper_starting_balance"] = float(paper_balance)
            save_settings(settings)
            st.success("Paper account balance saved. It applies to new signals.")

    with sb.expander("📨 Telegram notifications"):
        if telegram_configured():
            st.success("Telegram is configured.")
            if st.button("Send test notification"):
                with st.spinner("Sending..."):
                    ok, info = send_test_notification()
                (st.success if ok else st.error)(info)
            monitor_state = get_monitor_status()
            if monitor_state.get("telegram_error"):
                st.warning(f"Telegram polling issue: {monitor_state['telegram_error']}")
        else:
            st.warning("Telegram is not configured. The app works without it, but no notifications "
                       "will be sent.")
            st.markdown(
                "1. Create a bot with **@BotFather** and copy the token.\n"
                "2. Copy `.env.example` to `.env` and fill in `TELEGRAM_BOT_TOKEN`.\n"
                "3. Restart the app, send `/start` to your bot - it replies with your chat ID.\n"
                "4. Put that number in `TELEGRAM_CHAT_ID` in `.env` and restart again.\n\n"
                "Full guide: README.md")

    s = get_monitor_status()
    last_value = s.get("last_cycle")
    last = pd.Timestamp(last_value).strftime("%H:%M:%S UTC") if last_value else "no completed cycle yet"
    worker_fresh = False
    if last_value:
        last_ts = pd.Timestamp(last_value)
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        worker_fresh = (s.get("worker_state") == "running" and
                        (utc_now() - last_ts.to_pydatetime()).total_seconds() <= MONITOR_INTERVAL_SEC * 3)
    worker_label = "online" if worker_fresh else s.get("worker_state", "not detected")
    sb.caption(f"Background worker: {worker_label} | last cycle {last} | cycles: {s.get('cycles', 0)}")
    if s.get("cycle_seconds") is not None:
        metrics = s.get("metrics", {})
        request_count = metrics.get("market_requests", 0)
        hit_count = metrics.get("cache_hits", 0)
        hit_rate = (100.0 * hit_count / (hit_count + metrics.get("cache_misses", 0))) if (hit_count + metrics.get("cache_misses", 0)) else 0.0
        market_avg = (metrics.get("market_latency_ms_total", 0.0) / request_count) if request_count else 0.0
        db_queries = metrics.get("db_queries", 0)
        db_avg = (metrics.get("db_latency_ms_total", 0.0) / db_queries) if db_queries else 0.0
        sb.caption(f"Last scan: {s['cycle_seconds']}s | Yahoo avg: {market_avg:.0f}ms | DB avg: {db_avg:.1f}ms | cache: {hit_rate:.1f}%")
    if s.get("last_error"):
        sb.caption(f"Last monitor error: {s['last_error']}")

@st.fragment(run_every=REFRESH_SECONDS)
def dashboard_fragment():
    settings = load_settings()
    rows, problems = [], []
    symbols = watched_symbols(settings)
    snapshots = get_market_snapshots(symbols)
    for sym in symbols:
        snap = snapshots.get(sym)
        if snap is None:
            problems.append(f"{sym}: waiting for the background worker to publish its first snapshot")
            rows.append({"Asset": asset_label(sym), "Type": "Crypto" if is_crypto(sym) else "Index" if sym.startswith("^") else "Stock",
                         "Price": np.nan, "5m %": np.nan, "1h %": np.nan, "24h %": np.nan,
                         "Last candle (UTC)": "n/a", "Age (min)": np.nan, "Status": "Waiting for worker"})
            continue
        updated_at = snap.get("updated_at")
        worker_snapshot_age = float("inf")
        if updated_at is not None:
            ts = pd.Timestamp(updated_at)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            worker_snapshot_age = max(0.0, (utc_now() - ts.to_pydatetime()).total_seconds())
        worker_snapshot_stale = worker_snapshot_age > MONITOR_INTERVAL_SEC * 3
        if snap["error"]:
            problems.append(f"{sym}: {snap['error']}")
        if snap["error"] or snap.get("price") is None:
            rows.append({"Asset": asset_label(sym), "Type": snap["type"], "Price": np.nan, "5m %": np.nan,
                         "1h %": np.nan, "24h %": np.nan, "Last candle (UTC)": "n/a",
                         "Age (min)": np.nan, "Status": "No data"})
        else:
            state = "Worker snapshot stale" if worker_snapshot_stale else (
                "Stale (market closed or delayed)" if snap["stale"] else "OK")
            rows.append({"Asset": asset_label(sym), "Type": snap["type"], "Price": snap["price"],
                         "5m %": snap["chg_5m"], "1h %": snap["chg_1h"], "24h %": snap["chg_24h"],
                         "Last candle (UTC)": fmt_ts(snap["last_candle"]) if snap["last_candle"] is not None else "n/a",
                         "Age (min)": snap["age"], "Status": state})

    recent = get_alerts(since_minutes=60, limit=50)
    c1, c2, c3 = st.columns(3)
    c1.metric("Assets watched", len(symbols))
    c2.metric("Active alerts (last 60 min)", len(recent))
    c3.metric("Dashboard refreshed (UTC)", utc_now().strftime("%H:%M:%S"))
    st.caption("Data source: Yahoo Finance via yfinance (unofficial feed). Quotes can be delayed or "
               "incomplete and are NOT guaranteed real-time. Dashboard values are published by the "
               "separate background worker; a stale worker snapshot is explicitly marked. Auto-refresh "
               f"every {REFRESH_SECONDS} s.")
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
                           "Result": t["outcome"], "R (after cost)": t["r"],
                           "Equity change % (2% risk)": t["equity_return_pct"]} for t in trades])
    st.dataframe(table.style.format({"Entry": fmt_price, "Stop": fmt_price, "Target": fmt_price,
                                     "R (after cost)": "{:+.2f}",
                                     "Equity change % (2% risk)": "{:+.2f}%"}), hide_index=True)


def chart_analysis():
    settings = load_settings()
    c1, c2, c3 = st.columns([2, 3, 2])
    symbol = c1.selectbox("Asset", watched_symbols(settings), key="chart_symbol")
    tf = c2.radio("Timeframe", PATTERN_SCAN_TFS, index=1, horizontal=True, key="chart_tf")
    candles = c3.slider("Candles shown", 50, 400, 150, 10, key="chart_candles")

    res = fetch_candles(symbol, tf)
    if res.df is None:
        st.warning(res.error)
        return
    ind = get_indicators(symbol, tf, res)
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
    data_source = "Polymarket Perps API" if symbol.endswith("-PERP") else "Yahoo Finance via yfinance"
    st.caption(f"Data source: {data_source}. Downloaded {res.fetched_at:%Y-%m-%d %H:%M:%S} UTC | "
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
        st.caption("Paper replay only: entry at signal close, 2% of starting equity at risk per trade, target = 3R. "
                   "No real orders are sent. If one candle touches both stop and target it counts as a LOSS "
                   "(conservative). Results omit compounding and depend on candle data quality.")
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
    st.caption("Alerts are created by the separate background worker (checks every 60 s) and stored in the "
               "shared database. 'Potential' = not confirmed yet; 'Confirmed' = closed candle beyond "
               "the level with volume confirmation. Only confirmed patterns, rapid moves and volume spikes "
               "are sent to Telegram.")
    f1, f2, f3 = st.columns(3)
    asset = f1.selectbox("Asset", ["All"] + watched_symbols(settings), key="alerts_asset")
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


@st.fragment(run_every=REFRESH_SECONDS)
def predictions_fragment():
    """Dedicated, filterable and sortable history of every confirmed directional prediction."""
    st.subheader("Paper trade ledger")
    st.caption("Confirmed directional signals open simulated positions with 2% account risk and a 3R target. "
               "Positions close only when stop or target is touched; P&L is updated from closed candles, with no time exit.")
    filter_options = get_prediction_filter_options()
    assets = filter_options["asset"]
    timeframes = [tf for tf in TIMEFRAMES if tf in set(filter_options["timeframe"])]
    patterns = filter_options["pattern"]
    if not assets:
        st.info("No paper trades yet. A confirmed directional signal will appear here once detected.")
        st.caption("Potential signals are not counted as predictions. Existing alert history is not deleted.")
        return
    filters = st.columns(5)
    asset = filters[0].selectbox("Asset", ["All"] + assets, key="pred_asset")
    timeframe = filters[1].selectbox("Timeframe", ["All"] + timeframes, key="pred_timeframe")
    pattern = filters[2].selectbox("Pattern", ["All"] + patterns, key="pred_pattern")
    result_label = filters[3].selectbox("Status", ["All", "Open", "Target hit", "Stop hit"],
                                        key="pred_result")
    sort_label = filters[4].selectbox(
        "Sort by", ["Newest first", "Oldest first", "Highest P/L", "Lowest P/L", "Asset A–Z", "Pattern A–Z"],
        key="pred_sort")
    limit = st.selectbox("Rows", [50, 100, 200, 500, 1000, 5000], index=2, key="pred_limit")
    result_value = {"Open": "PENDING", "Target hit": "SUCCESS", "Stop hit": "FAILED"}.get(result_label)
    sort_value = {"Newest first": "newest", "Oldest first": "oldest", "Highest P/L": "best",
                  "Lowest P/L": "worst", "Asset A–Z": "asset", "Pattern A–Z": "pattern"}[sort_label]
    df = get_predictions(limit=limit, asset=None if asset == "All" else asset,
                         timeframe=None if timeframe == "All" else timeframe,
                         pattern=None if pattern == "All" else pattern,
                         outcome=result_value, sort_by=sort_value)
    if df.empty:
        st.info("No predictions match these filters.")
        return

    completed = df[df["outcome"].notna()]
    wins = int((completed["outcome"] == "SUCCESS").sum())
    losses = int((completed["outcome"] == "FAILED").sum())
    decided = wins + losses
    win_rate = wins / decided * 100 if decided else 0.0
    realized = float(completed["pnl_amount"].fillna(0).sum())
    open_pnl = float(df.loc[df["outcome"].isna(), "unrealized_pnl"].fillna(0).sum())
    metrics = st.columns(5)
    metrics[0].metric("Trades", len(df))
    metrics[1].metric("Open", int(df["outcome"].isna().sum()), f"unrealized ${open_pnl:+.2f}")
    metrics[2].metric("Realized P&L", f"${realized:+.2f}")
    metrics[3].metric("Win rate", f"{win_rate:.1f}%" if decided else "n/a")
    metrics[4].metric("Closed", len(completed), f"{wins} targets / {losses} stops")

    table = pd.DataFrame({
        "Signal time (UTC)": df["signal_time"],
        "Asset": df["asset"],
        "Pattern": df["pattern"],
        "Timeframe": df["timeframe"],
        "Direction": df["direction"].map({"bull": "LONG", "bear": "SHORT"}),
        "Entry price": df["entry_price"],
        "Stop": df["stop_price"],
        "Target": df["target_price"],
        "Position units": df["position_units"],
        "Risk $": df["risk_amount"],
        "Exit price": df["exit_price"],
        "Mark price": df["mark_price"],
        "Realized P&L $": df["pnl_amount"],
        "Open P&L $": df["unrealized_pnl"],
        "P&L % equity": df.apply(lambda row: row["pnl_equity_pct"] if pd.notna(row["outcome"]) else row["unrealized_pnl_pct"], axis=1),
        "P&L R": df.apply(lambda row: row["pnl_r"] if pd.notna(row["outcome"]) else row["unrealized_pnl"] / row["risk_amount"] if row["risk_amount"] else None, axis=1),
        "Status": df.apply(lambda row: row["exit_reason"] or row["outcome"] if pd.notna(row["outcome"]) else "OPEN", axis=1),
        "Last checked candle (UTC)": df["mark_time"],
        "Signal explanation": df["description"],
    })
    # Render nullable numeric columns explicitly. Styler's format callbacks
    # can raise on pandas nullable values (for example pd.NA in open trades).
    display_table = table.copy()

    def _safe_display(value, formatter):
        if value is None or pd.isna(value):
            return "—"
        try:
            return formatter(float(value))
        except (TypeError, ValueError, OverflowError):
            return "—"

    for column in ("Entry price", "Stop", "Target", "Exit price", "Mark price"):
        display_table[column] = display_table[column].map(lambda value: _safe_display(value, fmt_price))
    for column, formatter in {
        "Position units": lambda value: f"{value:.6f}",
        "Risk $": lambda value: f"${value:,.2f}",
        "Realized P&L $": lambda value: f"${value:+,.2f}",
        "Open P&L $": lambda value: f"${value:+,.2f}",
        "P&L % equity": lambda value: f"{value:+.2f}%",
        "P&L R": lambda value: f"{value:+.2f}R",
    }.items():
        display_table[column] = display_table[column].map(lambda value, fn=formatter: _safe_display(value, fn))
    st.dataframe(display_table, hide_index=True, use_container_width=True)
    st.caption("P&L is simulated and gross before fees, funding, spread, slippage and execution constraints. "
               "Each trade risks 2% of equity at entry; the target is 3 times the entry-to-stop distance. "
               "This ledger does not place real orders.")


@st.cache_resource(show_spinner=False)
def ensure_database_initialized():
    init_db()
    return True


def main():
    st.set_page_config(page_title="Trading Intelligence Monitor", page_icon="📈", layout="wide")
    global RT
    RT = get_runtime()
    ensure_database_initialized()

    render_sidebar()
    st.title("📈 Trading Intelligence Monitor")
    st.caption("Tracks stocks and crypto, detects patterns, and records simulated paper trades. "
               "It does not send real orders or manage money. " + DISCLAIMER)

    page = st.radio(
        "Section",
        ["📊 Dashboard", "📈 Chart Analysis", "🔔 Alerts & Patterns", "📜 Paper trade ledger"],
        horizontal=True,
        label_visibility="collapsed",
        key="main_page",
    )
    if page == "📊 Dashboard":
        dashboard_fragment()
    elif page == "📈 Chart Analysis":
        chart_analysis()
    elif page == "🔔 Alerts & Patterns":
        alerts_fragment()
    else:
        predictions_fragment()


if __name__ == "__main__":
    main()
