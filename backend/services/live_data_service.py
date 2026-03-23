from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import ccxt
import pandas as pd
import yfinance as yf

from backend.services.runtime_cache import runtime_cache


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YFINANCE_CACHE_DIR = PROJECT_ROOT / "data" / "yfinance_cache"
CRYPTO_BASES = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA"}
DEFAULT_LIMIT = 400
PRICE_CACHE_TTL_SECONDS = 300
STOCK_FETCH_TIMEOUT_SECONDS = 5


YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))
logging.getLogger("yfinance").setLevel(logging.ERROR)


def _normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper().lstrip("$")


def _is_crypto_symbol(symbol: str) -> bool:
    upper = _normalize_symbol(symbol)
    if "/" in upper:
        return True
    if upper.endswith("-USD") and upper.split("-", 1)[0] in CRYPTO_BASES:
        return True
    return upper in CRYPTO_BASES


def _format_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(
        columns={
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    formatted = renamed.loc[:, columns].copy()
    formatted["timestamp"] = pd.to_datetime(formatted["timestamp"]).dt.tz_localize(None)
    formatted = formatted.sort_values("timestamp").dropna(subset=["close", "volume"]).reset_index(drop=True)
    return formatted


def _try_yfinance_history(symbol: str, limit_days: int) -> pd.DataFrame | None:
    history = yf.Ticker(symbol).history(
        period=f"{max(int(limit_days), DEFAULT_LIMIT)}d",
        interval="1d",
        auto_adjust=False,
        timeout=STOCK_FETCH_TIMEOUT_SECONDS,
    )
    if history is None or history.empty:
        return None
    return _format_ohlcv_frame(history.reset_index())


def _stock_search_candidates(symbol: str) -> list[str]:
    normalized = _normalize_symbol(symbol)
    queries = [normalized]
    if normalized and "." not in normalized and "=" not in normalized:
        queries.extend([f"{normalized} NSE", f"{normalized} India", f"{normalized} stock"])

    candidates: list[str] = []
    seen: set[str] = set()
    for query in queries:
        try:
            search = yf.Search(query, max_results=8)
            quotes = getattr(search, "quotes", None) or []
        except Exception:
            continue

        for quote in quotes:
            candidate = str(quote.get("symbol") or "").upper().strip().lstrip("$")
            exchange = str(quote.get("exchange") or quote.get("fullExchangeName") or "").lower()
            if not candidate:
                continue
            if exchange.startswith("nse") and not candidate.endswith(".NS"):
                candidate = f"{candidate}.NS"
            elif exchange.startswith("bse") and not candidate.endswith(".BO"):
                candidate = f"{candidate}.BO"
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def _fetch_stock_data(symbol: str, limit_days: int) -> pd.DataFrame:
    normalized = _normalize_symbol(symbol)
    candidates = [normalized]
    if normalized and "." not in normalized and not normalized.endswith(".NS"):
        candidates.extend([f"{normalized}.NS", f"{normalized}.BO"])
    candidates.extend(_stock_search_candidates(normalized))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            frame = _try_yfinance_history(candidate, limit_days)
        except Exception:
            frame = None
        if frame is None or frame.empty:
            continue

        if candidate != normalized:
            try:
                from backend.services.symbol_resolver import _persist_all_asset, _infer_asset_type

                _persist_all_asset(candidate, name=normalized, asset_type=_infer_asset_type(candidate))
            except Exception:
                pass
        return frame

    raise RuntimeError(f"No stock data returned for {normalized}")


def _normalize_crypto_market_symbol(symbol: str) -> str:
    upper = _normalize_symbol(symbol)
    if "/" in upper:
        return upper
    if upper.endswith("-USD"):
        return f"{upper.split('-', 1)[0]}/USDT"
    return f"{upper}/USDT"


def _fetch_crypto_data(symbol: str, limit_days: int) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    try:
        market_symbol = _normalize_crypto_market_symbol(symbol)
        ohlcv = exchange.fetch_ohlcv(market_symbol, timeframe="1d", limit=max(int(limit_days), DEFAULT_LIMIT))
        if not ohlcv:
            raise RuntimeError(f"No crypto data returned for {market_symbol}")
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return _format_ohlcv_frame(df)
    finally:
        close_fn: Callable[[], object] | None = getattr(exchange, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


def fetch_asset_data(symbol: str, limit_days: int = DEFAULT_LIMIT) -> pd.DataFrame:
    normalized_symbol = _normalize_symbol(symbol)
    fetch_window = max(int(limit_days or DEFAULT_LIMIT), 30)
    cache_key = ("price", normalized_symbol, fetch_window)
    cached = runtime_cache.get(cache_key)
    if cached is not None:
        return cached.copy(deep=True)

    if _is_crypto_symbol(normalized_symbol):
        frame = _fetch_crypto_data(normalized_symbol, limit_days=fetch_window)
    else:
        frame = _fetch_stock_data(normalized_symbol, limit_days=fetch_window)

    runtime_cache.set(cache_key, frame.copy(deep=True), ttl_seconds=PRICE_CACHE_TTL_SECONDS)
    return frame
