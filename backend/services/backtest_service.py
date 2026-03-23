from __future__ import annotations

from collections import OrderedDict
import math
import re

import numpy as np
import pandas as pd

from backend.services.live_data_service import fetch_asset_data
from backend.services.runtime_cache import runtime_cache
from backend.services.symbol_resolver import resolve_symbol


DEFAULT_LOOKBACK_DAYS = 180
DEFAULT_ASSETS = [
    {"symbol": "AAPL", "weight": 0.4},
    {"symbol": "MSFT", "weight": 0.35},
    {"symbol": "NVDA", "weight": 0.25},
]
BACKTEST_CACHE_TTL_SECONDS = 300
TRADING_DAYS_PER_YEAR = 252.0


def _normalize_assets(assets: list[dict[str, object]] | None, capital: float = 0.0) -> tuple[list[dict[str, object]], float]:
    raw_assets = assets or DEFAULT_ASSETS
    deduped = OrderedDict()
    for item in raw_assets:
        symbol = resolve_symbol(str(item.get("symbol", "")).strip())
        amount = float(item.get("amount", 0.0) or 0.0)
        weight = float(item.get("weight", 0.0) or 0.0)
        if capital > 0.0 and amount > 0.0:
            weight = amount / capital
        if not math.isfinite(weight):
            weight = 0.0
        deduped[symbol] = deduped.get(symbol, 0.0) + weight

    cleaned = [{"symbol": symbol, "weight": weight} for symbol, weight in deduped.items() if weight > 0.0]
    if not cleaned:
        cleaned = [{"symbol": resolve_symbol(item["symbol"]), "weight": float(item["weight"])} for item in DEFAULT_ASSETS]

    total_weight = sum(item["weight"] for item in cleaned)
    if total_weight <= 0.0:
        equal_weight = 1.0 / len(cleaned)
        return ([{"symbol": item["symbol"], "weight": equal_weight} for item in cleaned], 1.0)

    if total_weight <= 1.0 + 1e-6:
        return ([{"symbol": item["symbol"], "weight": item["weight"]} for item in cleaned], total_weight)

    normalized = [{"symbol": item["symbol"], "weight": item["weight"] / total_weight} for item in cleaned]
    return normalized, 1.0


def _portfolio_stats(returns: pd.Series) -> dict[str, float]:
    returns = returns.astype(float)
    wealth = (1.0 + returns).cumprod()
    peaks = wealth.cummax()
    drawdown = 1.0 - wealth / (peaks + 1e-8)
    observed_years = max(len(returns) / TRADING_DAYS_PER_YEAR, 1.0 / TRADING_DAYS_PER_YEAR)
    final_wealth = float(wealth.iloc[-1]) if not wealth.empty else 1.0
    annualized_return = float(final_wealth ** (1.0 / observed_years) - 1.0) if final_wealth > 0.0 else -1.0

    return {
        "final_return": float(final_wealth - 1.0) if not wealth.empty else 0.0,
        "annualized_return": annualized_return,
        "sharpe": float((returns.mean() / (returns.std() + 1e-6)) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(returns) > 1 else 0.0,
        "volatility": float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(returns) > 1 else 0.0,
        "drawdown": float(drawdown.max()) if not drawdown.empty else 0.0,
    }


def _parse_horizon_range(investment_horizon: str | None) -> tuple[str, float, float]:
    label = str(investment_horizon or "3-5 years").strip() or "3-5 years"
    lowered = label.lower()
    if "<" in lowered or "month" in lowered:
        return label, 0.25, 0.5
    if "1-2" in lowered:
        return label, 1.0, 2.0
    if "3-5" in lowered:
        return label, 3.0, 5.0
    if "5+" in lowered:
        return label, 5.0, 8.0

    match = re.search(r"(\d+(?:\.\d+)?)", lowered)
    if match:
        years = max(float(match.group(1)), 0.25)
        return label, years, years
    return label, 3.0, 5.0


def _project_future_value(capital: float, annualized_return: float, investment_horizon: str | None) -> dict[str, float | str]:
    safe_capital = max(float(capital or 0.0), 0.0)
    label, low_years, high_years = _parse_horizon_range(investment_horizon)
    capped_return = max(min(float(annualized_return), 1.5), -0.95)

    value_low = safe_capital * ((1.0 + capped_return) ** low_years)
    value_high = safe_capital * ((1.0 + capped_return) ** high_years)

    return {
        "horizon_label": label,
        "years_low": low_years,
        "years_high": high_years,
        "annualized_return_pct": capped_return * 100.0,
        "projected_value_low": value_low,
        "projected_value_high": value_high,
        "projected_profit_low": value_low - safe_capital,
        "projected_profit_high": value_high - safe_capital,
    }


def run_backtest(
    assets: list[dict[str, object]] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    capital: float = 0.0,
    investment_horizon: str | None = None,
) -> dict[str, object]:
    normalized_assets, invested_fraction = _normalize_assets(assets, capital=capital)
    cache_key = (
        "backtest",
        int(lookback_days),
        round(float(capital or 0.0), 2),
        str(investment_horizon or ""),
        tuple((item["symbol"], round(float(item["weight"]), 6)) for item in normalized_assets),
    )
    cached = runtime_cache.get(cache_key)
    if cached is not None:
        return cached
    price_frames = []
    fetch_window = max(int(lookback_days), 30) + 30

    for item in normalized_assets:
        market_df = fetch_asset_data(item["symbol"], limit_days=fetch_window)
        close_frame = market_df[["timestamp", "close"]].copy()
        close_frame["timestamp"] = pd.to_datetime(close_frame["timestamp"])
        close_frame = close_frame.rename(columns={"close": item["symbol"]})
        price_frames.append(close_frame)

    merged = price_frames[0]
    for frame in price_frames[1:]:
        merged = merged.merge(frame, on="timestamp", how="inner")

    merged = merged.sort_values("timestamp").dropna().reset_index(drop=True)
    merged = merged.tail(max(int(lookback_days), 30) + 1)
    if len(merged) < 2:
        raise ValueError("Not enough aligned history to run a backtest")

    symbols = [item["symbol"] for item in normalized_assets]
    weights = np.asarray([item["weight"] for item in normalized_assets], dtype=np.float64)

    returns = merged[symbols].pct_change().dropna().reset_index(drop=True)
    timestamps = pd.to_datetime(merged["timestamp"]).iloc[1:].reset_index(drop=True)
    strategy_returns = pd.Series(returns.to_numpy(dtype=np.float64) @ weights, index=timestamps)
    baseline_returns = returns.mean(axis=1) * invested_fraction
    baseline_returns.index = timestamps

    strategy_wealth = (1.0 + strategy_returns).cumprod()
    baseline_wealth = (1.0 + baseline_returns).cumprod()
    metrics = _portfolio_stats(strategy_returns)
    baseline_metrics = _portfolio_stats(baseline_returns)
    projection = _project_future_value(capital, metrics["annualized_return"], investment_horizon)

    payload = {
        "assets": normalized_assets,
        "invested_fraction": invested_fraction,
        "cash_fraction": max(1.0 - invested_fraction, 0.0),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "projection": projection,
        "equity_curve": [
            {"timestamp": timestamp.strftime("%Y-%m-%d"), "value": float(value)}
            for timestamp, value in strategy_wealth.items()
        ],
        "baseline_curve": [
            {"timestamp": timestamp.strftime("%Y-%m-%d"), "value": float(value)}
            for timestamp, value in baseline_wealth.items()
        ],
    }
    runtime_cache.set(cache_key, payload, ttl_seconds=BACKTEST_CACHE_TTL_SECONDS)
    return payload
