from __future__ import annotations

import math
import re

import pandas as pd

from backend.services.live_data_service import fetch_asset_data


TRADING_DAYS_PER_YEAR = 252.0
FORECAST_DEFAULT_YEARS = 3.0
FORECAST_MIN_YEARS = 0.5
FORECAST_MAX_YEARS = 10.0
YEAR_WORDS = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
}
REGIME_DRIFT_ADJUSTMENT = {
    "bull": 0.03,
    "normal": 0.0,
    "volatile": -0.03,
    "crisis": -0.06,
}


def _clamp_years(value: float) -> float:
    return max(FORECAST_MIN_YEARS, min(float(value), FORECAST_MAX_YEARS))


def extract_forecast_horizon_years(query: str) -> float:
    normalized = str(query or "").strip().lower()

    year_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:year|yr)", normalized)
    if year_match:
        return _clamp_years(float(year_match.group(1)))

    month_match = re.search(r"(\d+(?:\.\d+)?)\s*month", normalized)
    if month_match:
        months = float(month_match.group(1))
        return _clamp_years(months / 12.0)

    for word, value in YEAR_WORDS.items():
        if re.search(rf"\b{word}\s+year", normalized):
            return _clamp_years(value)

    if "long term" in normalized:
        return 5.0
    if "short term" in normalized:
        return 1.0
    return FORECAST_DEFAULT_YEARS


def build_asset_forecast(
    symbol: str,
    years: float,
    alpha: float = 0.0,
    regime: str = "normal",
    confidence: float = 0.0,
) -> dict[str, float | str]:
    horizon_years = _clamp_years(float(years or FORECAST_DEFAULT_YEARS))
    data = fetch_asset_data(symbol, limit_days=540)
    if data.empty:
        raise ValueError(f"No market history available for {symbol}")

    close = pd.to_numeric(data["close"], errors="coerce").dropna()
    if close.size < 40:
        raise ValueError(f"Not enough price history to forecast {symbol}")

    latest_price = float(close.iloc[-1])
    latest_timestamp = pd.Timestamp(data.iloc[-1]["timestamp"]).strftime("%Y-%m-%d")

    daily_returns = close.pct_change().dropna()
    trailing_window = close.tail(min(len(close), 253))
    trailing_years = max((len(trailing_window) - 1) / TRADING_DAYS_PER_YEAR, 0.25)
    trailing_cagr = float((trailing_window.iloc[-1] / trailing_window.iloc[0]) ** (1.0 / trailing_years) - 1.0)
    annualized_volatility = float(daily_returns.tail(min(len(daily_returns), 252)).std() * math.sqrt(TRADING_DAYS_PER_YEAR))

    regime_adjustment = REGIME_DRIFT_ADJUSTMENT.get(str(regime or "normal").lower(), 0.0)
    alpha_adjustment = max(min(float(alpha) * 4.0, 0.12), -0.12)
    confidence_scale = 0.75 + min(max(float(confidence), 0.0), 1.0) * 0.5
    adjusted_annual_return = max(min((trailing_cagr + regime_adjustment + alpha_adjustment) * confidence_scale, 0.45), -0.35)

    downside_return = max(adjusted_annual_return - annualized_volatility * 0.55, -0.55)
    upside_return = min(adjusted_annual_return + annualized_volatility * 0.45, 0.65)

    base_price = latest_price * ((1.0 + adjusted_annual_return) ** horizon_years)
    downside_price = latest_price * ((1.0 + downside_return) ** horizon_years)
    upside_price = latest_price * ((1.0 + upside_return) ** horizon_years)

    return {
        "symbol": symbol,
        "years": horizon_years,
        "current_price": latest_price,
        "current_timestamp": latest_timestamp,
        "trailing_cagr": trailing_cagr,
        "annualized_volatility": annualized_volatility,
        "adjusted_annual_return": adjusted_annual_return,
        "base_price": base_price,
        "downside_price": downside_price,
        "upside_price": upside_price,
        "confidence": float(confidence),
        "alpha": float(alpha),
        "regime": str(regime),
    }
