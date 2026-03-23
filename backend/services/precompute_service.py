from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock, Thread
import time

import numpy as np
import pandas as pd

from backend.database.models.all_assets import AllAssets
from backend.db.models import FeaturesLatest
from backend.db.postgres import SessionLocal
from backend.services.feature_service import FEATURE_COLUMNS, generate_feature_history_for_asset
from backend.services.live_data_service import fetch_asset_data
from backend.services.model_registry import REGIME_LABEL_BY_ID, REGIME_TO_ID, SEQUENCE_LENGTH, get_active_features
from backend.services.runtime_cache import runtime_cache


PRECOMPUTE_HISTORY_LENGTH = 90
PRECOMPUTE_BATCH_LIMIT = 20
PRECOMPUTE_INTERVAL_SECONDS = 3600
PRECOMPUTE_CACHE_TTL_SECONDS = 300
_scheduler_started = False
_scheduler_lock = Lock()


def _rolling_percentile(series: pd.Series, window: int = 60) -> pd.Series:
    min_periods = max(10, window // 4)
    return series.rolling(window, min_periods=min_periods).apply(
        lambda values: pd.Series(values).rank(pct=True).iloc[-1],
        raw=False,
    )


def _coerce_utc_timestamp(value) -> datetime | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    dt = timestamp.to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _payload_timestamp(payload: dict[str, object] | None) -> datetime | None:
    if not isinstance(payload, dict):
        return None
    return _coerce_utc_timestamp(payload.get("timestamp"))


def _latest_market_timestamp(symbol: str) -> datetime | None:
    try:
        asset_df = fetch_asset_data(symbol, limit_days=45)
    except Exception:
        return None
    if asset_df is None or asset_df.empty or "timestamp" not in asset_df.columns:
        return None
    timestamps = pd.to_datetime(asset_df["timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return None
    return _coerce_utc_timestamp(timestamps.max())


def _payload_needs_refresh(payload: dict[str, object] | None, latest_market_timestamp: datetime | None) -> bool:
    if latest_market_timestamp is None:
        return False
    current_timestamp = _payload_timestamp(payload)
    if current_timestamp is None:
        return True
    return current_timestamp.date() < latest_market_timestamp.date()


def _infer_regime_id(momentum_20: float, volatility_20: float, rolling_vol_median: float) -> int:
    median = rolling_vol_median if np.isfinite(rolling_vol_median) and rolling_vol_median > 0 else volatility_20
    if momentum_20 < -0.08 or volatility_20 > median * 2.0:
        return REGIME_TO_ID["crisis"]
    if volatility_20 > median * 1.35:
        return REGIME_TO_ID["volatile"]
    if momentum_20 > 0.03:
        return REGIME_TO_ID["bull"]
    return REGIME_TO_ID["normal"]


def _build_inference_history(symbol: str) -> dict[str, object]:
    asset_df = fetch_asset_data(symbol)
    history_df = generate_feature_history_for_asset(asset_df)

    enriched = history_df.copy()
    market_return = enriched["log_return_1"].rolling(20, min_periods=5).mean()
    enriched["relative_return_1"] = enriched["log_return_1"] - market_return
    enriched["rank_return_1"] = _rolling_percentile(enriched["log_return_1"])
    enriched["vol_adjusted_return_1"] = enriched["log_return_1"] / (enriched["volatility_20"] + 1e-6)
    rolling_mean_5 = enriched["log_return_1"].rolling(5).mean()
    rolling_std_5 = enriched["log_return_1"].rolling(5).std()
    enriched["return_zscore_5"] = (enriched["log_return_1"] - rolling_mean_5) / (rolling_std_5 + 1e-6)
    enriched["rolling_rank_mean_10"] = enriched["rank_return_1"].rolling(10).mean()
    enriched["momentum_x_volatility"] = enriched["momentum_20"] * enriched["volatility_20"]

    rolling_vol_median = enriched["volatility_20"].rolling(60, min_periods=20).median()
    enriched["regime_id"] = [
        _infer_regime_id(float(momentum), float(volatility), float(median) if pd.notna(median) else float("nan"))
        for momentum, volatility, median in zip(
            enriched["momentum_20"],
            enriched["volatility_20"],
            rolling_vol_median,
        )
    ]
    enriched["regime"] = enriched["regime_id"].map(REGIME_LABEL_BY_ID)
    enriched["momentum_x_regime"] = enriched["momentum_20"] * enriched["regime_id"]

    active_features = list(get_active_features())
    valid = enriched.replace([np.inf, -np.inf], np.nan).dropna(subset=active_features).reset_index(drop=True)
    if len(valid) < SEQUENCE_LENGTH:
        raise ValueError(f"Not enough valid feature history for {symbol}")

    history_rows = []
    trimmed = valid.tail(PRECOMPUTE_HISTORY_LENGTH)
    for row in trimmed.itertuples(index=False):
        history_rows.append(
            {
                "timestamp": pd.Timestamp(row.timestamp).strftime("%Y-%m-%d"),
                "regime": str(row.regime),
                "regime_id": int(row.regime_id),
                "values": [float(getattr(row, feature)) for feature in active_features],
            }
        )

    latest = trimmed.iloc[-1]
    latest_features = {
        feature: float(latest[feature])
        for feature in FEATURE_COLUMNS
        if feature in trimmed.columns and pd.notna(latest[feature])
    }

    latest_timestamp = pd.Timestamp(latest["timestamp"])
    return {
        "symbol": symbol,
        "timestamp": latest_timestamp.to_pydatetime().replace(tzinfo=timezone.utc),
        "features_json": {
            "symbol": symbol,
            "timestamp": latest_timestamp.strftime("%Y-%m-%d"),
            "active_features": active_features,
            "regime": str(latest["regime"]),
            "regime_id": int(latest["regime_id"]),
            "latest_features": latest_features,
            "history": history_rows,
        },
    }


def precompute_symbol_features(symbol: str) -> dict[str, object]:
    normalized_symbol = (symbol or "").strip().upper()
    if not normalized_symbol:
        raise ValueError("Symbol must not be empty")

    payload = _build_inference_history(normalized_symbol)
    db = SessionLocal()
    try:
        record = db.query(FeaturesLatest).filter(FeaturesLatest.symbol == normalized_symbol).first()
        if record is None:
            record = FeaturesLatest(
                symbol=normalized_symbol,
                timestamp=payload["timestamp"],
                features_json=payload["features_json"],
            )
            db.add(record)
        else:
            record.timestamp = payload["timestamp"]
            record.features_json = payload["features_json"]

        db.commit()
        runtime_cache.set(("features-latest", normalized_symbol), payload["features_json"], ttl_seconds=PRECOMPUTE_CACHE_TTL_SECONDS)
        return payload["features_json"]
    finally:
        db.close()


def get_latest_features_payload(symbol: str, refresh_if_stale: bool = True) -> dict[str, object]:
    normalized_symbol = (symbol or "").strip().upper()
    cache_key = ("features-latest", normalized_symbol)
    latest_market_timestamp = None

    cached = runtime_cache.get(cache_key)
    if cached is not None:
        if not refresh_if_stale:
            return cached
        latest_market_timestamp = _latest_market_timestamp(normalized_symbol)
        if not _payload_needs_refresh(cached, latest_market_timestamp):
            return cached

    db = SessionLocal()
    try:
        record = db.query(FeaturesLatest).filter(FeaturesLatest.symbol == normalized_symbol).first()
        payload = record.features_json if record is not None and isinstance(record.features_json, dict) else None
    finally:
        db.close()

    if payload is None:
        payload = precompute_symbol_features(normalized_symbol)
    elif refresh_if_stale:
        latest_market_timestamp = latest_market_timestamp or _latest_market_timestamp(normalized_symbol)
        if _payload_needs_refresh(payload, latest_market_timestamp):
            try:
                payload = precompute_symbol_features(normalized_symbol)
            except Exception:
                pass

    runtime_cache.set(cache_key, payload, ttl_seconds=PRECOMPUTE_CACHE_TTL_SECONDS)
    return payload


def precompute_asset_batch(limit: int = PRECOMPUTE_BATCH_LIMIT, asset_type: str | None = None) -> int:
    normalized_limit = max(1, min(int(limit), PRECOMPUTE_BATCH_LIMIT))
    normalized_type = (asset_type or "").strip().lower()

    db = SessionLocal()
    try:
        query = db.query(AllAssets.symbol)
        if normalized_type:
            query = query.filter(AllAssets.asset_type.ilike(normalized_type))
        symbols = [symbol for (symbol,) in query.order_by(AllAssets.symbol.asc()).limit(normalized_limit).all()]
    finally:
        db.close()

    completed = 0
    for symbol in symbols:
        try:
            precompute_symbol_features(symbol)
            completed += 1
        except Exception:
            continue
    return completed


def _scheduler_loop() -> None:
    time.sleep(5)
    while True:
        try:
            precompute_asset_batch(limit=PRECOMPUTE_BATCH_LIMIT)
        except Exception:
            pass
        time.sleep(PRECOMPUTE_INTERVAL_SECONDS)


def start_precompute_scheduler() -> None:
    global _scheduler_started
    with _scheduler_lock:
        if _scheduler_started:
            return
        _scheduler_started = True
        Thread(target=_scheduler_loop, daemon=True, name="neuroquant-precompute-scheduler").start()
