from __future__ import annotations

import numpy as np
import torch

from backend.services.model_registry import REGIME_TO_ID, SEQUENCE_LENGTH, get_model
from backend.services.precompute_service import get_latest_features_payload
from backend.services.runtime_cache import runtime_cache


BUY_THRESHOLD = 0.02
AVOID_THRESHOLD = -0.02
INFERENCE_CACHE_TTL_SECONDS = 300
ALPHA_SERIES_CACHE_TTL_SECONDS = 300


def _predict_alpha_from_history(history: list[dict[str, object]], end_idx: int) -> float:
    sequence_rows = history[end_idx - SEQUENCE_LENGTH + 1 : end_idx + 1]
    feature_array = np.asarray([row["values"] for row in sequence_rows], dtype=np.float32)
    regime_id = int(sequence_rows[-1].get("regime_id", REGIME_TO_ID["normal"]))

    model = get_model()
    feature_tensor = torch.tensor(feature_array[np.newaxis, :, :], dtype=torch.float32)
    regime_tensor = torch.tensor([regime_id], dtype=torch.long)
    with torch.no_grad():
        return float(model(feature_tensor, regime_tensor).item())


def _recommendation_from_alpha(alpha: float) -> str:
    if alpha > BUY_THRESHOLD:
        return "BUY"
    if alpha < AVOID_THRESHOLD:
        return "AVOID"
    return "HOLD"


def _confidence_from_alpha(alpha: float) -> float:
    return float(min(abs(alpha) * 20.0, 1.0))


def infer_symbol(symbol: str, refresh_if_stale: bool = True) -> dict[str, object]:
    normalized_symbol = (symbol or "").strip().upper()
    payload = get_latest_features_payload(normalized_symbol, refresh_if_stale=refresh_if_stale)
    timestamp = str(payload.get("timestamp", ""))
    cache_key = ("fast-inference", normalized_symbol, timestamp)
    cached = runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    history = list(payload.get("history") or [])
    if len(history) < SEQUENCE_LENGTH:
        raise ValueError(f"Not enough precomputed history to score {normalized_symbol}")

    alpha = _predict_alpha_from_history(history, len(history) - 1)
    result = {
        "symbol": normalized_symbol,
        "timestamp": timestamp,
        "alpha": float(alpha),
        "recommendation": _recommendation_from_alpha(alpha),
        "confidence": _confidence_from_alpha(alpha),
        "regime": str(payload.get("regime", "normal")),
        "regime_id": int(payload.get("regime_id", REGIME_TO_ID["normal"])),
        "features": dict(payload.get("latest_features") or {}),
    }
    runtime_cache.set(cache_key, result, ttl_seconds=INFERENCE_CACHE_TTL_SECONDS)
    return result


def alpha_series(symbol: str, lookback: int = 30, refresh_if_stale: bool = True) -> list[dict[str, object]]:
    normalized_symbol = (symbol or "").strip().upper()
    payload = get_latest_features_payload(normalized_symbol, refresh_if_stale=refresh_if_stale)
    timestamp = str(payload.get("timestamp", ""))
    cache_key = ("fast-alpha-series", normalized_symbol, int(lookback), timestamp)
    cached = runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    history = list(payload.get("history") or [])
    if len(history) < SEQUENCE_LENGTH:
        raise ValueError(f"Not enough precomputed history to build alpha series for {normalized_symbol}")

    start_idx = max(SEQUENCE_LENGTH - 1, len(history) - max(int(lookback), 1))
    series = []
    for end_idx in range(start_idx, len(history)):
        alpha = _predict_alpha_from_history(history, end_idx)
        row = history[end_idx]
        series.append(
            {
                "timestamp": str(row.get("timestamp", "")),
                "alpha": float(alpha),
                "regime": str(row.get("regime", "normal")),
            }
        )

    runtime_cache.set(cache_key, series, ttl_seconds=ALPHA_SERIES_CACHE_TTL_SECONDS)
    return series
