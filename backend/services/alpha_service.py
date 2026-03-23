from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.alpha_agent.dataset_builder import (
    REGIME_TO_ID,
    SEQUENCE_LENGTH,
    _apply_temporal_feature_smoothing,
    get_active_features,
)
from agents.alpha_agent.neuroquant_model import NeuroQuantAlphaModel
from agents.feature_engineering_agent.generate_features import compute_features


MODEL_PATH = PROJECT_ROOT / "models" / "neuroquant_alpha_model.pt"
FEATURE_CLIP_VALUE = 5.0
ROLLING_NORMALIZATION_WINDOW = 60


def get_latest_alpha_signal() -> str:
    try:
        if not MODEL_PATH.exists():
            return "No alpha model available"
        return "Live alpha model ready"
    except Exception as exc:
        return f"Alpha unavailable: {exc}"


@lru_cache(maxsize=1)
def _load_alpha_model() -> NeuroQuantAlphaModel:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Alpha model not found at {MODEL_PATH}")

    active_features = get_active_features()
    model = NeuroQuantAlphaModel(feature_dim=len(active_features))
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _rolling_percentile(series: pd.Series, window: int = ROLLING_NORMALIZATION_WINDOW) -> pd.Series:
    min_periods = max(10, window // 4)
    return series.rolling(window, min_periods=min_periods).apply(
        lambda values: pd.Series(values).rank(pct=True).iloc[-1],
        raw=False,
    )


def _classify_regime(momentum_20: float, volatility_20: float, rolling_vol_median: float) -> str:
    if not np.isfinite(momentum_20) or not np.isfinite(volatility_20):
        return "normal"

    median = rolling_vol_median if np.isfinite(rolling_vol_median) and rolling_vol_median > 0 else volatility_20
    if momentum_20 < -0.08 or volatility_20 > median * 2.0:
        return "crisis"
    if volatility_20 > median * 1.35:
        return "volatile"
    if momentum_20 > 0.03:
        return "bull"
    return "normal"


def _normalize_live_feature(series: pd.Series) -> pd.Series:
    mean = series.rolling(ROLLING_NORMALIZATION_WINDOW, min_periods=20).mean()
    std = series.rolling(ROLLING_NORMALIZATION_WINDOW, min_periods=20).std()
    normalized = (series - mean) / (std + 1e-6)
    return normalized.clip(-FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE)


def generate_features_for_asset(dataframe: pd.DataFrame) -> dict[str, object]:
    if dataframe.empty:
        raise ValueError("No market data available for feature generation")

    feature_df = compute_features(dataframe.copy())
    feature_df = feature_df.sort_values("timestamp").reset_index(drop=True)

    rolling_market_return = feature_df["log_return_1"].rolling(20, min_periods=5).mean()
    feature_df["relative_return_1"] = feature_df["log_return_1"] - rolling_market_return
    feature_df["rank_return_1"] = _rolling_percentile(feature_df["log_return_1"])
    feature_df["vol_adjusted_return_1"] = feature_df["log_return_1"] / (feature_df["volatility_20"] + 1e-6)

    rolling_mean_5 = feature_df["log_return_1"].rolling(5).mean()
    rolling_std_5 = feature_df["log_return_1"].rolling(5).std()
    feature_df["return_zscore_5"] = (feature_df["log_return_1"] - rolling_mean_5) / (rolling_std_5 + 1e-6)
    feature_df["rolling_rank_mean_10"] = feature_df["rank_return_1"].rolling(10).mean()
    feature_df["momentum_x_volatility"] = feature_df["momentum_20"] * feature_df["volatility_20"]

    rolling_vol_median = feature_df["volatility_20"].rolling(60, min_periods=20).median()
    feature_df["regime_label"] = [
        _classify_regime(momentum, volatility, median)
        for momentum, volatility, median in zip(
            feature_df["momentum_20"],
            feature_df["volatility_20"],
            rolling_vol_median,
        )
    ]
    feature_df["regime_id"] = feature_df["regime_label"].map(REGIME_TO_ID).fillna(REGIME_TO_ID["normal"]).astype(int)
    feature_df["momentum_x_regime"] = feature_df["momentum_20"] * feature_df["regime_id"]

    active_features = get_active_features()
    available_features = [feature for feature in active_features if feature in feature_df.columns]
    feature_df = _apply_temporal_feature_smoothing(feature_df, available_features)

    raw_feature_df = feature_df.copy()
    for feature_name in active_features:
        if feature_name not in feature_df.columns:
            raise RuntimeError(f"Required active feature '{feature_name}' is missing from live inference frame")
        feature_df[feature_name] = _normalize_live_feature(pd.to_numeric(feature_df[feature_name], errors="coerce"))

    normalized = feature_df[active_features].replace([np.inf, -np.inf], np.nan)
    valid = normalized.dropna().reset_index(drop=True)
    if len(valid) < SEQUENCE_LENGTH:
        raise RuntimeError("Not enough valid feature rows to run model inference")

    sequence = valid.tail(SEQUENCE_LENGTH).to_numpy(dtype=np.float32, copy=False)
    latest_index = normalized.dropna().index[-1]
    latest_raw = raw_feature_df.loc[latest_index]

    volatility = float(latest_raw.get("volatility_20", 0.0) or 0.0)
    regime = str(latest_raw.get("regime_label", "normal"))
    regime_id = int(latest_raw.get("regime_id", REGIME_TO_ID["normal"]))

    return {
        "features": sequence,
        "volatility": volatility,
        "regime": regime,
        "regime_id": regime_id,
    }


def predict_alpha(features: np.ndarray, regime_id: int) -> float:
    model = _load_alpha_model()
    feature_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0)
    regime_tensor = torch.tensor([int(regime_id)], dtype=torch.long)

    with torch.no_grad():
        raw_alpha = float(model(feature_tensor, regime_tensor).item())
    return float(np.tanh(raw_alpha))


def generate_recommendation(alpha: float, volatility: float, regime: str) -> tuple[str, float]:
    regime_penalty = 0.0
    threshold = 0.01
    if regime == "volatile":
        threshold = 0.015
        regime_penalty = 0.04
    elif regime == "crisis":
        threshold = 0.02
        regime_penalty = 0.08

    if volatility > 0.05:
        threshold *= 1.2
        regime_penalty += 0.03

    if alpha > threshold:
        recommendation = "BUY"
    elif alpha < -threshold:
        recommendation = "AVOID"
    else:
        recommendation = "HOLD"

    strength = abs(alpha) / max(threshold, 1e-6)
    confidence = 0.55 + min(strength, 1.5) * 0.18 - min(volatility / 0.08, 1.0) * 0.05 - regime_penalty
    confidence = float(np.clip(confidence, 0.5, 0.95))
    return recommendation, round(confidence, 2)
