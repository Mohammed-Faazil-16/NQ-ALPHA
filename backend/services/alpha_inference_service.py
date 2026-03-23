from __future__ import annotations

from backend.services.fast_inference_service import alpha_series, infer_symbol
from backend.services.model_registry import warm_model_registry


def warm_alpha_inference() -> None:
    warm_model_registry()


def predict_alpha(symbol: str) -> float:
    return float(infer_symbol(symbol)["alpha"])


def predict_alpha_details(symbol: str) -> dict[str, object]:
    return infer_symbol(symbol)


def predict_alpha_series(symbol: str, lookback: int = 30) -> list[dict[str, object]]:
    return alpha_series(symbol, lookback=lookback)
