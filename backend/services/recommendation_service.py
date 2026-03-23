from __future__ import annotations

from backend.services.fast_inference_service import infer_symbol


def recommend_asset(symbol: str) -> dict[str, float | str]:
    normalized_symbol = (symbol or "").strip().upper()
    try:
        result = infer_symbol(normalized_symbol)
        return {
            "symbol": str(result["symbol"]),
            "alpha": float(result["alpha"]),
            "recommendation": str(result["recommendation"]),
            "confidence": float(result["confidence"]),
        }
    except Exception as exc:
        return {
            "symbol": normalized_symbol,
            "alpha": 0.0,
            "recommendation": "HOLD",
            "confidence": 0.05,
            "detail": f"Live market data is unavailable for {normalized_symbol}. {exc}",
        }
