from __future__ import annotations

from backend.services.asset_ingestion_service import search_assets
from backend.services.fast_inference_service import infer_symbol
from backend.services.symbol_resolver import resolve_symbol


ASSET_QUERY_HINTS = {
    "buy",
    "avoid",
    "hold",
    "invest",
    "stock",
    "share",
    "crypto",
    "commodity",
    "analyze",
    "analysis",
    "opportunity",
    "further strategy",
    "should i",
}


def _normalize_query(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _looks_like_asset_query(query: str) -> bool:
    normalized = _normalize_query(query)
    if not normalized:
        return False
    return any(hint in normalized for hint in ASSET_QUERY_HINTS)


def extract_asset_intelligence(query: str) -> dict[str, object] | None:
    normalized = _normalize_query(query)
    if not _looks_like_asset_query(normalized):
        return None

    candidates = search_assets(normalized, limit=3)
    if not candidates:
        return None

    chosen = candidates[0]
    symbol = resolve_symbol(chosen.get("symbol") or normalized)
    details = infer_symbol(symbol)
    details["matched_query"] = query
    details["asset_name"] = chosen.get("name") or symbol
    details["asset_type"] = chosen.get("asset_type") or "stock"
    details["source"] = chosen.get("source") or "search"
    return details
