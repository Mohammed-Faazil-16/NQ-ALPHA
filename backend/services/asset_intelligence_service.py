from __future__ import annotations

from backend.services.asset_ingestion_service import search_assets
from backend.services.fast_inference_service import infer_symbol
from backend.services.live_data_service import fetch_asset_data
from backend.services.symbol_resolver import extract_asset_candidates, resolve_symbol


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
    "price",
    "quote",
    "take on",
    "outlook",
}


def _normalize_query(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _looks_like_asset_query(query: str) -> bool:
    normalized = _normalize_query(query)
    if not normalized:
        return False
    return any(hint in normalized for hint in ASSET_QUERY_HINTS)


def _enrich_with_price(details: dict[str, object], symbol: str) -> dict[str, object]:
    try:
        data = fetch_asset_data(symbol, limit_days=30)
    except Exception:
        return details

    if data.empty:
        return details

    latest_row = data.iloc[-1]
    details["latest_price"] = float(latest_row["close"])
    details["latest_timestamp"] = str(latest_row["timestamp"].strftime("%Y-%m-%d"))
    return details


def extract_asset_intelligence(query: str) -> dict[str, object] | None:
    normalized = _normalize_query(query)
    if not _looks_like_asset_query(normalized):
        return None

    for candidate_query in extract_asset_candidates(normalized):
        candidates = search_assets(candidate_query, limit=3)
        if not candidates:
            continue

        chosen = candidates[0]
        symbol = resolve_symbol(chosen.get("symbol") or candidate_query)
        details = infer_symbol(symbol)
        details = _enrich_with_price(details, symbol)
        details["matched_query"] = query
        details["matched_asset_query"] = candidate_query
        details["asset_name"] = chosen.get("name") or symbol
        details["asset_type"] = chosen.get("asset_type") or "stock"
        details["source"] = chosen.get("source") or "search"
        return details

    return None
