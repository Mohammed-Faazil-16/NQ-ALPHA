from __future__ import annotations

from typing import Any

from backend.services.asset_ingestion_service import search_assets
from backend.services.fast_inference_service import infer_symbol
from backend.services.forecast_service import extract_forecast_horizon_years
from backend.services.live_data_service import fetch_asset_data
from backend.services.news_service import get_asset_news
from backend.services.scanner_service import scan_assets


BROAD_OPPORTUNITY_QUERY_HINTS = {
    "what stock",
    "which stock",
    "best stock",
    "best stocks",
    "best opportunity",
    "best opportunities",
    "opportunity right now",
    "opportunities right now",
    "suggest a stock",
    "suggest me a stock",
    "recommend a stock",
    "recommend some stocks",
    "confident about",
    "provide me with return",
    "provide me returns",
    "good return",
    "good returns",
    "1 or 2 years",
    "1-2 years",
    "next 1 year",
    "next 2 years",
}


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").strip().lower().split())


def is_broad_opportunity_query(query: str) -> bool:
    normalized = _normalize_query(query)
    if not normalized:
        return False
    return any(hint in normalized for hint in BROAD_OPPORTUNITY_QUERY_HINTS)


def _infer_asset_type(query: str, profile: dict[str, object]) -> str | None:
    normalized = _normalize_query(query)
    if any(word in normalized for word in ("crypto", "bitcoin", "ethereum")):
        return "crypto"
    if any(word in normalized for word in ("commodity", "commodities", "gold", "silver", "oil")):
        return "commodity"
    if any(word in normalized for word in ("stock", "stocks", "share", "equity", "equities")):
        return "stock"

    interests = [str(item).strip().lower() for item in (profile.get("interests") or []) if str(item).strip()]
    if len(interests) == 1:
        interest = interests[0]
        if interest.startswith("stock"):
            return "stock"
        if interest.startswith("crypto"):
            return "crypto"
        if interest.startswith("commod"):
            return "commodity"
    return None


def _lookup_asset_name(symbol: str) -> str:
    matches = search_assets(symbol, limit=1)
    if matches:
        return str(matches[0].get("name") or symbol)
    return symbol


def _latest_price_snapshot(symbol: str) -> dict[str, object]:
    try:
        data = fetch_asset_data(symbol, limit_days=45)
    except Exception:
        return {"price": None, "timestamp": None}

    if data.empty:
        return {"price": None, "timestamp": None}

    latest_row = data.iloc[-1]
    return {
        "price": float(latest_row["close"]),
        "timestamp": latest_row["timestamp"].strftime("%Y-%m-%d"),
    }


def _score_candidate(
    item: dict[str, object],
    profile: dict[str, object],
    horizon_years: float,
) -> float:
    alpha = float(item.get("alpha") or 0.0)
    confidence = float(item.get("confidence") or 0.0)
    recommendation = str(item.get("recommendation") or "HOLD").upper()
    regime = str(item.get("regime") or "normal").lower()
    risk_level = str(profile.get("risk_level") or "balanced").lower()

    score = alpha * 100.0 + confidence * 22.0
    if recommendation == "BUY":
        score += 8.0
    elif recommendation == "HOLD":
        score += 2.0
    else:
        score -= 10.0

    if regime == "bull":
        score += 4.0
    elif regime == "normal":
        score += 2.0
    elif regime == "volatile":
        score -= 4.0
    elif regime == "crisis":
        score -= 8.0

    if risk_level == "conservative":
        score -= max(float(item.get("volatility_penalty") or 0.0), 0.0) * 6.0
        if regime in {"volatile", "crisis"}:
            score -= 6.0
    elif risk_level == "aggressive":
        score += max(alpha, 0.0) * 18.0

    if horizon_years <= 2.0:
        score += confidence * 6.0
        if recommendation == "BUY":
            score += 2.0
    else:
        if regime in {"bull", "normal"}:
            score += 2.0

    return score


def build_opportunity_snapshot(user_query: str, profile: dict[str, object], max_picks: int = 3) -> dict[str, Any]:
    asset_type = _infer_asset_type(user_query, profile)
    horizon_years = extract_forecast_horizon_years(user_query)
    scan_payload = scan_assets(top_n=12, asset_type=asset_type)
    raw_results = list(scan_payload.get("results") or [])

    shortlist: list[dict[str, Any]] = []
    for result in raw_results[:6]:
        symbol = str(result.get("symbol") or "").upper()
        if not symbol:
            continue
        details = infer_symbol(symbol, refresh_if_stale=False)
        price_snapshot = _latest_price_snapshot(symbol)
        shortlist.append(
            {
                "symbol": symbol,
                "asset_name": _lookup_asset_name(symbol),
                "alpha": float(details.get("alpha") or 0.0),
                "recommendation": str(details.get("recommendation") or "HOLD").upper(),
                "confidence": float(details.get("confidence") or 0.0),
                "regime": str(details.get("regime") or "normal"),
                "latest_price": price_snapshot.get("price"),
                "latest_timestamp": price_snapshot.get("timestamp"),
            }
        )

    for item in shortlist:
        item["score"] = _score_candidate(item, profile, horizon_years)

    shortlist.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    selected = shortlist[: max(1, int(max_picks))]

    for item in selected[:2]:
        news_payload = get_asset_news(item["symbol"], asset_name=item["asset_name"], asset_type=asset_type or "stock", limit=2)
        item["news"] = news_payload.get("articles") or []

    return {
        "asset_type": asset_type or "all",
        "horizon_years": horizon_years,
        "candidates": selected,
        "partial": bool(scan_payload.get("partial")),
        "evaluated": int(scan_payload.get("evaluated") or len(raw_results)),
        "elapsed_seconds": float(scan_payload.get("elapsed_seconds") or 0.0),
    }
