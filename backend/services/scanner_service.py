from __future__ import annotations

from time import monotonic

from sqlalchemy import and_

from backend.database.models.all_assets import AllAssets
from backend.db.models import FeaturesLatest
from backend.db.postgres import SessionLocal
from backend.services.fast_inference_service import infer_symbol
from backend.services.precompute_service import precompute_asset_batch
from backend.services.runtime_cache import runtime_cache


DEFAULT_SCAN_CANDIDATES = 20
MAX_SCAN_CANDIDATES = 20
SCAN_CACHE_TTL_SECONDS = 300
SCAN_TIMEOUT_SECONDS = 5.0


def _load_precomputed_assets(top_n: int, asset_type: str | None) -> list[tuple[AllAssets, FeaturesLatest]]:
    db = SessionLocal()
    try:
        query = (
            db.query(AllAssets, FeaturesLatest)
            .join(FeaturesLatest, and_(FeaturesLatest.symbol == AllAssets.symbol))
        )
        if asset_type:
            query = query.filter(AllAssets.asset_type.ilike(asset_type))
        return query.order_by(FeaturesLatest.timestamp.desc(), AllAssets.symbol.asc()).limit(top_n).all()
    finally:
        db.close()


def scan_assets(top_n: int = DEFAULT_SCAN_CANDIDATES, asset_type: str | None = None) -> dict[str, object]:
    normalized_top_n = max(1, min(int(top_n), MAX_SCAN_CANDIDATES))
    normalized_asset_type = (asset_type or "").strip().lower()
    cache_key = ("scan", normalized_top_n, normalized_asset_type or "all")
    cached = runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    rows = _load_precomputed_assets(normalized_top_n, normalized_asset_type or None)
    if not rows:
        precompute_asset_batch(limit=normalized_top_n, asset_type=normalized_asset_type or None)
        rows = _load_precomputed_assets(normalized_top_n, normalized_asset_type or None)

    results = []
    start = monotonic()
    partial = False
    for asset, _feature_row in rows:
        if monotonic() - start > SCAN_TIMEOUT_SECONDS:
            partial = True
            break

        try:
            recommendation = infer_symbol(asset.symbol, refresh_if_stale=False)
        except Exception:
            continue

        results.append(
            {
                "symbol": asset.symbol,
                "name": asset.name or asset.symbol,
                "asset_type": asset.asset_type,
                "alpha": float(recommendation["alpha"]),
                "recommendation": str(recommendation["recommendation"]),
                "confidence": float(recommendation["confidence"]),
            }
        )

    results.sort(key=lambda item: item.get("alpha", 0.0), reverse=True)
    payload = {
        "results": results[:normalized_top_n],
        "partial": partial,
        "evaluated": len(results),
        "elapsed_seconds": round(monotonic() - start, 3),
    }
    runtime_cache.set(cache_key, payload, ttl_seconds=SCAN_CACHE_TTL_SECONDS)
    return payload


def warm_scan_cache() -> None:
    precompute_asset_batch(limit=10)
    scan_assets(top_n=10)
