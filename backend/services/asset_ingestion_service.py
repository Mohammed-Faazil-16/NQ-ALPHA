from __future__ import annotations

from sqlalchemy import func, or_

from agents.asset_universe_agent.load_full_assets import load_full_assets
from backend.database.models.all_assets import AllAssets
from backend.database.models.asset_universe import AssetUniverse
from backend.database.postgres import SessionLocal
from backend.services.symbol_resolver import STATIC_SYMBOL_MAP, resolve_symbol


def ingest_assets() -> dict[str, str]:
    load_full_assets()
    return {"status": "ok", "message": "Asset universe refreshed"}


def search_assets(query: str, limit: int = 20, asset_type: str | None = None) -> list[dict[str, str]]:
    normalized = " ".join((query or "").strip().lower().split())
    if not normalized:
        return []

    query_upper = normalized.replace(" ", "").upper().lstrip("$")
    results: list[dict[str, str]] = []
    seen: set[str] = set()

    def add_result(symbol: str, name: str, resolved_asset_type: str, source: str) -> None:
        symbol_upper = (symbol or "").strip().upper()
        if not symbol_upper or symbol_upper in seen:
            return
        if asset_type and resolved_asset_type and resolved_asset_type.lower() != asset_type.lower():
            return
        seen.add(symbol_upper)
        results.append(
            {
                "symbol": symbol_upper,
                "name": (name or symbol_upper).strip(),
                "asset_type": (resolved_asset_type or "stock").strip().lower(),
                "source": source,
            }
        )

    exact_static_symbol = STATIC_SYMBOL_MAP.get(normalized)
    if exact_static_symbol:
        inferred_type = "crypto" if exact_static_symbol.endswith("-USD") or "/" in exact_static_symbol else "stock"
        add_result(exact_static_symbol, normalized.title(), inferred_type, "static")
        return results[:limit]

    for alias, symbol in STATIC_SYMBOL_MAP.items():
        if normalized in alias or alias in normalized:
            inferred_type = "crypto" if symbol.endswith("-USD") or "/" in symbol else "stock"
            add_result(symbol, alias.title(), inferred_type, "static")
            if len(results) >= limit:
                return results[:limit]

    db = SessionLocal()
    try:
        all_assets_query = db.query(AllAssets).filter(
            or_(
                func.upper(AllAssets.symbol) == query_upper,
                func.lower(AllAssets.name) == normalized,
                AllAssets.symbol.ilike(f"%{query_upper}%"),
                AllAssets.name.ilike(f"%{normalized}%"),
            )
        )
        if asset_type:
            all_assets_query = all_assets_query.filter(func.lower(AllAssets.asset_type) == asset_type.lower())
        all_assets_rows = all_assets_query.order_by(AllAssets.symbol.asc()).limit(limit).all()
        for row in all_assets_rows:
            add_result(row.symbol, row.name or row.symbol, row.asset_type or "stock", "all-assets")
            if len(results) >= limit:
                return results[:limit]

        universe_rows = (
            db.query(AssetUniverse)
            .filter(AssetUniverse.symbol.ilike(f"%{query_upper}%"))
            .order_by(AssetUniverse.symbol.asc())
            .limit(limit)
            .all()
        )
        for row in universe_rows:
            add_result(row.symbol, row.symbol, "stock", "asset-universe")
            if len(results) >= limit:
                break
    finally:
        db.close()

    if len(results) < limit:
        try:
            resolved = resolve_symbol(query)
            inferred_type = "crypto" if resolved.endswith("-USD") or "/" in resolved else "stock"
            add_result(resolved, query.strip().title() or resolved, inferred_type, "resolver")
        except Exception:
            pass

    return results[:limit]
