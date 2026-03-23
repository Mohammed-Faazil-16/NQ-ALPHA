from fastapi import APIRouter, HTTPException, Query

from backend.services.asset_ingestion_service import search_assets
from backend.services.fast_inference_service import alpha_series, infer_symbol
from backend.services.live_data_service import fetch_asset_data
from backend.services.precompute_service import get_latest_features_payload
from backend.services.scanner_service import scan_assets
from backend.services.symbol_resolver import resolve_symbol


router = APIRouter()


@router.get("/assets/search")
def asset_search(q: str = Query(..., min_length=1), limit: int = Query(8, ge=1, le=20), asset_type: str | None = Query(None)):
    try:
        results = search_assets(q, limit=limit, asset_type=asset_type)
        return {"query": q, "results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/price")
def price(symbol: str = Query(..., min_length=1)):
    try:
        resolved_symbol = resolve_symbol(symbol)
        data = fetch_asset_data(resolved_symbol).tail(180).copy()
        latest_timestamp = data.iloc[-1].timestamp.strftime("%Y-%m-%d") if not data.empty else None
        return {
            "symbol": resolved_symbol,
            "latest_timestamp": latest_timestamp,
            "data": [
                {
                    "timestamp": row.timestamp.strftime("%Y-%m-%d"),
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": float(row.volume),
                }
                for row in data.itertuples(index=False)
            ],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/features")
def features(symbol: str = Query(..., min_length=1)):
    try:
        resolved_symbol = resolve_symbol(symbol)
        payload = get_latest_features_payload(resolved_symbol, refresh_if_stale=True)
        details = infer_symbol(resolved_symbol, refresh_if_stale=True)
        return {
            "symbol": resolved_symbol,
            "timestamp": str(payload.get("timestamp", "")),
            "features": dict(payload.get("latest_features") or {}),
            "regime": details["regime"],
            "regime_id": details["regime_id"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/alpha_series")
def get_alpha_series(symbol: str = Query(..., min_length=1), lookback: int = Query(30, ge=10, le=90)):
    try:
        resolved_symbol = resolve_symbol(symbol)
        series = alpha_series(resolved_symbol, lookback=lookback, refresh_if_stale=True)
        return {
            "symbol": resolved_symbol,
            "as_of": series[-1]["timestamp"] if series else None,
            "series": series,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/scan")
def scan(top_n: int = Query(20, ge=5, le=20), asset_type: str | None = Query(None)):
    try:
        return scan_assets(top_n=top_n, asset_type=asset_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
