from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.recommendation_service import recommend_asset
from backend.services.symbol_resolver import resolve_symbol


router = APIRouter()


class RecommendRequest(BaseModel):
    query: str


@router.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        symbol = resolve_symbol(req.query)
        result = recommend_asset(symbol)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
