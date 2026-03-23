from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.advisor_service import generate_financial_advice
from backend.services.alpha_service import generate_features_for_asset, generate_recommendation, predict_alpha
from backend.services.live_data_service import fetch_asset_data
from backend.services.memory_service import retrieve_context, store_message
from backend.services.symbol_resolver import resolve_symbol


router = APIRouter()


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    user_id: str
    response: str
    timestamp: datetime
    context_count: int


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        store_message(request.user_id, request.message, "user")
        memory_context = retrieve_context(request.user_id, request.message)
        assistant_response = generate_financial_advice(request.user_id, request.message)
        store_message(request.user_id, assistant_response, "assistant")

        return ChatResponse(
            user_id=request.user_id,
            response=assistant_response,
            timestamp=datetime.utcnow(),
            context_count=len(memory_context),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/recommend")
def recommend(request: RecommendRequest):
    try:
        symbol = resolve_symbol(request.query)
        asset_data = fetch_asset_data(symbol)
        feature_payload = generate_features_for_asset(asset_data)
        alpha = predict_alpha(feature_payload["features"], feature_payload["regime_id"])
        recommendation, confidence = generate_recommendation(
            alpha,
            float(feature_payload["volatility"]),
            str(feature_payload["regime"]),
        )
        return {
            "symbol": symbol,
            "alpha": round(float(alpha), 4),
            "recommendation": recommendation,
            "confidence": confidence,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/portfolio")
def portfolio():
    return [
        {"asset": "AAPL", "weight": 0.2},
        {"asset": "MSFT", "weight": 0.3},
        {"asset": "NVDA", "weight": 0.15},
        {"asset": "Cash", "weight": 0.35},
    ]


@router.get("/metrics")
def metrics():
    return {
        "equity_curve": [
            {"timestamp": "2023-01", "value": 100},
            {"timestamp": "2023-02", "value": 110},
            {"timestamp": "2023-03", "value": 106},
            {"timestamp": "2023-04", "value": 118},
            {"timestamp": "2023-05", "value": 126},
            {"timestamp": "2023-06", "value": 134},
        ]
    }


@router.get("/alpha")
def alpha():
    return [
        {"asset": "AAPL", "alpha": 0.12},
        {"asset": "MSFT", "alpha": 0.08},
        {"asset": "NVDA", "alpha": 0.17},
        {"asset": "AMZN", "alpha": 0.05},
        {"asset": "TSLA", "alpha": 0.03},
    ]
