from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.advisor_service import generate_financial_advice
from backend.services.llm_service import get_ollama_status
from backend.services.memory_service import store_message
from .market import router as market_router
from .recommend import router as recommend_router
from .strategy import router as strategy_router
from .system_guide import router as system_guide_router


router = APIRouter()
router.include_router(recommend_router)
router.include_router(market_router)
router.include_router(strategy_router)
router.include_router(system_guide_router)


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    user_id: str
    response: str
    timestamp: datetime
    context_count: int
    source: str | None = None
    model: str | None = None
    latency_seconds: float | None = None
    plan: dict[str, Any] | None = None


@router.get("/advisor/status")
def advisor_status():
    return get_ollama_status()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        store_message(request.user_id, request.message, "user", include_embedding=False)
        advisor_payload = generate_financial_advice(request.user_id, request.message)
        assistant_response = str(advisor_payload.get("text") or "")
        store_message(request.user_id, assistant_response, "assistant", include_embedding=False)

        return ChatResponse(
            user_id=request.user_id,
            response=assistant_response,
            timestamp=datetime.utcnow(),
            context_count=0,
            source=str(advisor_payload.get("source") or "unknown"),
            model=str(advisor_payload.get("model") or "") or None,
            latency_seconds=float(advisor_payload.get("latency_seconds") or 0.0),
            plan=advisor_payload.get("plan") if isinstance(advisor_payload.get("plan"), dict) else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
