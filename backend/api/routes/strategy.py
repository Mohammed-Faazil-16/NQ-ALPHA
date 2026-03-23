from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.backtest_service import run_backtest


router = APIRouter()


class BacktestAsset(BaseModel):
    symbol: str = Field(..., min_length=1)
    weight: float | None = None
    amount: float | None = Field(default=None, ge=0.0)


class BacktestRequest(BaseModel):
    assets: list[BacktestAsset] = Field(default_factory=list)
    lookback_days: int = Field(default=180, ge=30, le=3650)
    capital: float = Field(default=0.0, ge=0.0)
    investment_horizon: str | None = Field(default=None, min_length=1, max_length=50)


@router.post("/backtest")
def backtest(request: BacktestRequest):
    try:
        assets = [asset.model_dump() for asset in request.assets]
        return run_backtest(
            assets=assets,
            lookback_days=request.lookback_days,
            capital=request.capital,
            investment_horizon=request.investment_horizon,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
