from __future__ import annotations

from datetime import datetime

from backend.db.models import Portfolio, User
from backend.db.postgres import SessionLocal


DEFAULT_LOOKBACK_DAYS = 180
DEFAULT_CURRENCY = "INR"


def _clean_positions(assets: list[dict[str, object]] | None, capital: float = 0.0) -> list[dict[str, float | str]]:
    cleaned = []
    for asset in assets or []:
        symbol = str(asset.get("symbol") or "").strip().upper()
        amount = round(float(asset.get("amount") or 0.0), 2)
        weight = float(asset.get("weight") or 0.0)
        if amount > 0.0 and capital > 0.0:
            weight = amount / capital
        if not symbol or weight <= 0.0:
            continue
        cleaned.append({"symbol": symbol, "weight": weight})
    return cleaned


def _prepare_positions(positions: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    total = sum(float(position["weight"]) for position in positions)
    if total <= 0.0:
        return []
    if total <= 1.0 + 1e-6:
        return [
            {"symbol": str(position["symbol"]), "weight": float(position["weight"])}
            for position in positions
        ]

    return [
        {"symbol": str(position["symbol"]), "weight": float(position["weight"]) / total}
        for position in positions
    ]


def _build_allocation_payload(user: User | None, portfolio: Portfolio | None) -> dict[str, object]:
    capital = float(user.capital if user and user.capital is not None else 0.0)
    positions = _prepare_positions(_clean_positions((portfolio.positions_json if portfolio else []) or [], capital))
    allocations = [
        {
            "symbol": position["symbol"],
            "weight": round(float(position["weight"]), 6),
            "percent": round(float(position["weight"]) * 100.0, 2),
            "amount": round(capital * float(position["weight"]), 2),
        }
        for position in positions
    ]

    invested_amount = sum(float(item["amount"]) for item in allocations)
    if not allocations and portfolio and capital > 0.0:
        invested_amount = capital * float(portfolio.equity_pct or 0.0) / 100.0
    available_cash_amount = max(capital - invested_amount, 0.0)
    equity_pct = (invested_amount / capital * 100.0) if capital > 0.0 else float(portfolio.equity_pct if portfolio and portfolio.equity_pct is not None else 0.0)
    cash_pct = (available_cash_amount / capital * 100.0) if capital > 0.0 else float(portfolio.cash_pct if portfolio and portfolio.cash_pct is not None else 100.0)

    return {
        "capital": capital,
        "currency": DEFAULT_CURRENCY,
        "invested_amount": invested_amount,
        "available_cash_amount": available_cash_amount,
        "strategy": (portfolio.strategy if portfolio else "") or "",
        "lookback_days": int(portfolio.lookback_days if portfolio and portfolio.lookback_days else DEFAULT_LOOKBACK_DAYS),
        "equity_pct": equity_pct,
        "cash_pct": cash_pct,
        "allocations": allocations,
        "updated_at": (portfolio.last_updated if portfolio else None),
    }


def save_user_portfolio_plan(user_id: str, assets: list[dict[str, object]], lookback_days: int) -> dict[str, object]:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        portfolio = db.query(Portfolio).filter(Portfolio.user_id == user_id).first()
        if not portfolio:
            portfolio = Portfolio(user_id=user_id)
            db.add(portfolio)

        capital = float(user.capital if user and user.capital is not None else 0.0)
        cleaned = _clean_positions(assets, capital)
        prepared = _prepare_positions(cleaned)
        allocated_fraction = min(sum(float(item["weight"]) for item in prepared), 1.0)
        portfolio.positions_json = prepared
        portfolio.lookback_days = int(lookback_days or DEFAULT_LOOKBACK_DAYS)
        portfolio.equity_pct = round(allocated_fraction * 100.0, 2)
        portfolio.cash_pct = round(max(100.0 - portfolio.equity_pct, 0.0), 2)
        portfolio.last_updated = datetime.utcnow().isoformat()

        db.commit()
        db.refresh(portfolio)
        return _build_allocation_payload(user, portfolio)
    except Exception:
        db.rollback()
        return {
            "capital": 0.0,
            "currency": DEFAULT_CURRENCY,
            "invested_amount": 0.0,
            "available_cash_amount": 0.0,
            "strategy": "",
            "lookback_days": DEFAULT_LOOKBACK_DAYS,
            "equity_pct": 0.0,
            "cash_pct": 100.0,
            "allocations": [],
            "updated_at": None,
        }
    finally:
        db.close()


def build_portfolio_from_strategy(user_id: str, strategy: str):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        portfolio = db.query(Portfolio).filter(Portfolio.user_id == user_id).first()
        if not portfolio:
            portfolio = Portfolio(user_id=user_id)
            db.add(portfolio)

        strategy_text = (strategy or "").strip()
        lowered = strategy_text.lower()
        if "momentum" in lowered or "aggressive" in lowered:
            portfolio.equity_pct = 80.0
            portfolio.cash_pct = 20.0
        else:
            portfolio.equity_pct = 60.0
            portfolio.cash_pct = 40.0

        portfolio.strategy = strategy_text
        portfolio.last_updated = datetime.utcnow().isoformat()
        db.commit()
        db.refresh(portfolio)

        return _build_allocation_payload(user, portfolio)
    except Exception:
        db.rollback()
        return {
            "capital": 0.0,
            "currency": DEFAULT_CURRENCY,
            "invested_amount": 0.0,
            "available_cash_amount": 0.0,
            "equity_pct": 0.0,
            "cash_pct": 100.0,
            "allocations": [],
            "lookback_days": DEFAULT_LOOKBACK_DAYS,
            "strategy": "",
            "updated_at": None,
        }
    finally:
        db.close()


def get_user_portfolio(user_id: str):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        portfolio = db.query(Portfolio).filter(Portfolio.user_id == user_id).first()
        if not portfolio:
            capital = float(user.capital if user and user.capital is not None else 0.0)
            return {
                "equity": 0.0,
                "cash": 100.0,
                "strategy": "",
                "allocations": [],
                "capital": capital,
                "currency": DEFAULT_CURRENCY,
                "invested_amount": 0.0,
                "available_cash_amount": capital,
                "lookback_days": DEFAULT_LOOKBACK_DAYS,
            }

        allocation = _build_allocation_payload(user, portfolio)
        return {
            "equity": float(allocation["equity_pct"]),
            "cash": float(allocation["cash_pct"]),
            "strategy": allocation["strategy"],
            "allocations": allocation["allocations"],
            "capital": allocation["capital"],
            "currency": allocation["currency"],
            "invested_amount": allocation["invested_amount"],
            "available_cash_amount": allocation["available_cash_amount"],
            "lookback_days": allocation["lookback_days"],
            "updated_at": allocation["updated_at"],
        }
    except Exception:
        return {
            "equity": 0.0,
            "cash": 100.0,
            "strategy": "",
            "allocations": [],
            "capital": 0.0,
            "currency": DEFAULT_CURRENCY,
            "invested_amount": 0.0,
            "available_cash_amount": 0.0,
            "lookback_days": DEFAULT_LOOKBACK_DAYS,
        }
    finally:
        db.close()


def get_user_allocation_view(user_id: str) -> dict[str, object]:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        portfolio = db.query(Portfolio).filter(Portfolio.user_id == user_id).first()
        return _build_allocation_payload(user, portfolio)
    finally:
        db.close()
