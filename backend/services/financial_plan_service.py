from __future__ import annotations

import re
from typing import Any

from backend.db.models import FinancialPlan
from backend.db.postgres import SessionLocal


PLAN_FIELDS = {
    "strategy": "strategy",
    "risk level": "risk_level",
    "allocation": "allocation_summary",
    "reasoning": "reasoning",
}


def _clean_line(line: str) -> str:
    cleaned = str(line or "").replace("**", "").replace("__", "").strip()
    return cleaned.lstrip("*- ").strip()


def parse_advisor_response(response_text: str) -> dict[str, str]:
    parsed = {
        "strategy": "",
        "risk_level": "",
        "allocation_summary": "",
        "reasoning": "",
    }
    current_key = "reasoning"

    for raw_line in str(response_text or "").splitlines():
        line = _clean_line(raw_line)
        if not line:
            continue

        matched_field = None
        for label, target in PLAN_FIELDS.items():
            pattern = rf"^{re.escape(label)}(?:\s*:)?\s*(.*)$"
            match = re.match(pattern, line, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                parsed[target] = value
                current_key = target
                matched_field = target
                break

        if matched_field is not None:
            continue

        if current_key not in parsed:
            current_key = "reasoning"
        parsed[current_key] = f"{parsed[current_key]} {line}".strip()

    return parsed


def _serialize_plan(plan: FinancialPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "id": plan.id,
        "user_id": plan.user_id,
        "strategy": plan.strategy,
        "risk_level": plan.risk_level,
        "allocation_summary": plan.allocation_summary,
        "reasoning": plan.reasoning,
        "source": plan.source,
        "model": plan.model,
        "created_at": plan.created_at.isoformat() if plan.created_at else None,
        "updated_at": plan.updated_at.isoformat() if plan.updated_at else None,
    }


def save_financial_plan(user_id: str, response_text: str, source: str = "", model: str = "") -> dict[str, Any] | None:
    parsed = parse_advisor_response(response_text)
    if not any(parsed.values()):
        return None

    db = SessionLocal()
    try:
        plan = FinancialPlan(
            user_id=user_id,
            strategy=parsed["strategy"],
            risk_level=parsed["risk_level"] or "medium",
            allocation_summary=parsed["allocation_summary"],
            reasoning=parsed["reasoning"],
            source=str(source or ""),
            model=str(model or ""),
        )
        db.add(plan)
        db.commit()
        db.refresh(plan)
        return _serialize_plan(plan)
    except Exception:
        db.rollback()
        return None
    finally:
        db.close()


def get_latest_financial_plan(user_id: str) -> dict[str, Any] | None:
    db = SessionLocal()
    try:
        plan = (
            db.query(FinancialPlan)
            .filter(FinancialPlan.user_id == user_id)
            .order_by(FinancialPlan.updated_at.desc(), FinancialPlan.created_at.desc())
            .first()
        )
        return _serialize_plan(plan)
    except Exception:
        return None
    finally:
        db.close()


def get_latest_strategy(user_id: str) -> str | None:
    plan = get_latest_financial_plan(user_id)
    if not plan:
        return None
    strategy = str(plan.get("strategy") or "").strip()
    return strategy or None


def plan_to_response_text(plan: dict[str, Any] | None, fallback_reason: str | None = None) -> str:
    if not plan:
        reasoning = fallback_reason or "The advisor could not refresh the plan right now."
        return (
            "Strategy: Preserve capital while monitoring current conditions.\n"
            "Risk Level: Medium\n"
            "Allocation: Keep a diversified stance until the system stabilizes.\n"
            f"Reasoning: {reasoning}"
        )

    reasoning = str(plan.get("reasoning") or "").strip() or "This is the latest validated plan for the user."
    if fallback_reason:
        reasoning = f"{reasoning} {fallback_reason}".strip()

    return (
        f"Strategy: {str(plan.get('strategy') or 'Maintain current plan').strip()}\n"
        f"Risk Level: {str(plan.get('risk_level') or 'medium').strip()}\n"
        f"Allocation: {str(plan.get('allocation_summary') or 'Keep current allocation.').strip()}\n"
        f"Reasoning: {reasoning}"
    )
