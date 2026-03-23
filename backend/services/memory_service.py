from datetime import datetime

from sqlalchemy.orm import Session

from backend.config import settings
from backend.db.models import MemoryMessage, User
from backend.db.postgres import SessionLocal
from backend.services.embedding_service import get_embedding
from backend.services.financial_plan_service import get_latest_strategy as get_latest_financial_plan_strategy
from backend.services.financial_plan_service import save_financial_plan
from backend.vectorstore.chroma_client import get_collection


DEFAULT_PROFILE = {
    "full_name": "",
    "risk_level": "medium",
    "goals": "balanced growth",
    "investment_horizon": "5y",
    "capital": 0.0,
    "interests": [],
    "onboarding_complete": False,
}
STRATEGY_PREFIX = "LAST_STRATEGY::"


def _recent_messages_from_postgres(user_id: str, limit: int) -> list[str]:
    db: Session = SessionLocal()
    try:
        rows = (
            db.query(MemoryMessage)
            .filter(MemoryMessage.user_id == user_id)
            .filter(~MemoryMessage.message.like(f"{STRATEGY_PREFIX}%"))
            .order_by(MemoryMessage.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [row.message for row in reversed(rows)]
    except Exception:
        return []
    finally:
        db.close()


def store_message(user_id: str, message: str, role: str, include_embedding: bool = False) -> None:
    timestamp = datetime.utcnow()
    db: Session = SessionLocal()
    record_id: str | None = None

    try:
        db_record = MemoryMessage(
            user_id=user_id,
            message=message,
            role=role,
            timestamp=timestamp,
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        record_id = str(db_record.id)
    except Exception:
        db.rollback()
    finally:
        db.close()

    if not include_embedding:
        return

    try:
        embedding = get_embedding(message)
        if not embedding or record_id is None:
            return

        collection = get_collection()
        metadata = {
            "user_id": user_id,
            "role": role,
            "timestamp": timestamp.isoformat(),
        }
        try:
            collection.upsert(
                ids=[record_id],
                documents=[message],
                embeddings=[embedding],
                metadatas=[metadata],
            )
        except AttributeError:
            collection.add(
                ids=[record_id],
                documents=[message],
                embeddings=[embedding],
                metadatas=[metadata],
            )
    except Exception:
        return


def retrieve_context(user_id: str, query: str, top_k: int | None = None) -> list[str]:
    limit = min(top_k or settings.memory_top_k, 5)
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            return _recent_messages_from_postgres(user_id, limit)

        collection = get_collection()
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where={"user_id": user_id},
        )

        documents = result.get("documents") or []
        if not documents:
            return _recent_messages_from_postgres(user_id, limit)

        first_batch = documents[0] if documents else []
        context = [
            str(document)
            for document in first_batch
            if document and not str(document).startswith(STRATEGY_PREFIX)
        ]
        return context[:limit] or _recent_messages_from_postgres(user_id, limit)
    except Exception:
        return _recent_messages_from_postgres(user_id, limit)


def store_user_profile(user_id: str, risk_level: str, goals: str):
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return

        user.risk_level = risk_level
        user.goals = goals
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def get_user_profile(user_id: str):
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return DEFAULT_PROFILE.copy()

        return {
            "full_name": user.full_name,
            "risk_level": user.risk_level,
            "goals": user.goals,
            "investment_horizon": user.investment_horizon,
            "capital": user.capital,
            "interests": list(user.interests or []),
            "onboarding_complete": bool(user.onboarding_complete),
        }
    except Exception:
        return DEFAULT_PROFILE.copy()
    finally:
        db.close()


def store_last_strategy(user_id: str, strategy: str):
    normalized = strategy.strip()
    if not normalized:
        return
    if normalized.lower().startswith("strategy:"):
        normalized = normalized.split(":", 1)[1].strip()
    if not normalized:
        return
    save_financial_plan(
        user_id=user_id,
        response_text=f"Strategy: {normalized}\nRisk Level: Medium\nAllocation: Keep current allocation.\nReasoning: Strategy state was updated from the latest advisor output.",
        source="legacy-strategy-store",
        model="",
    )


def get_last_strategy(user_id: str):
    strategy = get_latest_financial_plan_strategy(user_id)
    if strategy:
        return strategy

    db: Session = SessionLocal()
    try:
        result = (
            db.query(MemoryMessage)
            .filter(MemoryMessage.user_id == user_id)
            .filter(MemoryMessage.message.like(f"{STRATEGY_PREFIX}%"))
            .order_by(MemoryMessage.id.desc())
            .first()
        )

        if result:
            return result.message.replace(STRATEGY_PREFIX, "", 1)
        return None
    except Exception:
        return None
    finally:
        db.close()
