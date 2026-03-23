from datetime import datetime
import uuid

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.postgres import Base


class MemoryMessage(Base):
    __tablename__ = "memory_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        index=True,
        nullable=False,
    )


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    full_name = Column(String, default="", nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    risk_level = Column(String, default="medium", nullable=False)
    goals = Column(String, default="balanced growth", nullable=False)
    investment_horizon = Column(String, default="5y", nullable=False)
    capital = Column(Float, default=0.0, nullable=False)
    interests = Column(JSON, default=list, nullable=False)
    onboarding_complete = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    strategy = Column(String, default="", nullable=False)
    equity_pct = Column(Float, default=0.0, nullable=False)
    cash_pct = Column(Float, default=0.0, nullable=False)
    positions_json = Column(JSON, default=list, nullable=False)
    lookback_days = Column(Integer, default=180, nullable=False)
    last_updated = Column(String)


class FinancialPlan(Base):
    __tablename__ = "financial_plans"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    strategy = Column(Text, default="", nullable=False)
    risk_level = Column(String, default="medium", nullable=False)
    allocation_summary = Column(Text, default="", nullable=False)
    reasoning = Column(Text, default="", nullable=False)
    source = Column(String, default="", nullable=False)
    model = Column(String, default="", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class FeaturesLatest(Base):
    __tablename__ = "features_latest"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    features_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
