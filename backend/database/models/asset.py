from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.sql import func

from backend.database.postgres import Base


class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True)
    name = Column(String)
    asset_type = Column(String)
    exchange = Column(String, nullable=True)
    sector = Column(String, nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
