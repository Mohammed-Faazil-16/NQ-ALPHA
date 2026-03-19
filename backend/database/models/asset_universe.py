from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from backend.database.postgres import Base


class AssetUniverse(Base):
    __tablename__ = "asset_universe"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True, unique=True)
    score = Column(Float)
    selected_at = Column(DateTime, default=func.now())
