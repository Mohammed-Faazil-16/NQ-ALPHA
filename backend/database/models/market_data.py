from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from backend.database.postgres import Base


class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    source = Column(String)
    created_at = Column(DateTime, default=func.now())
