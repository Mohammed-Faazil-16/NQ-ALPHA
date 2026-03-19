from sqlalchemy import Column, DateTime, Float, Integer, String

from backend.database.postgres import Base


class MarketRegime(Base):
    __tablename__ = "market_regimes"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    regime_score = Column(Float)
    regime_label = Column(String)
