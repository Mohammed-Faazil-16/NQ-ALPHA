from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from backend.database.postgres import Base


class Features(Base):
    __tablename__ = "features"

    id = Column(Integer, primary_key=True)

    symbol = Column(String, index=True)
    timestamp = Column(DateTime, index=True)

    log_return_1 = Column(Float)
    log_return_5 = Column(Float)
    log_return_20 = Column(Float)

    momentum_5 = Column(Float)
    momentum_20 = Column(Float)
    momentum_60 = Column(Float)

    volatility_10 = Column(Float)
    volatility_20 = Column(Float)
    volatility_60 = Column(Float)

    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)

    price_vs_sma20 = Column(Float)
    price_vs_sma50 = Column(Float)
    price_vs_sma200 = Column(Float)

    RSI_14 = Column(Float)

    MACD = Column(Float)
    MACD_signal = Column(Float)
    MACD_hist = Column(Float)

    volume_change = Column(Float)
    volume_zscore = Column(Float)

    created_at = Column(DateTime, default=func.now())