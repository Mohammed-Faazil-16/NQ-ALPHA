from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from backend.database.postgres import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    risk_tolerance = Column(String)
    capital = Column(Float)
    investment_goal = Column(String)
    created_at = Column(DateTime, default=func.now())
