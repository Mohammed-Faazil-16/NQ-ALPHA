from datetime import datetime

from pydantic import BaseModel


class UserCreate(BaseModel):
    name: str
    email: str
    risk_tolerance: str
    capital: float
    investment_goal: str


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    risk_tolerance: str
    capital: float
    investment_goal: str
    created_at: datetime

    class Config:
        orm_mode = True
