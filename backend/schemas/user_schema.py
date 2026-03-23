from datetime import datetime

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    full_name: str = Field(default="", max_length=255)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str


class UserProfileUpdate(BaseModel):
    capital: float = Field(default=0.0, ge=0.0)
    risk_level: str = Field(default="medium", min_length=2, max_length=50)
    interests: list[str] = Field(default_factory=list)
    goals: str = Field(default="balanced growth", min_length=2, max_length=255)
    investment_horizon: str = Field(default="5y", min_length=1, max_length=50)


class UserResponse(BaseModel):
    id: str
    full_name: str
    email: str
    risk_level: str
    goals: str
    investment_horizon: str
    capital: float
    interests: list[str]
    onboarding_complete: bool
    created_at: datetime

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    access_token: str
    user: UserResponse
