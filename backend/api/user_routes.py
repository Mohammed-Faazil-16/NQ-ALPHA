from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.db.models import User
from backend.db.postgres import SessionLocal
from backend.schemas.user_schema import AuthResponse, UserCreate, UserLogin, UserProfileUpdate, UserResponse
from backend.services.financial_plan_service import get_latest_financial_plan
from backend.services.portfolio_service import get_user_allocation_view, save_user_portfolio_plan
from backend.utils.auth import create_access_token, get_current_user, hash_password, verify_password


router = APIRouter(tags=["users"])


class PortfolioAssetInput(BaseModel):
    symbol: str = Field(..., min_length=1)
    weight: float | None = Field(default=None, ge=0.0)
    amount: float | None = Field(default=None, ge=0.0)


class PortfolioAllocationUpdate(BaseModel):
    assets: list[PortfolioAssetInput] = Field(default_factory=list)
    lookback_days: int = Field(default=180, ge=30, le=3650)


class FinancialPlanResponse(BaseModel):
    id: str | None = None
    user_id: str | None = None
    strategy: str = ""
    risk_level: str = "medium"
    allocation_summary: str = ""
    reasoning: str = ""
    source: str = ""
    model: str = ""
    created_at: str | None = None
    updated_at: str | None = None


def _serialize_user(user: User) -> UserResponse:
    return UserResponse.model_validate(
        {
            "id": user.id,
            "full_name": user.full_name or user.email.split("@", 1)[0],
            "email": user.email,
            "risk_level": user.risk_level,
            "goals": user.goals,
            "investment_horizon": user.investment_horizon,
            "capital": float(user.capital or 0.0),
            "interests": list(user.interests or []),
            "onboarding_complete": bool(user.onboarding_complete),
            "created_at": user.created_at,
        }
    )


@router.post("/auth/register", response_model=AuthResponse)
def register(user: UserCreate):
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == user.email.strip().lower()).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already exists")

        new_user = User(
            full_name=user.full_name.strip(),
            email=user.email.strip().lower(),
            password_hash=hash_password(user.password),
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        token = create_access_token({"user_id": new_user.id})
        return AuthResponse(access_token=token, user=_serialize_user(new_user))
    finally:
        db.close()


@router.post("/auth/login", response_model=AuthResponse)
def login(user: UserLogin):
    db = SessionLocal()
    try:
        db_user = db.query(User).filter(User.email == user.email.strip().lower()).first()
        if not db_user or not verify_password(user.password, db_user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = create_access_token({"user_id": db_user.id})
        return AuthResponse(access_token=token, user=_serialize_user(db_user))
    finally:
        db.close()


@router.get("/auth/me", response_model=UserResponse)
def auth_me(current_user: User = Depends(get_current_user)):
    return _serialize_user(current_user)


@router.get("/user/profile", response_model=UserResponse)
def get_profile(current_user: User = Depends(get_current_user)):
    return _serialize_user(current_user)


@router.post("/user/profile", response_model=UserResponse)
def update_profile(profile: UserProfileUpdate, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == current_user.id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.risk_level = profile.risk_level.strip().lower()
        user.goals = profile.goals.strip()
        user.investment_horizon = profile.investment_horizon.strip()
        user.capital = float(profile.capital)
        user.interests = [interest.strip().lower() for interest in profile.interests if interest.strip()]
        user.onboarding_complete = True
        db.commit()
        db.refresh(user)
        return _serialize_user(user)
    finally:
        db.close()


@router.get("/portfolio/allocation")
def get_portfolio_allocation(current_user: User = Depends(get_current_user)):
    return get_user_allocation_view(current_user.id)


@router.post("/portfolio/allocation")
def save_portfolio_allocation(payload: PortfolioAllocationUpdate, current_user: User = Depends(get_current_user)):
    assets = [asset.model_dump() for asset in payload.assets]
    return save_user_portfolio_plan(current_user.id, assets, payload.lookback_days)


@router.get("/financial-plan/current", response_model=FinancialPlanResponse | None)
def get_current_financial_plan(current_user: User = Depends(get_current_user)):
    plan = get_latest_financial_plan(current_user.id)
    if not plan:
        return None
    return FinancialPlanResponse.model_validate(plan)


@router.post("/users/signup")
def legacy_signup(user: UserCreate):
    auth = register(user)
    return {"message": "User created", "user_id": auth.user.id, "access_token": auth.access_token}


@router.post("/users/login")
def legacy_login(user: UserLogin):
    auth = login(user)
    return {"access_token": auth.access_token, "user_id": auth.user.id}


@router.post("/users/profile/{user_id}")
def legacy_update_profile(user_id: str, profile: UserProfileUpdate):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.risk_level = profile.risk_level.strip().lower()
        user.goals = profile.goals.strip()
        user.investment_horizon = profile.investment_horizon.strip()
        user.capital = float(profile.capital)
        user.interests = [interest.strip().lower() for interest in profile.interests if interest.strip()]
        user.onboarding_complete = True
        db.commit()
        return {"message": "Profile updated", "user_id": user.id}
    finally:
        db.close()


@router.get("/users/", response_model=list[UserResponse])
def list_users():
    db = SessionLocal()
    try:
        users = db.query(User).order_by(User.created_at.desc()).all()
        return [_serialize_user(user) for user in users]
    finally:
        db.close()
