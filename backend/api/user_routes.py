from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.database.models.user import User
from backend.database.postgres import get_db
from backend.schemas.user_schema import UserCreate, UserResponse


router = APIRouter()


@router.post("/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    new_user = User(
        name=user.name,
        email=user.email,
        risk_tolerance=user.risk_tolerance,
        capital=user.capital,
        investment_goal=user.investment_goal,
    )

    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="User with this email already exists")

    return new_user


@router.get("/", response_model=list[UserResponse])
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()
