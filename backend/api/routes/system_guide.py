from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.system_guide_service import build_system_guide_payload, validate_system_guide_password


router = APIRouter()


class SystemGuideRequest(BaseModel):
    password: str = Field(..., min_length=1)


@router.post("/system/guide")
def system_guide(payload: SystemGuideRequest):
    if not validate_system_guide_password(payload.password):
        raise HTTPException(status_code=403, detail="Invalid guide password")
    return build_system_guide_payload()
