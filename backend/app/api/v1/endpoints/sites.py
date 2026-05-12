"""Sites endpoint – convenience wrapper around projects."""
from fastapi import APIRouter, Depends
from app.core.auth import get_current_user
from app.models.models import User

router = APIRouter()

@router.get("/", summary="List active construction sites")
async def list_sites(current_user: User = Depends(get_current_user)):
    """Alias for active projects with site metadata."""
    return {"message": "Use /api/v1/projects?status=active for site listings"}
