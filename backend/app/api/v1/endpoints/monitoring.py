"""Internal monitoring and system status endpoints."""
from fastapi import APIRouter, Depends
from app.core.auth import get_current_user, require_roles
from app.models.models import User, UserRole

router = APIRouter()

@router.get("/system-status", summary="System health overview")
async def system_status(current_user: User = Depends(require_roles(UserRole.ADMIN))):
    """Return high-level system health for ops dashboard."""
    import psutil
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
        "status": "operational",
    }

@router.get("/queue-stats", summary="Celery queue statistics")
async def queue_stats(current_user: User = Depends(require_roles(UserRole.ADMIN))):
    """Return Celery queue depths."""
    try:
        from app.workers.tasks import celery_app
        inspect = celery_app.control.inspect()
        active = inspect.active() or {}
        reserved = inspect.reserved() or {}
        return {
            "active_tasks": sum(len(v) for v in active.values()),
            "reserved_tasks": sum(len(v) for v in reserved.values()),
            "workers": list(active.keys()),
        }
    except Exception:
        return {"active_tasks": 0, "reserved_tasks": 0, "workers": []}
