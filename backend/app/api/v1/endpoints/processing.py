"""Processing job endpoints – trigger reconstruction, detection, segmentation."""
from typing import Optional
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.auth import get_current_user
from app.db.session import get_db
from app.models.models import User
from app.workers.tasks import run_sfm_task, run_detection_task, compute_progress_task

router = APIRouter()

@router.post("/reconstruction", status_code=status.HTTP_202_ACCEPTED)
async def trigger_reconstruction(project_id: str, quality: str = "high",
                                  media_upload_ids: list = [],
                                  db: AsyncSession = Depends(get_db),
                                  current_user: User = Depends(get_current_user)):
    task = run_sfm_task.delay(project_id, media_upload_ids, quality)
    return {"job_id": task.id, "status": "queued", "project_id": project_id}

@router.post("/detection", status_code=status.HTTP_202_ACCEPTED)
async def trigger_detection(project_id: str, model: str = "yolov8",
                             db: AsyncSession = Depends(get_db),
                             current_user: User = Depends(get_current_user)):
    task = run_detection_task.delay(project_id, model_name=model)
    return {"job_id": task.id, "status": "queued"}

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str, current_user: User = Depends(get_current_user)):
    from app.workers.tasks import celery_app
    result = celery_app.AsyncResult(job_id)
    return {"job_id": job_id, "status": result.state,
            "progress": result.info.get("progress", 0) if isinstance(result.info, dict) else 0,
            "result": result.result if result.ready() else None}
