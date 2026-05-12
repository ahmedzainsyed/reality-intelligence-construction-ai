"""BIM comparison endpoints."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import uuid4

from app.core.auth import get_current_user
from app.db.session import get_db
from app.models.models import User, BIMModel, BIMComparison

router = APIRouter()


@router.post("/models", status_code=status.HTTP_201_CREATED, summary="Upload IFC/BIM model")
async def upload_bim_model(
    project_id: str = Form(...),
    name: str = Form(...),
    discipline: str = Form("structural"),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    file_data = await file.read()
    model_id = str(uuid4())

    from app.core.storage import get_storage_client
    from app.core.config import settings
    storage = get_storage_client()
    key = f"projects/{project_id}/bim/{model_id}/{file.filename}"
    await storage.upload_bytes(settings.MINIO_BUCKET_MEDIA, key, file_data,
                                content_type="application/octet-stream")

    bim = BIMModel(
        id=model_id, project_id=project_id, name=name,
        storage_path=key, discipline=discipline,
        file_size_mb=round(len(file_data) / 1e6, 2),
        is_active=True,
    )
    db.add(bim)
    await db.flush()
    return {"id": bim.id, "name": bim.name, "status": "uploaded"}


@router.get("/models", summary="List BIM models for project")
async def list_bim_models(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = await db.execute(
        select(BIMModel).where(BIMModel.project_id == project_id, BIMModel.is_active == True)
    )
    models = rows.scalars().all()
    return {"items": [{"id": m.id, "name": m.name, "discipline": m.discipline,
                        "file_size_mb": m.file_size_mb} for m in models]}


@router.post("/compare", status_code=status.HTTP_202_ACCEPTED, summary="Trigger BIM comparison")
async def trigger_bim_comparison(
    data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from app.workers.tasks import run_bim_comparison_task
    task = run_bim_comparison_task.delay(
        data["project_id"], data["bim_model_id"], data["reconstruction_id"]
    )
    return {"job_id": task.id, "status": "queued"}


@router.get("/comparisons/{comparison_id}", summary="Get BIM comparison result")
async def get_comparison(
    comparison_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    comp = await db.get(BIMComparison, comparison_id)
    if not comp:
        raise HTTPException(status_code=404, detail="Comparison not found")
    return {
        "id": comp.id, "overall_completion_percent": comp.overall_completion_percent,
        "mean_spatial_deviation_mm": comp.mean_spatial_deviation_mm,
        "max_spatial_deviation_mm": comp.max_spatial_deviation_mm,
        "schedule_deviation_days": comp.schedule_deviation_days,
        "computed_at": comp.computed_at,
    }
