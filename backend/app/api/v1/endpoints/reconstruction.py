"""
Reconstruction endpoint – 3D reconstruction job management.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.auth import get_current_user
from app.db.session import get_db
from app.models.models import User, Reconstruction3D, Project
from app.schemas.schemas import ReconstructionRequest, ReconstructionResponse

router = APIRouter()


@router.post("/", response_model=ReconstructionResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_reconstruction(
    req: ReconstructionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Trigger SfM + MVS 3D reconstruction pipeline."""
    from app.workers.tasks import run_sfm_task
    from uuid import uuid4

    # Validate project access
    proj = await db.execute(
        select(Project).where(
            Project.id == req.project_id,
            Project.organization_id == current_user.organization_id,
        )
    )
    if not proj.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Project not found")

    # Create DB record
    recon = Reconstruction3D(
        id=str(uuid4()),
        project_id=req.project_id,
        quality=req.quality,
        sfm_status="pending",
        mvs_status="pending",
    )
    db.add(recon)
    await db.flush()

    # Queue task
    task = run_sfm_task.delay(req.project_id, req.media_upload_ids, req.quality)

    return ReconstructionResponse(
        job_id=task.id,
        reconstruction_id=recon.id,
        status="queued",
        project_id=req.project_id,
        quality=req.quality,
        message=f"Reconstruction queued. {len(req.media_upload_ids)} videos will be processed.",
    )


@router.get("/", summary="List reconstructions for a project")
async def list_reconstructions(
    project_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = await db.execute(
        select(Reconstruction3D)
        .where(Reconstruction3D.project_id == project_id)
        .order_by(Reconstruction3D.created_at.desc())
        .limit(20)
    )
    recons = rows.scalars().all()
    return {"items": [
        {
            "id": r.id, "quality": r.quality, "sfm_status": r.sfm_status,
            "mvs_status": r.mvs_status, "num_images_registered": r.num_images_registered,
            "num_dense_points": r.num_dense_points, "created_at": r.created_at,
        }
        for r in recons
    ]}


@router.get("/{reconstruction_id}", summary="Get reconstruction details")
async def get_reconstruction(
    reconstruction_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    r = await db.get(Reconstruction3D, reconstruction_id)
    if not r:
        raise HTTPException(status_code=404, detail="Reconstruction not found")
    return {
        "id": r.id, "project_id": r.project_id, "quality": r.quality,
        "sfm_status": r.sfm_status, "mvs_status": r.mvs_status,
        "num_images_registered": r.num_images_registered,
        "num_images_total": r.num_images_total,
        "num_sparse_points": r.num_sparse_points,
        "num_dense_points": r.num_dense_points,
        "mean_reprojection_error": r.mean_reprojection_error,
        "point_cloud_path": r.point_cloud_path,
        "mesh_path": r.mesh_path,
        "sfm_duration_seconds": r.sfm_duration_seconds,
        "mvs_duration_seconds": r.mvs_duration_seconds,
        "created_at": r.created_at,
    }


@router.get("/{reconstruction_id}/pointcloud-url", summary="Get presigned URL for point cloud")
async def get_pointcloud_url(
    reconstruction_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    r = await db.get(Reconstruction3D, reconstruction_id)
    if not r or not r.point_cloud_path:
        raise HTTPException(status_code=404, detail="Point cloud not available")

    from app.core.storage import get_storage_client
    from app.core.config import settings
    storage = get_storage_client()
    url = storage.get_presigned_url(settings.MINIO_BUCKET_OUTPUTS, r.point_cloud_path, expires_hours=2)
    return {"url": url, "expires_in_seconds": 7200}


@router.get("/{reconstruction_id}/camera-poses", summary="Get camera poses")
async def get_camera_poses(
    reconstruction_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from app.models.models import CameraPose
    rows = await db.execute(
        select(CameraPose).where(CameraPose.reconstruction_id == reconstruction_id)
    )
    poses = rows.scalars().all()
    return {
        "reconstruction_id": reconstruction_id,
        "total_poses": len(poses),
        "poses": [
            {
                "image_name": p.image_name, "camera_id": p.camera_id,
                "qw": p.qw, "qx": p.qx, "qy": p.qy, "qz": p.qz,
                "tx": p.tx, "ty": p.ty, "tz": p.tz,
                "fx": p.fx, "fy": p.fy, "cx": p.cx, "cy": p.cy,
            }
            for p in poses
        ],
    }
