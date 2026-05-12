"""
Media Upload Endpoints

Handles chunked video uploads from:
- Drone footage
- CCTV streams
- Mobile walkthroughs
- 360° panoramic imagery

Features:
- Resumable chunked uploads
- Video metadata extraction
- Async validation
- MinIO/S3 storage
- Progress tracking
"""

import hashlib
import os
import tempfile
from typing import Optional, List
from uuid import uuid4

import aiofiles
import structlog
from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, Request,
    UploadFile, status, BackgroundTasks, Query
)
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.auth import get_current_user, require_roles
from app.core.storage import get_storage_client
from app.db.session import get_db
from app.models.models import User, MediaUpload, VideoSource, Project
from app.schemas.upload import (
    UploadInitResponse, UploadChunkResponse, UploadCompleteResponse,
    MediaUploadResponse, UploadListResponse
)
from app.services.upload_service import UploadService
from app.workers.tasks import extract_frames_task, validate_video_task

logger = structlog.get_logger(__name__)
router = APIRouter()


# =============================================================================
# UPLOAD INITIATION
# =============================================================================

@router.post(
    "/video/init",
    response_model=UploadInitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Initialize chunked video upload",
    description="""
    Initialize a chunked video upload session.
    Returns an upload_id to use for subsequent chunk uploads.
    Supports files up to 5GB via chunked multipart upload.
    """,
)
async def init_video_upload(
    project_id: str = Form(..., description="Project ID to associate media with"),
    filename: str = Form(..., description="Original filename"),
    file_size: int = Form(..., description="Total file size in bytes"),
    mime_type: str = Form(..., description="MIME type (video/mp4, etc.)"),
    source_type: VideoSource = Form(VideoSource.UNKNOWN, description="Video source type"),
    capture_date: Optional[str] = Form(None, description="Capture date ISO format"),
    camera_model: Optional[str] = Form(None, description="Camera/drone model"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(),
):
    """Initialize a resumable chunked upload session."""
    log = logger.bind(
        user_id=current_user.id,
        project_id=project_id,
        filename=filename,
        file_size_mb=round(file_size / 1024 / 1024, 2),
    )

    # Validate file size
    if file_size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {file_size} exceeds maximum {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    # Validate MIME type
    extension = filename.rsplit(".", 1)[-1].lower()
    if extension not in settings.SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported format '{extension}'. Supported: {settings.SUPPORTED_VIDEO_FORMATS}",
        )

    # Validate project exists and user has access
    project = await upload_service.get_project_for_user(db, project_id, current_user)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    # Generate upload session
    upload_id = str(uuid4())
    chunk_size = 10 * 1024 * 1024  # 10MB chunks
    total_chunks = (file_size + chunk_size - 1) // chunk_size

    # Initialize upload in MinIO (multipart upload)
    minio_path = f"projects/{project_id}/media/{upload_id}/{filename}"
    minio_upload_id = await upload_service.init_multipart_upload(minio_path, mime_type)

    # Create DB record in pending state
    media_record = await upload_service.create_upload_record(
        db=db,
        project_id=project_id,
        user_id=current_user.id,
        upload_id=upload_id,
        filename=filename,
        file_size=file_size,
        mime_type=mime_type,
        source_type=source_type,
        minio_path=minio_path,
        minio_upload_id=minio_upload_id,
        camera_model=camera_model,
        capture_date=capture_date,
    )

    log.info("Upload session initialized", upload_id=upload_id, total_chunks=total_chunks)

    return UploadInitResponse(
        upload_id=upload_id,
        minio_upload_id=minio_upload_id,
        chunk_size=chunk_size,
        total_chunks=total_chunks,
        storage_path=minio_path,
        expires_in_hours=24,
    )


# =============================================================================
# CHUNK UPLOAD
# =============================================================================

@router.put(
    "/video/{upload_id}/chunk/{chunk_number}",
    response_model=UploadChunkResponse,
    summary="Upload a video chunk",
)
async def upload_video_chunk(
    upload_id: str,
    chunk_number: int,
    chunk: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(),
):
    """Upload a single chunk of a video file."""
    # Read chunk data
    chunk_data = await chunk.read()
    chunk_size = len(chunk_data)

    if chunk_size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty chunk received")

    # Compute chunk checksum
    chunk_md5 = hashlib.md5(chunk_data).hexdigest()

    # Upload chunk to MinIO
    etag = await upload_service.upload_chunk(
        upload_id=upload_id,
        chunk_number=chunk_number,
        chunk_data=chunk_data,
    )

    logger.info(
        "Chunk uploaded",
        upload_id=upload_id,
        chunk_number=chunk_number,
        chunk_size_mb=round(chunk_size / 1024 / 1024, 2),
        etag=etag,
    )

    return UploadChunkResponse(
        upload_id=upload_id,
        chunk_number=chunk_number,
        chunk_size=chunk_size,
        etag=etag,
        checksum_md5=chunk_md5,
    )


# =============================================================================
# COMPLETE UPLOAD
# =============================================================================

@router.post(
    "/video/{upload_id}/complete",
    response_model=UploadCompleteResponse,
    summary="Complete chunked upload and trigger processing",
)
async def complete_video_upload(
    upload_id: str,
    parts: List[dict],  # [{PartNumber: int, ETag: str}]
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(),
):
    """
    Complete the multipart upload and trigger:
    1. Video validation
    2. Metadata extraction
    3. Frame extraction (queued as Celery task)
    """
    # Complete multipart upload in MinIO
    media_upload = await upload_service.complete_multipart_upload(
        db=db,
        upload_id=upload_id,
        parts=parts,
    )

    # Queue validation + frame extraction
    validate_task = validate_video_task.delay(str(media_upload.id))
    extract_task = extract_frames_task.apply_async(
        args=[str(media_upload.id)],
        countdown=5,  # Start after validation
        link=validate_task,  # Chain after validation completes
    )

    logger.info(
        "Upload completed, tasks queued",
        upload_id=upload_id,
        media_id=str(media_upload.id),
        validate_task_id=validate_task.id,
        extract_task_id=extract_task.id,
    )

    return UploadCompleteResponse(
        media_upload_id=str(media_upload.id),
        status="processing",
        validate_job_id=validate_task.id,
        extract_job_id=extract_task.id,
        message="Upload complete. Frame extraction queued.",
    )


# =============================================================================
# SIMPLE UPLOAD (Small files)
# =============================================================================

@router.post(
    "/video/simple",
    response_model=MediaUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Simple single-request video upload (< 100MB)",
)
async def simple_video_upload(
    project_id: str = Form(...),
    source_type: VideoSource = Form(VideoSource.UNKNOWN),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(),
):
    """Upload small videos (< 100MB) in a single request."""
    # Read file
    file_data = await file.read()
    file_size = len(file_data)

    # Size check for simple upload
    max_simple_size = 100 * 1024 * 1024  # 100MB
    if file_size > max_simple_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Use chunked upload for files > 100MB",
        )

    # Upload and create record
    media_upload = await upload_service.simple_upload(
        db=db,
        project_id=project_id,
        user_id=current_user.id,
        file_data=file_data,
        filename=file.filename,
        mime_type=file.content_type,
        source_type=source_type,
    )

    # Queue processing
    extract_frames_task.delay(str(media_upload.id))

    return MediaUploadResponse.model_validate(media_upload)


# =============================================================================
# LIST + GET UPLOADS
# =============================================================================

@router.get(
    "/",
    response_model=UploadListResponse,
    summary="List media uploads for a project",
)
async def list_uploads(
    project_id: str = Query(...),
    source_type: Optional[VideoSource] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(),
):
    """List all media uploads for a project with pagination."""
    uploads, total = await upload_service.list_uploads(
        db=db,
        project_id=project_id,
        user_id=current_user.id,
        source_type=source_type,
        page=page,
        page_size=page_size,
    )

    return UploadListResponse(
        items=[MediaUploadResponse.model_validate(u) for u in uploads],
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.get(
    "/{upload_id}",
    response_model=MediaUploadResponse,
    summary="Get media upload details",
)
async def get_upload(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(),
):
    """Get details of a specific media upload."""
    media_upload = await upload_service.get_upload(db, upload_id, current_user)
    if not media_upload:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found")
    return MediaUploadResponse.model_validate(media_upload)


@router.delete(
    "/{upload_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a media upload",
)
async def delete_upload(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(),
):
    """Delete a media upload and its associated files."""
    success = await upload_service.delete_upload(db, upload_id, current_user)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found")
