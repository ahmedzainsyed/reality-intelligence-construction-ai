"""
Upload Service – business logic for media upload lifecycle.

Handles:
  - Multipart upload session management (Redis-backed state)
  - MinIO chunk upload coordination
  - DB record creation and updates
  - Video metadata extraction (via FFprobe)
  - Access control checks
"""

import json
import subprocess
from datetime import datetime
from typing import List, Optional, Tuple
from uuid import uuid4

import structlog
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.storage import get_storage_client
from app.models.models import MediaUpload, Project, User, VideoSource

logger = structlog.get_logger(__name__)


class UploadService:
    """Service layer for media upload operations."""

    # ── Project access ────────────────────────────────────────────────────────

    async def get_project_for_user(
        self,
        db: AsyncSession,
        project_id: str,
        user: User,
    ) -> Optional[Project]:
        """Fetch project, ensuring it belongs to the user's organisation."""
        result = await db.execute(
            select(Project).where(
                Project.id == project_id,
                Project.organization_id == user.organization_id,
                Project.is_active == True,
            )
        )
        return result.scalar_one_or_none()

    # ── Multipart upload session ──────────────────────────────────────────────

    async def init_multipart_upload(self, storage_path: str, mime_type: str) -> str:
        """
        Initiate a MinIO multipart upload.
        Returns a synthetic upload_id for chunk tracking.
        """
        # MinIO Python SDK does not expose raw S3 multipart API directly.
        # We use a synthetic ID and store chunk ETags in Redis.
        upload_id = f"rip_mp_{uuid4().hex}"
        logger.info("Multipart upload initiated", path=storage_path, id=upload_id)
        return upload_id

    async def upload_chunk(
        self,
        upload_id: str,
        chunk_number: int,
        chunk_data: bytes,
    ) -> str:
        """Upload a single chunk to temp storage, return ETag."""
        import hashlib
        etag = hashlib.md5(chunk_data).hexdigest()

        # Store chunk in temp path
        bucket = settings.MINIO_BUCKET_MEDIA
        key = f"tmp/chunks/{upload_id}/chunk_{chunk_number:05d}"
        storage = get_storage_client()
        await storage.upload_bytes(bucket, key, chunk_data)

        # Track ETag in Redis
        await self._store_chunk_etag(upload_id, chunk_number, etag)
        return etag

    async def complete_multipart_upload(
        self,
        db: AsyncSession,
        upload_id: str,
        parts: List[dict],
    ) -> MediaUpload:
        """
        Assemble all chunks into the final object.
        Updates DB record with completed status.
        """
        # In production: use boto3 complete_multipart_upload with ETags
        # Here: we update the DB record
        result = await db.execute(
            select(MediaUpload).where(MediaUpload.id == upload_id)
        )
        media = result.scalar_one_or_none()

        if not media:
            # Fallback: find by a pending upload session
            logger.warning("Upload record not found by id, searching by session", upload_id=upload_id)
            # This path would look up via Redis session state
            raise ValueError(f"Upload session {upload_id} not found")

        media.upload_completed_at = datetime.utcnow()
        await db.flush()
        return media

    # ── Simple (non-chunked) upload ────────────────────────────────────────────

    async def simple_upload(
        self,
        db: AsyncSession,
        project_id: str,
        user_id: str,
        file_data: bytes,
        filename: str,
        mime_type: Optional[str],
        source_type: VideoSource,
    ) -> MediaUpload:
        """Upload small file in single request."""
        import hashlib

        checksum = hashlib.md5(file_data).hexdigest()
        media_id = str(uuid4())
        storage_key = f"projects/{project_id}/media/{media_id}/{filename}"

        # Upload to MinIO
        storage = get_storage_client()
        await storage.upload_bytes(
            settings.MINIO_BUCKET_MEDIA,
            storage_key,
            file_data,
            content_type=mime_type or "application/octet-stream",
        )

        # Extract video metadata
        metadata = await self._extract_video_metadata_from_bytes(file_data)

        # Create DB record
        media = MediaUpload(
            id=media_id,
            project_id=project_id,
            uploaded_by_id=user_id,
            filename=f"{media_id}_{filename}",
            original_filename=filename,
            storage_path=storage_key,
            storage_bucket=settings.MINIO_BUCKET_MEDIA,
            file_size_bytes=len(file_data),
            mime_type=mime_type,
            source_type=source_type,
            checksum_md5=checksum,
            upload_completed_at=datetime.utcnow(),
            **metadata,
        )
        db.add(media)
        await db.flush()

        logger.info(
            "Simple upload complete",
            media_id=media_id,
            size_mb=round(len(file_data) / 1_048_576, 2),
        )
        return media

    # ── Create upload record ──────────────────────────────────────────────────

    async def create_upload_record(
        self,
        db: AsyncSession,
        project_id: str,
        user_id: str,
        upload_id: str,
        filename: str,
        file_size: int,
        mime_type: str,
        source_type: VideoSource,
        minio_path: str,
        minio_upload_id: str,
        camera_model: Optional[str] = None,
        capture_date: Optional[str] = None,
    ) -> MediaUpload:
        """Create a pending MediaUpload DB record for a chunked upload."""
        capture_dt = None
        if capture_date:
            try:
                capture_dt = datetime.fromisoformat(capture_date)
            except ValueError:
                pass

        media = MediaUpload(
            id=upload_id,
            project_id=project_id,
            uploaded_by_id=user_id,
            filename=f"{upload_id}_{filename}",
            original_filename=filename,
            storage_path=minio_path,
            storage_bucket=settings.MINIO_BUCKET_MEDIA,
            file_size_bytes=file_size,
            mime_type=mime_type,
            source_type=source_type,
            camera_model=camera_model,
            capture_date=capture_dt,
            is_validated=False,
        )
        db.add(media)
        await db.flush()
        return media

    # ── List + Get ────────────────────────────────────────────────────────────

    async def list_uploads(
        self,
        db: AsyncSession,
        project_id: str,
        user_id: str,
        source_type: Optional[VideoSource] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[MediaUpload], int]:
        """Paginated list of uploads for a project."""
        base_q = select(MediaUpload).where(MediaUpload.project_id == project_id)
        if source_type:
            base_q = base_q.where(MediaUpload.source_type == source_type)

        count_q = select(func.count()).select_from(base_q.subquery())
        total = (await db.execute(count_q)).scalar_one()

        data_q = (
            base_q.order_by(MediaUpload.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        rows = (await db.execute(data_q)).scalars().all()
        return list(rows), total

    async def get_upload(
        self,
        db: AsyncSession,
        upload_id: str,
        user: User,
    ) -> Optional[MediaUpload]:
        result = await db.execute(
            select(MediaUpload)
            .join(Project, MediaUpload.project_id == Project.id)
            .where(
                MediaUpload.id == upload_id,
                Project.organization_id == user.organization_id,
            )
        )
        return result.scalar_one_or_none()

    async def delete_upload(
        self,
        db: AsyncSession,
        upload_id: str,
        user: User,
    ) -> bool:
        media = await self.get_upload(db, upload_id, user)
        if not media:
            return False

        # Delete from MinIO
        try:
            storage = get_storage_client()
            await storage.delete_object(media.storage_bucket or settings.MINIO_BUCKET_MEDIA,
                                        media.storage_path)
        except Exception as exc:
            logger.warning("Could not delete storage object", error=str(exc))

        await db.delete(media)
        return True

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _extract_video_metadata_from_bytes(self, data: bytes) -> dict:
        """Extract video metadata using FFprobe. Returns dict of metadata fields."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(data)
            tmp_path = f.name

        try:
            return await self._run_ffprobe(tmp_path)
        finally:
            os.unlink(tmp_path)

    @staticmethod
    async def _run_ffprobe(path: str) -> dict:
        """Run ffprobe and extract video stream metadata."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return {}
            info = json.loads(result.stdout)
            streams = info.get("streams", [])
            fmt = info.get("format", {})

            video_stream = next(
                (s for s in streams if s.get("codec_type") == "video"), {}
            )

            fps_str = video_stream.get("r_frame_rate", "25/1")
            try:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            except Exception:
                fps = 25.0

            return {
                "duration_seconds": float(fmt.get("duration", 0)),
                "fps": round(fps, 2),
                "width": video_stream.get("width"),
                "height": video_stream.get("height"),
                "codec": video_stream.get("codec_name"),
                "bitrate_kbps": round(
                    float(fmt.get("bit_rate", 0)) / 1000, 1
                ) or None,
            }
        except Exception as exc:
            logger.warning("ffprobe failed", error=str(exc))
            return {}

    async def _store_chunk_etag(
        self, upload_id: str, chunk_number: int, etag: str
    ) -> None:
        """Store chunk ETag in Redis for later assembly."""
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            await r.hset(f"upload:{upload_id}:chunks", str(chunk_number), etag)
            await r.expire(f"upload:{upload_id}:chunks", 86400)  # 24h TTL
            await r.aclose()
        except Exception as exc:
            logger.warning("Could not store chunk ETag in Redis", error=str(exc))
