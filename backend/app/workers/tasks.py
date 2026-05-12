"""
Celery Async Task Workers

Handles all compute-intensive ML and 3D reconstruction tasks:
- Frame extraction from videos
- Video validation
- Object detection pipeline
- Segmentation pipeline
- 3D reconstruction (SfM + MVS)
- Progress estimation
- Delay prediction
- BIM comparison

Architecture:
- Separate queues per priority (high, ml_gpu, reconstruction, analytics)
- Retry with exponential backoff
- Progress tracking via Redis
- GPU-aware routing
"""

import os
import time
import traceback
from datetime import datetime
from typing import Optional

import structlog
from celery import Celery, Task
from celery.signals import task_prerun, task_postrun, task_failure
from celery.utils.log import get_task_logger

from app.core.config import settings

# =============================================================================
# CELERY APP CONFIGURATION
# =============================================================================

celery_app = Celery(
    "rip_workers",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task routing
    task_routes={
        "app.workers.tasks.extract_frames_task": {"queue": "frame_extraction"},
        "app.workers.tasks.validate_video_task": {"queue": "validation"},
        "app.workers.tasks.run_detection_task": {"queue": "ml_gpu"},
        "app.workers.tasks.run_segmentation_task": {"queue": "ml_gpu"},
        "app.workers.tasks.run_sfm_task": {"queue": "reconstruction"},
        "app.workers.tasks.run_mvs_task": {"queue": "reconstruction"},
        "app.workers.tasks.compute_progress_task": {"queue": "analytics"},
        "app.workers.tasks.compute_delay_prediction_task": {"queue": "analytics"},
        "app.workers.tasks.run_bim_comparison_task": {"queue": "analytics"},
    },

    # Queue priorities
    task_queue_max_priority=10,
    task_default_priority=5,

    # Retry configuration
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_max_retries=3,

    # Result expiry (7 days)
    result_expires=604800,

    # Worker configuration
    worker_prefetch_multiplier=1,  # Important for long-running ML tasks
    worker_max_tasks_per_child=50,  # Prevent memory leaks

    # Heartbeat
    broker_heartbeat=10,
    broker_connection_timeout=30,

    # Task time limits
    task_soft_time_limit=3600,   # 1 hour soft limit
    task_time_limit=7200,        # 2 hour hard limit
)

logger = structlog.get_logger(__name__)
task_logger = get_task_logger(__name__)


# =============================================================================
# BASE TASK CLASS
# =============================================================================

class RIPTask(Task):
    """
    Base task class with:
    - Database session management
    - Progress tracking
    - Error reporting
    - GPU resource management
    """
    abstract = True
    _db = None

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure - update job status in DB."""
        task_logger.error(
            f"Task {self.name} failed",
            task_id=task_id,
            error=str(exc),
            traceback=str(einfo),
        )

    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        task_logger.info(f"Task {self.name} completed", task_id=task_id)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        task_logger.warning(
            f"Task {self.name} retrying",
            task_id=task_id,
            error=str(exc),
            retry_count=self.request.retries,
        )

    def update_job_progress(self, job_id: str, progress: float, step: str = ""):
        """Update job progress in database and Redis."""
        self.update_state(
            state="PROGRESS",
            meta={"progress": progress, "step": step, "job_id": job_id},
        )


# =============================================================================
# FRAME EXTRACTION TASK
# =============================================================================

@celery_app.task(
    bind=True,
    base=RIPTask,
    name="app.workers.tasks.extract_frames_task",
    max_retries=3,
    default_retry_delay=60,
    queue="frame_extraction",
)
def extract_frames_task(self, media_upload_id: str) -> dict:
    """
    Extract frames from uploaded video.

    Pipeline:
    1. Download video from MinIO
    2. Validate video integrity
    3. Extract frames at adaptive FPS
    4. Apply blur detection (Laplacian variance)
    5. Apply duplicate detection (SSIM)
    6. Apply motion-aware sampling (optical flow)
    7. Upload extracted frames to MinIO
    8. Update database records

    Args:
        media_upload_id: UUID of the MediaUpload record
    """
    import asyncio
    from app.ml_pipeline.frame_extraction.extractor import VideoFrameExtractor
    from app.services.upload_service import UploadService

    log = logger.bind(task="extract_frames", media_id=media_upload_id)
    log.info("Starting frame extraction")

    start_time = time.time()

    try:
        extractor = VideoFrameExtractor(
            target_fps=settings.FRAME_EXTRACTION_FPS,
            blur_threshold=settings.BLUR_THRESHOLD,
            ssim_threshold=settings.SSIM_THRESHOLD,
            max_frames=settings.MAX_FRAMES_PER_VIDEO,
        )

        # Run extraction (async in sync context)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            extractor.extract(
                media_upload_id=media_upload_id,
                progress_callback=lambda p, s: self.update_job_progress(media_upload_id, p, s),
            )
        )
        loop.close()

        duration = time.time() - start_time
        log.info(
            "Frame extraction completed",
            frames_extracted=result["frames_extracted"],
            frames_kept=result["frames_kept"],
            blur_filtered=result["blur_filtered"],
            duplicate_filtered=result["duplicate_filtered"],
            duration_seconds=round(duration, 2),
        )

        return result

    except Exception as exc:
        log.error("Frame extraction failed", error=str(exc), exc_info=True)
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)


# =============================================================================
# VIDEO VALIDATION TASK
# =============================================================================

@celery_app.task(
    bind=True,
    base=RIPTask,
    name="app.workers.tasks.validate_video_task",
    max_retries=2,
    queue="validation",
)
def validate_video_task(self, media_upload_id: str) -> dict:
    """
    Validate uploaded video file.

    Checks:
    - File integrity (can be opened by OpenCV/FFmpeg)
    - Minimum resolution (720p+ recommended)
    - Minimum duration (> 5 seconds)
    - Codec support
    - Extract video metadata
    """
    import asyncio
    from app.ml_pipeline.frame_extraction.validator import VideoValidator

    log = logger.bind(task="validate_video", media_id=media_upload_id)

    try:
        validator = VideoValidator()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(validator.validate(media_upload_id))
        loop.close()

        log.info("Video validated", valid=result["is_valid"], metadata=result.get("metadata", {}))
        return result

    except Exception as exc:
        log.error("Video validation failed", error=str(exc))
        raise self.retry(exc=exc, countdown=30)


# =============================================================================
# OBJECT DETECTION TASK
# =============================================================================

@celery_app.task(
    bind=True,
    base=RIPTask,
    name="app.workers.tasks.run_detection_task",
    max_retries=2,
    soft_time_limit=3600,
    queue="ml_gpu",
)
def run_detection_task(
    self,
    media_upload_id: str,
    model_name: str = "yolov8",
    confidence_threshold: float = 0.5,
    batch_size: int = 8,
) -> dict:
    """
    Run object detection on all extracted frames.

    Models:
    - yolov8: Fast inference, good for real-time
    - detectron2: More accurate, slower
    - ensemble: Both models merged

    Detects:
    - Workers (with PPE classification)
    - Cranes, excavators, concrete mixers
    - Structural elements (columns, slabs, beams, rebar)
    - Scaffolding and temporary structures
    - Material stockpiles
    - Safety hazards
    """
    import asyncio
    from app.ml_pipeline.detection.pipeline import DetectionPipeline

    log = logger.bind(
        task="object_detection",
        media_id=media_upload_id,
        model=model_name,
    )
    log.info("Starting object detection pipeline")

    try:
        pipeline = DetectionPipeline(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            batch_size=batch_size,
        )

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            pipeline.run(
                media_upload_id=media_upload_id,
                progress_callback=lambda p, s: self.update_job_progress(media_upload_id, p, s),
            )
        )
        loop.close()

        log.info("Detection completed", frames_processed=result["frames_processed"])
        return result

    except Exception as exc:
        log.error("Detection failed", error=str(exc), exc_info=True)
        raise self.retry(exc=exc, countdown=120)


# =============================================================================
# SEGMENTATION TASK
# =============================================================================

@celery_app.task(
    bind=True,
    base=RIPTask,
    name="app.workers.tasks.run_segmentation_task",
    max_retries=2,
    soft_time_limit=3600,
    queue="ml_gpu",
)
def run_segmentation_task(
    self,
    media_upload_id: str,
    model_name: str = "deeplabv3",
    use_sam: bool = False,
) -> dict:
    """
    Run semantic + instance segmentation on extracted frames.

    Semantic classes:
    - concrete, soil, steel, glass, wood, water, sky, vegetation

    Construction-specific classes:
    - foundation, columns, slabs, walls, roofing
    - active_work_zone, hazard_zone, scaffolding
    - machinery_area, material_storage
    """
    import asyncio
    from app.ml_pipeline.segmentation.pipeline import SegmentationPipeline

    log = logger.bind(task="segmentation", media_id=media_upload_id, model=model_name)

    try:
        pipeline = SegmentationPipeline(model_name=model_name, use_sam=use_sam)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            pipeline.run(
                media_upload_id=media_upload_id,
                progress_callback=lambda p, s: self.update_job_progress(media_upload_id, p, s),
            )
        )
        loop.close()

        return result

    except Exception as exc:
        log.error("Segmentation failed", error=str(exc), exc_info=True)
        raise self.retry(exc=exc, countdown=120)


# =============================================================================
# STRUCTURE FROM MOTION TASK
# =============================================================================

@celery_app.task(
    bind=True,
    base=RIPTask,
    name="app.workers.tasks.run_sfm_task",
    max_retries=1,
    soft_time_limit=7200,  # 2 hours
    time_limit=10800,       # 3 hours hard limit
    queue="reconstruction",
)
def run_sfm_task(
    self,
    project_id: str,
    media_upload_ids: list,
    quality: str = "high",
) -> dict:
    """
    Run Structure from Motion (SfM) using COLMAP.

    Pipeline:
    1. Feature extraction (SIFT/SuperPoint)
    2. Feature matching (exhaustive/sequential/vocabulary tree)
    3. Geometric verification (RANSAC)
    4. Incremental reconstruction
    5. Bundle adjustment
    6. Pose graph optimization

    Args:
        project_id: Construction project ID
        media_upload_ids: List of media IDs to reconstruct from
        quality: Reconstruction quality (low/medium/high/extreme)
    """
    import asyncio
    from app.ml_pipeline.reconstruction.sfm.pipeline import SfMPipeline

    log = logger.bind(
        task="sfm",
        project_id=project_id,
        num_videos=len(media_upload_ids),
        quality=quality,
    )
    log.info("Starting SfM reconstruction")

    try:
        pipeline = SfMPipeline(
            project_id=project_id,
            quality=quality,
            colmap_binary=settings.COLMAP_BINARY,
        )

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            pipeline.run(
                media_upload_ids=media_upload_ids,
                progress_callback=lambda p, s: self.update_job_progress(project_id, p, s),
            )
        )
        loop.close()

        log.info(
            "SfM completed",
            images_registered=result.get("num_images_registered"),
            sparse_points=result.get("num_sparse_points"),
        )
        return result

    except Exception as exc:
        log.error("SfM failed", error=str(exc), exc_info=True)
        raise self.retry(exc=exc, countdown=300)


# =============================================================================
# MULTI-VIEW STEREO TASK
# =============================================================================

@celery_app.task(
    bind=True,
    base=RIPTask,
    name="app.workers.tasks.run_mvs_task",
    max_retries=1,
    soft_time_limit=10800,  # 3 hours
    time_limit=14400,
    queue="reconstruction",
)
def run_mvs_task(
    self,
    reconstruction_id: str,
    quality: str = "high",
) -> dict:
    """
    Run Multi-View Stereo (MVS) for dense reconstruction.

    Pipeline:
    1. Depth map computation (PatchMatch stereo)
    2. Depth map fusion
    3. Dense point cloud generation
    4. Surface reconstruction (Poisson/Delaunay)
    5. Mesh simplification
    6. Texture mapping

    Args:
        reconstruction_id: UUID of the Reconstruction3D record (after SfM)
        quality: MVS quality setting
    """
    import asyncio
    from app.ml_pipeline.reconstruction.mvs.pipeline import MVSPipeline

    log = logger.bind(task="mvs", reconstruction_id=reconstruction_id)
    log.info("Starting MVS dense reconstruction")

    try:
        pipeline = MVSPipeline(quality=quality)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            pipeline.run(
                reconstruction_id=reconstruction_id,
                progress_callback=lambda p, s: self.update_job_progress(reconstruction_id, p, s),
            )
        )
        loop.close()

        log.info("MVS completed", dense_points=result.get("num_dense_points"))
        return result

    except Exception as exc:
        log.error("MVS failed", error=str(exc), exc_info=True)
        raise self.retry(exc=exc, countdown=600)


# =============================================================================
# PROGRESS ESTIMATION TASK
# =============================================================================

@celery_app.task(
    bind=True,
    base=RIPTask,
    name="app.workers.tasks.compute_progress_task",
    max_retries=3,
    queue="analytics",
)
def compute_progress_task(
    self,
    project_id: str,
    reconstruction_id: Optional[str] = None,
) -> dict:
    """
    Compute construction progress from 3D reconstruction + detections.

    Algorithm:
    1. Load dense point cloud from latest reconstruction
    2. Segment point cloud by structural component type
    3. Compare with BIM model volumes (if available)
    4. Aggregate detection statistics from all frames
    5. Apply progress estimation model (XGBoost)
    6. Compute spatial completion heatmap
    7. Store ProgressSnapshot to database
    """
    import asyncio
    from app.ml_pipeline.analytics.progress.estimator import ProgressEstimator

    log = logger.bind(task="progress_estimation", project_id=project_id)

    try:
        estimator = ProgressEstimator()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            estimator.compute(project_id=project_id, reconstruction_id=reconstruction_id)
        )
        loop.close()

        log.info("Progress computed", overall=result.get("overall_progress_percent"))
        return result

    except Exception as exc:
        log.error("Progress computation failed", error=str(exc), exc_info=True)
        raise self.retry(exc=exc, countdown=60)


# =============================================================================
# DELAY PREDICTION TASK
# =============================================================================

@celery_app.task(
    bind=True,
    base=RIPTask,
    name="app.workers.tasks.compute_delay_prediction_task",
    max_retries=3,
    queue="analytics",
)
def compute_delay_prediction_task(self, project_id: str) -> dict:
    """
    Generate ML delay predictions using XGBoost + LSTM ensemble.

    Features used:
    - Progress velocity (actual vs planned)
    - Weather data (external API)
    - Equipment utilization rate
    - Worker activity density
    - Material availability signals
    - Historical project similarity
    - Day of week / seasonal patterns
    """
    import asyncio
    from app.ml_pipeline.analytics.delay_prediction.predictor import DelayPredictor

    log = logger.bind(task="delay_prediction", project_id=project_id)

    try:
        predictor = DelayPredictor()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(predictor.predict(project_id=project_id))
        loop.close()

        log.info("Delay predicted", delay_days=result.get("predicted_delay_days"))
        return result

    except Exception as exc:
        log.error("Delay prediction failed", error=str(exc))
        raise self.retry(exc=exc, countdown=60)


# =============================================================================
# BIM COMPARISON TASK
# =============================================================================

@celery_app.task(
    bind=True,
    base=RIPTask,
    name="app.workers.tasks.run_bim_comparison_task",
    max_retries=2,
    queue="analytics",
)
def run_bim_comparison_task(
    self,
    project_id: str,
    bim_model_id: str,
    reconstruction_id: str,
) -> dict:
    """
    Compare BIM model against actual 3D reconstruction.

    Pipeline:
    1. Load IFC model elements
    2. Load actual point cloud
    3. Align coordinate systems (ICP)
    4. Per-element completion estimation
    5. Spatial deviation analysis
    6. Schedule mismatch analysis
    7. Generate comparison report
    """
    import asyncio
    from app.ml_pipeline.analytics.bim.comparator import BIMComparator

    log = logger.bind(task="bim_comparison", project_id=project_id, bim_id=bim_model_id)

    try:
        comparator = BIMComparator()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            comparator.compare(
                project_id=project_id,
                bim_model_id=bim_model_id,
                reconstruction_id=reconstruction_id,
            )
        )
        loop.close()

        log.info("BIM comparison completed", completion=result.get("overall_completion_percent"))
        return result

    except Exception as exc:
        log.error("BIM comparison failed", error=str(exc))
        raise self.retry(exc=exc, countdown=120)


# =============================================================================
# FULL PIPELINE ORCHESTRATION
# =============================================================================

@celery_app.task(
    bind=True,
    base=RIPTask,
    name="app.workers.tasks.run_full_pipeline",
    queue="high",
)
def run_full_pipeline(
    self,
    project_id: str,
    media_upload_ids: list,
    reconstruction_quality: str = "high",
) -> dict:
    """
    Orchestrate complete end-to-end pipeline:
    1. Frame extraction (parallel per video)
    2. Object detection (GPU)
    3. Semantic segmentation (GPU)
    4. SfM reconstruction
    5. MVS dense reconstruction
    6. Progress estimation
    7. Delay prediction
    8. BIM comparison (if BIM available)
    """
    from celery import group, chain

    log = logger.bind(task="full_pipeline", project_id=project_id)
    log.info("Starting full processing pipeline")

    # Stage 1: Parallel frame extraction + validation
    extraction_group = group(
        extract_frames_task.s(upload_id) for upload_id in media_upload_ids
    )

    # Stage 2: Parallel detection + segmentation
    detection_group = group(
        run_detection_task.s(upload_id) for upload_id in media_upload_ids
    )
    segmentation_group = group(
        run_segmentation_task.s(upload_id) for upload_id in media_upload_ids
    )

    # Full pipeline chain
    pipeline = chain(
        extraction_group,
        (detection_group | segmentation_group),
        run_sfm_task.s(project_id=project_id, media_upload_ids=media_upload_ids),
        run_mvs_task.s(),
        compute_progress_task.s(project_id=project_id),
        compute_delay_prediction_task.s(project_id=project_id),
    )

    result = pipeline.apply_async()
    log.info("Full pipeline dispatched", chain_id=result.id)

    return {"pipeline_id": result.id, "project_id": project_id}
