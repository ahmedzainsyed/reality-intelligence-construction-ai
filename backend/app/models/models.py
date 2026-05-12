"""
SQLAlchemy ORM Models

Defines all database models for the Reality Intelligence Platform:
- Users and Organizations
- Construction Projects and Sites
- Media Assets (videos, images)
- Processing Jobs
- Reconstruction Results
- Detection Results
- Analytics and Progress Snapshots
- BIM Models
"""

import uuid
from datetime import datetime
from typing import Optional, List
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text,
    ForeignKey, JSON, Enum, BigInteger, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


def generate_uuid() -> str:
    return str(uuid.uuid4())


# =============================================================================
# ENUMERATIONS
# =============================================================================

class UserRole(str, PyEnum):
    ADMIN = "admin"
    PROJECT_MANAGER = "project_manager"
    SITE_ENGINEER = "site_engineer"
    VIEWER = "viewer"


class JobStatus(str, PyEnum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoSource(str, PyEnum):
    DRONE = "drone"
    CCTV = "cctv"
    MOBILE = "mobile"
    PANORAMIC_360 = "360_panoramic"
    UNKNOWN = "unknown"


class ReconstructionQuality(str, PyEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class ProgressStatus(str, PyEnum):
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    DELAYED = "delayed"
    CRITICAL = "critical"


# =============================================================================
# ORGANIZATION + USER MODELS
# =============================================================================

class Organization(Base):
    """Multi-tenant organization model."""
    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    subscription_tier = Column(String(50), default="starter")  # starter | pro | enterprise
    max_sites = Column(Integer, default=5)
    max_storage_gb = Column(Float, default=100.0)
    is_active = Column(Boolean, default=True)
    settings = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    users = relationship("User", back_populates="organization")
    projects = relationship("Project", back_populates="organization")


class User(Base):
    """Platform user model with RBAC."""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    organization_id = Column(UUID(as_uuid=False), ForeignKey("organizations.id"), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False)
    full_name = Column(String(255))
    hashed_password = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    last_login_at = Column(DateTime(timezone=True))
    avatar_url = Column(String(500))
    preferences = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    organization = relationship("Organization", back_populates="users")
    created_projects = relationship("Project", back_populates="created_by")

    __table_args__ = (
        Index("idx_users_org_email", "organization_id", "email"),
    )


# =============================================================================
# PROJECT + SITE MODELS
# =============================================================================

class Project(Base):
    """Construction project model."""
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    organization_id = Column(UUID(as_uuid=False), ForeignKey("organizations.id"), nullable=False)
    created_by_id = Column(UUID(as_uuid=False), ForeignKey("users.id"))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    location = Column(String(500))
    latitude = Column(Float)
    longitude = Column(Float)
    start_date = Column(DateTime(timezone=True))
    planned_end_date = Column(DateTime(timezone=True))
    actual_end_date = Column(DateTime(timezone=True))
    project_type = Column(String(100))  # residential | commercial | infrastructure
    total_area_sqm = Column(Float)
    total_floors = Column(Integer)
    budget_usd = Column(BigInteger)
    status = Column(String(50), default="active")
    thumbnail_url = Column(String(500))
    metadata = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    organization = relationship("Organization", back_populates="projects")
    created_by = relationship("User", back_populates="created_projects")
    media_uploads = relationship("MediaUpload", back_populates="project")
    processing_jobs = relationship("ProcessingJob", back_populates="project")
    reconstructions = relationship("Reconstruction3D", back_populates="project")
    progress_snapshots = relationship("ProgressSnapshot", back_populates="project")
    bim_models = relationship("BIMModel", back_populates="project")

    __table_args__ = (
        Index("idx_projects_org_status", "organization_id", "status"),
    )


# =============================================================================
# MEDIA MODELS
# =============================================================================

class MediaUpload(Base):
    """Uploaded video/image asset."""
    __tablename__ = "media_uploads"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.id"), nullable=False)
    uploaded_by_id = Column(UUID(as_uuid=False), ForeignKey("users.id"))
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500))
    storage_path = Column(String(1000), nullable=False)  # S3/MinIO path
    storage_bucket = Column(String(255))
    file_size_bytes = Column(BigInteger)
    mime_type = Column(String(100))
    source_type = Column(Enum(VideoSource), default=VideoSource.UNKNOWN)

    # Video metadata
    duration_seconds = Column(Float)
    fps = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    codec = Column(String(50))
    bitrate_kbps = Column(Float)

    # Camera/drone metadata
    camera_model = Column(String(255))
    camera_serial = Column(String(255))
    focal_length_mm = Column(Float)
    sensor_width_mm = Column(Float)
    gps_coordinates = Column(JSONB)  # List of GPS waypoints

    # Processing status
    is_validated = Column(Boolean, default=False)
    validation_errors = Column(JSONB, default=[])
    checksum_md5 = Column(String(32))
    frame_count_extracted = Column(Integer, default=0)

    capture_date = Column(DateTime(timezone=True))
    upload_completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    project = relationship("Project", back_populates="media_uploads")
    extracted_frames = relationship("ExtractedFrame", back_populates="media_upload")
    processing_jobs = relationship("ProcessingJob", back_populates="media_upload")


class ExtractedFrame(Base):
    """Individual frame extracted from a video."""
    __tablename__ = "extracted_frames"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    media_upload_id = Column(UUID(as_uuid=False), ForeignKey("media_uploads.id"), nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp_seconds = Column(Float, nullable=False)
    storage_path = Column(String(1000), nullable=False)
    file_size_bytes = Column(Integer)

    # Quality metrics
    blur_score = Column(Float)          # Laplacian variance
    is_blurry = Column(Boolean, default=False)
    is_duplicate = Column(Boolean, default=False)
    ssim_with_prev = Column(Float)
    optical_flow_magnitude = Column(Float)

    # Spatial info
    width = Column(Integer)
    height = Column(Integer)
    gps_lat = Column(Float)
    gps_lon = Column(Float)
    gps_alt = Column(Float)
    camera_pitch = Column(Float)
    camera_yaw = Column(Float)
    camera_roll = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    media_upload = relationship("MediaUpload", back_populates="extracted_frames")
    detections = relationship("DetectionResult", back_populates="frame")
    segmentation = relationship("SegmentationResult", back_populates="frame", uselist=False)

    __table_args__ = (
        Index("idx_frames_upload_number", "media_upload_id", "frame_number"),
    )


# =============================================================================
# PROCESSING JOB MODELS
# =============================================================================

class ProcessingJob(Base):
    """Async processing job (Celery task tracker)."""
    __tablename__ = "processing_jobs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.id"), nullable=False)
    media_upload_id = Column(UUID(as_uuid=False), ForeignKey("media_uploads.id"))
    celery_task_id = Column(String(255), index=True)
    job_type = Column(String(100), nullable=False)  # frame_extraction | detection | reconstruction | analytics
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    priority = Column(Integer, default=5)  # 1-10, higher = more urgent
    progress_percent = Column(Float, default=0.0)
    current_step = Column(String(255))
    total_steps = Column(Integer)
    result_data = Column(JSONB)
    error_message = Column(Text)
    error_traceback = Column(Text)
    config = Column(JSONB, default={})
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Float)
    worker_id = Column(String(255))
    gpu_used = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    project = relationship("Project", back_populates="processing_jobs")
    media_upload = relationship("MediaUpload", back_populates="processing_jobs")

    __table_args__ = (
        Index("idx_jobs_project_status", "project_id", "status"),
        Index("idx_jobs_celery_id", "celery_task_id"),
    )


# =============================================================================
# ML RESULTS MODELS
# =============================================================================

class DetectionResult(Base):
    """Object detection results per frame."""
    __tablename__ = "detection_results"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    frame_id = Column(UUID(as_uuid=False), ForeignKey("extracted_frames.id"), nullable=False)
    model_name = Column(String(100))
    model_version = Column(String(50))
    inference_time_ms = Column(Float)

    # Detections stored as JSONB array of:
    # {class_id, class_name, confidence, bbox: [x1,y1,x2,y2], mask_rle, track_id}
    detections = Column(JSONB, default=[])
    detection_count = Column(Integer, default=0)

    # Aggregated counts by class
    worker_count = Column(Integer, default=0)
    crane_count = Column(Integer, default=0)
    excavator_count = Column(Integer, default=0)
    vehicle_count = Column(Integer, default=0)
    ppe_count = Column(Integer, default=0)
    structural_element_count = Column(Integer, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    frame = relationship("ExtractedFrame", back_populates="detections")

    __table_args__ = (
        Index("idx_detections_frame", "frame_id"),
    )


class SegmentationResult(Base):
    """Semantic + instance segmentation results per frame."""
    __tablename__ = "segmentation_results"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    frame_id = Column(UUID(as_uuid=False), ForeignKey("extracted_frames.id"), nullable=False, unique=True)
    model_name = Column(String(100))
    inference_time_ms = Column(Float)

    # Pixel area proportions per semantic class (0.0 - 1.0)
    class_coverage = Column(JSONB, default={})  # {concrete: 0.3, soil: 0.2, sky: 0.1, ...}

    # Instance segmentation masks stored as compressed RLE
    instances = Column(JSONB, default=[])
    instance_count = Column(Integer, default=0)

    # Hazard zones
    hazard_zone_coverage = Column(Float, default=0.0)
    active_work_zone_coverage = Column(Float, default=0.0)

    # Segmentation mask storage path
    semantic_mask_path = Column(String(1000))
    instance_mask_path = Column(String(1000))

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    frame = relationship("ExtractedFrame", back_populates="segmentation")


# =============================================================================
# 3D RECONSTRUCTION MODELS
# =============================================================================

class Reconstruction3D(Base):
    """3D reconstruction result for a project site."""
    __tablename__ = "reconstructions_3d"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.id"), nullable=False)
    processing_job_id = Column(UUID(as_uuid=False), ForeignKey("processing_jobs.id"))
    name = Column(String(255))
    quality = Column(Enum(ReconstructionQuality), default=ReconstructionQuality.HIGH)

    # SfM results
    sfm_status = Column(String(50), default="pending")
    num_images_registered = Column(Integer, default=0)
    num_images_total = Column(Integer, default=0)
    num_sparse_points = Column(Integer, default=0)
    num_cameras = Column(Integer, default=0)
    mean_reprojection_error = Column(Float)
    sfm_workspace_path = Column(String(1000))

    # MVS results
    mvs_status = Column(String(50), default="pending")
    num_dense_points = Column(Integer, default=0)
    point_cloud_path = Column(String(1000))
    point_cloud_size_mb = Column(Float)
    mesh_path = Column(String(1000))
    textured_mesh_path = Column(String(1000))

    # Bounding box
    bbox_min_x = Column(Float)
    bbox_min_y = Column(Float)
    bbox_min_z = Column(Float)
    bbox_max_x = Column(Float)
    bbox_max_y = Column(Float)
    bbox_max_z = Column(Float)

    # Processing performance
    sfm_duration_seconds = Column(Float)
    mvs_duration_seconds = Column(Float)
    gpu_memory_gb = Column(Float)

    capture_date = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    project = relationship("Project", back_populates="reconstructions")
    camera_poses = relationship("CameraPose", back_populates="reconstruction")


class CameraPose(Base):
    """Individual camera pose from SfM."""
    __tablename__ = "camera_poses"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    reconstruction_id = Column(UUID(as_uuid=False), ForeignKey("reconstructions_3d.id"), nullable=False)
    frame_id = Column(UUID(as_uuid=False), ForeignKey("extracted_frames.id"))
    image_name = Column(String(500))
    camera_id = Column(Integer)

    # Rotation (quaternion)
    qw = Column(Float)
    qx = Column(Float)
    qy = Column(Float)
    qz = Column(Float)

    # Translation
    tx = Column(Float)
    ty = Column(Float)
    tz = Column(Float)

    # Camera intrinsics
    fx = Column(Float)
    fy = Column(Float)
    cx = Column(Float)
    cy = Column(Float)
    k1 = Column(Float)
    k2 = Column(Float)

    reprojection_error = Column(Float)

    reconstruction = relationship("Reconstruction3D", back_populates="camera_poses")


# =============================================================================
# ANALYTICS MODELS
# =============================================================================

class ProgressSnapshot(Base):
    """Construction progress snapshot at a point in time."""
    __tablename__ = "progress_snapshots"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.id"), nullable=False)
    reconstruction_id = Column(UUID(as_uuid=False), ForeignKey("reconstructions_3d.id"))
    snapshot_date = Column(DateTime(timezone=True), nullable=False)

    # Overall progress
    overall_progress_percent = Column(Float, default=0.0)
    planned_progress_percent = Column(Float, default=0.0)
    progress_variance_percent = Column(Float, default=0.0)
    status = Column(Enum(ProgressStatus), default=ProgressStatus.ON_TRACK)

    # Structural progress
    foundation_completion = Column(Float, default=0.0)
    structural_frame_completion = Column(Float, default=0.0)
    slab_completion = Column(Float, default=0.0)
    walls_completion = Column(Float, default=0.0)
    mep_completion = Column(Float, default=0.0)
    finishing_completion = Column(Float, default=0.0)

    # Activity metrics
    active_workers = Column(Integer, default=0)
    active_equipment = Column(Integer, default=0)
    material_utilization_percent = Column(Float)
    safety_violations_detected = Column(Integer, default=0)

    # Delay prediction
    predicted_delay_days = Column(Float, default=0.0)
    delay_probability = Column(Float, default=0.0)
    risk_level = Column(String(50), default="low")

    # Volume metrics (from point cloud)
    concrete_volume_m3 = Column(Float)
    excavation_volume_m3 = Column(Float)
    rebar_linear_meters = Column(Float)

    # Heatmap data
    activity_heatmap = Column(JSONB)  # Grid of activity scores
    risk_heatmap = Column(JSONB)

    computed_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    project = relationship("Project", back_populates="progress_snapshots")

    __table_args__ = (
        Index("idx_progress_project_date", "project_id", "snapshot_date"),
    )


class DelayPrediction(Base):
    """ML-generated delay prediction."""
    __tablename__ = "delay_predictions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.id"), nullable=False)
    model_version = Column(String(50))
    predicted_delay_days = Column(Float, nullable=False)
    delay_probability = Column(Float, nullable=False)
    confidence_interval_low = Column(Float)
    confidence_interval_high = Column(Float)

    # Risk factors
    risk_factors = Column(JSONB, default=[])  # [{factor: "weather", impact: 0.3, ...}]
    top_risk_factor = Column(String(255))

    # Feature importance
    feature_importances = Column(JSONB, default={})

    # Prediction context
    prediction_date = Column(DateTime(timezone=True), nullable=False)
    target_completion_date = Column(DateTime(timezone=True))
    revised_completion_date = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_delay_project_date", "project_id", "prediction_date"),
    )


# =============================================================================
# BIM MODELS
# =============================================================================

class BIMModel(Base):
    """IFC/BIM model uploaded for comparison."""
    __tablename__ = "bim_models"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    ifc_version = Column(String(20))
    storage_path = Column(String(1000))
    file_size_mb = Column(Float)
    element_count = Column(Integer)
    discipline = Column(String(100))  # architectural | structural | mep
    version = Column(String(50))
    is_active = Column(Boolean, default=True)
    metadata = Column(JSONB, default={})
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("Project", back_populates="bim_models")
    comparisons = relationship("BIMComparison", back_populates="bim_model")


class BIMComparison(Base):
    """BIM vs actual reconstruction comparison result."""
    __tablename__ = "bim_comparisons"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.id"), nullable=False)
    bim_model_id = Column(UUID(as_uuid=False), ForeignKey("bim_models.id"), nullable=False)
    reconstruction_id = Column(UUID(as_uuid=False), ForeignKey("reconstructions_3d.id"), nullable=False)

    overall_completion_percent = Column(Float)
    mean_spatial_deviation_mm = Column(Float)
    max_spatial_deviation_mm = Column(Float)

    # Per-element comparison
    element_comparisons = Column(JSONB, default=[])
    missing_elements = Column(JSONB, default=[])
    extra_elements = Column(JSONB, default=[])

    # Schedule comparison
    schedule_deviation_days = Column(Float)
    critical_path_status = Column(String(100))

    comparison_metadata = Column(JSONB, default={})
    computed_at = Column(DateTime(timezone=True), server_default=func.now())

    bim_model = relationship("BIMModel", back_populates="comparisons")
