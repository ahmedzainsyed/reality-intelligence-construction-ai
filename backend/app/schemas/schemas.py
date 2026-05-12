"""
Pydantic v2 request/response schemas.

Covers:
  - Auth (Token, UserCreate, UserResponse)
  - Uploads (init, chunk, complete, list)
  - Analytics (progress, delay, KPI, timeline, heatmap, alerts)
  - Processing (job status)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


# =============================================================================
# AUTH SCHEMAS
# =============================================================================

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Seconds until expiry")
    refresh_token: Optional[str] = None


class TokenPayload(BaseModel):
    sub: str
    org: Optional[str] = None
    exp: int
    type: str = "access"


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=2, max_length=255)
    organization_name: Optional[str] = Field(None, min_length=2, max_length=255)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    role: str
    organization_id: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# =============================================================================
# UPLOAD SCHEMAS
# =============================================================================

class UploadInitResponse(BaseModel):
    upload_id: str
    minio_upload_id: str
    chunk_size: int
    total_chunks: int
    storage_path: str
    expires_in_hours: int = 24


class UploadChunkResponse(BaseModel):
    upload_id: str
    chunk_number: int
    chunk_size: int
    etag: str
    checksum_md5: str


class UploadCompleteResponse(BaseModel):
    media_upload_id: str
    status: str
    validate_job_id: Optional[str] = None
    extract_job_id: Optional[str] = None
    message: str


class MediaUploadResponse(BaseModel):
    id: str
    project_id: str
    filename: str
    original_filename: Optional[str]
    source_type: str
    file_size_bytes: Optional[int]
    mime_type: Optional[str]
    duration_seconds: Optional[float]
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    camera_model: Optional[str]
    is_validated: bool
    frame_count_extracted: int
    capture_date: Optional[datetime]
    created_at: datetime

    model_config = {"from_attributes": True}


class UploadListResponse(BaseModel):
    items: List[MediaUploadResponse]
    total: int
    page: int
    page_size: int
    pages: int


# =============================================================================
# ANALYTICS SCHEMAS
# =============================================================================

class ProgressBreakdown(BaseModel):
    foundation_completion: float = 0.0
    structural_frame_completion: float = 0.0
    slab_completion: float = 0.0
    walls_completion: float = 0.0
    mep_completion: float = 0.0
    finishing_completion: float = 0.0


class ProgressResponse(BaseModel):
    id: str
    project_id: str
    snapshot_date: datetime
    overall_progress_percent: float
    planned_progress_percent: float
    progress_variance_percent: float
    status: str
    active_workers: int
    active_equipment: int
    safety_violations_detected: int
    predicted_delay_days: float
    delay_probability: float
    risk_level: str
    breakdown: Optional[ProgressBreakdown] = None

    model_config = {"from_attributes": True}


class RiskFactor(BaseModel):
    factor: str
    feature: str
    importance: float
    current_value: float


class DelayPredictionResponse(BaseModel):
    id: Optional[str] = None
    project_id: str
    predicted_delay_days: float
    delay_probability: float
    confidence_interval_low: float
    confidence_interval_high: float
    risk_level: str
    top_risk_factor: Optional[str]
    risk_factors: List[RiskFactor] = []
    feature_importances: Dict[str, float] = {}
    prediction_date: datetime
    revised_completion_date: Optional[datetime]
    model_version: Optional[str]

    model_config = {"from_attributes": True}


class KPIMetric(BaseModel):
    value: float
    unit: str
    delta_pct: Optional[float] = None
    trend: str = "stable"   # up | down | stable


class KPIDashboardResponse(BaseModel):
    project_id: str
    period_days: int
    computed_at: datetime
    schedule_performance_index: KPIMetric
    overall_progress: KPIMetric
    worker_productivity: KPIMetric
    equipment_utilisation: KPIMetric
    safety_score: KPIMetric
    material_burn_rate: KPIMetric
    cost_performance_index: Optional[KPIMetric] = None


class TimelineSnapshot(BaseModel):
    snapshot_date: datetime
    overall_progress_percent: float
    planned_progress_percent: float
    progress_variance_percent: float
    active_workers: int
    active_equipment: int
    status: str


class TimelineResponse(BaseModel):
    project_id: str
    granularity: str
    start_date: datetime
    end_date: datetime
    snapshots: List[TimelineSnapshot]
    total_snapshots: int


class HeatmapCell(BaseModel):
    x: int
    y: int
    value: float
    label: Optional[str] = None


class HeatmapResponse(BaseModel):
    project_id: str
    heatmap_type: str
    floor: Optional[int]
    snapshot_date: Optional[datetime]
    grid_rows: int
    grid_cols: int
    cells: List[HeatmapCell]
    min_value: float
    max_value: float
    unit: str


class EquipmentType(BaseModel):
    type_name: str
    count: int
    utilisation_pct: float
    idle_hours: float
    active_hours: float


class EquipmentUtilizationResponse(BaseModel):
    project_id: str
    period_days: int
    total_equipment: int
    overall_utilisation_pct: float
    equipment_types: List[EquipmentType]


class Alert(BaseModel):
    id: str
    project_id: str
    alert_type: str
    severity: str   # critical | high | medium | low
    title: str
    description: str
    location: Optional[str]
    is_resolved: bool
    created_at: datetime
    resolved_at: Optional[datetime]


class AlertsResponse(BaseModel):
    project_id: str
    total: int
    critical: int
    high: int
    medium: int
    low: int
    alerts: List[Alert]


# =============================================================================
# PROCESSING JOB SCHEMAS
# =============================================================================

class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    progress_percent: float
    current_step: Optional[str]
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]

    model_config = {"from_attributes": True}


class ReconstructionRequest(BaseModel):
    project_id: str
    media_upload_ids: List[str] = Field(min_length=1)
    quality: str = Field(default="high", pattern="^(low|medium|high|extreme)$")
    run_mvs: bool = True
    camera_model: str = "OPENCV"


class ReconstructionResponse(BaseModel):
    job_id: str
    reconstruction_id: Optional[str] = None
    status: str
    project_id: str
    quality: str
    message: str
