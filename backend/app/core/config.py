"""
Application Configuration Module

Uses Pydantic Settings for type-safe, environment-driven configuration.
Supports .env files, environment variables, and AWS Secrets Manager.
"""

from functools import lru_cache
from typing import List, Optional, Union
from pydantic import AnyHttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Production-grade application settings.
    All values can be overridden via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # Application
    # ==========================================================================
    APP_NAME: str = "Reality Intelligence Platform"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"  # development | staging | production
    DEBUG: bool = False
    PORT: int = 8000
    WORKERS: int = 4
    LOG_LEVEL: str = "INFO"
    SHOW_DOCS: bool = True

    # ==========================================================================
    # API Configuration
    # ==========================================================================
    API_V1_PREFIX: str = "/api/v1"
    ALLOWED_HOSTS: List[str] = ["*"]
    ALLOWED_ORIGINS: List[Union[str, AnyHttpUrl]] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://app.reality-intelligence.io",
    ]

    # ==========================================================================
    # Security & Authentication
    # ==========================================================================
    SECRET_KEY: str = "dev-secret-key-change-in-production-minimum-32-chars"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    PASSWORD_MIN_LENGTH: int = 8

    # ==========================================================================
    # Database
    # ==========================================================================
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/rip_db"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_ECHO: bool = False

    # Test DB
    TEST_DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/rip_test_db"

    # ==========================================================================
    # Redis
    # ==========================================================================
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 100
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ==========================================================================
    # Storage (MinIO / AWS S3)
    # ==========================================================================
    STORAGE_BACKEND: str = "minio"  # minio | s3
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_MEDIA: str = "rip-media"
    MINIO_BUCKET_MODELS: str = "rip-models"
    MINIO_BUCKET_OUTPUTS: str = "rip-outputs"

    # AWS S3 (Production)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_S3_BUCKET: Optional[str] = None
    AWS_REGION: str = "us-east-1"

    # ==========================================================================
    # ML Services
    # ==========================================================================
    TRITON_SERVER_URL: str = "localhost:8001"
    TRITON_SERVER_GRPC_URL: str = "localhost:8001"
    TRITON_SERVER_HTTP_URL: str = "http://localhost:8000"
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    WANDB_API_KEY: Optional[str] = None
    WANDB_PROJECT: str = "reality-intelligence-platform"

    # Model configurations
    YOLOV8_MODEL_PATH: str = "models/yolov8_construction.pt"
    DETECTRON2_MODEL_PATH: str = "models/detectron2_construction.pkl"
    SAM_MODEL_PATH: str = "models/sam_vit_h_4b8939.pth"
    DEEPLABV3_MODEL_PATH: str = "models/deeplabv3_construction.pth"

    # ==========================================================================
    # Processing Configuration
    # ==========================================================================
    MAX_UPLOAD_SIZE_MB: int = 5000  # 5GB
    FRAME_EXTRACTION_FPS: float = 2.0
    BLUR_THRESHOLD: float = 100.0  # Laplacian variance threshold
    SSIM_THRESHOLD: float = 0.95   # Duplicate frame threshold
    MAX_FRAMES_PER_VIDEO: int = 5000
    SUPPORTED_VIDEO_FORMATS: List[str] = ["mp4", "avi", "mov", "mkv", "webm"]
    SUPPORTED_IMAGE_FORMATS: List[str] = ["jpg", "jpeg", "png", "tiff", "webp"]

    # COLMAP configuration
    COLMAP_BINARY: str = "colmap"
    COLMAP_WORKSPACE: str = "/tmp/colmap_workspace"
    RECONSTRUCTION_QUALITY: str = "high"  # low | medium | high | extreme

    # ==========================================================================
    # Rate Limiting
    # ==========================================================================
    RATE_LIMIT_DEFAULT: str = "100/minute"
    RATE_LIMIT_UPLOAD: str = "10/minute"
    RATE_LIMIT_RECONSTRUCTION: str = "5/hour"
    RATE_LIMIT_AUTH: str = "20/minute"

    # ==========================================================================
    # Monitoring
    # ==========================================================================
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_ENABLED: bool = True
    GRAFANA_URL: str = "http://localhost:3001"

    # ==========================================================================
    # Email (Notifications)
    # ==========================================================================
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: str = "noreply@reality-intelligence.io"
    EMAILS_FROM_NAME: str = "Reality Intelligence Platform"

    # ==========================================================================
    # Feature Flags
    # ==========================================================================
    ENABLE_GAUSSIAN_SPLATTING: bool = False  # Experimental
    ENABLE_BIM_COMPARISON: bool = True
    ENABLE_DELAY_PREDICTION: bool = True
    ENABLE_REAL_TIME_PROCESSING: bool = False

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production", "testing"}
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}")
        return v

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        if not v.startswith(("postgresql", "sqlite")):
            raise ValueError("DATABASE_URL must be a PostgreSQL or SQLite connection string")
        return v

    @model_validator(mode="after")
    def configure_environment_defaults(self) -> "Settings":
        """Apply environment-specific defaults."""
        if self.ENVIRONMENT == "production":
            self.SHOW_DOCS = False
            self.DEBUG = False
            self.DATABASE_ECHO = False
        elif self.ENVIRONMENT == "development":
            self.SHOW_DOCS = True
            self.DEBUG = True
        return self

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def database_url_sync(self) -> str:
        """Synchronous database URL (for Alembic migrations)."""
        return self.DATABASE_URL.replace("+asyncpg", "")

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.
    Use this function instead of instantiating Settings directly.
    """
    return Settings()


# Global settings instance
settings = get_settings()
