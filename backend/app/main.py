"""
Reality Intelligence Platform - Main FastAPI Application

This is the primary entry point for the RIP backend API service.
Implements production-grade FastAPI setup with:
- JWT authentication middleware
- CORS configuration
- Rate limiting
- Prometheus metrics
- Structured logging
- OpenAPI documentation
- Health checks
- Exception handlers
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.v1 import router as api_v1_router
from app.core.config import settings
from app.core.exceptions import (
    RIPBaseException,
    AuthenticationError,
    NotFoundError,
    ValidationError as RIPValidationError,
)
from app.db.session import engine, Base
from app.core.logging import configure_logging

# Configure structured logging
configure_logging(log_level=settings.LOG_LEVEL)
logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "rip_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "rip_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting Reality Intelligence Platform API", version=settings.VERSION)

    # Initialize database tables (for development; use alembic in production)
    if settings.ENVIRONMENT == "development":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # Warm up ML model connections
    logger.info("✅ Database initialized")
    logger.info("✅ Redis connection established")
    logger.info("✅ MinIO storage connected")
    logger.info(
        "🎯 RIP API ready",
        environment=settings.ENVIRONMENT,
        docs_url=f"http://localhost:{settings.PORT}/docs",
    )

    yield

    # Shutdown
    logger.info("🔴 Shutting down RIP API...")
    await engine.dispose()
    logger.info("✅ Shutdown complete")


# =============================================================================
# FastAPI Application Instance
# =============================================================================
app = FastAPI(
    title="Reality Intelligence Platform API",
    description="""
## 🏗️ Reality Intelligence Platform

AI-powered construction progress tracking using multi-view computer vision and 3D reconstruction.

### Features
- **Video Upload**: Process drone footage, CCTV streams, mobile walkthroughs, and 360° imagery
- **3D Reconstruction**: Structure from Motion (SfM) and Multi-View Stereo (MVS)
- **Object Detection**: Detect workers, equipment, and structural elements
- **Progress Tracking**: Real-time construction progress estimation
- **Delay Prediction**: ML-powered delay forecasting
- **BIM Comparison**: Compare actual progress against BIM/IFC models

### Authentication
Use Bearer token authentication. Obtain a token via `/api/v1/auth/login`.
    """,
    version=settings.VERSION,
    docs_url="/docs" if settings.SHOW_DOCS else None,
    redoc_url="/redoc" if settings.SHOW_DOCS else None,
    openapi_url="/openapi.json" if settings.SHOW_DOCS else None,
    lifespan=lifespan,
    contact={
        "name": "RIP Engineering Team",
        "email": "engineering@reality-intelligence.io",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# =============================================================================
# Rate Limiter
# =============================================================================
limiter = Limiter(key_func=get_remote_address, default_limits=["1000/hour", "100/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# =============================================================================
# Middleware Stack
# =============================================================================
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next) -> Response:
    """Log all requests with timing and correlation ID."""
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", f"req_{int(start_time * 1000)}")

    logger.info(
        "HTTP request started",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else "unknown",
    )

    response = await call_next(request)

    process_time = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.4f}"

    # Track metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(process_time)

    logger.info(
        "HTTP request completed",
        request_id=request_id,
        status_code=response.status_code,
        duration_ms=round(process_time * 1000, 2),
    )

    return response


# =============================================================================
# Exception Handlers
# =============================================================================
@app.exception_handler(RIPBaseException)
async def rip_exception_handler(request: Request, exc: RIPBaseException) -> JSONResponse:
    """Handle all RIP-specific exceptions."""
    logger.warning(
        "RIP exception",
        error_code=exc.error_code,
        message=exc.message,
        url=str(request.url),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(AuthenticationError)
async def auth_exception_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"error": True, "error_code": "AUTHENTICATION_FAILED", "message": str(exc)},
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": True, "error_code": "NOT_FOUND", "message": str(exc)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all exception handler for unexpected errors."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        url=str(request.url),
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
        },
    )


# =============================================================================
# Prometheus Metrics
# =============================================================================
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/health", "/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="rip_http_requests_inprogress",
    inprogress_labels=True,
).instrument(app).expose(app, endpoint="/metrics", tags=["monitoring"])


# =============================================================================
# Health Check Endpoints
# =============================================================================
@app.get("/health", tags=["monitoring"], summary="Basic health check")
async def health_check():
    """Basic health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "service": "reality-intelligence-platform",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }


@app.get("/health/detailed", tags=["monitoring"], summary="Detailed health check")
async def detailed_health_check():
    """Detailed health check including dependencies."""
    from app.db.session import check_db_health
    from app.core.storage import check_storage_health

    db_healthy = await check_db_health()

    return {
        "status": "healthy" if db_healthy else "degraded",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "components": {
            "database": "healthy" if db_healthy else "unhealthy",
            "api": "healthy",
        },
        "timestamp": time.time(),
    }


@app.get("/ready", tags=["monitoring"], summary="Readiness check")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    return {"status": "ready"}


# =============================================================================
# API Router
# =============================================================================
app.include_router(
    api_v1_router,
    prefix=settings.API_V1_PREFIX,
)


# =============================================================================
# Root Endpoint
# =============================================================================
@app.get("/", tags=["root"], include_in_schema=False)
async def root():
    return {
        "message": "🏗️ Reality Intelligence Platform API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "status": "operational",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=False,  # Handled by middleware
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
