# =============================================================================
# Reality Intelligence Platform – Backend Dockerfile
# Multi-stage build for minimal production image
# =============================================================================

# ── Stage 1: Dependencies ─────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System deps for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python deps into a virtual env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Production image ─────────────────────────────────────────────────
FROM python:3.11-slim AS production

LABEL maintainer="engineering@reality-intelligence.io"
LABEL version="1.0.0"
LABEL description="Reality Intelligence Platform – Backend API"

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    # COLMAP runtime dependency
    colmap \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1000 ripuser && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home ripuser

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copy application code
COPY --chown=ripuser:ripuser . .

# Create directories
RUN mkdir -p /app/logs /app/models /tmp/rip_sfm && \
    chown -R ripuser:ripuser /app /tmp/rip_sfm

USER ripuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Default: run Gunicorn with Uvicorn workers
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keepalive", "5", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]
