# Reality Intelligence Platform – Architecture Documentation

## System Overview

The Reality Intelligence Platform (RIP) is a distributed, GPU-accelerated AI system
for real-time construction site intelligence. It processes multi-source visual data
(drone, CCTV, mobile, 360°) and delivers 3D reconstruction, progress tracking,
delay prediction, and BIM comparison.

---

## Architecture Principles

| Principle | Implementation |
|-----------|---------------|
| **Scalability** | Kubernetes HPA, distributed Celery workers, read replicas |
| **Fault Tolerance** | Task retries, circuit breakers, health checks, PDBs |
| **GPU Efficiency** | Mixed precision (FP16), batched inference, TensorRT |
| **Data Isolation** | Per-organization data partitioning via org_id FKs |
| **Observability** | Prometheus + Grafana + structured logging + Sentry |
| **Security** | JWT + RBAC, mTLS, secrets management, non-root containers |

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INGESTION LAYER                                       │
│                                                                             │
│  Drone API  ──┐                                                             │
│  CCTV SDK   ──┼──► NGINX (Rate Limit + TLS) ──► FastAPI Upload Service     │
│  Mobile App ──┤                                   │                         │
│  360° Cam   ──┘                                   ▼                         │
│                                           MinIO / AWS S3                    │
│                                           (Chunked Storage)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                            │
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ASYNC PROCESSING LAYER                                  │
│                                                                             │
│  Redis Broker                                                               │
│       │                                                                     │
│       ├── Queue: frame_extraction  → Celery Worker (CPU, 8 cores)          │
│       │                             VideoFrameExtractor                     │
│       │                             BlurDetector (Laplacian)               │
│       │                             DuplicateDetector (SSIM)               │
│       │                             MotionAnalyzer (Lucas-Kanade)           │
│       │                                                                     │
│       ├── Queue: ml_gpu            → Celery Worker (GPU, A100)             │
│       │                             YOLOv8ConstructionDetector              │
│       │                             DeepLabV3+ Segmentation                │
│       │                             SAM Instance Segmentation              │
│       │                                                                     │
│       ├── Queue: reconstruction    → Celery Worker (GPU + CPU)             │
│       │                             COLMAP SfM Pipeline                    │
│       │                             COLMAP MVS Pipeline                    │
│       │                             Open3D Post-processing                  │
│       │                                                                     │
│       └── Queue: analytics         → Celery Worker (CPU)                   │
│                                     XGBoost + LSTM Delay Predictor         │
│                                     Progress Estimator                     │
│                                     BIM Comparator (IFC)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                            │
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                            │
│                                                                             │
│  PostgreSQL 16                                                              │
│  ├── organizations, users (RBAC)                                           │
│  ├── projects, media_uploads                                               │
│  ├── extracted_frames                                                      │
│  ├── processing_jobs (Celery tracking)                                     │
│  ├── detection_results, segmentation_results                               │
│  ├── reconstructions_3d, camera_poses                                      │
│  ├── progress_snapshots, delay_predictions                                 │
│  └── bim_models, bim_comparisons                                           │
│                                                                             │
│  MinIO / S3                                                                 │
│  ├── rip-media/     (raw uploads)                                          │
│  ├── rip-models/    (ML model weights)                                     │
│  └── rip-outputs/   (point clouds, meshes, frames)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ML Pipeline Detail

### Detection Pipeline
```
Input Frame
    │
    ▼
YOLOv8-L (1280px, FP16)
    │
    ├── Detections: bbox, class, confidence
    │
    ▼
Detectron2 Mask R-CNN (ensemble, optional)
    │
    ▼
Post-processing (NMS, min-score filter)
    │
    ▼
DB: detection_results table
```

### 3D Reconstruction Pipeline
```
Extracted Frames (100-5000 images)
    │
    ▼ Stage 1: Feature Extraction (SIFT/SuperPoint)
    │   ~8192 keypoints/image, GPU-accelerated
    │
    ▼ Stage 2: Feature Matching
    │   Exhaustive (high quality): O(N²) pairs
    │   Sequential (drone): N × overlap_window pairs
    │   Vocab tree (large datasets): ~N × 15 pairs
    │
    ▼ Stage 3: Geometric Verification (RANSAC)
    │   Fundamental/Essential matrix estimation
    │   Min 15 inliers to keep pair
    │
    ▼ Stage 4: Incremental SfM (COLMAP Mapper)
    │   Camera pose estimation
    │   Triangulation
    │   Bundle adjustment (Ceres Solver)
    │   Pose graph optimization
    │
    ▼ Stage 5: Image Undistortion
    │
    ▼ Stage 6: PatchMatch Stereo (GPU)
    │   Per-image depth + normal maps
    │   Geometric consistency filter
    │
    ▼ Stage 7: Depth Map Fusion
    │   → fused.ply (dense point cloud, 5-50M pts)
    │
    ▼ Stage 8: Open3D Post-processing
    │   Statistical outlier removal
    │   Voxel downsampling (5cm)
    │   Normal estimation
    │
    ▼ Stage 9: Poisson Surface Reconstruction
    │   → mesh_poisson.ply
    │
    Output: dense cloud + mesh → MinIO
```

### Delay Prediction Pipeline
```
Project DB Records
    │
    ├── Tabular Features (25 features)
    │   Progress delta, velocity, equipment utilization,
    │   seasonal patterns, org history, material lag
    │   │
    │   └──► XGBoost Gradient Boosted Trees
    │           ↓ predicted_delay_days (regression)
    │           ↓ feature_importances (SHAP-compatible)
    │
    └── Sequential Features (30-day × 5-feature time series)
        Overall progress, planned progress, worker count,
        equipment count, safety violations
        │
        └──► LSTM (2-layer, 128 hidden units, dropout=0.3)
                ↓ predicted_delay_days
                ↓ delay_probability
    │
    ▼ Ensemble (XGB 60% + LSTM 40%)
    │
    Output: delay_days, probability, CI [low, high], risk_factors
```

---

## Database Schema Summary

```
organizations (1) ──── (N) users
organizations (1) ──── (N) projects
projects      (1) ──── (N) media_uploads
projects      (1) ──── (N) processing_jobs
projects      (1) ──── (N) reconstructions_3d
projects      (1) ──── (N) progress_snapshots
projects      (1) ──── (N) delay_predictions
projects      (1) ──── (N) bim_models

media_uploads  (1) ──── (N) extracted_frames
extracted_frames (1) ── (N) detection_results
extracted_frames (1) ── (1) segmentation_results

reconstructions_3d (1) ── (N) camera_poses
bim_models     (1) ──── (N) bim_comparisons
```

---

## API Design

### REST Conventions
- All resources versioned under `/api/v1/`
- Async operations return `202 Accepted` + `task_id`
- Pagination: `?page=1&page_size=20`
- Filtering: `?project_id=x&status=completed`
- Sorting: `?sort_by=created_at&order=desc`
- Response envelope: `{data: ..., meta: {page, total}}`
- Errors: `{error: true, error_code: "...", message: "..."}`

### Authentication Flow
```
POST /api/v1/auth/login
  → {access_token, refresh_token, expires_in}

Authorization: Bearer <access_token>

POST /api/v1/auth/refresh
  → {access_token, expires_in}
```

### Long-running Operations
```
POST /api/v1/processing/reconstruction
  → 202 {job_id: "uuid", status: "queued"}

GET /api/v1/jobs/{job_id}
  → {status: "processing", progress: 45.2, step: "Running MVS"}

WebSocket /ws/jobs/{job_id}
  → streaming {progress, step, message}
```

---

## Security Architecture

```
Internet
    │
    ▼ TLS 1.3 (Let's Encrypt / ACM)
NGINX (rate limiting, WAF headers)
    │
    ▼ JWT Bearer token (HS256)
FastAPI (RBAC middleware)
    │
    ├── Role: admin        → all operations
    ├── Role: project_mgr  → CRUD on own projects
    ├── Role: site_eng     → read + upload
    └── Role: viewer       → read only

Data isolation: every query filtered by organization_id
Row-level security enforced at service layer
Secrets in Kubernetes Secrets (or HashiCorp Vault)
```

---

## Deployment Topology

### Development
```
docker-compose.yml
  postgres, redis, minio, backend, frontend,
  celery-workers × 4, mlflow, prometheus, grafana
```

### Production (Kubernetes on EKS/GKE)
```
Namespace: rip-production
  backend                 (Deployment, 3–20 replicas, HPA)
  frontend                (Deployment, 2 replicas)
  celery-worker-cpu       (Deployment, 4–16 replicas, HPA)
  celery-worker-gpu       (Deployment, 2–8 replicas, GPU nodes)
  reconstruction-worker   (StatefulSet, 1 replica, 64GB RAM)
  postgres                (StatefulSet or RDS)
  redis                   (StatefulSet or ElastiCache)
  minio                   (StatefulSet or S3)
  prometheus, grafana     (Deployment)
  triton-server           (Deployment, GPU nodes)

Node groups:
  general:    m6i.2xlarge × 5–20 (API + CPU workers)
  gpu:        p3.2xlarge  × 2–8  (ML + reconstruction)
  memory:     r6i.4xlarge × 1–3  (large COLMAP jobs)
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| API P50 latency | < 50ms | Database-backed endpoints |
| API P95 latency | < 200ms | Complex analytics |
| Upload throughput | > 100 MB/s | MinIO direct upload |
| Frame extraction | > 500 fps | CPU parallelism |
| YOLOv8 inference | > 250 img/s | A100, FP16, batch=8 |
| SAM inference | > 30 img/s | A100, full image |
| SfM (1000 images) | < 10 min | 32-core CPU |
| MVS (1000 images) | < 30 min | A100 GPU |
| End-to-end pipeline | < 60 min | 2-hour drone flight |
| DB query (p95) | < 10ms | With proper indexes |

---

## Capacity Planning

| Scale | Sites | Monthly uploads | Required infra |
|-------|-------|----------------|----------------|
| Starter | 10 | 2TB | 2× m6i.xlarge + 1× p3.2xlarge |
| Growth | 100 | 20TB | 5× m6i.2xlarge + 3× p3.2xlarge |
| Enterprise | 1000+ | 200TB+ | Auto-scaling EKS + S3 + RDS Multi-AZ |
