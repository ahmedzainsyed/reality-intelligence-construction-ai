"""
API V1 Router - Aggregates all endpoint modules.
"""
from fastapi import APIRouter
from app.api.v1.endpoints import (
    auth, projects, uploads, processing,
    analytics, reconstruction, bim, sites, monitoring
)

router = APIRouter()

router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
router.include_router(projects.router, prefix="/projects", tags=["Projects"])
router.include_router(uploads.router, prefix="/uploads", tags=["Media Uploads"])
router.include_router(processing.router, prefix="/processing", tags=["Processing"])
router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
router.include_router(reconstruction.router, prefix="/reconstruction", tags=["3D Reconstruction"])
router.include_router(bim.router, prefix="/bim", tags=["BIM Comparison"])
router.include_router(sites.router, prefix="/sites", tags=["Sites"])
router.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"])
