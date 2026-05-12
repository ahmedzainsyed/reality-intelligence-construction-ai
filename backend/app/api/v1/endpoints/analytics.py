"""
Analytics API Endpoints

Provides construction intelligence analytics:
- Real-time progress estimation
- Delay prediction
- Equipment utilization
- Material inventory tracking
- Site KPI dashboards
- Temporal site evolution
- Spatial heatmaps
"""

from datetime import datetime, timedelta
from typing import Optional, List

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.db.session import get_db
from app.models.models import User
from app.schemas.analytics import (
    ProgressResponse, DelayPredictionResponse, KPIDashboardResponse,
    TimelineResponse, HeatmapResponse, EquipmentUtilizationResponse,
    SiteEvolutionResponse, AlertsResponse,
)
from app.services.analytics_service import AnalyticsService

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/progress/{project_id}",
    response_model=ProgressResponse,
    summary="Get construction progress analytics",
)
async def get_progress(
    project_id: str,
    snapshot_date: Optional[datetime] = Query(None, description="Point-in-time query (defaults to latest)"),
    include_breakdown: bool = Query(True, description="Include per-component breakdown"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(),
):
    """
    Get construction progress for a project.

    Returns:
    - Overall completion percentage
    - Per-component breakdown (foundation, structure, MEP, finishing)
    - Planned vs actual comparison
    - Progress velocity (% per week)
    """
    log = logger.bind(project_id=project_id, user_id=current_user.id)

    snapshot = await analytics_service.get_latest_progress(
        db=db,
        project_id=project_id,
        user_id=current_user.id,
        snapshot_date=snapshot_date,
    )

    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No progress data available. Run reconstruction and progress estimation first.",
        )

    log.info("Progress retrieved", overall=snapshot.overall_progress_percent)
    return ProgressResponse.model_validate(snapshot)


@router.get(
    "/delays/{project_id}",
    response_model=DelayPredictionResponse,
    summary="Get delay predictions",
)
async def get_delay_predictions(
    project_id: str,
    include_risk_factors: bool = Query(True),
    include_scenarios: bool = Query(False, description="Include best/worst case scenarios"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(),
):
    """
    Get ML-powered delay predictions for a construction project.

    Uses XGBoost + LSTM ensemble trained on historical project data.
    Returns:
    - Predicted delay in days
    - Probability of delay
    - Top risk factors with impact scores
    - Confidence intervals
    - Revised completion date
    """
    prediction = await analytics_service.get_delay_prediction(
        db=db,
        project_id=project_id,
        user_id=current_user.id,
    )

    if not prediction:
        # Trigger prediction computation
        await analytics_service.compute_delay_prediction(db, project_id)
        prediction = await analytics_service.get_delay_prediction(
            db=db,
            project_id=project_id,
            user_id=current_user.id,
        )

    return DelayPredictionResponse.model_validate(prediction)


@router.get(
    "/kpi/{project_id}",
    response_model=KPIDashboardResponse,
    summary="Get KPI dashboard data",
)
async def get_kpi_dashboard(
    project_id: str,
    period_days: int = Query(30, description="Number of days for trending"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(),
):
    """
    Get comprehensive KPI dashboard for construction project.

    Includes:
    - Schedule Performance Index (SPI)
    - Cost Performance Index (CPI)
    - Worker productivity metrics
    - Equipment utilization rates
    - Safety incident tracking
    - Material burn rates
    """
    kpis = await analytics_service.compute_kpis(
        db=db,
        project_id=project_id,
        user_id=current_user.id,
        period_days=period_days,
    )
    return kpis


@router.get(
    "/timeline/{project_id}",
    response_model=TimelineResponse,
    summary="Get temporal site evolution timeline",
)
async def get_timeline(
    project_id: str,
    granularity: str = Query("weekly", description="daily | weekly | monthly"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(),
):
    """
    Get temporal evolution of the construction site.

    Returns time-series data showing construction progress
    across each snapshot date, enabling timeline visualization.
    """
    end_date = end_date or datetime.utcnow()
    start_date = start_date or (end_date - timedelta(days=365))

    timeline = await analytics_service.get_timeline(
        db=db,
        project_id=project_id,
        user_id=current_user.id,
        start_date=start_date,
        end_date=end_date,
        granularity=granularity,
    )

    return timeline


@router.get(
    "/heatmap/{project_id}",
    response_model=HeatmapResponse,
    summary="Get spatial activity heatmap",
)
async def get_heatmap(
    project_id: str,
    heatmap_type: str = Query("activity", description="activity | risk | progress | equipment"),
    floor: Optional[int] = Query(None, description="Filter by floor number"),
    snapshot_date: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(),
):
    """
    Get spatial heatmap data for construction site.

    Types:
    - activity: Worker/equipment density
    - risk: Safety risk zones
    - progress: Construction completion by zone
    - equipment: Equipment utilization by zone
    """
    heatmap = await analytics_service.generate_heatmap(
        db=db,
        project_id=project_id,
        heatmap_type=heatmap_type,
        floor=floor,
        snapshot_date=snapshot_date,
    )

    return heatmap


@router.get(
    "/equipment/{project_id}",
    response_model=EquipmentUtilizationResponse,
    summary="Get equipment utilization analytics",
)
async def get_equipment_utilization(
    project_id: str,
    period_days: int = Query(7),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(),
):
    """
    Get equipment utilization metrics:
    - Active vs idle time
    - Equipment count by type
    - Utilization rate per equipment type
    - Productivity index
    """
    return await analytics_service.get_equipment_utilization(
        db=db,
        project_id=project_id,
        period_days=period_days,
    )


@router.get(
    "/alerts/{project_id}",
    response_model=AlertsResponse,
    summary="Get active site alerts",
)
async def get_alerts(
    project_id: str,
    severity: Optional[str] = Query(None, description="critical | high | medium | low"),
    resolved: bool = Query(False, description="Include resolved alerts"),
    limit: int = Query(50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(),
):
    """
    Get active construction site alerts:
    - Safety violations
    - Schedule deviations
    - Equipment breakdowns
    - Material shortages
    - Progress blockers
    """
    return await analytics_service.get_alerts(
        db=db,
        project_id=project_id,
        severity=severity,
        resolved=resolved,
        limit=limit,
    )


@router.post(
    "/progress/{project_id}/compute",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger progress estimation computation",
)
async def trigger_progress_computation(
    project_id: str,
    reconstruction_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(),
):
    """
    Trigger ML-based progress estimation pipeline.
    Uses latest 3D reconstruction if no reconstruction_id provided.
    """
    from app.workers.tasks import compute_progress_task

    task = compute_progress_task.delay(project_id, reconstruction_id)

    return {
        "message": "Progress computation queued",
        "task_id": task.id,
        "project_id": project_id,
    }


@router.get(
    "/summary/{project_id}",
    summary="Get project analytics summary",
)
async def get_analytics_summary(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(),
):
    """Comprehensive single-call analytics summary for dashboard."""
    return await analytics_service.get_full_summary(
        db=db,
        project_id=project_id,
        user_id=current_user.id,
    )
