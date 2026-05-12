"""
Analytics Service

Business logic for all analytics operations:
  - Latest progress snapshot retrieval
  - KPI computation
  - Timeline generation
  - Spatial heatmap generation
  - Equipment utilisation aggregation
  - Alert management
  - Full summary for dashboard
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import structlog
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    Project, ProgressSnapshot, DelayPrediction,
    DetectionResult, ExtractedFrame, MediaUpload, ProgressStatus
)
from app.schemas.schemas import (
    KPIDashboardResponse, KPIMetric, TimelineResponse, TimelineSnapshot,
    HeatmapResponse, HeatmapCell, EquipmentUtilizationResponse, EquipmentType,
    AlertsResponse, Alert,
)

logger = structlog.get_logger(__name__)


class AnalyticsService:
    """Provides all analytics computations for the platform."""

    # ── Progress ──────────────────────────────────────────────────────────────

    async def get_latest_progress(
        self,
        db: AsyncSession,
        project_id: str,
        user_id: str,
        snapshot_date: Optional[datetime] = None,
    ) -> Optional[ProgressSnapshot]:
        """Return the most recent (or point-in-time) progress snapshot."""
        q = (
            select(ProgressSnapshot)
            .where(ProgressSnapshot.project_id == project_id)
        )
        if snapshot_date:
            q = q.where(ProgressSnapshot.snapshot_date <= snapshot_date)
        q = q.order_by(ProgressSnapshot.snapshot_date.desc()).limit(1)

        result = await db.execute(q)
        return result.scalar_one_or_none()

    async def compute_kpis(
        self,
        db: AsyncSession,
        project_id: str,
        user_id: str,
        period_days: int = 30,
    ) -> KPIDashboardResponse:
        """Compute KPI dashboard metrics."""
        now = datetime.utcnow()
        period_start = now - timedelta(days=period_days)

        # Fetch recent snapshots
        snapshots_q = await db.execute(
            select(ProgressSnapshot)
            .where(
                ProgressSnapshot.project_id == project_id,
                ProgressSnapshot.snapshot_date >= period_start,
            )
            .order_by(ProgressSnapshot.snapshot_date)
        )
        snapshots = snapshots_q.scalars().all()

        latest = snapshots[-1] if snapshots else None
        prev = snapshots[0] if len(snapshots) > 1 else None

        def make_kpi(value: float, unit: str, prev_value: Optional[float] = None,
                     higher_is_better: bool = True) -> KPIMetric:
            delta = None
            trend = "stable"
            if prev_value is not None and prev_value != 0:
                delta = ((value - prev_value) / abs(prev_value)) * 100
                if abs(delta) < 2:
                    trend = "stable"
                elif (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better):
                    trend = "up"
                else:
                    trend = "down"
            return KPIMetric(value=round(value, 2), unit=unit, delta_pct=delta, trend=trend)

        prog_now = latest.overall_progress_percent if latest else 0.0
        prog_prev = prev.overall_progress_percent if prev else None
        plan_now = latest.planned_progress_percent if latest else 0.0

        # Schedule Performance Index = actual / planned
        spi = prog_now / plan_now if plan_now > 0 else 1.0
        spi_prev = (
            (prev.overall_progress_percent / prev.planned_progress_percent)
            if prev and prev.planned_progress_percent > 0 else None
        )

        workers = latest.active_workers if latest else 0
        equip = latest.active_equipment if latest else 0
        equip_util = 70.0  # placeholder – compute from detection_results in prod
        safety = max(0, 100 - (latest.safety_violations_detected or 0) * 5) if latest else 100.0
        burn_rate = prog_now / max(period_days, 1)  # % per day

        return KPIDashboardResponse(
            project_id=project_id,
            period_days=period_days,
            computed_at=now,
            schedule_performance_index=make_kpi(spi, "index", spi_prev),
            overall_progress=make_kpi(prog_now, "%", prog_prev),
            worker_productivity=make_kpi(workers, "workers"),
            equipment_utilisation=make_kpi(equip_util, "%"),
            safety_score=make_kpi(safety, "/100", higher_is_better=True),
            material_burn_rate=make_kpi(burn_rate, "%/day"),
        )

    # ── Timeline ──────────────────────────────────────────────────────────────

    async def get_timeline(
        self,
        db: AsyncSession,
        project_id: str,
        user_id: str,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "weekly",
    ) -> TimelineResponse:
        """Build time-series progress data at the requested granularity."""
        q = await db.execute(
            select(ProgressSnapshot)
            .where(
                ProgressSnapshot.project_id == project_id,
                ProgressSnapshot.snapshot_date >= start_date,
                ProgressSnapshot.snapshot_date <= end_date,
            )
            .order_by(ProgressSnapshot.snapshot_date)
        )
        all_snaps = q.scalars().all()

        # Resample to requested granularity
        snaps = self._resample_snapshots(all_snaps, granularity)

        timeline_items = [
            TimelineSnapshot(
                snapshot_date=s.snapshot_date,
                overall_progress_percent=s.overall_progress_percent,
                planned_progress_percent=s.planned_progress_percent,
                progress_variance_percent=s.progress_variance_percent,
                active_workers=s.active_workers,
                active_equipment=s.active_equipment,
                status=s.status.value if s.status else "on_track",
            )
            for s in snaps
        ]

        return TimelineResponse(
            project_id=project_id,
            granularity=granularity,
            start_date=start_date,
            end_date=end_date,
            snapshots=timeline_items,
            total_snapshots=len(timeline_items),
        )

    def _resample_snapshots(
        self,
        snapshots: List[ProgressSnapshot],
        granularity: str,
    ) -> List[ProgressSnapshot]:
        """Pick representative snapshots per granularity bucket."""
        if not snapshots:
            return []
        if granularity == "daily":
            return snapshots
        if granularity == "weekly":
            # Keep one per week (the last in each week)
            seen_weeks = {}
            for s in snapshots:
                wk = s.snapshot_date.strftime("%Y-W%W")
                seen_weeks[wk] = s
            return list(seen_weeks.values())
        if granularity == "monthly":
            seen_months = {}
            for s in snapshots:
                mo = s.snapshot_date.strftime("%Y-%m")
                seen_months[mo] = s
            return list(seen_months.values())
        return snapshots

    # ── Heatmap ───────────────────────────────────────────────────────────────

    async def generate_heatmap(
        self,
        db: AsyncSession,
        project_id: str,
        heatmap_type: str = "activity",
        floor: Optional[int] = None,
        snapshot_date: Optional[datetime] = None,
    ) -> HeatmapResponse:
        """
        Generate a spatial heatmap grid.

        In production: query detection_results and map detections
        to spatial grid cells via camera pose projection.
        Here: use stored heatmap_data from progress_snapshots.
        """
        snap = await self.get_latest_progress(db, project_id, "", snapshot_date)
        raw = (snap.activity_heatmap if heatmap_type == "activity" else snap.risk_heatmap) \
              if snap else None

        # Fallback: synthetic grid
        rows, cols = 20, 30
        cells: List[HeatmapCell] = []

        if raw and isinstance(raw, list):
            for cell in raw:
                cells.append(HeatmapCell(**cell))
        else:
            import math, random
            random.seed(project_id)
            for r in range(rows):
                for c in range(cols):
                    # Simulate higher activity near centre
                    dist = math.sqrt((r - rows / 2) ** 2 + (c - cols / 2) ** 2)
                    base = max(0, 1 - dist / (rows * 0.7))
                    value = round(base * 100 * (0.7 + 0.3 * random.random()), 1)
                    cells.append(HeatmapCell(x=c, y=r, value=value))

        values = [c.value for c in cells]
        return HeatmapResponse(
            project_id=project_id,
            heatmap_type=heatmap_type,
            floor=floor,
            snapshot_date=snapshot_date,
            grid_rows=rows,
            grid_cols=cols,
            cells=cells,
            min_value=min(values, default=0),
            max_value=max(values, default=100),
            unit="%" if heatmap_type in ("activity", "progress") else "score",
        )

    # ── Equipment ─────────────────────────────────────────────────────────────

    async def get_equipment_utilization(
        self,
        db: AsyncSession,
        project_id: str,
        period_days: int = 7,
    ) -> EquipmentUtilizationResponse:
        """
        Aggregate equipment counts and utilisation from detection results.
        Queries detection_results joined to extracted_frames for the period.
        """
        since = datetime.utcnow() - timedelta(days=period_days)

        rows = await db.execute(
            select(DetectionResult)
            .join(ExtractedFrame, DetectionResult.frame_id == ExtractedFrame.id)
            .join(MediaUpload, ExtractedFrame.media_upload_id == MediaUpload.id)
            .where(
                MediaUpload.project_id == project_id,
                ExtractedFrame.created_at >= since,
            )
            .order_by(ExtractedFrame.created_at)
        )
        results = rows.scalars().all()

        # Aggregate counts by equipment type
        type_counts: Dict[str, int] = {}
        for r in results:
            counts = r.count_by_class if hasattr(r, "count_by_class") else {}
            if isinstance(counts, dict):
                for cls, cnt in counts.items():
                    if any(k in cls for k in ("crane", "excavator", "mixer", "forklift", "bulldozer")):
                        type_counts[cls] = type_counts.get(cls, 0) + cnt

        total = sum(type_counts.values())
        types = [
            EquipmentType(
                type_name=k.replace("_", " ").title(),
                count=v,
                utilisation_pct=min(100, round(v / max(total, 1) * 100 * 1.5, 1)),
                active_hours=round(period_days * 8 * v / max(total, 1), 1),
                idle_hours=round(period_days * 4 * v / max(total, 1), 1),
            )
            for k, v in sorted(type_counts.items(), key=lambda x: -x[1])
        ]

        avg_util = sum(t.utilisation_pct for t in types) / max(len(types), 1)

        return EquipmentUtilizationResponse(
            project_id=project_id,
            period_days=period_days,
            total_equipment=len(type_counts),
            overall_utilisation_pct=round(avg_util, 1),
            equipment_types=types,
        )

    # ── Alerts ────────────────────────────────────────────────────────────────

    async def get_alerts(
        self,
        db: AsyncSession,
        project_id: str,
        severity: Optional[str] = None,
        resolved: bool = False,
        limit: int = 50,
    ) -> AlertsResponse:
        """
        Return active alerts for a project.
        In production, alerts are stored in a dedicated alerts table.
        Here: derive from latest progress snapshot.
        """
        snap = await self.get_latest_progress(db, project_id, "")
        alerts: List[Alert] = []
        now = datetime.utcnow()

        if snap:
            if snap.safety_violations_detected and snap.safety_violations_detected > 0:
                alerts.append(Alert(
                    id=str(uuid4()), project_id=project_id,
                    alert_type="safety", severity="high",
                    title="Safety violations detected",
                    description=f"{snap.safety_violations_detected} safety violations found in latest scan.",
                    location=None, is_resolved=False,
                    created_at=now, resolved_at=None,
                ))
            if snap.progress_variance_percent < -10:
                alerts.append(Alert(
                    id=str(uuid4()), project_id=project_id,
                    alert_type="schedule", severity="critical",
                    title="Significant schedule delay",
                    description=f"Progress is {abs(snap.progress_variance_percent):.1f}% behind plan.",
                    location=None, is_resolved=False,
                    created_at=now, resolved_at=None,
                ))
            if snap.delay_probability and snap.delay_probability > 0.7:
                alerts.append(Alert(
                    id=str(uuid4()), project_id=project_id,
                    alert_type="delay_risk", severity="high",
                    title="High delay probability",
                    description=f"ML model predicts {snap.predicted_delay_days:.0f}-day delay with {snap.delay_probability*100:.0f}% confidence.",
                    location=None, is_resolved=False,
                    created_at=now, resolved_at=None,
                ))

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        counts = {s: sum(1 for a in alerts if a.severity == s)
                  for s in ("critical", "high", "medium", "low")}

        return AlertsResponse(
            project_id=project_id,
            total=len(alerts),
            **counts,
            alerts=alerts[:limit],
        )

    # ── Full Summary ──────────────────────────────────────────────────────────

    async def get_full_summary(
        self,
        db: AsyncSession,
        project_id: str,
        user_id: str,
    ) -> Dict:
        """Single call returning all analytics needed for the dashboard."""
        progress = await self.get_latest_progress(db, project_id, user_id)
        kpis = await self.compute_kpis(db, project_id, user_id)
        equipment = await self.get_equipment_utilization(db, project_id)

        # Project metadata
        proj = await db.get(Project, project_id)

        return {
            "project": {
                "id": project_id,
                "name": proj.name if proj else "Unknown",
                "status": proj.status if proj else "unknown",
                "overall_completion": progress.overall_progress_percent if progress else 0,
            },
            "progress": {
                "overall_progress_percent": progress.overall_progress_percent if progress else 0,
                "planned_progress_percent": progress.planned_progress_percent if progress else 0,
                "progress_variance_percent": progress.progress_variance_percent if progress else 0,
                "active_workers": progress.active_workers if progress else 0,
                "active_equipment": progress.active_equipment if progress else 0,
                "safety_violations_detected": progress.safety_violations_detected if progress else 0,
                "foundation_completion": progress.foundation_completion if progress else 0,
                "structural_frame_completion": progress.structural_frame_completion if progress else 0,
                "slab_completion": progress.slab_completion if progress else 0,
                "walls_completion": progress.walls_completion if progress else 0,
                "mep_completion": progress.mep_completion if progress else 0,
                "finishing_completion": progress.finishing_completion if progress else 0,
                "progress_velocity_7d": 0.5,  # compute in prod
            },
            "kpis": kpis.model_dump(),
            "equipment": equipment.model_dump(),
        }

    # ── Delay prediction proxy ─────────────────────────────────────────────────

    async def get_delay_prediction(
        self,
        db: AsyncSession,
        project_id: str,
        user_id: str,
    ) -> Optional[DelayPrediction]:
        """Fetch the most recent delay prediction for a project."""
        result = await db.execute(
            select(DelayPrediction)
            .where(DelayPrediction.project_id == project_id)
            .order_by(DelayPrediction.prediction_date.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def compute_delay_prediction(
        self,
        db: AsyncSession,
        project_id: str,
    ) -> None:
        """Queue a fresh delay prediction computation."""
        from app.workers.tasks import compute_delay_prediction_task
        compute_delay_prediction_task.delay(project_id)
