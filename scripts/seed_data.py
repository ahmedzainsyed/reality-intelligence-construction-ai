#!/usr/bin/env python3
"""
Database Seed Script

Populates the development database with realistic sample data:
  - 1 demo organisation
  - 4 demo users (admin, PM, engineer, viewer)
  - 3 construction projects
  - Sample media uploads
  - Sample progress snapshots
  - Sample delay predictions

Usage:
  python scripts/seed_data.py
  # or via make:
  make seed
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


async def seed():
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from app.core.config import settings
    from app.core.security import get_password_hash
    from app.models.models import (
        Base, Organization, User, Project, MediaUpload,
        ProgressSnapshot, DelayPrediction, VideoSource, UserRole, ProgressStatus
    )

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as db:
        # ── Organisation ──────────────────────────────────────────────────────
        org_id = str(uuid4())
        org = Organization(
            id=org_id,
            name="BuildCorp International",
            slug="buildcorp",
            subscription_tier="enterprise",
            max_sites=100,
            max_storage_gb=10000,
        )
        db.add(org)

        # ── Users ─────────────────────────────────────────────────────────────
        users_data = [
            ("admin@reality-intelligence.io",   "admin",    "Admin User",         UserRole.ADMIN),
            ("demo@reality-intelligence.io",    "demo",     "Demo User",          UserRole.PROJECT_MANAGER),
            ("pm@buildcorp.com",                "pm",       "Sarah Johnson",      UserRole.PROJECT_MANAGER),
            ("engineer@buildcorp.com",          "engineer", "Raj Patel",          UserRole.SITE_ENGINEER),
            ("viewer@buildcorp.com",            "viewer",   "Mary Chen",          UserRole.VIEWER),
        ]
        user_ids = []
        for email, username, name, role in users_data:
            uid = str(uuid4())
            user = User(
                id=uid,
                organization_id=org_id,
                email=email,
                username=username,
                full_name=name,
                hashed_password=get_password_hash("Demo2024!"),
                role=role,
                is_active=True,
                is_verified=True,
            )
            db.add(user)
            user_ids.append(uid)

        admin_id = user_ids[0]

        # ── Projects ──────────────────────────────────────────────────────────
        projects_data = [
            {
                "name": "Mumbai Commercial Tower – Block A",
                "location": "Bandra Kurla Complex, Mumbai, Maharashtra",
                "latitude": 19.0596, "longitude": 72.8656,
                "project_type": "commercial",
                "total_area_sqm": 45000,
                "total_floors": 32,
                "budget_usd": 85_000_000,
                "status": "active",
                "start_date": datetime.utcnow() - timedelta(days=180),
                "planned_end_date": datetime.utcnow() + timedelta(days=365),
                "overall_completion": 42.3,
            },
            {
                "name": "Pune Residential Township – Phase 2",
                "location": "Hinjewadi, Pune, Maharashtra",
                "latitude": 18.5897, "longitude": 73.7388,
                "project_type": "residential",
                "total_area_sqm": 120000,
                "total_floors": 12,
                "budget_usd": 45_000_000,
                "status": "active",
                "start_date": datetime.utcnow() - timedelta(days=90),
                "planned_end_date": datetime.utcnow() + timedelta(days=540),
                "overall_completion": 18.7,
            },
            {
                "name": "Hyderabad Metro Station – Raidurgam",
                "location": "Raidurgam, Hyderabad, Telangana",
                "latitude": 17.4239, "longitude": 78.3495,
                "project_type": "infrastructure",
                "total_area_sqm": 8500,
                "total_floors": 3,
                "budget_usd": 32_000_000,
                "status": "delayed",
                "start_date": datetime.utcnow() - timedelta(days=300),
                "planned_end_date": datetime.utcnow() + timedelta(days=120),
                "overall_completion": 71.5,
            },
        ]

        project_ids = []
        for p in projects_data:
            pid = str(uuid4())
            project = Project(
                id=pid,
                organization_id=org_id,
                created_by_id=admin_id,
                name=p["name"],
                location=p["location"],
                latitude=p["latitude"],
                longitude=p["longitude"],
                project_type=p["project_type"],
                total_area_sqm=p["total_area_sqm"],
                total_floors=p["total_floors"],
                budget_usd=p["budget_usd"],
                status=p["status"],
                start_date=p["start_date"],
                planned_end_date=p["planned_end_date"],
                is_active=True,
            )
            db.add(project)
            project_ids.append((pid, p))

        await db.flush()

        # ── Media Uploads ─────────────────────────────────────────────────────
        source_types = [VideoSource.DRONE, VideoSource.CCTV, VideoSource.MOBILE]
        for pid, pdata in project_ids:
            for i in range(4):
                uid = str(uuid4())
                upload = MediaUpload(
                    id=uid,
                    project_id=pid,
                    uploaded_by_id=admin_id,
                    filename=f"{uid}_site_footage_{i}.mp4",
                    original_filename=f"site_footage_{i}.mp4",
                    storage_path=f"projects/{pid}/media/{uid}/site_footage_{i}.mp4",
                    storage_bucket="rip-media",
                    file_size_bytes=random.randint(200_000_000, 2_000_000_000),
                    mime_type="video/mp4",
                    source_type=random.choice(source_types),
                    duration_seconds=random.uniform(60, 1800),
                    fps=30.0,
                    width=3840, height=2160,
                    codec="h264",
                    is_validated=True,
                    frame_count_extracted=random.randint(500, 3000),
                    upload_completed_at=datetime.utcnow() - timedelta(days=random.randint(1, 60)),
                )
                db.add(upload)

        await db.flush()

        # ── Progress Snapshots (60-day history) ───────────────────────────────
        for pid, pdata in project_ids:
            base_progress = pdata["overall_completion"] - 15
            for days_ago in range(60, 0, -7):  # Weekly snapshots
                snap_date = datetime.utcnow() - timedelta(days=days_ago)
                progress = max(0, base_progress + (60 - days_ago) * 0.25
                               + random.uniform(-1, 1))
                planned = max(0, base_progress + (60 - days_ago) * 0.30)
                variance = progress - planned

                if variance >= 0:
                    status = ProgressStatus.ON_TRACK
                elif variance >= -5:
                    status = ProgressStatus.AT_RISK
                elif variance >= -15:
                    status = ProgressStatus.DELAYED
                else:
                    status = ProgressStatus.CRITICAL

                snap = ProgressSnapshot(
                    id=str(uuid4()),
                    project_id=pid,
                    snapshot_date=snap_date,
                    overall_progress_percent=round(progress, 2),
                    planned_progress_percent=round(planned, 2),
                    progress_variance_percent=round(variance, 2),
                    status=status,
                    foundation_completion=min(100, progress * 2.2),
                    structural_frame_completion=min(100, max(0, progress * 1.8 - 20)),
                    slab_completion=min(100, max(0, progress * 1.6 - 15)),
                    walls_completion=min(100, max(0, progress * 1.4 - 10)),
                    mep_completion=min(100, max(0, progress * 0.8 - 5)),
                    finishing_completion=min(100, max(0, progress * 0.5 - 3)),
                    active_workers=random.randint(40, 180),
                    active_equipment=random.randint(5, 25),
                    safety_violations_detected=random.randint(0, 3),
                    predicted_delay_days=max(0, -variance * 3 + random.uniform(-5, 5)),
                    delay_probability=min(1.0, max(0, 0.5 - variance * 0.05)),
                    risk_level=status.value.replace("on_track", "low")
                                            .replace("at_risk", "medium")
                                            .replace("delayed", "high")
                                            .replace("critical", "critical"),
                    activity_heatmap=None,
                )
                db.add(snap)

        # ── Delay Predictions ─────────────────────────────────────────────────
        for pid, pdata in project_ids:
            is_delayed = pdata["status"] == "delayed"
            pred = DelayPrediction(
                id=str(uuid4()),
                project_id=pid,
                model_version="ensemble-v1",
                predicted_delay_days=random.uniform(15, 45) if is_delayed else random.uniform(0, 10),
                delay_probability=random.uniform(0.6, 0.9) if is_delayed else random.uniform(0.1, 0.3),
                confidence_interval_low=5 if is_delayed else 0,
                confidence_interval_high=60 if is_delayed else 20,
                risk_factors=[
                    {"factor": "Schedule variance", "feature": "progress_delta_pct",
                     "importance": 0.42, "current_value": -12.5},
                    {"factor": "Weather / monsoon season", "feature": "is_monsoon_season",
                     "importance": 0.25, "current_value": 1.0},
                    {"factor": "Equipment underutilisation", "feature": "equipment_utilisation_pct",
                     "importance": 0.18, "current_value": 58.0},
                    {"factor": "Material supply delays", "feature": "material_delivery_lag_days",
                     "importance": 0.15, "current_value": 7.0},
                ],
                top_risk_factor="Schedule variance",
                feature_importances={
                    "progress_delta_pct": 0.42, "is_monsoon_season": 0.25,
                    "equipment_utilisation_pct": 0.18, "material_delivery_lag_days": 0.15,
                },
                prediction_date=datetime.utcnow(),
                target_completion_date=pdata["planned_end_date"],
                revised_completion_date=(
                    pdata["planned_end_date"] + timedelta(days=25)
                    if is_delayed else pdata["planned_end_date"]
                ),
            )
            db.add(pred)

        await db.commit()

    logger.info(
        "Database seeded successfully",
        org="BuildCorp International",
        users=len(users_data),
        projects=len(projects_data),
    )
    print("\n✅ Database seeded!")
    print("   Org:     BuildCorp International")
    print(f"   Users:   {len(users_data)} created")
    print(f"   Projects: {len(projects_data)} created")
    print("\n   Demo login:")
    print("   Email:    demo@reality-intelligence.io")
    print("   Password: Demo2024!\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    asyncio.run(seed())
