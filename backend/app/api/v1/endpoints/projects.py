"""
Projects API Endpoints – CRUD for construction projects.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.auth import get_current_user, require_roles
from app.db.session import get_db
from app.models.models import User, Project, UserRole

router = APIRouter()


@router.get("/", summary="List projects")
async def list_projects(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = select(Project).where(
        Project.organization_id == current_user.organization_id,
        Project.is_active == True,
    )
    if status:
        q = q.where(Project.status == status)

    total_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(total_q)).scalar_one()

    rows = (await db.execute(
        q.order_by(Project.created_at.desc())
         .offset((page - 1) * page_size)
         .limit(page_size)
    )).scalars().all()

    return {
        "items": [
            {
                "id": p.id, "name": p.name, "status": p.status,
                "location": p.location, "project_type": p.project_type,
                "total_area_sqm": p.total_area_sqm, "total_floors": p.total_floors,
                "start_date": p.start_date, "planned_end_date": p.planned_end_date,
                "created_at": p.created_at,
            }
            for p in rows
        ],
        "total": total, "page": page, "page_size": page_size,
    }


@router.post("/", status_code=status.HTTP_201_CREATED, summary="Create project")
async def create_project(
    data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.PROJECT_MANAGER)),
):
    from uuid import uuid4
    from datetime import datetime

    project = Project(
        id=str(uuid4()),
        organization_id=current_user.organization_id,
        created_by_id=current_user.id,
        name=data.get("name"),
        description=data.get("description"),
        location=data.get("location"),
        latitude=data.get("latitude"),
        longitude=data.get("longitude"),
        project_type=data.get("project_type", "commercial"),
        total_area_sqm=data.get("total_area_sqm"),
        total_floors=data.get("total_floors"),
        budget_usd=data.get("budget_usd"),
        start_date=datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None,
        planned_end_date=datetime.fromisoformat(data["planned_end_date"]) if data.get("planned_end_date") else None,
        status="active",
        is_active=True,
    )
    db.add(project)
    await db.flush()
    return {"id": project.id, "name": project.name, "status": project.status}


@router.get("/{project_id}", summary="Get project")
async def get_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Project).where(
            Project.id == project_id,
            Project.organization_id == current_user.organization_id,
        )
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {
        "id": project.id, "name": project.name, "description": project.description,
        "location": project.location, "latitude": project.latitude, "longitude": project.longitude,
        "project_type": project.project_type, "total_area_sqm": project.total_area_sqm,
        "total_floors": project.total_floors, "budget_usd": project.budget_usd,
        "start_date": project.start_date, "planned_end_date": project.planned_end_date,
        "status": project.status, "created_at": project.created_at,
    }


@router.patch("/{project_id}", summary="Update project")
async def update_project(
    project_id: str,
    data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.PROJECT_MANAGER)),
):
    result = await db.execute(
        select(Project).where(
            Project.id == project_id,
            Project.organization_id == current_user.organization_id,
        )
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    for field in ("name", "description", "location", "status", "total_area_sqm", "total_floors"):
        if field in data:
            setattr(project, field, data[field])
    await db.flush()
    return {"id": project.id, "name": project.name, "status": project.status}


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete project")
async def delete_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN)),
):
    result = await db.execute(
        select(Project).where(
            Project.id == project_id,
            Project.organization_id == current_user.organization_id,
        )
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    project.is_active = False
    await db.flush()
