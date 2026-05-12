"""Initial schema migration

Revision ID: 001_initial_schema
Revises:
Create Date: 2024-01-15 10:00:00.000000

Creates all tables for the Reality Intelligence Platform:
  - organizations
  - users
  - projects
  - media_uploads
  - extracted_frames
  - processing_jobs
  - detection_results
  - segmentation_results
  - reconstructions_3d
  - camera_poses
  - progress_snapshots
  - delay_predictions
  - bim_models
  - bim_comparisons
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── ENUMERATIONS ──────────────────────────────────────────────────────────
    op.execute("CREATE TYPE user_role AS ENUM ('admin','project_manager','site_engineer','viewer')")
    op.execute("CREATE TYPE job_status AS ENUM ('pending','queued','processing','completed','failed','cancelled')")
    op.execute("CREATE TYPE video_source AS ENUM ('drone','cctv','mobile','360_panoramic','unknown')")
    op.execute("CREATE TYPE recon_quality AS ENUM ('low','medium','high','extreme')")
    op.execute("CREATE TYPE progress_status AS ENUM ('on_track','at_risk','delayed','critical')")

    # ── ORGANIZATIONS ─────────────────────────────────────────────────────────
    op.create_table(
        "organizations",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), unique=True, nullable=False),
        sa.Column("subscription_tier", sa.String(50), server_default="starter"),
        sa.Column("max_sites", sa.Integer(), server_default="5"),
        sa.Column("max_storage_gb", sa.Float(), server_default="100"),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("settings", JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )

    # ── USERS ─────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("organization_id", UUID(as_uuid=False),
                  sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("username", sa.String(100), unique=True, nullable=False),
        sa.Column("full_name", sa.String(255)),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.Enum("admin","project_manager","site_engineer","viewer",
                                   name="user_role"), nullable=False, server_default="viewer"),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("is_verified", sa.Boolean(), server_default="false"),
        sa.Column("last_login_at", sa.DateTime(timezone=True)),
        sa.Column("avatar_url", sa.String(500)),
        sa.Column("preferences", JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index("idx_users_org_email", "users", ["organization_id", "email"])

    # ── PROJECTS ──────────────────────────────────────────────────────────────
    op.create_table(
        "projects",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("organization_id", UUID(as_uuid=False),
                  sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("created_by_id", UUID(as_uuid=False), sa.ForeignKey("users.id")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("location", sa.String(500)),
        sa.Column("latitude", sa.Float()),
        sa.Column("longitude", sa.Float()),
        sa.Column("start_date", sa.DateTime(timezone=True)),
        sa.Column("planned_end_date", sa.DateTime(timezone=True)),
        sa.Column("actual_end_date", sa.DateTime(timezone=True)),
        sa.Column("project_type", sa.String(100)),
        sa.Column("total_area_sqm", sa.Float()),
        sa.Column("total_floors", sa.Integer()),
        sa.Column("budget_usd", sa.BigInteger()),
        sa.Column("status", sa.String(50), server_default="active"),
        sa.Column("thumbnail_url", sa.String(500)),
        sa.Column("metadata", JSONB(), server_default="{}"),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index("idx_projects_org_status", "projects", ["organization_id", "status"])

    # ── MEDIA UPLOADS ─────────────────────────────────────────────────────────
    op.create_table(
        "media_uploads",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("project_id", UUID(as_uuid=False),
                  sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("uploaded_by_id", UUID(as_uuid=False), sa.ForeignKey("users.id")),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("original_filename", sa.String(500)),
        sa.Column("storage_path", sa.String(1000), nullable=False),
        sa.Column("storage_bucket", sa.String(255)),
        sa.Column("file_size_bytes", sa.BigInteger()),
        sa.Column("mime_type", sa.String(100)),
        sa.Column("source_type", sa.Enum("drone","cctv","mobile","360_panoramic","unknown",
                                          name="video_source"), server_default="unknown"),
        sa.Column("duration_seconds", sa.Float()),
        sa.Column("fps", sa.Float()),
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        sa.Column("codec", sa.String(50)),
        sa.Column("bitrate_kbps", sa.Float()),
        sa.Column("camera_model", sa.String(255)),
        sa.Column("camera_serial", sa.String(255)),
        sa.Column("focal_length_mm", sa.Float()),
        sa.Column("sensor_width_mm", sa.Float()),
        sa.Column("gps_coordinates", JSONB()),
        sa.Column("is_validated", sa.Boolean(), server_default="false"),
        sa.Column("validation_errors", JSONB(), server_default="[]"),
        sa.Column("checksum_md5", sa.String(32)),
        sa.Column("frame_count_extracted", sa.Integer(), server_default="0"),
        sa.Column("capture_date", sa.DateTime(timezone=True)),
        sa.Column("upload_completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── EXTRACTED FRAMES ──────────────────────────────────────────────────────
    op.create_table(
        "extracted_frames",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("media_upload_id", UUID(as_uuid=False),
                  sa.ForeignKey("media_uploads.id"), nullable=False),
        sa.Column("frame_number", sa.Integer(), nullable=False),
        sa.Column("timestamp_seconds", sa.Float(), nullable=False),
        sa.Column("storage_path", sa.String(1000), nullable=False),
        sa.Column("file_size_bytes", sa.Integer()),
        sa.Column("blur_score", sa.Float()),
        sa.Column("is_blurry", sa.Boolean(), server_default="false"),
        sa.Column("is_duplicate", sa.Boolean(), server_default="false"),
        sa.Column("ssim_with_prev", sa.Float()),
        sa.Column("optical_flow_magnitude", sa.Float()),
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        sa.Column("gps_lat", sa.Float()),
        sa.Column("gps_lon", sa.Float()),
        sa.Column("gps_alt", sa.Float()),
        sa.Column("camera_pitch", sa.Float()),
        sa.Column("camera_yaw", sa.Float()),
        sa.Column("camera_roll", sa.Float()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_frames_upload_number", "extracted_frames",
                    ["media_upload_id", "frame_number"])

    # ── PROCESSING JOBS ───────────────────────────────────────────────────────
    op.create_table(
        "processing_jobs",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("project_id", UUID(as_uuid=False),
                  sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("media_upload_id", UUID(as_uuid=False), sa.ForeignKey("media_uploads.id")),
        sa.Column("celery_task_id", sa.String(255)),
        sa.Column("job_type", sa.String(100), nullable=False),
        sa.Column("status", sa.Enum("pending","queued","processing","completed","failed","cancelled",
                                     name="job_status"), server_default="pending"),
        sa.Column("priority", sa.Integer(), server_default="5"),
        sa.Column("progress_percent", sa.Float(), server_default="0"),
        sa.Column("current_step", sa.String(255)),
        sa.Column("total_steps", sa.Integer()),
        sa.Column("result_data", JSONB()),
        sa.Column("error_message", sa.Text()),
        sa.Column("error_traceback", sa.Text()),
        sa.Column("config", JSONB(), server_default="{}"),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("duration_seconds", sa.Float()),
        sa.Column("worker_id", sa.String(255)),
        sa.Column("gpu_used", sa.Boolean(), server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_jobs_project_status", "processing_jobs", ["project_id", "status"])
    op.create_index("idx_jobs_celery_id", "processing_jobs", ["celery_task_id"])

    # ── DETECTION RESULTS ─────────────────────────────────────────────────────
    op.create_table(
        "detection_results",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("frame_id", UUID(as_uuid=False),
                  sa.ForeignKey("extracted_frames.id"), nullable=False),
        sa.Column("model_name", sa.String(100)),
        sa.Column("model_version", sa.String(50)),
        sa.Column("inference_time_ms", sa.Float()),
        sa.Column("detections", JSONB(), server_default="[]"),
        sa.Column("detection_count", sa.Integer(), server_default="0"),
        sa.Column("worker_count", sa.Integer(), server_default="0"),
        sa.Column("crane_count", sa.Integer(), server_default="0"),
        sa.Column("excavator_count", sa.Integer(), server_default="0"),
        sa.Column("vehicle_count", sa.Integer(), server_default="0"),
        sa.Column("ppe_count", sa.Integer(), server_default="0"),
        sa.Column("structural_element_count", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_detections_frame", "detection_results", ["frame_id"])

    # ── SEGMENTATION RESULTS ──────────────────────────────────────────────────
    op.create_table(
        "segmentation_results",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("frame_id", UUID(as_uuid=False),
                  sa.ForeignKey("extracted_frames.id"), nullable=False, unique=True),
        sa.Column("model_name", sa.String(100)),
        sa.Column("inference_time_ms", sa.Float()),
        sa.Column("class_coverage", JSONB(), server_default="{}"),
        sa.Column("instances", JSONB(), server_default="[]"),
        sa.Column("instance_count", sa.Integer(), server_default="0"),
        sa.Column("hazard_zone_coverage", sa.Float(), server_default="0"),
        sa.Column("active_work_zone_coverage", sa.Float(), server_default="0"),
        sa.Column("semantic_mask_path", sa.String(1000)),
        sa.Column("instance_mask_path", sa.String(1000)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── RECONSTRUCTIONS 3D ────────────────────────────────────────────────────
    op.create_table(
        "reconstructions_3d",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("project_id", UUID(as_uuid=False),
                  sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("processing_job_id", UUID(as_uuid=False), sa.ForeignKey("processing_jobs.id")),
        sa.Column("name", sa.String(255)),
        sa.Column("quality", sa.Enum("low","medium","high","extreme", name="recon_quality"),
                  server_default="high"),
        sa.Column("sfm_status", sa.String(50), server_default="pending"),
        sa.Column("num_images_registered", sa.Integer(), server_default="0"),
        sa.Column("num_images_total", sa.Integer(), server_default="0"),
        sa.Column("num_sparse_points", sa.Integer(), server_default="0"),
        sa.Column("num_cameras", sa.Integer(), server_default="0"),
        sa.Column("mean_reprojection_error", sa.Float()),
        sa.Column("sfm_workspace_path", sa.String(1000)),
        sa.Column("mvs_status", sa.String(50), server_default="pending"),
        sa.Column("num_dense_points", sa.Integer(), server_default="0"),
        sa.Column("point_cloud_path", sa.String(1000)),
        sa.Column("point_cloud_size_mb", sa.Float()),
        sa.Column("mesh_path", sa.String(1000)),
        sa.Column("textured_mesh_path", sa.String(1000)),
        sa.Column("bbox_min_x", sa.Float()), sa.Column("bbox_min_y", sa.Float()),
        sa.Column("bbox_min_z", sa.Float()), sa.Column("bbox_max_x", sa.Float()),
        sa.Column("bbox_max_y", sa.Float()), sa.Column("bbox_max_z", sa.Float()),
        sa.Column("sfm_duration_seconds", sa.Float()),
        sa.Column("mvs_duration_seconds", sa.Float()),
        sa.Column("gpu_memory_gb", sa.Float()),
        sa.Column("capture_date", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── CAMERA POSES ──────────────────────────────────────────────────────────
    op.create_table(
        "camera_poses",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("reconstruction_id", UUID(as_uuid=False),
                  sa.ForeignKey("reconstructions_3d.id"), nullable=False),
        sa.Column("frame_id", UUID(as_uuid=False), sa.ForeignKey("extracted_frames.id")),
        sa.Column("image_name", sa.String(500)),
        sa.Column("camera_id", sa.Integer()),
        sa.Column("qw", sa.Float()), sa.Column("qx", sa.Float()),
        sa.Column("qy", sa.Float()), sa.Column("qz", sa.Float()),
        sa.Column("tx", sa.Float()), sa.Column("ty", sa.Float()), sa.Column("tz", sa.Float()),
        sa.Column("fx", sa.Float()), sa.Column("fy", sa.Float()),
        sa.Column("cx", sa.Float()), sa.Column("cy", sa.Float()),
        sa.Column("k1", sa.Float()), sa.Column("k2", sa.Float()),
        sa.Column("reprojection_error", sa.Float()),
    )

    # ── PROGRESS SNAPSHOTS ────────────────────────────────────────────────────
    op.create_table(
        "progress_snapshots",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("project_id", UUID(as_uuid=False),
                  sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("reconstruction_id", UUID(as_uuid=False),
                  sa.ForeignKey("reconstructions_3d.id")),
        sa.Column("snapshot_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("overall_progress_percent", sa.Float(), server_default="0"),
        sa.Column("planned_progress_percent", sa.Float(), server_default="0"),
        sa.Column("progress_variance_percent", sa.Float(), server_default="0"),
        sa.Column("status", sa.Enum("on_track","at_risk","delayed","critical",
                                     name="progress_status"), server_default="on_track"),
        sa.Column("foundation_completion", sa.Float(), server_default="0"),
        sa.Column("structural_frame_completion", sa.Float(), server_default="0"),
        sa.Column("slab_completion", sa.Float(), server_default="0"),
        sa.Column("walls_completion", sa.Float(), server_default="0"),
        sa.Column("mep_completion", sa.Float(), server_default="0"),
        sa.Column("finishing_completion", sa.Float(), server_default="0"),
        sa.Column("active_workers", sa.Integer(), server_default="0"),
        sa.Column("active_equipment", sa.Integer(), server_default="0"),
        sa.Column("material_utilization_percent", sa.Float()),
        sa.Column("safety_violations_detected", sa.Integer(), server_default="0"),
        sa.Column("predicted_delay_days", sa.Float(), server_default="0"),
        sa.Column("delay_probability", sa.Float(), server_default="0"),
        sa.Column("risk_level", sa.String(50), server_default="low"),
        sa.Column("concrete_volume_m3", sa.Float()),
        sa.Column("excavation_volume_m3", sa.Float()),
        sa.Column("rebar_linear_meters", sa.Float()),
        sa.Column("activity_heatmap", JSONB()),
        sa.Column("risk_heatmap", JSONB()),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_progress_project_date", "progress_snapshots",
                    ["project_id", "snapshot_date"])

    # ── DELAY PREDICTIONS ─────────────────────────────────────────────────────
    op.create_table(
        "delay_predictions",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("project_id", UUID(as_uuid=False),
                  sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("model_version", sa.String(50)),
        sa.Column("predicted_delay_days", sa.Float(), nullable=False),
        sa.Column("delay_probability", sa.Float(), nullable=False),
        sa.Column("confidence_interval_low", sa.Float()),
        sa.Column("confidence_interval_high", sa.Float()),
        sa.Column("risk_factors", JSONB(), server_default="[]"),
        sa.Column("top_risk_factor", sa.String(255)),
        sa.Column("feature_importances", JSONB(), server_default="{}"),
        sa.Column("prediction_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("target_completion_date", sa.DateTime(timezone=True)),
        sa.Column("revised_completion_date", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_delay_project_date", "delay_predictions",
                    ["project_id", "prediction_date"])

    # ── BIM MODELS ────────────────────────────────────────────────────────────
    op.create_table(
        "bim_models",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("project_id", UUID(as_uuid=False),
                  sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("ifc_version", sa.String(20)),
        sa.Column("storage_path", sa.String(1000)),
        sa.Column("file_size_mb", sa.Float()),
        sa.Column("element_count", sa.Integer()),
        sa.Column("discipline", sa.String(100)),
        sa.Column("version", sa.String(50)),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("metadata", JSONB(), server_default="{}"),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── BIM COMPARISONS ───────────────────────────────────────────────────────
    op.create_table(
        "bim_comparisons",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("project_id", UUID(as_uuid=False),
                  sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("bim_model_id", UUID(as_uuid=False),
                  sa.ForeignKey("bim_models.id"), nullable=False),
        sa.Column("reconstruction_id", UUID(as_uuid=False),
                  sa.ForeignKey("reconstructions_3d.id"), nullable=False),
        sa.Column("overall_completion_percent", sa.Float()),
        sa.Column("mean_spatial_deviation_mm", sa.Float()),
        sa.Column("max_spatial_deviation_mm", sa.Float()),
        sa.Column("element_comparisons", JSONB(), server_default="[]"),
        sa.Column("missing_elements", JSONB(), server_default="[]"),
        sa.Column("extra_elements", JSONB(), server_default="[]"),
        sa.Column("schedule_deviation_days", sa.Float()),
        sa.Column("critical_path_status", sa.String(100)),
        sa.Column("comparison_metadata", JSONB(), server_default="{}"),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("bim_comparisons")
    op.drop_table("bim_models")
    op.drop_table("delay_predictions")
    op.drop_table("progress_snapshots")
    op.drop_table("camera_poses")
    op.drop_table("reconstructions_3d")
    op.drop_table("segmentation_results")
    op.drop_table("detection_results")
    op.drop_table("processing_jobs")
    op.drop_table("extracted_frames")
    op.drop_table("media_uploads")
    op.drop_table("projects")
    op.drop_table("users")
    op.drop_table("organizations")

    op.execute("DROP TYPE IF EXISTS progress_status")
    op.execute("DROP TYPE IF EXISTS recon_quality")
    op.execute("DROP TYPE IF EXISTS video_source")
    op.execute("DROP TYPE IF EXISTS job_status")
    op.execute("DROP TYPE IF EXISTS user_role")
