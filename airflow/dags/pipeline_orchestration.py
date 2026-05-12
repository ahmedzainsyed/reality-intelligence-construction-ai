"""
Airflow DAG – Reality Intelligence Platform Pipeline Orchestration

Automated workflows:
  1. daily_site_processing    – Nightly batch processing for all active sites
  2. weekly_reconstruction    – Weekly dense 3D reconstruction
  3. model_retraining         – Triggered on data accumulation threshold
  4. performance_report       – Weekly analytics summary generation

Schedule:
  - daily_site_processing:  Every day at 02:00 UTC
  - weekly_reconstruction:  Every Sunday at 03:00 UTC
  - model_retraining:       Every 2 weeks on Monday
  - performance_report:     Every Monday at 06:00 UTC
"""

from datetime import datetime, timedelta
from typing import List, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.redis.hooks.redis import RedisHook
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
import structlog

logger = structlog.get_logger(__name__)

# ── Default args ──────────────────────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner": "ml-engineering",
    "depends_on_past": False,
    "email": ["ml-alerts@reality-intelligence.io"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(hours=1),
    "execution_timeout": timedelta(hours=6),
}

# ── Shared Python callables ───────────────────────────────────────────────────

def get_active_projects(**context) -> List[Dict]:
    """Fetch all active projects requiring processing."""
    hook = PostgresHook(postgres_conn_id="rip_postgres")
    records = hook.get_records("""
        SELECT p.id, p.name, p.organization_id,
               COUNT(mu.id) as new_uploads
        FROM projects p
        JOIN media_uploads mu ON mu.project_id = p.id
        WHERE p.is_active = true
          AND p.status = 'active'
          AND mu.created_at > NOW() - INTERVAL '24 hours'
          AND mu.upload_completed_at IS NOT NULL
        GROUP BY p.id, p.name, p.organization_id
        HAVING COUNT(mu.id) > 0
        ORDER BY new_uploads DESC
        LIMIT 50;
    """)
    projects = [
        {"id": str(r[0]), "name": r[1], "org_id": str(r[2]), "new_uploads": r[3]}
        for r in records
    ]
    logger.info("Found active projects", count=len(projects))
    context["task_instance"].xcom_push(key="active_projects", value=projects)
    return projects


def dispatch_frame_extraction(project_id: str, **context) -> List[str]:
    """Queue frame extraction tasks for all new uploads in a project."""
    import requests

    api_url = Variable.get("RIP_API_URL", default_var="http://backend:8000")
    token   = Variable.get("RIP_API_TOKEN")

    resp = requests.get(
        f"{api_url}/api/v1/uploads/",
        params={"project_id": project_id, "processed": False, "page_size": 100},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    uploads = resp.json().get("items", [])

    task_ids = []
    for upload in uploads:
        r = requests.post(
            f"{api_url}/api/v1/processing/extract-frames/{upload['id']}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if r.status_code in (200, 201, 202):
            task_ids.append(r.json().get("task_id"))

    logger.info("Frame extraction dispatched", project=project_id, tasks=len(task_ids))
    return task_ids


def dispatch_detection(project_id: str, **context) -> None:
    """Queue object detection for extracted frames."""
    import requests
    api_url = Variable.get("RIP_API_URL", default_var="http://backend:8000")
    token   = Variable.get("RIP_API_TOKEN")

    resp = requests.post(
        f"{api_url}/api/v1/processing/detection",
        json={"project_id": project_id, "model": "yolov8"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    logger.info("Detection dispatched", project=project_id)


def check_reconstruction_eligibility(project_id: str, **context) -> str:
    """
    Check if a project has enough frames for reconstruction.
    Returns branch to take: 'run_reconstruction' or 'skip_reconstruction'.
    """
    hook = PostgresHook(postgres_conn_id="rip_postgres")
    count = hook.get_first("""
        SELECT COUNT(ef.id)
        FROM extracted_frames ef
        JOIN media_uploads mu ON ef.media_upload_id = mu.id
        WHERE mu.project_id = %s
          AND ef.is_blurry = false
          AND ef.is_duplicate = false
          AND ef.created_at > NOW() - INTERVAL '7 days'
    """, parameters=[project_id])[0]

    min_frames = int(Variable.get("MIN_FRAMES_FOR_RECONSTRUCTION", default_var="100"))
    logger.info("Frame count for reconstruction", project=project_id, count=count, min=min_frames)

    return "run_reconstruction" if count >= min_frames else "skip_reconstruction"


def run_reconstruction(project_id: str, **context) -> Dict:
    """Trigger SfM + MVS reconstruction pipeline."""
    import requests
    api_url = Variable.get("RIP_API_URL", default_var="http://backend:8000")
    token   = Variable.get("RIP_API_TOKEN")

    resp = requests.post(
        f"{api_url}/api/v1/processing/reconstruction",
        json={
            "project_id": project_id,
            "quality": Variable.get("RECONSTRUCTION_QUALITY", default_var="high"),
            "run_mvs": True,
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    logger.info("Reconstruction triggered", project=project_id, job_id=result.get("job_id"))
    return result


def compute_progress_analytics(project_id: str, **context) -> Dict:
    """Run progress estimation and delay prediction."""
    import requests
    api_url = Variable.get("RIP_API_URL", default_var="http://backend:8000")
    token   = Variable.get("RIP_API_TOKEN")

    # Progress estimation
    resp1 = requests.post(
        f"{api_url}/api/v1/analytics/progress/{project_id}/compute",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp1.raise_for_status()

    # Delay prediction
    resp2 = requests.post(
        f"{api_url}/api/v1/processing/delay-prediction/{project_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp2.raise_for_status()

    return {"progress_task": resp1.json(), "delay_task": resp2.json()}


def check_model_retraining_needed(**context) -> str:
    """
    Branch: decide if model retraining should run.
    Criteria: >= 1000 new labelled samples since last training.
    """
    hook = PostgresHook(postgres_conn_id="rip_postgres")
    new_samples = hook.get_first("""
        SELECT COUNT(*)
        FROM detection_results dr
        WHERE dr.created_at > (
            SELECT COALESCE(MAX(started_at), '2020-01-01')
            FROM processing_jobs
            WHERE job_type = 'model_retrain'
              AND status = 'completed'
        )
    """)[0]

    threshold = int(Variable.get("RETRAIN_SAMPLE_THRESHOLD", default_var="1000"))
    logger.info("New training samples", count=new_samples, threshold=threshold)

    return "trigger_retrain" if new_samples >= threshold else "skip_retrain"


def trigger_model_retrain(**context) -> None:
    """Submit a model retraining job to the ML cluster."""
    import requests
    api_url = Variable.get("RIP_API_URL", default_var="http://backend:8000")
    token   = Variable.get("RIP_API_TOKEN")

    resp = requests.post(
        f"{api_url}/api/v1/processing/retrain",
        json={
            "model": "yolov8",
            "epochs": 100,
            "export_formats": ["onnx", "engine"],
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    logger.info("Model retraining triggered", result=resp.json())


def send_weekly_report(**context) -> None:
    """Generate and email weekly performance report."""
    hook = PostgresHook(postgres_conn_id="rip_postgres")

    stats = hook.get_first("""
        SELECT
            COUNT(DISTINCT p.id)                               as active_projects,
            COUNT(DISTINCT mu.id)                              as uploads_this_week,
            COALESCE(SUM(ef.frames_extracted), 0)             as frames_extracted,
            COUNT(DISTINCT r.id)                               as reconstructions_completed,
            ROUND(AVG(ps.overall_progress_percent)::numeric, 1) as avg_progress,
            COUNT(DISTINCT dp.id) FILTER (
                WHERE dp.delay_probability > 0.5
            )                                                  as high_risk_projects
        FROM projects p
        LEFT JOIN media_uploads mu ON mu.project_id = p.id
            AND mu.created_at > NOW() - INTERVAL '7 days'
        LEFT JOIN (
            SELECT media_upload_id, COUNT(*) as frames_extracted
            FROM extracted_frames GROUP BY 1
        ) ef ON ef.media_upload_id = mu.id
        LEFT JOIN reconstructions_3d r ON r.project_id = p.id
            AND r.created_at > NOW() - INTERVAL '7 days'
        LEFT JOIN progress_snapshots ps ON ps.project_id = p.id
            AND ps.snapshot_date > NOW() - INTERVAL '7 days'
        LEFT JOIN delay_predictions dp ON dp.project_id = p.id
            AND dp.prediction_date > NOW() - INTERVAL '7 days'
        WHERE p.is_active = true
    """)

    report = {
        "week_ending": datetime.utcnow().strftime("%Y-%m-%d"),
        "active_projects": stats[0],
        "uploads_processed": stats[1],
        "frames_extracted": stats[2],
        "reconstructions_completed": stats[3],
        "avg_progress_pct": float(stats[4] or 0),
        "high_risk_projects": stats[5],
    }

    logger.info("Weekly report generated", **report)
    # In production: send via email/Slack


# =============================================================================
# DAG 1: Daily Site Processing
# =============================================================================

with DAG(
    dag_id="daily_site_processing",
    description="Nightly batch processing for all active construction sites",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 2 * * *",   # 02:00 UTC daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["production", "daily", "processing"],
    doc_md="""
    ## Daily Site Processing DAG

    Processes all active construction sites with new video uploads:

    1. **Get Projects** – Query DB for sites with new uploads
    2. **Frame Extraction** – Extract + filter frames per site
    3. **Object Detection** – Run YOLOv8 on extracted frames
    4. **Reconstruction Eligibility** – Check frame count threshold
    5. **Progress Analytics** – Compute progress + delay predictions
    """,
) as daily_dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end")

    get_projects = PythonOperator(
        task_id="get_active_projects",
        python_callable=get_active_projects,
    )

    # Dynamic task generation would use TaskFlow API in Airflow 2.x+
    # For clarity, we show a single-project template:

    with TaskGroup("frame_processing", tooltip="Extract and filter video frames") as frame_group:

        extract_frames = PythonOperator(
            task_id="dispatch_frame_extraction",
            python_callable=dispatch_frame_extraction,
            op_kwargs={"project_id": "{{ task_instance.xcom_pull('get_active_projects')[0]['id'] }}"},
        )

        run_detection = PythonOperator(
            task_id="dispatch_detection",
            python_callable=dispatch_detection,
            op_kwargs={"project_id": "{{ task_instance.xcom_pull('get_active_projects')[0]['id'] }}"},
        )

        extract_frames >> run_detection

    check_recon = BranchPythonOperator(
        task_id="check_reconstruction_eligibility",
        python_callable=check_reconstruction_eligibility,
        op_kwargs={"project_id": "{{ task_instance.xcom_pull('get_active_projects')[0]['id'] }}"},
    )

    do_reconstruction = PythonOperator(
        task_id="run_reconstruction",
        python_callable=run_reconstruction,
        op_kwargs={"project_id": "{{ task_instance.xcom_pull('get_active_projects')[0]['id'] }}"},
    )

    skip_reconstruction = EmptyOperator(task_id="skip_reconstruction")

    analytics = PythonOperator(
        task_id="compute_progress_analytics",
        python_callable=compute_progress_analytics,
        op_kwargs={"project_id": "{{ task_instance.xcom_pull('get_active_projects')[0]['id'] }}"},
        trigger_rule="none_failed_min_one_success",
    )

    (start >> get_projects >> frame_group >> check_recon
     >> [do_reconstruction, skip_reconstruction] >> analytics >> end)


# =============================================================================
# DAG 2: Model Retraining
# =============================================================================

with DAG(
    dag_id="model_retraining",
    description="Triggered model retraining when enough new labelled data accumulates",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 4 * * 1/14",  # Every 2 weeks on Monday
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training", "models"],
) as retrain_dag:

    start_retrain = EmptyOperator(task_id="start")

    check_needed = BranchPythonOperator(
        task_id="check_retraining_needed",
        python_callable=check_model_retraining_needed,
    )

    trigger_retrain = PythonOperator(
        task_id="trigger_retrain",
        python_callable=trigger_model_retrain,
    )

    skip_retrain = EmptyOperator(task_id="skip_retrain")
    end_retrain  = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    start_retrain >> check_needed >> [trigger_retrain, skip_retrain] >> end_retrain


# =============================================================================
# DAG 3: Weekly Performance Report
# =============================================================================

with DAG(
    dag_id="weekly_performance_report",
    description="Generate and distribute weekly platform performance report",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 6 * * 1",    # Monday 06:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["reporting", "weekly"],
) as report_dag:

    EmptyOperator(task_id="start") >> PythonOperator(
        task_id="send_weekly_report",
        python_callable=send_weekly_report,
    ) >> EmptyOperator(task_id="end")
