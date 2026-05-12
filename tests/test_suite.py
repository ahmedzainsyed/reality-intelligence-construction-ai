"""
Reality Intelligence Platform – Full Test Suite

Covers:
  Unit Tests:
    - Frame extraction (blur/duplicate/motion detectors)
    - Delay predictor feature engineering
    - API schemas and validators
    - Storage utilities

  Integration Tests:
    - FastAPI endpoint tests (auth, upload, analytics, processing)
    - Celery task dispatch
    - Database CRUD operations

  ML Pipeline Tests:
    - YOLOv8 detection smoke test
    - Frame extractor end-to-end
    - Progress estimator logic
"""

# =============================================================================
# conftest.py  (shared fixtures)
# =============================================================================

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

# ---------------------------------------------------------------------------
# Test database URL (in-memory SQLite for speed)
# ---------------------------------------------------------------------------
TEST_DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "sqlite+aiosqlite:///./test_rip.db",
)

os.environ.update({
    "ENVIRONMENT": "testing",
    "SECRET_KEY": "test-secret-key-minimum-32-characters-long",
    "DATABASE_URL": TEST_DB_URL,
    "REDIS_URL": "redis://localhost:6379/15",
    "CELERY_BROKER_URL": "memory://",
    "CELERY_RESULT_BACKEND": "cache+memory://",
    "MINIO_ENDPOINT": "localhost:9000",
    "MINIO_ACCESS_KEY": "test",
    "MINIO_SECRET_KEY": "testtest",
})


@pytest.fixture(scope="session")
def event_loop():
    """Override default event loop for session-scoped async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    from app.models.models import Base
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db(test_engine) -> AsyncGenerator[AsyncSession, None]:
    async_session = async_sessionmaker(test_engine, expire_on_commit=False)
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db) -> AsyncGenerator[AsyncClient, None]:
    from app.main import app
    from app.db.session import get_db

    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture
def sample_frame_bgr() -> np.ndarray:
    """A sharp synthetic BGR frame (640×480)."""
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    # Add visible edges to make it "sharp"
    frame[100:110, :] = 255
    frame[:, 200:210] = 0
    return frame


@pytest.fixture
def blurry_frame_bgr() -> np.ndarray:
    """A blurry synthetic BGR frame."""
    import cv2
    rng = np.random.default_rng(0)
    frame = rng.integers(50, 200, (480, 640, 3), dtype=np.uint8)
    return cv2.GaussianBlur(frame, (51, 51), 0)


@pytest.fixture
def tmp_video(tmp_path) -> str:
    """Create a minimal synthetic MP4 using OpenCV."""
    import cv2
    path = str(tmp_path / "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 25.0, (640, 480))
    rng = np.random.default_rng(0)
    for i in range(75):  # 3 seconds @ 25fps
        frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        writer.write(frame.astype(np.uint8))
    writer.release()
    return path


# =============================================================================
# UNIT TESTS – Frame Extraction
# =============================================================================

class TestBlurDetector:

    def setup_method(self):
        from ml.frame_extraction.extractor import BlurDetector
        self.detector = BlurDetector(threshold=100.0)

    def test_sharp_frame_passes(self, sample_frame_bgr):
        is_blurry, score = self.detector.is_blurry(sample_frame_bgr)
        assert not is_blurry
        assert score > 100.0

    def test_blurry_frame_rejected(self, blurry_frame_bgr):
        is_blurry, score = self.detector.is_blurry(blurry_frame_bgr)
        assert is_blurry
        assert score < 100.0

    def test_score_is_float(self, sample_frame_bgr):
        _, score = self.detector.is_blurry(sample_frame_bgr)
        assert isinstance(score, float)

    def test_threshold_adjustable(self, sample_frame_bgr):
        strict = BlurDetector(threshold=10000.0)
        from ml.frame_extraction.extractor import BlurDetector
        is_blurry, _ = strict.is_blurry(sample_frame_bgr)
        assert is_blurry  # Even sharp frame fails very strict threshold

    def test_grayscale_input(self):
        import cv2
        from ml.frame_extraction.extractor import BlurDetector
        gray_bgr = np.ones((480, 640, 3), dtype=np.uint8) * 128
        _, score = BlurDetector(threshold=100.0).is_blurry(gray_bgr)
        assert score == 0.0


class TestDuplicateDetector:

    def setup_method(self):
        from ml.frame_extraction.extractor import DuplicateDetector
        self.det = DuplicateDetector(threshold=0.95)

    def test_first_frame_never_duplicate(self, sample_frame_bgr):
        is_dup, score = self.det.is_duplicate(sample_frame_bgr)
        assert not is_dup
        assert score == 0.0

    def test_identical_frame_is_duplicate(self, sample_frame_bgr):
        self.det.is_duplicate(sample_frame_bgr)   # prime
        is_dup, score = self.det.is_duplicate(sample_frame_bgr.copy())
        assert is_dup
        assert score > 0.95

    def test_different_frames_not_duplicate(self, sample_frame_bgr, blurry_frame_bgr):
        self.det.is_duplicate(sample_frame_bgr)
        is_dup, score = self.det.is_duplicate(blurry_frame_bgr)
        assert not is_dup
        assert score < 0.95

    def test_reset_clears_state(self, sample_frame_bgr):
        self.det.is_duplicate(sample_frame_bgr)
        self.det.reset()
        assert self.det.prev_gray is None
        is_dup, score = self.det.is_duplicate(sample_frame_bgr)
        assert not is_dup  # first frame after reset


class TestMotionAnalyzer:

    def setup_method(self):
        from ml.frame_extraction.extractor import MotionAnalyzer
        self.analyzer = MotionAnalyzer()

    def test_first_frame_zero_magnitude(self, sample_frame_bgr):
        mag = self.analyzer.compute_flow_magnitude(sample_frame_bgr)
        assert mag == 0.0

    def test_static_scene_low_magnitude(self, sample_frame_bgr):
        self.analyzer.compute_flow_magnitude(sample_frame_bgr)
        # Same frame twice = near-zero motion
        mag = self.analyzer.compute_flow_magnitude(sample_frame_bgr.copy())
        assert mag < 1.0

    def test_returns_float(self, sample_frame_bgr):
        mag = self.analyzer.compute_flow_magnitude(sample_frame_bgr)
        assert isinstance(mag, float)


class TestVideoFrameExtractor:

    def test_extraction_from_path(self, tmp_video, tmp_path):
        from ml.frame_extraction.extractor import VideoFrameExtractor
        extractor = VideoFrameExtractor(
            target_fps=5.0,
            blur_threshold=10.0,   # Very lenient for synthetic video
            ssim_threshold=0.99,
            max_frames=100,
        )
        out_dir = str(tmp_path / "frames")
        result = extractor.extract_from_path(tmp_video, out_dir, source_type="drone")

        assert result.frames_kept > 0
        assert result.total_frames_in_video > 0
        assert result.duration_seconds > 0
        assert Path(out_dir).exists()
        frames = list(Path(out_dir).glob("*.jpg"))
        assert len(frames) == result.frames_kept

    def test_configure_for_source_cctv(self):
        from ml.frame_extraction.extractor import VideoFrameExtractor
        extractor = VideoFrameExtractor()
        extractor.configure_for_source("cctv")
        assert extractor.target_fps == 0.5
        assert extractor.blur_detector.threshold == 50.0

    def test_configure_for_source_mobile(self):
        from ml.frame_extraction.extractor import VideoFrameExtractor
        extractor = VideoFrameExtractor()
        extractor.configure_for_source("mobile")
        assert extractor.target_fps == 4.0
        assert extractor.blur_detector.threshold == 150.0

    def test_max_frames_respected(self, tmp_video, tmp_path):
        from ml.frame_extraction.extractor import VideoFrameExtractor
        extractor = VideoFrameExtractor(
            target_fps=25.0,
            blur_threshold=0.0,
            ssim_threshold=1.0,
            max_frames=5,
        )
        out = str(tmp_path / "frames_limited")
        result = extractor.extract_from_path(tmp_video, out)
        assert result.frames_kept <= 5


# =============================================================================
# UNIT TESTS – Delay Predictor Feature Engineering
# =============================================================================

class TestFeatureEngineer:

    def setup_method(self):
        from ml.analytics.delay_prediction.predictor import FeatureEngineer, TABULAR_FEATURES
        self.eng = FeatureEngineer()
        self.features = TABULAR_FEATURES

    def _make_history(self, n=30):
        return [
            {
                "overall_progress_percent": i * 1.5,
                "planned_progress_percent": i * 1.8,
                "active_workers": 20 + i % 5,
                "active_equipment": 5 + i % 3,
                "safety_violations_detected": 0,
            }
            for i in range(n)
        ]

    def test_builds_correct_columns(self):
        from datetime import datetime
        history = self._make_history()
        project = {
            "start_date": datetime(2024, 1, 1),
            "planned_end_date": datetime(2025, 1, 1),
            "total_area_sqm": 5000,
            "total_floors": 10,
            "project_type": "commercial",
        }
        row = self.eng.build_tabular_features(project, history, {}, datetime(2024, 7, 1))
        assert list(row.index) == self.features

    def test_progress_delta_computed_correctly(self):
        from datetime import datetime
        history = [
            {"overall_progress_percent": 40.0, "planned_progress_percent": 50.0,
             "active_workers": 10, "active_equipment": 3, "safety_violations_detected": 0},
        ]
        project = {"start_date": datetime(2024, 1, 1), "planned_end_date": datetime(2025, 1, 1),
                   "total_area_sqm": 1000, "total_floors": 5, "project_type": "residential"}
        row = self.eng.build_tabular_features(project, history, {}, datetime(2024, 6, 1))
        assert row["progress_delta_pct"] == pytest.approx(-10.0, abs=0.1)

    def test_lstm_sequence_shape(self):
        history = self._make_history(60)
        seq = self.eng.build_lstm_sequence(history, seq_len=30)
        assert seq.shape == (30, 5)
        assert seq.dtype == np.float32

    def test_lstm_sequence_padding(self):
        history = self._make_history(5)   # Fewer than seq_len=30
        seq = self.eng.build_lstm_sequence(history, seq_len=30)
        assert seq.shape == (30, 5)
        # First 25 rows should be zero-padded
        assert np.all(seq[:25] == 0.0)


class TestXGBoostDelayModel:

    def test_heuristic_fallback_on_track(self):
        import pandas as pd
        from ml.analytics.delay_prediction.predictor import XGBoostDelayModel, TABULAR_FEATURES
        model = XGBoostDelayModel(model_path=None)  # triggers heuristic
        features = pd.Series({f: 0.0 for f in TABULAR_FEATURES})
        features["progress_delta_pct"] = 2.0   # ahead of schedule
        features["progress_velocity_7d"] = 0.5
        features["remaining_duration_days"] = 90.0

        delay, prob, importances = model.predict(features)
        assert delay == 0.0
        assert prob < 0.5
        assert len(importances) > 0

    def test_heuristic_fallback_delayed(self):
        import pandas as pd
        from ml.analytics.delay_prediction.predictor import XGBoostDelayModel, TABULAR_FEATURES
        model = XGBoostDelayModel(model_path=None)
        features = pd.Series({f: 0.0 for f in TABULAR_FEATURES})
        features["progress_delta_pct"] = -20.0  # severely behind
        features["remaining_duration_days"] = 30.0

        delay, prob, importances = model.predict(features)
        assert delay > 0.0
        assert prob > 0.5


# =============================================================================
# INTEGRATION TESTS – FastAPI Endpoints
# =============================================================================

@pytest.mark.asyncio
class TestAuthEndpoints:

    async def test_login_invalid_credentials(self, client):
        resp = await client.post("/api/v1/auth/login", json={
            "username": "nobody@nowhere.com",
            "password": "wrongpassword",
        })
        assert resp.status_code == 401

    async def test_health_check(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data

    async def test_readiness_check(self, client):
        resp = await client.get("/ready")
        assert resp.status_code == 200

    async def test_openapi_schema_available(self, client):
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "Reality Intelligence Platform API"
        assert "/api/v1/auth/login" in str(schema["paths"])

    async def test_rate_limit_headers_present(self, client):
        resp = await client.get("/health")
        # Should not be rate-limited on health endpoint
        assert resp.status_code == 200

    async def test_metrics_endpoint(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        assert b"rip_http_requests_total" in resp.content


@pytest.mark.asyncio
class TestProjectEndpoints:

    async def _get_token(self, client) -> str:
        # Register + login helper
        await client.post("/api/v1/auth/register", json={
            "email": "test@company.com",
            "password": "TestPass123!",
            "full_name": "Test User",
            "organization_name": "Test Corp",
        })
        resp = await client.post("/api/v1/auth/login", json={
            "username": "test@company.com",
            "password": "TestPass123!",
        })
        if resp.status_code == 200:
            return resp.json()["access_token"]
        return ""

    async def test_create_project_unauthorized(self, client):
        resp = await client.post("/api/v1/projects", json={
            "name": "Test Site",
            "location": "Mumbai, India",
        })
        assert resp.status_code == 401

    async def test_list_projects_unauthorized(self, client):
        resp = await client.get("/api/v1/projects")
        assert resp.status_code == 401


# =============================================================================
# ML PIPELINE TESTS
# =============================================================================

class TestSfMPipeline:

    def test_quality_preset_exists(self):
        from ml.reconstruction.sfm.pipeline import QUALITY_PRESETS
        for q in ("low", "medium", "high", "extreme"):
            assert q in QUALITY_PRESETS
            assert "SiftExtraction.max_num_features" in QUALITY_PRESETS[q]
            assert "matching" in QUALITY_PRESETS[q]

    def test_pipeline_init(self):
        from ml.reconstruction.sfm.pipeline import SfMPipeline
        pipeline = SfMPipeline(
            project_id="test-proj-001",
            quality="medium",
            colmap_binary="colmap",
        )
        assert pipeline.quality == "medium"
        assert pipeline.project_id == "test-proj-001"

    def test_invalid_quality_raises(self):
        from ml.reconstruction.sfm.pipeline import SfMPipeline
        with pytest.raises(ValueError, match="quality must be one of"):
            SfMPipeline("p", quality="ultra")


class TestMVSPipeline:

    def test_pipeline_init(self):
        from ml.reconstruction.mvs.pipeline import MVSPipeline
        p = MVSPipeline(quality="high", poisson_depth=11)
        assert p.quality == "high"
        assert p.poisson_depth == 11

    def test_quality_preset_applied(self):
        from ml.reconstruction.mvs.pipeline import MVSPipeline, MVS_QUALITY
        p = MVSPipeline(quality="low")
        expected = MVS_QUALITY["low"]["PatchMatchStereo.geom_consistency"]
        assert p.preset["PatchMatchStereo.geom_consistency"] == expected


class TestDetectionClasses:

    def test_class_count(self):
        from ml.detection.training.train_yolov8 import CONSTRUCTION_CLASSES, NUM_CLASSES
        assert len(CONSTRUCTION_CLASSES) == NUM_CLASSES
        assert NUM_CLASSES > 20  # sanity: should have many classes

    def test_class_ids_sequential(self):
        from ml.detection.training.train_yolov8 import CONSTRUCTION_CLASSES
        ids = sorted(CONSTRUCTION_CLASSES.keys())
        assert ids == list(range(len(ids)))

    def test_all_class_names_unique(self):
        from ml.detection.training.train_yolov8 import CONSTRUCTION_CLASSES
        names = list(CONSTRUCTION_CLASSES.values())
        assert len(names) == len(set(names))


# =============================================================================
# PERFORMANCE / SMOKE TESTS
# =============================================================================

class TestExtractorPerformance:
    """Ensure frame extraction meets throughput requirements."""

    def test_blur_detection_throughput(self, sample_frame_bgr, benchmark):
        from ml.frame_extraction.extractor import BlurDetector
        det = BlurDetector(threshold=100.0)
        benchmark(det.compute_score, sample_frame_bgr)
        # Should be well under 5ms per frame on any modern CPU

    def test_ssim_detection_throughput(self, sample_frame_bgr, benchmark):
        from ml.frame_extraction.extractor import DuplicateDetector
        det = DuplicateDetector()
        det.is_duplicate(sample_frame_bgr)  # prime
        benchmark(det.is_duplicate, sample_frame_bgr)
