"""
Construction Delay Prediction Engine

Ensemble model combining:
  - XGBoost (tabular features, interpretable)
  - LSTM (temporal progress sequence)

Features:
  - Progress velocity (actual vs planned)
  - Equipment utilisation rate
  - Worker activity density
  - Weather exposure index
  - Historical project similarity
  - Seasonal/calendar patterns
  - Material supply signals

Outputs:
  - predicted_delay_days   (point estimate)
  - delay_probability      (P(delay > 0))
  - confidence_interval    [low, high] 90 %
  - risk_factors           ranked list with impact scores
  - revised_completion_date
"""

import asyncio
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# ─── Feature names ────────────────────────────────────────────────────────────

TABULAR_FEATURES = [
    # Progress signals
    "progress_actual_pct",
    "progress_planned_pct",
    "progress_delta_pct",            # actual – planned
    "progress_velocity_7d",          # % per day, last 7 days
    "progress_velocity_30d",         # % per day, last 30 days
    "velocity_trend",                # velocity_7d / velocity_30d – 1
    # Activity signals
    "mean_daily_workers",
    "mean_daily_equipment",
    "equipment_utilisation_pct",
    "idle_equipment_ratio",
    # Project characteristics
    "project_age_days",
    "planned_duration_days",
    "remaining_duration_days",
    "total_area_sqm_log",
    "total_floors",
    "project_type_enc",              # residential=0, commercial=1, infra=2
    # Temporal patterns
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_monsoon_season",
    "holidays_next_7d",
    # Historical
    "similar_project_avg_delay_days",
    "org_historical_delay_ratio",    # org avg actual / planned duration
    # Supply chain proxy
    "material_delivery_lag_days",
]

SEQUENCE_LENGTH = 30  # days of history for LSTM


# ─── Feature Engineering ──────────────────────────────────────────────────────

class FeatureEngineer:
    """Extract and engineer features from raw DB records."""

    def build_tabular_features(
        self,
        project: Dict,
        progress_history: List[Dict],   # sorted chronologically
        detection_stats: Dict,
        reference_date: datetime,
    ) -> pd.Series:
        """
        Build a single feature row for XGBoost inference.

        Args:
            project:          Project metadata dict
            progress_history: List of ProgressSnapshot dicts
            detection_stats:  Aggregated detection stats (workers, equipment)
            reference_date:   Date of prediction

        Returns:
            pd.Series with exactly TABULAR_FEATURES columns
        """
        if not progress_history:
            return pd.Series({f: 0.0 for f in TABULAR_FEATURES})

        latest = progress_history[-1]

        # Progress signals
        prog_actual = float(latest.get("overall_progress_percent", 0.0))
        prog_planned = float(latest.get("planned_progress_percent", 0.0))
        prog_delta = prog_actual - prog_planned

        # Velocity: % per day over last N days
        vel_7d = self._compute_velocity(progress_history, days=7)
        vel_30d = self._compute_velocity(progress_history, days=30)
        vel_trend = (vel_7d / vel_30d - 1) if vel_30d > 1e-6 else 0.0

        # Project timing
        start = project.get("start_date")
        planned_end = project.get("planned_end_date")
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(planned_end, str):
            planned_end = datetime.fromisoformat(planned_end)

        age_days = (reference_date - start).days if start else 0
        planned_dur = (planned_end - start).days if (start and planned_end) else 365
        remaining_days = max(0, (planned_end - reference_date).days) if planned_end else 0

        # Activity
        mean_workers = float(detection_stats.get("mean_daily_workers", 0))
        mean_equip = float(detection_stats.get("mean_daily_equipment", 0))
        equip_util = float(detection_stats.get("equipment_utilisation_pct", 0))
        idle_ratio = 1.0 - equip_util / 100.0

        # Temporal encoding (cyclic)
        dow = reference_date.weekday()
        month = reference_date.month
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        month_sin = np.sin(2 * np.pi * (month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12)

        # Monsoon heuristic (June–September in South Asia)
        is_monsoon = 1.0 if 6 <= month <= 9 else 0.0

        # Project type encoding
        ptype_map = {"residential": 0, "commercial": 1, "infrastructure": 2}
        ptype_enc = float(ptype_map.get(project.get("project_type", ""), 1))

        # Log-transform area
        area = float(project.get("total_area_sqm", 1000))
        area_log = np.log1p(area)

        # Placeholder for historical and supply chain features
        similar_delay = float(project.get("_similar_avg_delay_days", 0))
        org_delay_ratio = float(project.get("_org_delay_ratio", 1.0))
        material_lag = float(project.get("_material_lag_days", 0))
        holidays = float(project.get("_holidays_next_7d", 0))

        row = {
            "progress_actual_pct": prog_actual,
            "progress_planned_pct": prog_planned,
            "progress_delta_pct": prog_delta,
            "progress_velocity_7d": vel_7d,
            "progress_velocity_30d": vel_30d,
            "velocity_trend": vel_trend,
            "mean_daily_workers": mean_workers,
            "mean_daily_equipment": mean_equip,
            "equipment_utilisation_pct": equip_util,
            "idle_equipment_ratio": idle_ratio,
            "project_age_days": float(age_days),
            "planned_duration_days": float(planned_dur),
            "remaining_duration_days": float(remaining_days),
            "total_area_sqm_log": area_log,
            "total_floors": float(project.get("total_floors", 1)),
            "project_type_enc": ptype_enc,
            "day_of_week_sin": dow_sin,
            "day_of_week_cos": dow_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "is_monsoon_season": is_monsoon,
            "holidays_next_7d": holidays,
            "similar_project_avg_delay_days": similar_delay,
            "org_historical_delay_ratio": org_delay_ratio,
            "material_delivery_lag_days": material_lag,
        }

        return pd.Series(row)[TABULAR_FEATURES]

    def build_lstm_sequence(
        self, progress_history: List[Dict], seq_len: int = SEQUENCE_LENGTH
    ) -> np.ndarray:
        """
        Build (seq_len, n_features) array for LSTM from daily progress records.
        Pads with zeros if fewer than seq_len records available.
        """
        seq_features = [
            "overall_progress_percent",
            "planned_progress_percent",
            "active_workers",
            "active_equipment",
            "safety_violations_detected",
        ]
        n_feat = len(seq_features)

        seq = np.zeros((seq_len, n_feat), dtype=np.float32)
        records = progress_history[-seq_len:]  # take last seq_len days

        for i, rec in enumerate(records):
            for j, col in enumerate(seq_features):
                seq[i, j] = float(rec.get(col, 0.0))

        return seq  # shape: (seq_len, n_feat)

    @staticmethod
    def _compute_velocity(history: List[Dict], days: int) -> float:
        """Compute progress velocity in %/day over last `days` days."""
        if len(history) < 2:
            return 0.0
        recent = history[-days:]
        if len(recent) < 2:
            return 0.0
        delta_pct = float(recent[-1].get("overall_progress_percent", 0)) - float(
            recent[0].get("overall_progress_percent", 0)
        )
        delta_days = max(1, len(recent) - 1)
        return delta_pct / delta_days


# ─── XGBoost Model ────────────────────────────────────────────────────────────

class XGBoostDelayModel:
    """Gradient-boosted tree for tabular delay regression."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path

    def load(self):
        if self.model_path and Path(self.model_path).exists():
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            logger.info("XGBoost model loaded", path=self.model_path)
        else:
            logger.warning("XGBoost model not found – using heuristic fallback")

    def predict(self, features: pd.Series) -> Tuple[float, float, Dict]:
        """
        Returns (predicted_delay_days, delay_probability, feature_importances).
        """
        if self.model is None:
            return self._heuristic_fallback(features)

        import xgboost as xgb
        dmat = xgb.DMatrix(features.values.reshape(1, -1), feature_names=TABULAR_FEATURES)
        pred = float(self.model.predict(dmat)[0])

        # Feature importance (gain)
        scores = self.model.get_score(importance_type="gain")
        total = sum(scores.values()) or 1.0
        importances = {k: round(v / total, 4) for k, v in sorted(
            scores.items(), key=lambda x: -x[1]
        )[:10]}

        # Heuristic delay probability from prediction
        delay_prob = min(1.0, max(0.0, pred / 60.0)) if pred > 0 else 0.1

        return max(0.0, pred), delay_prob, importances

    def _heuristic_fallback(self, features: pd.Series) -> Tuple[float, float, Dict]:
        """Rule-based fallback when model not loaded."""
        delta = features.get("progress_delta_pct", 0.0)
        vel_7d = features.get("progress_velocity_7d", 0.0)
        remaining = features.get("remaining_duration_days", 180.0)

        # If progress is significantly behind planned
        if delta < -10:
            delay_days = abs(delta) * 1.5
            delay_prob = min(0.95, 0.5 + abs(delta) / 40.0)
        elif delta < -5:
            delay_days = abs(delta) * 0.8
            delay_prob = 0.35
        elif vel_7d < 0.1 and remaining < 60:
            delay_days = remaining * 0.3
            delay_prob = 0.6
        else:
            delay_days = 0.0
            delay_prob = 0.1

        importances = {
            "progress_delta_pct": 0.45,
            "progress_velocity_7d": 0.25,
            "remaining_duration_days": 0.15,
            "equipment_utilisation_pct": 0.10,
            "is_monsoon_season": 0.05,
        }
        return delay_days, delay_prob, importances


# ─── LSTM Model ───────────────────────────────────────────────────────────────

class LSTMDelayModel:
    """Temporal sequence model for delay detection."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model = None
        self.model_path = model_path
        self.device = device

    def load(self):
        if self.model_path and Path(self.model_path).exists():
            import torch
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            logger.info("LSTM model loaded", path=self.model_path)
        else:
            logger.warning("LSTM model not found – skipping LSTM branch")

    def predict(self, sequence: np.ndarray) -> Tuple[float, float]:
        """
        Args:
            sequence: (seq_len, n_features) numpy array

        Returns:
            (predicted_delay_days, delay_probability)
        """
        if self.model is None:
            return 0.0, 0.5

        import torch
        x = torch.tensor(sequence[np.newaxis], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(x)  # expected shape: (1, 2) → [delay, prob]
        delay = float(out[0, 0].cpu())
        prob = float(torch.sigmoid(out[0, 1]).cpu())
        return max(0.0, delay), min(1.0, max(0.0, prob))


# ─── Ensemble Predictor ───────────────────────────────────────────────────────

class DelayPredictor:
    """
    Ensemble predictor combining XGBoost + LSTM.
    XGBoost weight: 0.6  |  LSTM weight: 0.4
    """

    XGB_WEIGHT = 0.6
    LSTM_WEIGHT = 0.4

    def __init__(
        self,
        xgb_model_path: Optional[str] = None,
        lstm_model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        from app.core.config import settings

        xgb_path = xgb_model_path or settings.YOLOV8_MODEL_PATH.replace(
            "yolov8_construction.pt", "delay_xgb.model"
        )
        lstm_path = lstm_model_path or settings.YOLOV8_MODEL_PATH.replace(
            "yolov8_construction.pt", "delay_lstm.pt"
        )

        self.xgb_model = XGBoostDelayModel(xgb_path)
        self.lstm_model = LSTMDelayModel(lstm_path, device=device)
        self.feat_engineer = FeatureEngineer()

        self.xgb_model.load()
        self.lstm_model.load()

    async def predict(self, project_id: str) -> Dict:
        """
        Generate delay prediction for a construction project.

        Returns prediction dict ready for DelayPrediction DB model.
        """
        from app.db.session import get_async_session

        async with get_async_session() as db:
            project, history, det_stats = await self._load_project_data(db, project_id)

        now = datetime.utcnow()

        # ── Build features ────────────────────────────────────────────
        tab_features = self.feat_engineer.build_tabular_features(
            project=project,
            progress_history=history,
            detection_stats=det_stats,
            reference_date=now,
        )
        seq = self.feat_engineer.build_lstm_sequence(history)

        # ── Run models ────────────────────────────────────────────────
        xgb_delay, xgb_prob, importances = self.xgb_model.predict(tab_features)
        lstm_delay, lstm_prob = self.lstm_model.predict(seq)

        # ── Ensemble ──────────────────────────────────────────────────
        # Weight by model availability
        if self.lstm_model.model is not None:
            ensemble_delay = self.XGB_WEIGHT * xgb_delay + self.LSTM_WEIGHT * lstm_delay
            ensemble_prob = self.XGB_WEIGHT * xgb_prob + self.LSTM_WEIGHT * lstm_prob
        else:
            ensemble_delay = xgb_delay
            ensemble_prob = xgb_prob

        # Confidence interval (±20% heuristic; replace with quantile regression)
        ci_low = max(0.0, ensemble_delay * 0.7)
        ci_high = ensemble_delay * 1.4

        # Revised completion date
        planned_end = project.get("planned_end_date")
        if isinstance(planned_end, str):
            planned_end = datetime.fromisoformat(planned_end)
        revised_end = (planned_end + timedelta(days=ensemble_delay)) if planned_end else None

        # Risk factors from feature importance
        risk_factors = self._build_risk_factors(importances, tab_features)
        top_factor = risk_factors[0]["factor"] if risk_factors else "unknown"

        # Risk level
        if ensemble_prob >= 0.7:
            risk_level = "critical"
        elif ensemble_prob >= 0.5:
            risk_level = "high"
        elif ensemble_prob >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        result = {
            "project_id": project_id,
            "predicted_delay_days": round(ensemble_delay, 1),
            "delay_probability": round(ensemble_prob, 4),
            "confidence_interval_low": round(ci_low, 1),
            "confidence_interval_high": round(ci_high, 1),
            "risk_level": risk_level,
            "top_risk_factor": top_factor,
            "risk_factors": risk_factors,
            "feature_importances": importances,
            "prediction_date": now.isoformat(),
            "revised_completion_date": revised_end.isoformat() if revised_end else None,
            "model_versions": {"xgboost": "1.0", "lstm": "1.0"},
            "xgb_prediction": round(xgb_delay, 1),
            "lstm_prediction": round(lstm_delay, 1),
        }

        # Persist to DB
        await self._save_prediction(project_id, result)

        logger.info(
            "Delay prediction generated",
            project_id=project_id,
            delay_days=result["predicted_delay_days"],
            probability=result["delay_probability"],
            risk=risk_level,
        )
        return result

    @staticmethod
    def _build_risk_factors(importances: Dict[str, float], features: pd.Series) -> List[Dict]:
        """Convert feature importances → human-readable risk factors."""
        FACTOR_LABELS = {
            "progress_delta_pct": "Schedule variance",
            "progress_velocity_7d": "Recent progress slowdown",
            "is_monsoon_season": "Weather / monsoon season",
            "equipment_utilisation_pct": "Equipment underutilisation",
            "idle_equipment_ratio": "High equipment idle time",
            "mean_daily_workers": "Workforce shortage",
            "material_delivery_lag_days": "Material supply delays",
            "remaining_duration_days": "Tight remaining schedule",
            "org_historical_delay_ratio": "Organisation delay history",
            "velocity_trend": "Decelerating progress",
        }
        factors = []
        for feat, imp in list(importances.items())[:6]:
            val = float(features.get(feat, 0.0))
            factors.append({
                "factor": FACTOR_LABELS.get(feat, feat),
                "feature": feat,
                "importance": round(imp, 4),
                "current_value": round(val, 3),
            })
        return factors

    async def _load_project_data(self, db, project_id: str):
        """Load project, progress history, and detection stats from DB."""
        from sqlalchemy import select
        from app.models.models import Project, ProgressSnapshot, DetectionResult

        # Project
        proj_row = await db.execute(
            select(Project).where(Project.id == project_id)
        )
        project_obj = proj_row.scalar_one_or_none()
        if not project_obj:
            raise ValueError(f"Project {project_id} not found")
        project = {
            "id": project_obj.id,
            "start_date": project_obj.start_date,
            "planned_end_date": project_obj.planned_end_date,
            "total_area_sqm": project_obj.total_area_sqm,
            "total_floors": project_obj.total_floors,
            "project_type": project_obj.project_type,
        }

        # Progress history (last 60 days)
        hist_rows = await db.execute(
            select(ProgressSnapshot)
            .where(ProgressSnapshot.project_id == project_id)
            .order_by(ProgressSnapshot.snapshot_date)
            .limit(90)
        )
        history = [
            {
                "overall_progress_percent": r.overall_progress_percent,
                "planned_progress_percent": r.planned_progress_percent,
                "active_workers": r.active_workers,
                "active_equipment": r.active_equipment,
                "safety_violations_detected": r.safety_violations_detected,
                "snapshot_date": r.snapshot_date.isoformat() if r.snapshot_date else None,
            }
            for r in hist_rows.scalars()
        ]

        # Aggregated detection stats (placeholder – join with DetectionResult in prod)
        det_stats = {
            "mean_daily_workers": history[-1].get("active_workers", 0) if history else 0,
            "mean_daily_equipment": history[-1].get("active_equipment", 0) if history else 0,
            "equipment_utilisation_pct": 70.0,
        }

        return project, history, det_stats

    async def _save_prediction(self, project_id: str, result: Dict):
        """Persist prediction to delay_predictions table."""
        try:
            from app.db.session import get_async_session
            from app.models.models import DelayPrediction
            from datetime import datetime

            async with get_async_session() as db:
                pred = DelayPrediction(
                    project_id=project_id,
                    model_version="ensemble-v1",
                    predicted_delay_days=result["predicted_delay_days"],
                    delay_probability=result["delay_probability"],
                    confidence_interval_low=result["confidence_interval_low"],
                    confidence_interval_high=result["confidence_interval_high"],
                    risk_factors=result["risk_factors"],
                    top_risk_factor=result["top_risk_factor"],
                    feature_importances=result["feature_importances"],
                    prediction_date=datetime.utcnow(),
                    revised_completion_date=(
                        datetime.fromisoformat(result["revised_completion_date"])
                        if result.get("revised_completion_date") else None
                    ),
                )
                db.add(pred)
                await db.commit()
        except Exception as exc:
            logger.error("Failed to save prediction", error=str(exc))
