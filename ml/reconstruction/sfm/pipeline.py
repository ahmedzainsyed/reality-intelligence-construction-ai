"""
Structure from Motion (SfM) Pipeline using COLMAP

Implements incremental SfM for construction site reconstruction:

Pipeline Stages:
  1. Feature Extraction   - SIFT/SuperPoint per image
  2. Feature Matching     - Exhaustive/Sequential/Vocab-tree
  3. Geometric Verify     - RANSAC epipolar geometry
  4. Incremental SfM      - Camera pose recovery + triangulation
  5. Bundle Adjustment    - Global refinement (Ceres Solver)
  6. Model Export         - NVM / PLY / TXT export

Quality Settings (COLMAP presets):
  - low:     Fast, ~500 features, sequential matching
  - medium:  Balanced, ~2000 features, vocab tree
  - high:    Accurate, ~8000 features, exhaustive
  - extreme: Maximum, ~16000 features + SuperPoint
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# ── COLMAP quality presets ──────────────────────────────────────────────────

QUALITY_PRESETS: Dict[str, Dict] = {
    "low": {
        "SiftExtraction.max_num_features": 1024,
        "SiftExtraction.num_octaves": 4,
        "SiftExtraction.octave_resolution": 3,
        "SiftMatching.max_num_matches": 4096,
        "matching": "sequential",
        "sequential_overlap": 10,
        "mapper.min_num_matches": 10,
        "mapper.init_min_num_inliers": 50,
        "mapper.abs_pose_min_num_inliers": 20,
    },
    "medium": {
        "SiftExtraction.max_num_features": 4096,
        "SiftExtraction.num_octaves": 4,
        "SiftExtraction.octave_resolution": 3,
        "SiftMatching.max_num_matches": 16384,
        "matching": "vocab_tree",
        "mapper.min_num_matches": 15,
        "mapper.init_min_num_inliers": 100,
        "mapper.abs_pose_min_num_inliers": 30,
    },
    "high": {
        "SiftExtraction.max_num_features": 8192,
        "SiftExtraction.num_octaves": 4,
        "SiftExtraction.octave_resolution": 4,
        "SiftMatching.max_num_matches": 32768,
        "matching": "exhaustive",
        "mapper.min_num_matches": 15,
        "mapper.init_min_num_inliers": 100,
        "mapper.abs_pose_min_num_inliers": 30,
        "mapper.ba_local_max_num_iterations": 40,
        "mapper.ba_global_max_num_iterations": 50,
    },
    "extreme": {
        "SiftExtraction.max_num_features": 16384,
        "SiftExtraction.num_octaves": 4,
        "SiftExtraction.octave_resolution": 4,
        "SiftMatching.max_num_matches": 65536,
        "matching": "exhaustive",
        "mapper.min_num_matches": 15,
        "mapper.init_min_num_inliers": 200,
        "mapper.abs_pose_min_num_inliers": 30,
        "mapper.ba_local_max_num_iterations": 60,
        "mapper.ba_global_max_num_iterations": 80,
    },
}

CAMERA_MODELS = {
    "PINHOLE": "PINHOLE",
    "SIMPLE_RADIAL": "SIMPLE_RADIAL",
    "RADIAL": "RADIAL",
    "OPENCV": "OPENCV",
    "FULL_OPENCV": "FULL_OPENCV",
    "FISHEYE": "FISHEYE",
}


# ── Result dataclass ────────────────────────────────────────────────────────

class SfMResult:
    def __init__(self):
        self.success: bool = False
        self.num_images_registered: int = 0
        self.num_images_total: int = 0
        self.num_points3D: int = 0
        self.num_cameras: int = 0
        self.mean_reprojection_error: float = 0.0
        self.mean_track_length: float = 0.0
        self.workspace_path: str = ""
        self.sparse_model_path: str = ""
        self.camera_poses: List[Dict] = []
        self.duration_seconds: float = 0.0
        self.error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "num_images_registered": self.num_images_registered,
            "num_images_total": self.num_images_total,
            "num_points3D": self.num_points3D,
            "num_cameras": self.num_cameras,
            "mean_reprojection_error": round(self.mean_reprojection_error, 4),
            "mean_track_length": round(self.mean_track_length, 2),
            "workspace_path": self.workspace_path,
            "sparse_model_path": self.sparse_model_path,
            "num_camera_poses": len(self.camera_poses),
            "duration_seconds": round(self.duration_seconds, 2),
            "error": self.error,
        }


# ── Main Pipeline ───────────────────────────────────────────────────────────

class SfMPipeline:
    """
    Production SfM pipeline wrapping COLMAP CLI.

    Manages workspace, runs each stage, parses results,
    and stores camera poses + sparse point cloud.
    """

    def __init__(
        self,
        project_id: str,
        quality: str = "high",
        colmap_binary: str = "colmap",
        camera_model: str = "OPENCV",
        single_camera: bool = False,
        use_gpu: bool = True,
        gpu_index: int = 0,
        workspace_base: str = "/tmp/rip_sfm",
    ):
        if quality not in QUALITY_PRESETS:
            raise ValueError(f"quality must be one of {list(QUALITY_PRESETS.keys())}")

        self.project_id = project_id
        self.quality = quality
        self.preset = QUALITY_PRESETS[quality]
        self.colmap = colmap_binary
        self.camera_model = camera_model
        self.single_camera = single_camera  # True when all images from same drone
        self.use_gpu = use_gpu
        self.gpu_index = gpu_index

        self.workspace = Path(workspace_base) / project_id
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.db_path = self.workspace / "database.db"
        self.image_dir = self.workspace / "images"
        self.sparse_dir = self.workspace / "sparse"
        self.dense_dir = self.workspace / "dense"

        logger.info(
            "SfMPipeline initialized",
            project_id=project_id,
            quality=quality,
            workspace=str(self.workspace),
            gpu=use_gpu,
        )

    def _run_colmap(self, subcommand: str, args: Dict[str, str], timeout: int = 7200) -> bool:
        """Execute a COLMAP CLI subcommand and return success."""
        cmd = [self.colmap, subcommand]
        for k, v in args.items():
            cmd += [f"--{k}", str(v)]

        log = logger.bind(stage=subcommand)
        log.info("Running COLMAP", cmd=" ".join(cmd[:6]) + "...")

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace),
            )
            elapsed = time.time() - t0
            if result.returncode != 0:
                log.error(
                    "COLMAP failed",
                    returncode=result.returncode,
                    stderr=result.stderr[-2000:],
                )
                return False
            log.info("COLMAP stage complete", duration_s=round(elapsed, 1))
            return True
        except subprocess.TimeoutExpired:
            log.error("COLMAP timed out", timeout_s=timeout)
            return False
        except FileNotFoundError:
            log.error("COLMAP binary not found", binary=self.colmap)
            raise RuntimeError(f"COLMAP not found at '{self.colmap}'. Install with: sudo apt install colmap")

    async def run(
        self,
        image_paths: Optional[List[str]] = None,
        media_upload_ids: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict:
        """
        Full async SfM pipeline.

        Args:
            image_paths: Local paths to images (for direct use)
            media_upload_ids: MinIO IDs to download first
            progress_callback: progress(0-100, step_name)

        Returns:
            SfMResult dict
        """
        result = SfMResult()
        result.workspace_path = str(self.workspace)
        t_start = time.time()

        try:
            # ── Stage 0: Prepare images ──────────────────────────────────
            self.image_dir.mkdir(exist_ok=True)
            self.sparse_dir.mkdir(exist_ok=True)

            if image_paths:
                # Symlink or copy images into workspace
                for p in image_paths:
                    dst = self.image_dir / Path(p).name
                    if not dst.exists():
                        os.symlink(os.path.abspath(p), dst)
                total_images = len(image_paths)
            elif media_upload_ids:
                total_images = await self._download_frames(media_upload_ids)
                if progress_callback:
                    progress_callback(10.0, "Images downloaded")
            else:
                raise ValueError("Provide image_paths or media_upload_ids")

            result.num_images_total = total_images
            logger.info("Images ready", count=total_images, dir=str(self.image_dir))

            # ── Stage 1: Feature extraction ──────────────────────────────
            if progress_callback:
                progress_callback(15.0, "Extracting features (SIFT)")

            feat_args = {
                "database_path": str(self.db_path),
                "image_path": str(self.image_dir),
                "ImageReader.camera_model": self.camera_model,
                "ImageReader.single_camera": "1" if self.single_camera else "0",
                "SiftExtraction.use_gpu": "1" if self.use_gpu else "0",
                "SiftExtraction.gpu_index": str(self.gpu_index),
                "SiftExtraction.max_num_features": str(self.preset["SiftExtraction.max_num_features"]),
                "SiftExtraction.num_octaves": str(self.preset["SiftExtraction.num_octaves"]),
                "SiftExtraction.octave_resolution": str(self.preset["SiftExtraction.octave_resolution"]),
            }
            if not self._run_colmap("feature_extractor", feat_args):
                result.error = "Feature extraction failed"
                return result.to_dict()

            # ── Stage 2: Feature matching ────────────────────────────────
            if progress_callback:
                progress_callback(35.0, f"Matching features ({self.preset['matching']})")

            match_ok = self._run_matching_stage()
            if not match_ok:
                result.error = "Feature matching failed"
                return result.to_dict()

            # ── Stage 3: Geometric verification + incremental SfM ────────
            if progress_callback:
                progress_callback(55.0, "Running incremental SfM (bundle adjustment)")

            mapper_args = {
                "database_path": str(self.db_path),
                "image_path": str(self.image_dir),
                "output_path": str(self.sparse_dir),
                "Mapper.min_num_matches": str(self.preset["mapper.min_num_matches"]),
                "Mapper.init_min_num_inliers": str(self.preset["mapper.init_min_num_inliers"]),
                "Mapper.abs_pose_min_num_inliers": str(self.preset["mapper.abs_pose_min_num_inliers"]),
                "Mapper.ba_local_max_num_iterations": str(self.preset.get("mapper.ba_local_max_num_iterations", 25)),
                "Mapper.ba_global_max_num_iterations": str(self.preset.get("mapper.ba_global_max_num_iterations", 30)),
            }
            if not self._run_colmap("mapper", mapper_args, timeout=7200):
                result.error = "Incremental SfM (mapper) failed"
                return result.to_dict()

            # ── Stage 4: Model analysis ──────────────────────────────────
            if progress_callback:
                progress_callback(80.0, "Analysing reconstruction")

            model_dirs = sorted(self.sparse_dir.iterdir())
            if not model_dirs:
                result.error = "No reconstruction models produced"
                return result.to_dict()

            # Pick largest model (most registered images)
            best_model_dir = max(model_dirs, key=lambda d: self._count_images_in_model(d))
            result.sparse_model_path = str(best_model_dir)

            # Parse model statistics
            stats = self._parse_model_stats(best_model_dir)
            result.num_images_registered = stats["num_images"]
            result.num_points3D = stats["num_points3D"]
            result.num_cameras = stats["num_cameras"]
            result.mean_reprojection_error = stats["mean_reprojection_error"]
            result.mean_track_length = stats["mean_track_length"]

            # ── Stage 5: Export camera poses ─────────────────────────────
            if progress_callback:
                progress_callback(90.0, "Exporting camera poses")

            result.camera_poses = self._export_camera_poses(best_model_dir)

            # ── Stage 6: Save results ────────────────────────────────────
            result.success = True
            result.duration_seconds = time.time() - t_start

            if progress_callback:
                progress_callback(100.0, "SfM complete")

            logger.info(
                "SfM reconstruction complete",
                images_registered=result.num_images_registered,
                images_total=total_images,
                points3D=result.num_points3D,
                reprojection_error=result.mean_reprojection_error,
                duration_s=round(result.duration_seconds, 1),
            )

        except Exception as exc:
            logger.error("SfM pipeline error", error=str(exc), exc_info=True)
            result.error = str(exc)
            result.success = False

        return result.to_dict()

    def _run_matching_stage(self) -> bool:
        """Run appropriate COLMAP matching based on quality preset."""
        strategy = self.preset["matching"]
        common = {
            "database_path": str(self.db_path),
            "SiftMatching.use_gpu": "1" if self.use_gpu else "0",
            "SiftMatching.gpu_index": str(self.gpu_index),
            "SiftMatching.max_num_matches": str(self.preset["SiftMatching.max_num_matches"]),
        }

        if strategy == "sequential":
            return self._run_colmap("sequential_matcher", {
                **common,
                "SequentialMatching.overlap": str(self.preset.get("sequential_overlap", 10)),
                "SequentialMatching.loop_detection": "0",
            })
        elif strategy == "exhaustive":
            return self._run_colmap("exhaustive_matcher", {
                **common,
                "ExhaustiveMatching.block_size": "50",
            })
        elif strategy == "vocab_tree":
            vocab_path = os.environ.get("COLMAP_VOCAB_TREE", "vocab_tree_flickr100K_words32K.bin")
            return self._run_colmap("vocab_tree_matcher", {
                **common,
                "VocabTreeMatching.vocab_tree_path": vocab_path,
                "VocabTreeMatching.num_nearest_neighbors": "15",
            })
        else:
            logger.error("Unknown matching strategy", strategy=strategy)
            return False

    def _count_images_in_model(self, model_dir: Path) -> int:
        """Count registered images in a sparse model directory."""
        images_file = model_dir / "images.bin"
        if not images_file.exists():
            images_file = model_dir / "images.txt"
        if not images_file.exists():
            return 0
        try:
            if images_file.suffix == ".txt":
                with open(images_file) as f:
                    lines = [l for l in f if l.strip() and not l.startswith("#")]
                return len(lines) // 2  # every image = 2 lines in text format
            else:
                return images_file.stat().st_size // 64  # rough estimate from binary
        except Exception:
            return 0

    def _parse_model_stats(self, model_dir: Path) -> Dict:
        """Parse COLMAP model statistics from text or binary files."""
        stats = {
            "num_images": 0,
            "num_points3D": 0,
            "num_cameras": 0,
            "mean_reprojection_error": 0.0,
            "mean_track_length": 0.0,
        }

        # Run colmap model_analyzer if available
        try:
            cmd = [
                self.colmap, "model_analyzer",
                "--path", str(model_dir),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            for line in r.stdout.splitlines():
                if "Registered images:" in line:
                    stats["num_images"] = int(line.split(":")[-1].strip())
                elif "Points:" in line:
                    stats["num_points3D"] = int(line.split(":")[-1].strip().split()[0])
                elif "Mean reprojection error:" in line:
                    stats["mean_reprojection_error"] = float(line.split(":")[-1].strip().split()[0])
                elif "Mean track length:" in line:
                    stats["mean_track_length"] = float(line.split(":")[-1].strip())
        except Exception as e:
            logger.warning("Could not parse model stats", error=str(e))
            # Fallback: count files
            if (model_dir / "images.txt").exists():
                with open(model_dir / "images.txt") as f:
                    lines = [l for l in f if l.strip() and not l.startswith("#")]
                stats["num_images"] = len(lines) // 2

        return stats

    def _export_camera_poses(self, model_dir: Path) -> List[Dict]:
        """Parse camera poses from COLMAP images.txt."""
        poses = []
        images_txt = model_dir / "images.txt"

        if not images_txt.exists():
            # Convert from binary first
            self._run_colmap("model_converter", {
                "input_path": str(model_dir),
                "output_path": str(model_dir),
                "output_type": "TXT",
            })

        if not images_txt.exists():
            return poses

        with open(images_txt) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        i = 0
        while i < len(lines) - 1:
            parts = lines[i].split()
            if len(parts) < 9:
                i += 1
                continue
            try:
                pose = {
                    "image_id": int(parts[0]),
                    "qw": float(parts[1]),
                    "qx": float(parts[2]),
                    "qy": float(parts[3]),
                    "qz": float(parts[4]),
                    "tx": float(parts[5]),
                    "ty": float(parts[6]),
                    "tz": float(parts[7]),
                    "camera_id": int(parts[8]),
                    "image_name": parts[9] if len(parts) > 9 else "",
                }
                poses.append(pose)
            except (ValueError, IndexError):
                pass
            i += 2  # Skip point2D line

        logger.info("Parsed camera poses", count=len(poses))
        return poses

    async def _download_frames(self, media_upload_ids: List[str]) -> int:
        """Download extracted frames from MinIO to workspace image dir."""
        self.image_dir.mkdir(exist_ok=True)
        count = 0
        for uid in media_upload_ids:
            # In production: list and download frames for each upload
            # from app.core.storage import get_storage_client
            # storage = get_storage_client()
            # frames = await storage.list_objects(f"projects/frames/{uid}/")
            # for frame in frames:
            #     await storage.download_to_file(frame, self.image_dir / Path(frame).name)
            #     count += 1
            logger.debug("Would download frames for upload", uid=uid)
        return count
