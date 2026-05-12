"""
Multi-View Stereo (MVS) Dense Reconstruction Pipeline

Builds dense 3D point clouds and surface meshes from SfM results.

Pipeline:
  1. Undistort images using COLMAP camera models
  2. PatchMatch Stereo  – per-image depth + normal maps
  3. Stereo Fusion      – fuse depth maps → fused.ply (dense cloud)
  4. Poisson Surface    – Open3D Poisson mesh reconstruction
  5. Mesh cleaning      – remove low-density artifacts
  6. Texture mapping    – project colour onto mesh

Output artefacts (all stored to MinIO):
  - dense/fused.ply          Raw dense point cloud
  - dense/meshed_poisson.ply Poisson mesh
  - dense/meshed_clean.ply   Cleaned mesh
  - dense/cloud_stats.json   Statistics
"""

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

MVS_QUALITY = {
    "low":     {"PatchMatchStereo.geom_consistency": "false", "PatchMatchStereo.max_image_size": 1000},
    "medium":  {"PatchMatchStereo.geom_consistency": "true",  "PatchMatchStereo.max_image_size": 2000},
    "high":    {"PatchMatchStereo.geom_consistency": "true",  "PatchMatchStereo.max_image_size": 3200,
                "PatchMatchStereo.num_samples": "15"},
    "extreme": {"PatchMatchStereo.geom_consistency": "true",  "PatchMatchStereo.max_image_size": -1,
                "PatchMatchStereo.num_samples": "20"},
}


class MVSPipeline:
    """
    Dense reconstruction via COLMAP PatchMatch + Open3D surface reconstruction.
    """

    def __init__(
        self,
        quality: str = "high",
        colmap_binary: str = "colmap",
        use_gpu: bool = True,
        gpu_index: int = 0,
        poisson_depth: int = 11,
        voxel_size: float = 0.05,
    ):
        self.quality = quality
        self.preset = MVS_QUALITY.get(quality, MVS_QUALITY["high"])
        self.colmap = colmap_binary
        self.use_gpu = use_gpu
        self.gpu_index = gpu_index
        self.poisson_depth = poisson_depth   # octree depth for Poisson (higher = finer)
        self.voxel_size = voxel_size         # downsampling voxel size (metres)

    def _run(self, cmd: list, timeout: int = 10800) -> bool:
        logger.info("Running command", cmd=" ".join(cmd[:5]) + "...")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            logger.error("Command failed", stderr=r.stderr[-2000:])
            return False
        return True

    async def run(
        self,
        reconstruction_id: str,
        sfm_workspace: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict:
        """
        Run full MVS pipeline.

        Args:
            reconstruction_id: UUID of Reconstruction3D DB record
            sfm_workspace: Override workspace path (for local testing)
            progress_callback: (percent, step_name)

        Returns:
            dict with paths and statistics
        """
        from app.core.config import settings

        if sfm_workspace is None:
            sfm_workspace = os.path.join(settings.COLMAP_WORKSPACE, reconstruction_id)

        workspace = Path(sfm_workspace)
        sparse_dir = workspace / "sparse" / "0"   # best sparse model
        dense_dir  = workspace / "dense"
        dense_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.time()
        result = {
            "reconstruction_id": reconstruction_id,
            "success": False,
            "dense_dir": str(dense_dir),
            "fused_cloud_path": "",
            "mesh_path": "",
            "num_dense_points": 0,
            "duration_seconds": 0.0,
        }

        try:
            # ── 1. Undistort images ─────────────────────────────────────
            if progress_callback:
                progress_callback(5.0, "Undistorting images")

            ok = self._run([
                self.colmap, "image_undistorter",
                "--image_path",   str(workspace / "images"),
                "--input_path",   str(sparse_dir),
                "--output_path",  str(dense_dir),
                "--output_type",  "COLMAP",
                "--max_image_size", self.preset.get("PatchMatchStereo.max_image_size", "2000"),
            ])
            if not ok:
                result["error"] = "Image undistortion failed"
                return result

            # ── 2. PatchMatch Stereo ────────────────────────────────────
            if progress_callback:
                progress_callback(20.0, "PatchMatch stereo (depth maps)")

            pms_args = [
                self.colmap, "patch_match_stereo",
                "--workspace_path", str(dense_dir),
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.geom_consistency",
                    self.preset.get("PatchMatchStereo.geom_consistency", "true"),
                "--PatchMatchStereo.gpu_index", str(self.gpu_index),
            ]
            if "PatchMatchStereo.num_samples" in self.preset:
                pms_args += [
                    "--PatchMatchStereo.num_samples",
                    self.preset["PatchMatchStereo.num_samples"],
                ]

            if not self._run(pms_args, timeout=10800):
                result["error"] = "PatchMatch stereo failed"
                return result

            # ── 3. Stereo Fusion ────────────────────────────────────────
            if progress_callback:
                progress_callback(60.0, "Depth map fusion")

            fused_path = dense_dir / "fused.ply"
            ok = self._run([
                self.colmap, "stereo_fusion",
                "--workspace_path",  str(dense_dir),
                "--workspace_format", "COLMAP",
                "--input_type",      "geometric",
                "--output_path",     str(fused_path),
                "--StereoFusion.min_num_pixels", "5",
                "--StereoFusion.max_reproj_error", "2.0",
                "--StereoFusion.max_depth_error", "0.01",
            ])
            if not ok or not fused_path.exists():
                result["error"] = "Stereo fusion failed"
                return result

            if progress_callback:
                progress_callback(75.0, "Point cloud post-processing")

            # ── 4. Open3D post-processing ───────────────────────────────
            cloud_stats = await asyncio.to_thread(
                self._postprocess_point_cloud, fused_path, dense_dir
            )
            result.update(cloud_stats)

            # ── 5. Poisson surface reconstruction ───────────────────────
            if progress_callback:
                progress_callback(88.0, "Surface mesh reconstruction (Poisson)")

            mesh_path = dense_dir / "mesh_poisson.ply"
            await asyncio.to_thread(
                self._poisson_mesh, cloud_stats.get("cleaned_cloud_path", str(fused_path)), mesh_path
            )

            result["mesh_path"] = str(mesh_path) if mesh_path.exists() else ""
            result["fused_cloud_path"] = str(fused_path)
            result["success"] = True
            result["duration_seconds"] = round(time.time() - t_start, 2)

            if progress_callback:
                progress_callback(100.0, "MVS complete")

            logger.info(
                "MVS complete",
                dense_points=result["num_dense_points"],
                mesh=result["mesh_path"],
                duration_s=result["duration_seconds"],
            )

        except Exception as exc:
            logger.error("MVS pipeline error", error=str(exc), exc_info=True)
            result["error"] = str(exc)

        return result

    def _postprocess_point_cloud(self, ply_path: Path, out_dir: Path) -> Dict:
        """
        Open3D post-processing:
        - Statistical outlier removal
        - Voxel downsampling
        - Normal estimation
        """
        try:
            import open3d as o3d
        except ImportError:
            logger.warning("open3d not installed – skipping post-processing")
            return {"num_dense_points": 0, "cleaned_cloud_path": str(ply_path)}

        logger.info("Loading point cloud", path=str(ply_path))
        pcd = o3d.io.read_point_cloud(str(ply_path))
        raw_count = len(pcd.points)
        logger.info("Raw point count", n=raw_count)

        # Statistical outlier removal
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        after_stat = len(pcd_clean.points)
        logger.info("After outlier removal", n=after_stat, removed=raw_count - after_stat)

        # Voxel downsample
        pcd_down = pcd_clean.voxel_down_sample(voxel_size=self.voxel_size)
        after_down = len(pcd_down.points)
        logger.info("After voxel downsample", n=after_down, voxel_size=self.voxel_size)

        # Estimate normals (needed for Poisson)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
        pcd_down.orient_normals_consistent_tangent_plane(k=15)

        # Save cleaned cloud
        cleaned_path = out_dir / "fused_clean.ply"
        o3d.io.write_point_cloud(str(cleaned_path), pcd_down)

        # Compute bounding box
        bbox = pcd_down.get_axis_aligned_bounding_box()
        mn = bbox.min_bound
        mx = bbox.max_bound

        return {
            "num_dense_points": after_down,
            "raw_points": raw_count,
            "cleaned_cloud_path": str(cleaned_path),
            "bbox_min": [float(mn[0]), float(mn[1]), float(mn[2])],
            "bbox_max": [float(mx[0]), float(mx[1]), float(mx[2])],
            "point_cloud_size_mb": round(cleaned_path.stat().st_size / 1e6, 2),
        }

    def _poisson_mesh(self, cloud_path: str, mesh_path: Path):
        """
        Poisson surface reconstruction via Open3D.
        Requires normals in point cloud.
        """
        try:
            import open3d as o3d
        except ImportError:
            logger.warning("open3d not available for mesh reconstruction")
            return

        pcd = o3d.io.read_point_cloud(cloud_path)
        if not pcd.has_normals():
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(k=15)

        logger.info("Running Poisson surface reconstruction", depth=self.poisson_depth)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.poisson_depth, width=0, scale=1.1, linear_fit=False
        )

        # Remove low-density triangles (artefacts around boundary)
        density_thresh = np.quantile(np.asarray(densities), 0.01)
        verts_to_remove = np.asarray(densities) < density_thresh
        mesh.remove_vertices_by_mask(verts_to_remove)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

        logger.info(
            "Mesh reconstructed",
            triangles=len(mesh.triangles),
            vertices=len(mesh.vertices),
        )
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
