"""Backend workflow state for the browser-based 3D region growing tool.

This module intentionally avoids importing PyQt so it can be used by FastAPI.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from geometric_analyzer import GeometricAnalyzer
from mesh_processor import MeshProcessor
from point_cloud_io import PointCloudFileHandler
from RegionGrowing import RegionGrowingSegmentation
from utils import filter_point_cloud


INTERFACE_PART_COLOR_CYCLE = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
]

NORMAL_VECTOR_SCALE_FRACTION = 0.01
VIEW_NORMAL_RADIUS = 0.05
VIEW_NORMAL_MAX_NN = 50

DEFAULT_CONFIG = {
    "thresholds": {
        "smoothness": 0.9,
        "curvature": 0.1,
        "basal_proximity": 0.05,
    },
    "filters": {
        "k_neighbors": 10,
        "std_ratio": 2.0,
        "vertical_std": 1.0,
        "cluster_cleanup": True,
        "adaptive_dbscan_eps": False,
        "cluster_eps": 0.02,
        "cluster_dbscan_min_points": 20,
        "cluster_min_pct": 0.01,
        "basal_clipping": False,
        "basal_clip_threshold": 0.0,
        "adaptive_k_neighbors": True,
    },
    "region_growing": {
        "voxel_size": 0.02,
        "neighbor_count": 50,
        "distance_threshold": 0.05,
    },
    "normals": {
        "method": "pymeshlab",
        "k": 200,
    },
}


def _deep_update(base: dict[str, Any], upd: dict[str, Any] | None) -> dict[str, Any]:
    for key, value in (upd or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _deep_get(dct: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = dct
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _load_config() -> dict[str, Any]:
    cfg = {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in DEFAULT_CONFIG.items()
    }
    try:
        import yaml
    except Exception:
        return cfg

    for parent in [Path.cwd(), MODULE_DIR, *MODULE_DIR.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as handle:
                    user_cfg = yaml.safe_load(handle) or {}
                return _deep_update(cfg, user_cfg)
            except Exception as exc:
                logging.warning("Failed to load config %s: %s", candidate, exc)
                return cfg
    return cfg


def _compute_part_color(part_index: int, is_lateral: bool) -> np.ndarray:
    base = np.array(INTERFACE_PART_COLOR_CYCLE[part_index % len(INTERFACE_PART_COLOR_CYCLE)], dtype=float)
    if is_lateral:
        base = np.clip(base + 0.3, 0.0, 1.0)
    return base


def _array_to_list(array: np.ndarray, precision: int | None = None) -> list[list[float]]:
    arr = np.asarray(array)
    if precision is not None:
        arr = np.round(arr.astype(float), precision)
    return arr.tolist()


def _point_bounds(points: np.ndarray) -> dict[str, list[float]]:
    if points.size == 0:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    return {
        "min": np.min(points, axis=0).astype(float).tolist(),
        "max": np.max(points, axis=0).astype(float).tolist(),
    }


def _make_point_cloud(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if colors is None:
        colors = np.full((len(points), 3), 0.5, dtype=float)
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
    return pcd


def _dense_basal_points(
    all_points: np.ndarray,
    selected_indices: list[int],
    close_loop: bool,
) -> np.ndarray:
    """Pure version of BasalPointAlgorithm.run without any PyQt dependency."""
    if len(selected_indices) < 2:
        raise ValueError("Select at least two interface points for each part.")

    basal_points_coords = all_points[np.asarray(selected_indices, dtype=int)]
    dense_points: list[np.ndarray] = []
    visited_points: set[tuple[float, float, float]] = set()

    num_points = len(basal_points_coords)
    end_range = num_points if close_loop else num_points - 1
    max_step_size = 0.1
    max_iterations = 1000
    min_progress = 0.01

    for idx in range(end_range):
        p1 = basal_points_coords[idx]
        p2 = basal_points_coords[0] if (idx == num_points - 1 and close_loop) else basal_points_coords[idx + 1]
        pair_dense_points: list[np.ndarray] = []

        current_point = p1
        if not dense_points or not np.allclose(current_point, dense_points[-1], atol=1e-3):
            pair_dense_points.append(current_point)
        visited_points.add(tuple(current_point))

        iteration_count = 0
        previous_distance = float(np.linalg.norm(p2 - current_point))

        while not np.allclose(current_point, p2, atol=1e-3):
            iteration_count += 1
            if iteration_count > max_iterations:
                break

            direction = p2 - current_point
            distance_to_p2 = float(np.linalg.norm(direction))
            if distance_to_p2 < 1e-6:
                break

            direction /= distance_to_p2
            vectors = all_points - current_point
            projections = np.dot(vectors, direction)
            mask = (projections > 0) & (projections < min(distance_to_p2, max_step_size))
            candidates = all_points[mask]
            if len(candidates) == 0:
                break

            distances_to_target = np.linalg.norm(candidates - p2, axis=1)
            distances_to_current = np.linalg.norm(candidates - current_point, axis=1)
            progress_mask = distances_to_target < (distance_to_p2 - min_progress)
            if not np.any(progress_mask):
                break

            candidates = candidates[progress_mask]
            distances_to_current = distances_to_current[progress_mask]

            next_point = None
            for candidate_idx in np.argsort(distances_to_current):
                candidate = candidates[candidate_idx]
                if tuple(candidate) not in visited_points:
                    next_point = candidate
                    break
            if next_point is None:
                break

            new_distance = float(np.linalg.norm(next_point - p2))
            if new_distance >= previous_distance:
                break
            previous_distance = new_distance

            pair_dense_points.append(next_point)
            visited_points.add(tuple(next_point))
            current_point = next_point

        if not np.allclose(current_point, p2, atol=1e-3):
            pair_dense_points.append(p2)
        dense_points.extend(pair_dense_points)

    return np.asarray(dense_points, dtype=float)


@dataclass
class WorkflowStatus:
    point_cloud_loaded: bool = False
    seeds_ready: bool = False
    interface_ready: bool = False
    segmentation_ready: bool = False
    mesh_prepared: bool = False
    mesh_completed: bool = False
    analysis_completed: bool = False


@dataclass
class WebWorkflowSession:
    session_id: str
    run_dir: Path
    config: dict[str, Any] = field(default_factory=_load_config)
    status: WorkflowStatus = field(default_factory=WorkflowStatus)

    def __post_init__(self) -> None:
        self.upload_dir = self.run_dir / "uploads"
        self.output_dir = self.run_dir / "outputs"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.file_handler = PointCloudFileHandler()
        noise_settings = {
            "sor_neighbors": int(_deep_get(self.config, "filters.k_neighbors", 10)),
            "sor_std_ratio": float(_deep_get(self.config, "filters.std_ratio", 2.0)),
            "cluster_cleanup": bool(_deep_get(self.config, "filters.cluster_cleanup", True)),
            "cluster_eps": float(_deep_get(self.config, "filters.cluster_eps", 0.02)),
            "cluster_dbscan_min_points": int(_deep_get(self.config, "filters.cluster_dbscan_min_points", 20)),
            "cluster_min_pct": float(_deep_get(self.config, "filters.cluster_min_pct", 0.01)),
            "basal_clipping": bool(_deep_get(self.config, "filters.basal_clipping", False)),
            "basal_clip_threshold": float(_deep_get(self.config, "filters.basal_clip_threshold", 0.0)),
        }
        self.mesh_processor = MeshProcessor(noise_settings=noise_settings)
        self.geometric_analyzer = GeometricAnalyzer()
        self.reset_runtime()

    def reset_runtime(self) -> None:
        self.input_path: Path | None = None
        self.current_pbr_file: str | None = None
        self.pcd: o3d.geometry.PointCloud | None = None
        self.raw_view_points: np.ndarray | None = None
        self.raw_view_colors: np.ndarray | None = None
        self.raw_view_normals: np.ndarray | None = None
        self.scene_bounds: dict[str, list[float]] | None = None
        self.seed_view_points: np.ndarray | None = None
        self.seed_view_colors: np.ndarray | None = None
        self.seed_view_normals: np.ndarray | None = None
        self.interface_view_points: np.ndarray | None = None
        self.interface_view_colors: np.ndarray | None = None
        self.interface_view_normals: np.ndarray | None = None
        self.interface_preview_view_points: np.ndarray | None = None
        self.interface_preview_view_colors: np.ndarray | None = None
        self.interface_preview_view_normals: np.ndarray | None = None
        self.interface_preview_metadata: dict[str, Any] | None = None
        self.interface_preview_dense_parts: list[np.ndarray] = []
        self.interface_preview_basal_points: np.ndarray | None = None
        self.epsg_code: int | None = None
        self.rock_seeds: list[int] = []
        self.pedestal_seeds: list[int] = []
        self.basal_points: np.ndarray | None = None
        self.dense_basal_parts: list[np.ndarray] = []
        self.dense_basal_parts_is_lateral: list[bool] = []
        self.basal_parts_metadata: dict[str, Any] = {
            "parts": [],
            "close_loop": False,
            "num_parts": 0,
            "has_lateral_parts": False,
            "palette": [],
        }
        self.interface_source: Literal["manual", "auto"] | None = None
        self.segmenter: RegionGrowingSegmentation | None = None
        self.segmented_pcd: o3d.geometry.PointCloud | None = None
        self.segmented_labels: np.ndarray | None = None
        self.prepared_mesh_data: dict[str, Any] | None = None
        self.normals_display_ready = False
        self.noise_removal_history: list[dict[str, Any]] = []
        self.segmented_pcd_file_path: str | None = None
        self.mesh_path: str | None = None
        self.analysis_csv_path: str | None = None
        self.status = WorkflowStatus()

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status.__dict__,
            "current_file": self.current_pbr_file,
            "epsg_code": self.epsg_code,
            "point_count": len(self.pcd.points) if self.pcd is not None else 0,
            "seeds": {
                "rock": self.rock_seeds,
                "pedestal": self.pedestal_seeds,
            },
            "interface_source": self.interface_source,
            "outputs": {
                "segmented": self.segmented_pcd_file_path,
                "mesh": self.mesh_path,
                "analysis": self.analysis_csv_path,
            },
        }

    def load_point_cloud(self, file_path: Path) -> dict[str, Any]:
        self.reset_runtime()
        self.input_path = file_path
        self.current_pbr_file = file_path.stem
        self.pcd, _, self.epsg_code = self.file_handler.load_las_as_open3d_point_cloud(file_path)
        self._snapshot_raw_view()
        self.status.point_cloud_loaded = True
        return self.summary()

    def _require_pcd(self) -> o3d.geometry.PointCloud:
        if self.pcd is None:
            raise ValueError("Load a point cloud before running this action.")
        return self.pcd

    def _snapshot_raw_view(self) -> None:
        pcd = self._require_pcd()
        self.raw_view_points = np.asarray(pcd.points).copy()
        colors = np.asarray(pcd.colors)
        self.raw_view_colors = colors.copy() if colors.size else np.full((len(self.raw_view_points), 3), 0.5, dtype=float)
        self.raw_view_normals = self._ensure_view_normals(pcd).copy()
        self.scene_bounds = _point_bounds(self.raw_view_points)

    def _snapshot_seed_view(self) -> None:
        pcd = self._require_pcd()
        self.seed_view_points = np.asarray(pcd.points).copy()
        self.seed_view_colors = np.asarray(pcd.colors).copy()
        self.seed_view_normals = self._ensure_view_normals(pcd).copy()

    def _snapshot_interface_view(self) -> None:
        pcd = self._require_pcd()
        self.interface_view_points = np.asarray(pcd.points).copy()
        self.interface_view_colors = np.asarray(pcd.colors).copy()
        self.interface_view_normals = self._ensure_view_normals(pcd).copy()

    def _clear_interface_preview_state(self) -> None:
        self.interface_preview_view_points = None
        self.interface_preview_view_colors = None
        self.interface_preview_view_normals = None
        self.interface_preview_metadata = None
        self.interface_preview_dense_parts = []
        self.interface_preview_basal_points = None

    def _ensure_view_normals(self, pcd: o3d.geometry.PointCloud | None = None) -> np.ndarray:
        point_cloud = pcd if pcd is not None else self._require_pcd()
        points = np.asarray(point_cloud.points)
        if len(points) == 0:
            return np.empty((0, 3), dtype=float)

        normals = np.asarray(point_cloud.normals)
        if normals.shape != points.shape:
            try:
                point_cloud.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=VIEW_NORMAL_RADIUS,
                        max_nn=VIEW_NORMAL_MAX_NN,
                    )
                )
                normals = np.asarray(point_cloud.normals)
            except Exception as exc:
                logging.warning("Failed to estimate viewer normals: %s", exc)
                normals = np.zeros_like(points, dtype=float)
                normals[:, 2] = 1.0

        if normals.shape != points.shape:
            normals = np.zeros_like(points, dtype=float)
            normals[:, 2] = 1.0

        normals = np.asarray(normals, dtype=float).copy()
        norms = np.linalg.norm(normals, axis=1)
        valid = np.all(np.isfinite(normals), axis=1) & (norms > 1e-12)
        if not np.all(valid):
            normals[~valid] = [0.0, 0.0, 1.0]
            norms = np.linalg.norm(normals, axis=1)
        normals = normals / np.maximum(norms[:, None], 1e-12)
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
        return normals

    def _cached_normals_for_points(
        self,
        points: np.ndarray | None,
        cached_normals: np.ndarray | None,
    ) -> np.ndarray | None:
        if points is None:
            return None
        point_array = np.asarray(points)
        normal_array = np.asarray(cached_normals) if cached_normals is not None else None
        if normal_array is not None and normal_array.shape == point_array.shape:
            return normal_array

        pcd = self._require_pcd()
        if len(pcd.points) == len(point_array):
            return self._ensure_view_normals(pcd)
        return None

    def _scene_bounds_for_payload(self, fallback_points: np.ndarray) -> dict[str, list[float]]:
        if self.scene_bounds is not None:
            return self.scene_bounds
        if self.raw_view_points is not None:
            self.scene_bounds = _point_bounds(self.raw_view_points)
            return self.scene_bounds
        return _point_bounds(fallback_points)

    def auto_seeds(self) -> dict[str, Any]:
        pcd = self._require_pcd()
        filtered_pcd, _ = filter_point_cloud(
            pcd,
            filter_type="sor",
            use_vertical_filter=True,
            k_neighbors=int(_deep_get(self.config, "filters.k_neighbors", 10)),
            std_ratio=float(_deep_get(self.config, "filters.std_ratio", 2.0)),
            vertical_std=float(_deep_get(self.config, "filters.vertical_std", 1.0)),
            adaptive_k=bool(_deep_get(self.config, "filters.adaptive_k_neighbors", True)),
        )
        self.pcd = filtered_pcd
        points = np.asarray(self.pcd.points)
        self._snapshot_raw_view()
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        centroid = (min_bound + max_bound) / 2.0
        distances = np.linalg.norm(points[:, :2] - centroid[:2], axis=1)
        highest_point_index = int(np.argmax(points[:, 2] - distances))
        bottommost_point_index = int(np.argmin(points[:, 2]))

        self.rock_seeds = [highest_point_index]
        self.pedestal_seeds = [bottommost_point_index]
        self.pcd.colors = o3d.utility.Vector3dVector(np.full(points.shape, 0.5, dtype=float))
        self._snapshot_seed_view()
        self.status.seeds_ready = True
        return {
            "rock_seed_indices": self.rock_seeds,
            "pedestal_seed_indices": self.pedestal_seeds,
            "summary": self.summary(),
        }

    def manual_seeds(self, rock_seed_indices: list[int], pedestal_seed_indices: list[int]) -> dict[str, Any]:
        pcd = self._require_pcd()
        point_count = len(pcd.points)
        self.rock_seeds = [self._validate_index(idx, point_count, "rock seed") for idx in rock_seed_indices]
        self.pedestal_seeds = [self._validate_index(idx, point_count, "pedestal seed") for idx in pedestal_seed_indices]
        pcd.colors = o3d.utility.Vector3dVector(np.full((point_count, 3), 0.5, dtype=float))
        self._snapshot_seed_view()
        self.status.seeds_ready = bool(self.rock_seeds and self.pedestal_seeds)
        return self.summary()

    @staticmethod
    def _validate_index(index: int, point_count: int, label: str) -> int:
        idx = int(index)
        if idx < 0 or idx >= point_count:
            raise ValueError(f"Invalid {label} index {idx}; point count is {point_count}.")
        return idx

    def _validate_interface_parts(
        self,
        parts: list[dict[str, Any]],
    ) -> tuple[list[list[int]], list[bool]]:
        pcd = self._require_pcd()
        if not parts:
            raise ValueError("Provide at least one interface part.")

        part_indices_list: list[list[int]] = []
        lateral_flags: list[bool] = []
        point_count = len(pcd.points)
        for part_idx, part in enumerate(parts):
            selected = [
                self._validate_index(idx, point_count, f"interface part {part_idx + 1}")
                for idx in part.get("selected_indices", [])
            ]
            if len(selected) < 2:
                raise ValueError(f"Interface part {part_idx + 1} needs at least two selected points.")
            part_indices_list.append(selected)
            lateral_flags.append(bool(part.get("is_lateral", False)))
        return part_indices_list, lateral_flags

    def interpolate_interface(self, parts: list[dict[str, Any]], close_loop: bool) -> dict[str, Any]:
        pcd = self._require_pcd()
        part_indices_list, lateral_flags = self._validate_interface_parts(parts)
        metadata, dense_parts, basal_indices = self._compute_basal_metadata(part_indices_list, lateral_flags, close_loop)
        colors = self._build_basal_color_array(metadata)
        self.normals_display_ready = False
        self.interface_preview_view_points = np.asarray(pcd.points).copy()
        self.interface_preview_view_colors = colors
        self.interface_preview_view_normals = self._ensure_view_normals(pcd).copy()
        self.interface_preview_metadata = metadata
        self.interface_preview_dense_parts = dense_parts
        self.interface_preview_basal_points = basal_indices
        return {
            "basal_point_count": int(len(basal_indices)),
            "metadata": metadata,
            "summary": self.summary(),
        }

    def clear_interface_preview(self) -> dict[str, Any]:
        self.normals_display_ready = False
        self._clear_interface_preview_state()
        return self.summary()

    def set_interface(self, parts: list[dict[str, Any]], close_loop: bool) -> dict[str, Any]:
        pcd = self._require_pcd()
        part_indices_list, lateral_flags = self._validate_interface_parts(parts)

        metadata, dense_parts, basal_indices = self._compute_basal_metadata(part_indices_list, lateral_flags, close_loop)
        colors = self._build_basal_color_array(metadata)
        self.normals_display_ready = False
        pcd.colors = o3d.utility.Vector3dVector(colors)
        self._snapshot_interface_view()
        self.basal_points = basal_indices
        self.dense_basal_parts = dense_parts
        self.dense_basal_parts_is_lateral = [bool(part["is_lateral"]) for part in metadata["parts"]]
        self.basal_parts_metadata = metadata
        self.interface_source = "manual"
        self._clear_interface_preview_state()
        self.status.interface_ready = True
        return {
            "basal_point_count": int(len(basal_indices)),
            "metadata": metadata,
            "summary": self.summary(),
        }

    def _compute_basal_metadata(
        self,
        part_indices_list: list[list[int]],
        lateral_flags: list[bool],
        close_loop: bool,
    ) -> tuple[dict[str, Any], list[np.ndarray], np.ndarray]:
        pcd = self._require_pcd()
        points = np.asarray(pcd.points)
        tree = cKDTree(points)

        global_close_loop = bool(close_loop)
        multi_part_close_loop = global_close_loop and len(part_indices_list) > 1
        single_part_close_loop = global_close_loop and len(part_indices_list) == 1
        first_part_indices = np.asarray(part_indices_list[0], dtype=int)

        metadata: dict[str, Any] = {
            "parts": [],
            "close_loop": global_close_loop,
            "num_parts": 0,
            "has_lateral_parts": False,
        }
        dense_parts: list[np.ndarray] = []
        all_indices: list[int] = []

        for idx, indices in enumerate(part_indices_list):
            is_lateral = bool(lateral_flags[idx] if idx < len(lateral_flags) else False)
            close_current = single_part_close_loop
            run_indices = np.asarray(indices, dtype=int)
            selected_points = points[run_indices]
            if multi_part_close_loop and idx == len(part_indices_list) - 1:
                run_indices = np.concatenate([run_indices, first_part_indices[:1]])
                close_current = False

            dense_part = _dense_basal_points(points, run_indices.tolist(), close_current)
            if dense_part.size == 0:
                raise ValueError(f"Failed to generate interface curve for part {idx + 1}.")

            _, point_indices = tree.query(dense_part)
            point_indices = np.asarray(point_indices, dtype=int)
            part_color = _compute_part_color(idx, is_lateral)
            metadata["parts"].append({
                "id": idx + 1,
                "is_lateral": is_lateral,
                "selected_indices": np.asarray(indices, dtype=int).tolist(),
                "original_points": selected_points.astype(float).tolist(),
                "dense_points": dense_part.astype(float).tolist(),
                "point_indices": point_indices.tolist(),
                "num_points": int(len(point_indices)),
                "color": part_color.tolist(),
            })
            dense_parts.append(dense_part)
            all_indices.extend(point_indices.tolist())

        metadata["num_parts"] = len(metadata["parts"])
        metadata["has_lateral_parts"] = any(part["is_lateral"] for part in metadata["parts"])
        metadata["palette"] = [list(color) for color in INTERFACE_PART_COLOR_CYCLE]
        if not all_indices:
            raise ValueError("No interface constraint points were generated from the selected inputs.")
        return metadata, dense_parts, np.unique(np.asarray(all_indices, dtype=int))

    def _build_basal_color_array(self, metadata: dict[str, Any]) -> np.ndarray:
        pcd = self._require_pcd()
        points = np.asarray(pcd.points)
        colors = np.full((len(points), 3), 0.5, dtype=float)
        for idx, part in enumerate(metadata.get("parts", []) or []):
            point_indices = np.asarray(part.get("point_indices", []), dtype=int)
            if point_indices.size == 0:
                continue
            color = np.asarray(part.get("color") or _compute_part_color(idx, bool(part.get("is_lateral"))), dtype=float)
            colors[point_indices] = color
        return colors

    def _has_manual_interface(self) -> bool:
        return self.interface_source == "manual" and bool((self.basal_parts_metadata or {}).get("parts"))

    def _manual_basal_coords_for_segmentation(self) -> np.ndarray | None:
        if not self._has_manual_interface() or self.basal_points is None or len(self.basal_points) == 0:
            return None
        return np.asarray(self._require_pcd().points)[self.basal_points]

    def _active_basal_export_data(self) -> dict[str, Any] | np.ndarray | None:
        if bool((self.basal_parts_metadata or {}).get("parts")):
            return self.basal_parts_metadata
        return self.basal_points

    def _invalidate_mesh_outputs(self) -> None:
        self.mesh_processor.reconstructed_mesh = None
        self.mesh_processor.temp_mesh_path = None
        self.mesh_path = None
        self.analysis_csv_path = None
        self.status.mesh_completed = False
        self.status.analysis_completed = False

    def _set_automatic_interface_from_segmentation(self) -> np.ndarray:
        if self.segmented_labels is None:
            raise ValueError("Run segmentation before detecting an automatic interface.")
        pcd = self._require_pcd()
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        if colors.shape != points.shape:
            colors = np.full((len(points), 3), 0.5, dtype=float)
        else:
            colors = colors.copy()

        basal_mask = np.asarray(self.detect_basal_points_optimized(points, self.segmented_labels), dtype=bool)
        basal_indices = np.flatnonzero(basal_mask)
        if len(basal_indices):
            colors[basal_indices] = np.asarray([0.0, 1.0, 0.0], dtype=float)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            self.interface_view_points = points.copy()
            self.interface_view_colors = colors.copy()
            self.interface_view_normals = self._ensure_view_normals(pcd).copy()
            if self.segmented_pcd is not None:
                self.segmented_pcd.colors = o3d.utility.Vector3dVector(colors)

        self.basal_points = basal_indices.astype(int)
        self.dense_basal_parts = []
        self.dense_basal_parts_is_lateral = []
        self.basal_parts_metadata = {
            "parts": [],
            "close_loop": False,
            "num_parts": 0,
            "has_lateral_parts": False,
            "palette": [],
        }
        self.interface_source = "auto"
        self.status.interface_ready = bool(len(basal_indices))
        return self.basal_points

    def segment(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        pcd = self._require_pcd()
        if not self.rock_seeds or not self.pedestal_seeds:
            raise ValueError("Seed selection is required before segmentation.")
        params = params or {}
        smoothness = float(params.get("smoothness_threshold", _deep_get(self.config, "thresholds.smoothness", 0.9)))
        curvature = float(params.get("curvature_threshold", _deep_get(self.config, "thresholds.curvature", 0.1)))
        basal_proximity = float(params.get("basal_proximity_threshold", _deep_get(self.config, "thresholds.basal_proximity", 0.05)))
        voxel_size = float(params.get("voxel_size", _deep_get(self.config, "region_growing.voxel_size", 0.02)))
        neighbor_count = int(params.get("neighbor_count", _deep_get(self.config, "region_growing.neighbor_count", 50)))
        distance_threshold = float(params.get("distance_threshold", _deep_get(self.config, "region_growing.distance_threshold", 0.05)))

        basal_coords = self._manual_basal_coords_for_segmentation()
        used_manual_interface_constraint = basal_coords is not None

        self.segmenter = RegionGrowingSegmentation(
            pcd,
            downsample=True,
            voxel_size=voxel_size,
            num_neighbors=neighbor_count,
            smoothness_threshold=smoothness,
            distance_threshold=distance_threshold,
            curvature_threshold=curvature,
            rock_seeds=self.rock_seeds,
            pedestal_seeds=self.pedestal_seeds,
            basal_points=basal_coords,
            basal_proximity_threshold=basal_proximity,
            stepwise_visualize=False,
        )
        self.segmenter.segment()
        self.segmenter.conditional_label_propagation()
        colored_pcd = self.segmenter.color_point_cloud()
        labels = np.asarray(self.segmenter.labels)
        labels[labels == -1] = 0

        self.pcd = colored_pcd
        self._ensure_view_normals(self.pcd)
        self.segmented_pcd = colored_pcd
        self.segmented_labels = labels
        self.status.segmentation_ready = True
        auto_interface_indices = np.asarray([], dtype=int)
        if not self._has_manual_interface():
            auto_interface_indices = self._set_automatic_interface_from_segmentation()

        self.segmented_pcd_file_path = self.file_handler.save_point_cloud(
            self.pcd,
            self.output_dir / f"{self.current_pbr_file or 'point_cloud'}_segmented.las",
            labels,
            self._active_basal_export_data(),
            plain=False,
        )
        return {
            "label_counts": {
                "pedestal": int(np.sum(labels == 0)),
                "rock": int(np.sum(labels == 1)),
            },
            "used_manual_interface_constraint": used_manual_interface_constraint,
            "auto_interface_generated": bool(len(auto_interface_indices)),
            "auto_interface_point_count": int(len(auto_interface_indices)),
            "download": self.segmented_pcd_file_path,
            "summary": self.summary(),
        }

    def prepare_mesh(self) -> dict[str, Any]:
        if self.segmented_pcd is None or self.segmented_labels is None:
            raise ValueError("Run segmentation before preparing the mesh.")
        pcd = self._require_pcd()
        if self.basal_points is None or len(self.basal_points) == 0:
            self._set_automatic_interface_from_segmentation()

        result = self.mesh_processor.prepare_bottom_face(
            pcd,
            labels=self.segmented_labels,
            basal_points=self.basal_points,
            dense_basal_parts=self.dense_basal_parts,
            dense_basal_parts_is_lateral=self.dense_basal_parts_is_lateral,
            use_dbscan_cleaning=False,
            basal_parts_metadata=self.basal_parts_metadata,
        )
        rock_pcd, bottom_pcd, combined_points, combined_colors, combined_normals = self.mesh_processor.compute_normals_for_visualization(
            result.rock_points,
            result.bottom_points,
            k=int(_deep_get(self.config, "normals.k", 200)),
        )
        self.prepared_mesh_data = {
            "rock_pcd": rock_pcd,
            "bottom_pcd": bottom_pcd,
            "combined_points": combined_points,
            "combined_colors": combined_colors,
            "combined_normals": combined_normals,
            "preparation_result": result,
        }
        self.normals_display_ready = False
        self.status.mesh_prepared = True
        self._invalidate_mesh_outputs()
        normal_diagnostics = self._normal_diagnostics(combined_points, combined_normals)
        return {
            "rock_point_count": int(len(rock_pcd.points)),
            "bottom_point_count": int(len(bottom_pcd.points)),
            "normal_segment_count": normal_diagnostics["segment_count"],
            "normal_diagnostics": normal_diagnostics,
            "summary": self.summary(),
        }

    def compute_normals(self, method: Literal["pymeshlab", "open3d"] = "pymeshlab", k: int = 200) -> dict[str, Any]:
        if not self.prepared_mesh_data:
            raise ValueError("Prepare the mesh before computing normals.")
        rock_points = np.asarray(self.prepared_mesh_data["rock_pcd"].points)
        bottom_points = np.asarray(self.prepared_mesh_data["bottom_pcd"].points)
        rock_pcd, bottom_pcd, combined_points, combined_colors, combined_normals = self._compute_prepared_normals(
            rock_points,
            bottom_points,
            method=method,
            k=k,
        )
        self.prepared_mesh_data.update({
            "rock_pcd": rock_pcd,
            "bottom_pcd": bottom_pcd,
            "combined_points": combined_points,
            "combined_colors": combined_colors,
            "combined_normals": combined_normals,
        })
        self.normals_display_ready = True
        self._invalidate_mesh_outputs()
        normal_diagnostics = self._normal_diagnostics(combined_points, combined_normals)
        return {
            "method": method,
            "k": int(k),
            "normal_segment_count": normal_diagnostics["segment_count"],
            "normal_diagnostics": normal_diagnostics,
            "normal_display_ready": True,
            "summary": self.summary(),
        }

    def _compute_prepared_normals(
        self,
        rock_points: np.ndarray,
        bottom_points: np.ndarray,
        method: Literal["pymeshlab", "open3d"] | str | None = None,
        k: int | None = None,
    ) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, np.ndarray, np.ndarray, np.ndarray]:
        normal_method = str(method or _deep_get(self.config, "normals.method", "pymeshlab")).lower()
        if normal_method == "open3d":
            compute_fn = self.mesh_processor.compute_normals_for_visualization_separate
        else:
            compute_fn = self.mesh_processor.compute_normals_for_visualization

        normal_k = max(3, int(k if k is not None else _deep_get(self.config, "normals.k", 200)))
        try:
            return compute_fn(
                np.asarray(rock_points, dtype=np.float64),
                np.asarray(bottom_points, dtype=np.float64),
                k=normal_k,
            )
        except Exception as exc:
            logging.warning(
                "Normal computation with %s failed; using basic Open3D fallback: %s",
                normal_method,
                exc,
            )
            return self._compute_basic_open3d_normals(rock_points, bottom_points, normal_k)

    def _compute_basic_open3d_normals(
        self,
        rock_points: np.ndarray,
        bottom_points: np.ndarray,
        k: int,
    ) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, np.ndarray, np.ndarray, np.ndarray]:
        rock_pcd = self._point_cloud_with_basic_normals(np.asarray(rock_points, dtype=np.float64), k)
        bottom_pcd = self._point_cloud_with_basic_normals(
            np.asarray(bottom_points, dtype=np.float64),
            k,
            preferred_direction=np.array([0.0, 0.0, -1.0]),
        )
        combined_points = np.vstack((np.asarray(rock_pcd.points), np.asarray(bottom_pcd.points)))
        combined_colors = np.vstack((
            np.full((len(np.asarray(rock_pcd.points)), 3), [1.0, 0.0, 0.0]),
            np.full((len(np.asarray(bottom_pcd.points)), 3), [0.0, 1.0, 0.0]),
        ))
        combined_normals = np.vstack((np.asarray(rock_pcd.normals), np.asarray(bottom_pcd.normals)))
        return rock_pcd, bottom_pcd, combined_points, combined_colors, combined_normals

    def _point_cloud_with_basic_normals(
        self,
        points: np.ndarray,
        k: int,
        preferred_direction: np.ndarray | None = None,
    ) -> o3d.geometry.PointCloud:
        point_array = np.asarray(points, dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_array)
        if len(point_array) == 0:
            pcd.normals = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            return pcd

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max(1, min(int(k), len(point_array)))))
        pcd.normalize_normals()
        normals = np.asarray(pcd.normals, dtype=np.float64).copy()
        if normals.shape != point_array.shape or not np.all(np.isfinite(normals)):
            normals = np.zeros_like(point_array)
            normals[:, 2] = 1.0

        if preferred_direction is not None:
            direction = preferred_direction / max(float(np.linalg.norm(preferred_direction)), 1e-12)
            flips = np.einsum("ij,j->i", normals, direction) < 0
        else:
            radial = point_array - np.mean(point_array, axis=0)
            flips = np.einsum("ij,ij->i", normals, radial) < 0
        normals[flips] *= -1.0
        pcd.normals = o3d.utility.Vector3dVector(normals)
        return pcd

    def _update_prepared_mesh_data(
        self,
        rock_pcd: o3d.geometry.PointCloud,
        bottom_pcd: o3d.geometry.PointCloud,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rock_points = np.asarray(rock_pcd.points)
        bottom_points = np.asarray(bottom_pcd.points)
        combined_points = np.vstack((rock_points, bottom_points))
        combined_colors = np.vstack((
            np.full((len(rock_points), 3), [1.0, 0.0, 0.0]),
            np.full((len(bottom_points), 3), [0.0, 1.0, 0.0]),
        ))
        rock_normals = np.asarray(rock_pcd.normals) if rock_pcd.has_normals() else np.zeros_like(rock_points)
        bottom_normals = np.asarray(bottom_pcd.normals) if bottom_pcd.has_normals() else np.zeros_like(bottom_points)
        combined_normals = np.vstack((rock_normals, bottom_normals))
        self.prepared_mesh_data.update({
            "rock_pcd": rock_pcd,
            "bottom_pcd": bottom_pcd,
            "combined_points": combined_points,
            "combined_colors": combined_colors,
            "combined_normals": combined_normals,
        })
        return combined_points, combined_colors, combined_normals

    def remove_noise(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.prepared_mesh_data:
            raise ValueError("Prepare the mesh before removing noise.")
        params = params or {}
        method = str(params.get("method", "sor")).lower()
        if method not in {"sor", "dbscan", "sor_dbscan"}:
            raise ValueError("Denoise method must be 'sor', 'dbscan', or 'sor_dbscan'.")
        self.noise_removal_history.append(dict(self.prepared_mesh_data))

        rock_pcd = self.prepared_mesh_data["rock_pcd"]
        bottom_pcd = self.prepared_mesh_data["bottom_pcd"]
        initial_count = len(rock_pcd.points)

        if method in {"sor", "sor_dbscan"}:
            sor_neighbors = max(3, int(params.get("sor_neighbors", _deep_get(self.config, "filters.k_neighbors", 10))))
            sor_std_ratio = max(0.01, float(params.get("sor_std_ratio", _deep_get(self.config, "filters.std_ratio", 2.0))))
            rock_pcd, _ = rock_pcd.remove_statistical_outlier(
                nb_neighbors=sor_neighbors,
                std_ratio=sor_std_ratio,
            )

        if method in {"dbscan", "sor_dbscan"}:
            dbscan_eps = max(1e-6, float(params.get("dbscan_eps", _deep_get(self.config, "filters.cluster_eps", 0.02))))
            dbscan_min_points = max(1, int(params.get(
                "dbscan_min_points",
                _deep_get(self.config, "filters.cluster_dbscan_min_points", 20),
            )))
            rock_pcd = self.mesh_processor.clean_outliers_dbscan(
                rock_pcd,
                eps=dbscan_eps,
                min_samples=dbscan_min_points,
                return_inlier_indices=False,
            )

        normal_method = str(params.get("normal_method", _deep_get(self.config, "normals.method", "pymeshlab"))).lower()
        normal_k = max(3, int(params.get("normal_k", _deep_get(self.config, "normals.k", 200))))
        rock_pcd, bottom_pcd, combined_points, _, combined_normals = self._compute_prepared_normals(
            np.asarray(rock_pcd.points),
            np.asarray(bottom_pcd.points),
            method=normal_method,
            k=normal_k,
        )
        self._update_prepared_mesh_data(rock_pcd, bottom_pcd)
        self.normals_display_ready = False
        self._invalidate_mesh_outputs()
        normal_diagnostics = self._normal_diagnostics(combined_points, combined_normals)
        return {
            "method": method,
            "initial_rock_point_count": int(initial_count),
            "rock_point_count": int(len(rock_pcd.points)),
            "removed_rock_point_count": int(max(0, initial_count - len(rock_pcd.points))),
            "normal_method": normal_method,
            "normal_k": normal_k,
            "normal_segment_count": normal_diagnostics["segment_count"],
            "normal_diagnostics": normal_diagnostics,
            "summary": self.summary(),
        }

    def manual_remove_prepared_points(self, selected_indices: list[int]) -> dict[str, Any]:
        if not self.prepared_mesh_data:
            raise ValueError("Prepare the mesh before manually removing points.")
        if not selected_indices:
            raise ValueError("Select at least one prepared point to remove.")

        rock_pcd = self.prepared_mesh_data["rock_pcd"]
        bottom_pcd = self.prepared_mesh_data["bottom_pcd"]
        rock_count = len(rock_pcd.points)
        bottom_count = len(bottom_pcd.points)
        total_count = rock_count + bottom_count
        initial_rock_count = int(rock_count)
        initial_bottom_count = int(bottom_count)
        selected = np.unique(np.asarray(selected_indices, dtype=int))
        valid = selected[(selected >= 0) & (selected < total_count)]
        rock_removable = valid[valid < rock_count]
        bottom_removable = valid[valid >= rock_count] - rock_count
        if len(valid) == 0:
            raise ValueError("The polygon did not select any removable prepared points.")
        if rock_count > 0 and len(rock_removable) >= rock_count:
            raise ValueError("Manual removal would remove all prepared rock points.")
        if bottom_count > 0 and len(bottom_removable) >= bottom_count:
            raise ValueError("Manual removal would remove all interpolated bottom-face points.")

        self.noise_removal_history.append(dict(self.prepared_mesh_data))
        rock_keep_indices = np.setdiff1d(np.arange(rock_count, dtype=int), rock_removable, assume_unique=False)
        bottom_keep_indices = np.setdiff1d(np.arange(bottom_count, dtype=int), bottom_removable, assume_unique=False)
        rock_pcd = rock_pcd.select_by_index(rock_keep_indices.astype(int).tolist())
        bottom_pcd = bottom_pcd.select_by_index(bottom_keep_indices.astype(int).tolist())
        self._update_prepared_mesh_data(rock_pcd, bottom_pcd)
        self.normals_display_ready = False
        self._invalidate_mesh_outputs()
        return {
            "removed_rock_point_count": int(len(rock_removable)),
            "removed_interpolated_point_count": int(len(bottom_removable)),
            "ignored_out_of_range_point_count": int(len(selected) - len(valid)),
            "initial_rock_point_count": initial_rock_count,
            "initial_interpolated_point_count": initial_bottom_count,
            "rock_point_count": int(len(rock_pcd.points)),
            "interpolated_point_count": int(len(bottom_pcd.points)),
            "summary": self.summary(),
        }

    def undo_noise(self) -> dict[str, Any]:
        if not self.noise_removal_history:
            raise ValueError("No noise removal step is available to undo.")
        self.prepared_mesh_data = self.noise_removal_history.pop()
        self.normals_display_ready = False
        self._invalidate_mesh_outputs()
        return self.summary()

    def prepared_normal_diagnostics(self) -> dict[str, Any]:
        if not self.prepared_mesh_data:
            raise ValueError("Prepare the mesh before inspecting normals.")
        return self._normal_diagnostics(
            self.prepared_mesh_data["combined_points"],
            self.prepared_mesh_data.get("combined_normals"),
        )

    def reconstruct_mesh(self, depth: int = 8) -> dict[str, Any]:
        if not self.prepared_mesh_data:
            raise ValueError("Prepare the mesh before reconstruction.")
        rock_pcd = self.prepared_mesh_data["rock_pcd"]
        bottom_pcd = self.prepared_mesh_data["bottom_pcd"]
        if not rock_pcd.has_normals() or not bottom_pcd.has_normals():
            raise ValueError("Compute normals before mesh reconstruction.")

        payload = {
            "rock_points": np.asarray(rock_pcd.points, dtype=np.float64).copy(),
            "rock_normals": np.asarray(rock_pcd.normals, dtype=np.float64).copy(),
            "bottom_points": np.asarray(bottom_pcd.points, dtype=np.float64).copy(),
            "bottom_normals": np.asarray(bottom_pcd.normals, dtype=np.float64).copy(),
            "depth": int(depth),
        }

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=MeshProcessor.poisson_worker_entrypoint,
            args=(child_conn, payload),
            name="web_poisson_reconstruction_worker",
        )
        process.start()
        child_conn.close()

        worker_result = None
        try:
            process.join()
            if parent_conn.poll():
                worker_result = parent_conn.recv()
            else:
                worker_result = {"success": False, "message": "Worker exited without returning a result."}
        finally:
            parent_conn.close()
            if process.is_alive():
                process.terminate()
                process.join()

        if not worker_result or not worker_result.get("success"):
            message = worker_result.get("message", "Mesh reconstruction failed.") if worker_result else "Mesh reconstruction failed."
            if worker_result and worker_result.get("traceback"):
                logging.error("Poisson worker traceback:\n%s", worker_result["traceback"])
            raise RuntimeError(message)

        mesh_path = worker_result.get("mesh_path")
        if not mesh_path or not os.path.exists(mesh_path):
            raise RuntimeError("Worker completed but produced no mesh output.")

        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh is None or len(mesh.triangles) == 0:
            raise RuntimeError("Worker returned an empty mesh.")

        self.mesh_processor.reconstructed_mesh = mesh
        self.mesh_processor.temp_mesh_path = mesh_path
        self.mesh_path = self.mesh_processor.save_mesh(self.output_dir / f"{self.current_pbr_file or 'point_cloud'}_mesh.ply")
        self.status.mesh_completed = True
        return {
            "mesh_path": self.mesh_path,
            "vertex_count": int(len(mesh.vertices)),
            "triangle_count": int(len(mesh.triangles)),
            "summary": self.summary(),
        }

    def analyze(self) -> dict[str, Any]:
        if self.mesh_processor.reconstructed_mesh is None:
            raise ValueError("Complete mesh reconstruction before analysis.")
        pcd = self._require_pcd()
        if self.basal_points is None or len(self.basal_points) == 0:
            raise ValueError("Define interface constraints before running analysis.")
        if self.segmented_labels is None:
            raise ValueError("Run segmentation before analysis.")

        pedestal_points = np.asarray(pcd.points)[self.segmented_labels == 0]
        if len(pedestal_points) == 0:
            raise ValueError("No pedestal points are available for beta-angle analysis.")
        basal_coords = np.asarray(pcd.points)[self.basal_points]
        results = self.geometric_analyzer.compute_geometric_properties(
            self.mesh_processor.reconstructed_mesh,
            basal_coords,
            pedestal_points,
            lateral_flags=None,
        )
        self.analysis_csv_path = self.geometric_analyzer.save_results(
            results,
            self.current_pbr_file or "point_cloud",
            self.input_path or "",
            self.segmented_pcd_file_path or "",
            self.mesh_path or "",
            smoothness_threshold=float(_deep_get(self.config, "thresholds.smoothness", 0.9)),
            curvature_threshold=float(_deep_get(self.config, "thresholds.curvature", 0.1)),
            proximity_threshold=float(_deep_get(self.config, "thresholds.basal_proximity", 0.05)),
            user=None,
            epsg_code=self.epsg_code,
            output_csv=self.output_dir / f"{self.current_pbr_file or 'point_cloud'}_analysis.csv",
            include_user=False,
        )
        self.status.analysis_completed = True
        serializable_results = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in results.items()
        }
        return {
            "results": serializable_results,
            "analysis_csv_path": self.analysis_csv_path,
            "summary": self.summary(),
        }

    def viewer_payload(self, view_name: str, mesh_url: str | None = None) -> dict[str, Any]:
        if view_name == "mesh":
            if not self.mesh_path:
                raise ValueError("No reconstructed mesh is available.")
            payload: dict[str, Any] = {"kind": "mesh", "url": mesh_url, "show_wireframe": True}
            if self.scene_bounds is not None:
                payload["scene_bounds"] = self.scene_bounds
            return payload

        if view_name == "raw":
            if self.raw_view_points is None or self.raw_view_colors is None:
                pcd = self._require_pcd()
                self._snapshot_raw_view()
            return self._point_payload(
                self.raw_view_points,
                self.raw_view_colors,
                np.arange(len(self.raw_view_points), dtype=int),
                markers=[],
                normals=self._cached_normals_for_points(self.raw_view_points, self.raw_view_normals),
            )
        elif view_name == "seeds":
            points = self.seed_view_points if self.seed_view_points is not None else self.raw_view_points
            colors = self.seed_view_colors if self.seed_view_colors is not None else self.raw_view_colors
            normals = self.seed_view_normals if self.seed_view_normals is not None else self.raw_view_normals
            if points is None or colors is None:
                pcd = self._require_pcd()
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                normals = self._ensure_view_normals(pcd)
            return self._point_payload(
                points,
                colors,
                np.arange(len(points), dtype=int),
                markers=self._current_markers(include_interface=False),
                normals=self._cached_normals_for_points(points, normals),
            )
        elif view_name == "interface":
            if self.interface_preview_view_points is not None and self.interface_preview_view_colors is not None:
                points = self.interface_preview_view_points
                colors = self.interface_preview_view_colors
                normals = self.interface_preview_view_normals
                interface_metadata = self.interface_preview_metadata
            else:
                points = self.interface_view_points
                colors = self.interface_view_colors
                normals = self.interface_view_normals
                interface_metadata = self.basal_parts_metadata
            if points is None or colors is None:
                points = self.seed_view_points if self.seed_view_points is not None else self.raw_view_points
                colors = self.seed_view_colors if self.seed_view_colors is not None else self.raw_view_colors
                normals = self.seed_view_normals if self.seed_view_normals is not None else self.raw_view_normals
            if points is None or colors is None:
                pcd = self._require_pcd()
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                normals = self._ensure_view_normals(pcd)
            payload = self._point_payload(
                points,
                colors,
                np.arange(len(points), dtype=int),
                markers=self._current_markers(include_interface=True, interface_metadata=interface_metadata),
                normals=self._cached_normals_for_points(points, normals),
            )
            interface_points, interface_normals = self._interface_normal_arrays(interface_metadata)
            if self.normals_display_ready and interface_points is not None and interface_normals is not None:
                payload["normal_segments"] = self._normal_segments(
                    interface_points,
                    interface_normals,
                    scale_points=np.asarray(points),
                    scale_fraction=NORMAL_VECTOR_SCALE_FRACTION,
                )
                payload["normal_diagnostics"] = self._normal_diagnostics(
                    interface_points,
                    interface_normals,
                    scale_points=np.asarray(points),
                    scale_fraction=NORMAL_VECTOR_SCALE_FRACTION,
                )
            return payload
        elif view_name == "segmented":
            pcd = self.segmented_pcd
            markers = []
        elif view_name == "mesh_prepared":
            if not self.prepared_mesh_data:
                raise ValueError("No prepared mesh point cloud is available.")
            payload = self._point_payload(
                self.prepared_mesh_data["combined_points"],
                self.prepared_mesh_data["combined_colors"],
                np.arange(len(self.prepared_mesh_data["combined_points"]), dtype=int),
                markers=[],
                normals=self.prepared_mesh_data.get("combined_normals"),
            )
            payload["rock_point_count"] = int(len(self.prepared_mesh_data["rock_pcd"].points))
            payload["bottom_point_count"] = int(len(self.prepared_mesh_data["bottom_pcd"].points))
            if self.normals_display_ready:
                payload["normal_segments"] = self._normal_segments(
                    self.prepared_mesh_data["combined_points"],
                    self.prepared_mesh_data.get("combined_normals"),
                )
                payload["normal_diagnostics"] = self._normal_diagnostics(
                    self.prepared_mesh_data["combined_points"],
                    self.prepared_mesh_data.get("combined_normals"),
                )
            return payload
        else:
            pcd = self._require_pcd()
            markers = self._current_markers()

        if pcd is None:
            raise ValueError(f"View '{view_name}' is not available yet.")

        return self._point_payload(
            np.asarray(pcd.points),
            np.asarray(pcd.colors),
            np.arange(len(pcd.points), dtype=int),
            markers=markers,
            normals=self._ensure_view_normals(pcd),
        )

    def _point_payload(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        source_indices: np.ndarray,
        markers: list[dict[str, Any]],
        normals: np.ndarray | None = None,
    ) -> dict[str, Any]:
        sampled_points = np.asarray(points)
        sampled_colors = np.asarray(colors) if colors is not None and len(colors) else np.full((len(sampled_points), 3), 0.5)
        sampled_indices = np.asarray(source_indices)
        payload = {
            "kind": "pointCloud",
            "points": _array_to_list(sampled_points, precision=6),
            "colors": _array_to_list(sampled_colors, precision=4),
            "indices": sampled_indices.astype(int).tolist(),
            "bounds": _point_bounds(np.asarray(points)),
            "scene_bounds": self._scene_bounds_for_payload(np.asarray(points)),
            "markers": markers,
            "total_points": int(len(points)),
            "rendered_points": int(len(sampled_points)),
        }
        sampled_normals = np.asarray(normals) if normals is not None else None
        if sampled_normals is not None and sampled_normals.shape == sampled_points.shape:
            payload["normals"] = _array_to_list(sampled_normals, precision=5)
        return payload

    def _interface_normal_arrays(
        self,
        interface_metadata: dict[str, Any] | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not interface_metadata:
            return None, None
        dense_parts: list[np.ndarray] = []
        for part in interface_metadata.get("parts", []) or []:
            dense_points = np.asarray(part.get("dense_points", []), dtype=float)
            if dense_points.ndim == 2 and dense_points.shape[1] == 3 and len(dense_points):
                dense_parts.append(dense_points)
                continue
            pcd = self._require_pcd()
            all_points = np.asarray(pcd.points)
            indices = np.asarray(part.get("point_indices", []), dtype=int)
            valid = indices[(indices >= 0) & (indices < len(all_points))]
            if len(valid):
                dense_parts.append(all_points[valid])
        if not dense_parts:
            return None, None
        interface_points = np.vstack(dense_parts)
        normals = self._estimate_local_normals(interface_points)
        return interface_points, normals

    def _estimate_local_normals(
        self,
        query_points: np.ndarray,
        max_neighbors: int = 50,
    ) -> np.ndarray:
        pcd = self._require_pcd()
        all_points = np.asarray(pcd.points, dtype=float)
        query_array = np.asarray(query_points, dtype=float)
        if len(query_array) == 0:
            return np.zeros((0, 3), dtype=float)
        if len(all_points) < 3:
            normals = np.zeros_like(query_array)
            normals[:, 2] = 1.0
            return normals

        tree = cKDTree(all_points)
        source_normals = np.asarray(pcd.normals, dtype=float) if pcd.has_normals() else None
        if source_normals is not None and source_normals.shape == all_points.shape:
            _, nearest_indices = tree.query(query_array, k=1)
            normals = source_normals[np.asarray(nearest_indices, dtype=int)].copy()
        else:
            neighbor_count = max(3, min(int(max_neighbors), len(all_points)))
            _, neighbor_indices = tree.query(query_array, k=neighbor_count)
            neighbor_indices = np.asarray(neighbor_indices, dtype=int)
            if neighbor_indices.ndim == 1:
                neighbor_indices = neighbor_indices[:, None]
            normals = np.zeros_like(query_array)
            cloud_centroid = np.mean(all_points, axis=0)
            for idx, neighbors_for_point in enumerate(neighbor_indices):
                neighbors = all_points[neighbors_for_point]
                if len(neighbors) < 3:
                    normals[idx] = [0.0, 0.0, 1.0]
                    continue
                centered = neighbors - np.mean(neighbors, axis=0)
                covariance = centered.T @ centered
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
                except np.linalg.LinAlgError:
                    normal = np.array([0.0, 0.0, 1.0], dtype=float)
                norm = float(np.linalg.norm(normal))
                if not np.isfinite(norm) or norm <= 1e-12:
                    normal = np.array([0.0, 0.0, 1.0], dtype=float)
                else:
                    normal = normal / norm
                radial = query_array[idx] - cloud_centroid
                if float(np.linalg.norm(radial)) > 1e-12:
                    if float(np.dot(normal, radial)) < 0:
                        normal *= -1.0
                elif normal[2] < 0:
                    normal *= -1.0
                normals[idx] = normal

        norms = np.linalg.norm(normals, axis=1)
        invalid = (~np.isfinite(norms)) | (norms <= 1e-12)
        if np.any(invalid):
            normals[invalid] = [0.0, 0.0, 1.0]
            norms = np.linalg.norm(normals, axis=1)
        return normals / np.maximum(norms[:, None], 1e-12)

    def _normal_segments(
        self,
        points: np.ndarray,
        normals: np.ndarray | None,
        max_segments: int = 6000,
        scale_points: np.ndarray | None = None,
        scale_fraction: float = NORMAL_VECTOR_SCALE_FRACTION,
    ) -> list[list[list[float]]]:
        point_array = np.asarray(points, dtype=float)
        if normals is None or len(point_array) == 0:
            return []
        normal_array = np.asarray(normals, dtype=float)
        if normal_array.shape != point_array.shape:
            return []

        scale_array = np.asarray(scale_points, dtype=float) if scale_points is not None else point_array
        bounds = _point_bounds(scale_array)
        extent = np.asarray(bounds["max"], dtype=float) - np.asarray(bounds["min"], dtype=float)
        scale = max(float(np.linalg.norm(extent)) * float(scale_fraction), 0.001)
        stride = max(1, int(np.ceil(len(point_array) / max(1, max_segments))))
        segments: list[list[list[float]]] = []
        for idx in range(0, len(point_array), stride):
            normal = normal_array[idx]
            norm = float(np.linalg.norm(normal))
            if not np.isfinite(norm) or norm <= 1e-12:
                continue
            start = point_array[idx]
            if not np.all(np.isfinite(start)):
                continue
            end = start + (normal / norm) * scale
            if not np.all(np.isfinite(end)):
                continue
            segments.append([
                [round(float(value), 6) for value in start],
                [round(float(value), 6) for value in end],
            ])
        return segments

    def _normal_diagnostics(
        self,
        points: np.ndarray,
        normals: np.ndarray | None,
        max_segments: int = 6000,
        scale_points: np.ndarray | None = None,
        scale_fraction: float = NORMAL_VECTOR_SCALE_FRACTION,
    ) -> dict[str, Any]:
        point_array = np.asarray(points, dtype=float)
        base = {
            "point_count": int(len(point_array)),
            "normal_shape": None,
            "finite_normal_count": 0,
            "nonzero_normal_count": 0,
            "segment_count": 0,
            "stride": 0,
            "scale": 0.0,
            "min_norm": 0.0,
            "mean_norm": 0.0,
            "max_norm": 0.0,
            "status": "missing",
        }
        if normals is None or len(point_array) == 0:
            return base

        normal_array = np.asarray(normals, dtype=float)
        base["normal_shape"] = [int(value) for value in normal_array.shape]
        if normal_array.shape != point_array.shape:
            base["status"] = "shape_mismatch"
            return base

        finite_mask = np.all(np.isfinite(normal_array), axis=1)
        norms = np.linalg.norm(normal_array, axis=1)
        nonzero_mask = finite_mask & np.isfinite(norms) & (norms > 1e-12)
        finite_norms = norms[nonzero_mask]
        base["finite_normal_count"] = int(np.count_nonzero(finite_mask))
        base["nonzero_normal_count"] = int(np.count_nonzero(nonzero_mask))
        if len(finite_norms):
            base["min_norm"] = float(np.min(finite_norms))
            base["mean_norm"] = float(np.mean(finite_norms))
            base["max_norm"] = float(np.max(finite_norms))

        scale_array = np.asarray(scale_points, dtype=float) if scale_points is not None else point_array
        bounds = _point_bounds(scale_array)
        extent = np.asarray(bounds["max"], dtype=float) - np.asarray(bounds["min"], dtype=float)
        base["scale"] = float(max(float(np.linalg.norm(extent)) * float(scale_fraction), 0.001))
        base["stride"] = int(max(1, int(np.ceil(len(point_array) / max(1, max_segments)))))
        base["segment_count"] = int(len(self._normal_segments(
            point_array,
            normal_array,
            max_segments=max_segments,
            scale_points=scale_array,
            scale_fraction=scale_fraction,
        )))
        base["status"] = "ok" if base["segment_count"] else "empty_segments"
        return base

    def _current_markers(
        self,
        include_interface: bool = True,
        interface_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.pcd is None:
            return []
        points = np.asarray(self.pcd.points)
        markers: list[dict[str, Any]] = []
        for idx in self.rock_seeds:
            if 0 <= idx < len(points):
                markers.append({"index": int(idx), "point": points[idx].astype(float).tolist(), "color": [1, 0, 0], "label": "Rock seed"})
        for idx in self.pedestal_seeds:
            if 0 <= idx < len(points):
                markers.append({"index": int(idx), "point": points[idx].astype(float).tolist(), "color": [0, 0.2, 1], "label": "Pedestal seed"})
        if include_interface:
            metadata = interface_metadata if interface_metadata is not None else self.basal_parts_metadata
            for part in metadata.get("parts", []) or []:
                color = part.get("color") or [0, 1, 0]
                for idx in part.get("selected_indices", []) or []:
                    if 0 <= idx < len(points):
                        markers.append({"index": int(idx), "point": points[idx].astype(float).tolist(), "color": color, "label": f"Interface {part['id']}"})
        return markers

    def download_path(self, kind: str) -> Path:
        lookup = {
            "segmented": self.segmented_pcd_file_path,
            "segmented_pcd": self.segmented_pcd_file_path,
            "mesh": self.mesh_path,
            "analysis": self.analysis_csv_path,
            "analysis_csv": self.analysis_csv_path,
        }
        value = lookup.get(kind)
        if not value:
            raise ValueError(f"No downloadable file is available for '{kind}'.")
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(str(path))
        return path

    @staticmethod
    def detect_basal_points_optimized(points: np.ndarray, labels: np.ndarray, k: int = 30, threshold: float = 0.35) -> np.ndarray:
        tree = cKDTree(points)
        _, indices = tree.query(points, k=min(k, len(points)))
        neighborhood_labels = labels[indices]
        rock_ratios = np.sum(neighborhood_labels == 1, axis=1) / neighborhood_labels.shape[1]
        return (threshold <= rock_ratios) & (rock_ratios <= (1 - threshold))


def copy_upload_to_session(src, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as handle:
        shutil.copyfileobj(src, handle)
    return target


def temp_traceback() -> str:
    return traceback.format_exc()
