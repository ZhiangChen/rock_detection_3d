"""Backend workflow state for the browser-based 3D region growing tool.

This module intentionally avoids importing PyQt so it can be used by FastAPI.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import json
import re
import shutil
import sys
import tempfile
import traceback
import zipfile
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import laspy
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


INTERFACE_GREEN = (0.0, 1.0, 0.0)
INTERFACE_PART_COLOR_CYCLE = [INTERFACE_GREEN]
BRANCH_COLOR_PALETTE = [
    [0.93, 0.18, 0.14],
    [0.10, 0.36, 0.95],
    [0.96, 0.62, 0.05],
    [0.00, 0.62, 0.45],
    [0.55, 0.28, 0.90],
    [0.90, 0.20, 0.58],
    [0.42, 0.70, 0.12],
    [0.13, 0.66, 0.86],
    [0.75, 0.38, 0.00],
    [0.35, 0.35, 0.35],
]

NORMAL_VECTOR_SCALE_FRACTION = 0.01
VIEW_NORMAL_RADIUS = 0.05
VIEW_NORMAL_MAX_NN = 50
DRAFT_TARGET_ANCHORS = 40
DRAFT_MIN_ANCHORS = 12
DRAFT_MAX_ANCHORS = 80
DRAFT_HISTORY_LIMIT = 20
BRUSH_ADD_MIN_ANCHORS = 3
BRUSH_ADD_MAX_ANCHORS = 18
BRUSH_REMOVE_MAX_ANCHORS = 12
BRUSH_JUMP_MEDIAN_MULTIPLIER = 12.0

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
        "label_propagation_distance": 0.05,
    },
    "normals": {
        "method": "pymeshlab",
        "k": 200,
    },
}

PROJECT_FORMAT = "rock_detection_3d.project"
PROJECT_SCHEMA_VERSION = 1
PROJECT_ARCHIVE_SUFFIX = ".rd3dproj"


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
    return np.array(INTERFACE_GREEN, dtype=float)


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


def _make_point_cloud_with_normals(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    pcd = _make_point_cloud(points, colors)
    normal_array = np.asarray(normals, dtype=float) if normals is not None else None
    point_array = np.asarray(points)
    if normal_array is not None and normal_array.shape == point_array.shape:
        pcd.normals = o3d.utility.Vector3dVector(normal_array)
    return pcd


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _safe_filename(name: str | None, default: str, suffix: str | None = None) -> str:
    raw_name = Path(str(name or default)).name.strip() or default
    if suffix and raw_name.lower().endswith(suffix.lower()):
        stem = raw_name[: -len(suffix)]
    else:
        stem = Path(raw_name).stem if Path(raw_name).suffix else raw_name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or default
    return f"{stem}{suffix or Path(raw_name).suffix}"


def _safe_archive_name(name: str | None, default: str) -> str:
    raw_name = Path(str(name or default)).name.strip() or default
    suffix = Path(raw_name).suffix
    stem = Path(raw_name).stem if suffix else raw_name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or Path(default).stem
    return f"{stem}{suffix}"


def _validate_archive_members(names: list[str]) -> None:
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Project archive contains an unsafe path: {name}")


def _npz_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    array = data[key]
    if array.size == 0:
        return np.empty(array.shape, dtype=array.dtype)
    return np.asarray(array)


def _metadata_dense_parts(metadata: dict[str, Any] | None) -> list[np.ndarray]:
    dense_parts: list[np.ndarray] = []
    for part in (metadata or {}).get("parts", []) or []:
        dense = np.asarray(part.get("dense_points", []), dtype=float)
        if dense.ndim == 2 and dense.shape[1] == 3:
            dense_parts.append(dense)
    return dense_parts


def _metadata_lateral_flags(metadata: dict[str, Any] | None) -> list[bool]:
    return [bool(part.get("is_lateral", False)) for part in (metadata or {}).get("parts", []) or []]


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
    manual_interface_ready: bool = False
    auto_interface_ready: bool = False
    interface_draft_ready: bool = False
    interface_ready: bool = False
    segmentation_ready: bool = False
    voxel_segmentation_ready: bool = False
    mesh_prepared: bool = False
    mesh_completed: bool = False
    analysis_completed: bool = False
    last_segmentation_mode: Literal["rg", "icrg"] | None = None


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
        self.manual_basal_points: np.ndarray | None = None
        self.manual_dense_basal_parts: list[np.ndarray] = []
        self.manual_dense_basal_parts_is_lateral: list[bool] = []
        self.manual_basal_parts_metadata: dict[str, Any] = {
            "parts": [],
            "close_loop": False,
            "num_parts": 0,
            "has_lateral_parts": False,
            "palette": [],
        }
        self.auto_basal_points: np.ndarray | None = None
        self.auto_basal_parts_metadata: dict[str, Any] = {
            "parts": [],
            "close_loop": False,
            "num_parts": 0,
            "has_lateral_parts": False,
            "palette": [],
        }
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
        self.display_interface_source: Literal["manual", "auto"] | None = None
        self.interface_edit_draft: dict[str, Any] | None = None
        self.segmenter: RegionGrowingSegmentation | None = None
        self.segmented_pcd: o3d.geometry.PointCloud | None = None
        self.segmented_labels: np.ndarray | None = None
        self.segmented_branch_ids: np.ndarray | None = None
        self.segmented_branches: list[dict[str, Any]] = []
        self.region_growing_dense_labels: np.ndarray | None = None
        self.region_growing_dense_branch_ids: np.ndarray | None = None
        self.voxel_segmented_points: np.ndarray | None = None
        self.voxel_segmented_colors: np.ndarray | None = None
        self.voxel_segmented_normals: np.ndarray | None = None
        self.voxel_segmented_labels: np.ndarray | None = None
        self.voxel_segmented_branch_ids: np.ndarray | None = None
        self.voxel_segmented_branches: list[dict[str, Any]] = []
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
            "display_interface_source": self.display_interface_source,
            "manual_interface_ready": self.status.manual_interface_ready,
            "auto_interface_ready": self.status.auto_interface_ready,
            "interface_draft_ready": self.status.interface_draft_ready,
            "last_segmentation_mode": self.status.last_segmentation_mode,
            "interface_draft": self._interface_draft_summary(),
            "outputs": {
                "segmented": self.segmented_pcd_file_path,
                "mesh": self.mesh_path,
                "analysis": self.analysis_csv_path,
            },
        }

    def _project_status_from_dict(self, status_data: dict[str, Any] | None) -> None:
        self.status = WorkflowStatus()
        for key, value in (status_data or {}).items():
            if hasattr(self.status, key):
                setattr(self.status, key, value)

    def _add_point_state(
        self,
        arrays: dict[str, np.ndarray],
        prefix: str,
        points: np.ndarray | None,
        colors: np.ndarray | None,
        normals: np.ndarray | None = None,
    ) -> None:
        point_array = np.asarray(points, dtype=np.float64) if points is not None else np.empty((0, 3), dtype=np.float64)
        color_array = np.asarray(colors, dtype=np.float64) if colors is not None else np.empty((0, 3), dtype=np.float64)
        normal_array = np.asarray(normals, dtype=np.float64) if normals is not None else np.empty((0, 3), dtype=np.float64)
        arrays[f"{prefix}_points"] = point_array
        arrays[f"{prefix}_colors"] = color_array
        arrays[f"{prefix}_normals"] = normal_array

    def _write_working_state_npz(self, path: Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        if self.pcd is not None:
            self._add_point_state(
                arrays,
                "pcd",
                np.asarray(self.pcd.points),
                np.asarray(self.pcd.colors),
                np.asarray(self.pcd.normals) if self.pcd.has_normals() else None,
            )
        self._add_point_state(arrays, "raw_view", self.raw_view_points, self.raw_view_colors, self.raw_view_normals)
        self._add_point_state(arrays, "seed_view", self.seed_view_points, self.seed_view_colors, self.seed_view_normals)
        self._add_point_state(arrays, "interface_view", self.interface_view_points, self.interface_view_colors, self.interface_view_normals)
        self._add_point_state(
            arrays,
            "interface_preview_view",
            self.interface_preview_view_points,
            self.interface_preview_view_colors,
            self.interface_preview_view_normals,
        )
        np.savez_compressed(path, **arrays)

    def _write_segmented_state_npz(self, path: Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        if self.segmented_pcd is not None:
            self._add_point_state(
                arrays,
                "segmented",
                np.asarray(self.segmented_pcd.points),
                np.asarray(self.segmented_pcd.colors),
                np.asarray(self.segmented_pcd.normals) if self.segmented_pcd.has_normals() else None,
            )
        arrays["labels"] = (
            np.asarray(self.segmented_labels, dtype=np.int32)
            if self.segmented_labels is not None
            else np.empty((0,), dtype=np.int32)
        )
        arrays["segmented_branch_ids"] = (
            np.asarray(self.segmented_branch_ids, dtype=np.int32)
            if self.segmented_branch_ids is not None
            else np.empty((0,), dtype=np.int32)
        )
        arrays["segmented_branches_json"] = np.asarray([json.dumps(self.segmented_branches or [])])
        self._add_point_state(
            arrays,
            "voxel_segmented",
            self.voxel_segmented_points,
            self.voxel_segmented_colors,
            self.voxel_segmented_normals,
        )
        arrays["voxel_labels"] = (
            np.asarray(self.voxel_segmented_labels, dtype=np.int32)
            if self.voxel_segmented_labels is not None
            else np.empty((0,), dtype=np.int32)
        )
        arrays["voxel_branch_ids"] = (
            np.asarray(self.voxel_segmented_branch_ids, dtype=np.int32)
            if self.voxel_segmented_branch_ids is not None
            else np.empty((0,), dtype=np.int32)
        )
        arrays["voxel_branches_json"] = np.asarray([json.dumps(self.voxel_segmented_branches or [])])
        np.savez_compressed(path, **arrays)

    def _write_prepared_mesh_npz(self, path: Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        if self.prepared_mesh_data:
            rock_pcd = self.prepared_mesh_data["rock_pcd"]
            bottom_pcd = self.prepared_mesh_data["bottom_pcd"]
            self._add_point_state(
                arrays,
                "rock",
                np.asarray(rock_pcd.points),
                np.asarray(rock_pcd.colors),
                np.asarray(rock_pcd.normals) if rock_pcd.has_normals() else None,
            )
            self._add_point_state(
                arrays,
                "bottom",
                np.asarray(bottom_pcd.points),
                np.asarray(bottom_pcd.colors),
                np.asarray(bottom_pcd.normals) if bottom_pcd.has_normals() else None,
            )
            self._add_point_state(
                arrays,
                "combined",
                self.prepared_mesh_data.get("combined_points"),
                self.prepared_mesh_data.get("combined_colors"),
                self.prepared_mesh_data.get("combined_normals"),
            )
        np.savez_compressed(path, **arrays)

    def _restore_point_state(self, data: np.lib.npyio.NpzFile, prefix: str) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        points = _npz_array(data, f"{prefix}_points")
        if points is None or points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            return None, None, None
        colors = _npz_array(data, f"{prefix}_colors")
        normals = _npz_array(data, f"{prefix}_normals")
        if colors is None or colors.shape != points.shape:
            colors = np.full(points.shape, 0.5, dtype=float)
        if normals is not None and normals.shape != points.shape:
            normals = None
        return points, colors, normals

    def _restore_working_state_npz(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            points, colors, normals = self._restore_point_state(data, "pcd")
            if points is not None:
                self.pcd = _make_point_cloud_with_normals(points, colors, normals)
            for prefix, attr_prefix in [
                ("raw_view", "raw_view"),
                ("seed_view", "seed_view"),
                ("interface_view", "interface_view"),
                ("interface_preview_view", "interface_preview_view"),
            ]:
                view_points, view_colors, view_normals = self._restore_point_state(data, prefix)
                setattr(self, f"{attr_prefix}_points", view_points)
                setattr(self, f"{attr_prefix}_colors", view_colors)
                setattr(self, f"{attr_prefix}_normals", view_normals)
        if self.pcd is not None and (self.raw_view_points is None or self.raw_view_colors is None):
            self._snapshot_raw_view()

    def _restore_segmented_state_npz(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            points, colors, normals = self._restore_point_state(data, "segmented")
            if points is not None:
                self.segmented_pcd = _make_point_cloud_with_normals(points, colors, normals)
            labels = _npz_array(data, "labels")
            if labels is not None and labels.ndim == 1 and len(labels):
                self.segmented_labels = np.asarray(labels, dtype=int)
            segmented_branch_ids = _npz_array(data, "segmented_branch_ids")
            if (
                segmented_branch_ids is not None
                and segmented_branch_ids.ndim == 1
                and points is not None
                and len(segmented_branch_ids) == len(points)
            ):
                self.segmented_branch_ids = np.asarray(segmented_branch_ids, dtype=int)
            segmented_branches_json = _npz_array(data, "segmented_branches_json")
            if segmented_branches_json is not None and segmented_branches_json.size:
                try:
                    self.segmented_branches = json.loads(str(segmented_branches_json.reshape(-1)[0]))
                except (TypeError, json.JSONDecodeError):
                    self.segmented_branches = []
            voxel_points, voxel_colors, voxel_normals = self._restore_point_state(data, "voxel_segmented")
            if voxel_points is not None:
                self.voxel_segmented_points = voxel_points
                self.voxel_segmented_colors = voxel_colors
                self.voxel_segmented_normals = voxel_normals
            voxel_labels = _npz_array(data, "voxel_labels")
            if voxel_labels is not None and voxel_labels.ndim == 1 and len(voxel_labels):
                self.voxel_segmented_labels = np.asarray(voxel_labels, dtype=int)
            voxel_branch_ids = _npz_array(data, "voxel_branch_ids")
            if (
                voxel_branch_ids is not None
                and voxel_branch_ids.ndim == 1
                and voxel_points is not None
                and len(voxel_branch_ids) == len(voxel_points)
            ):
                self.voxel_segmented_branch_ids = np.asarray(voxel_branch_ids, dtype=int)
            branches_json = _npz_array(data, "voxel_branches_json")
            if branches_json is not None and branches_json.size:
                try:
                    self.voxel_segmented_branches = json.loads(str(branches_json.reshape(-1)[0]))
                except (TypeError, json.JSONDecodeError):
                    self.voxel_segmented_branches = []

    def _restore_prepared_mesh_npz(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            rock_points, rock_colors, rock_normals = self._restore_point_state(data, "rock")
            bottom_points, bottom_colors, bottom_normals = self._restore_point_state(data, "bottom")
            if rock_points is None or bottom_points is None:
                return
            rock_pcd = _make_point_cloud_with_normals(rock_points, rock_colors, rock_normals)
            bottom_pcd = _make_point_cloud_with_normals(bottom_points, bottom_colors, bottom_normals)
            combined_points, combined_colors, combined_normals = self._restore_point_state(data, "combined")
            if combined_points is None:
                combined_points = np.vstack((rock_points, bottom_points))
                combined_colors = np.vstack((
                    np.full((len(rock_points), 3), [1.0, 0.0, 0.0]),
                    np.full((len(bottom_points), 3), [0.0, 1.0, 0.0]),
                ))
                rock_normal_array = np.asarray(rock_pcd.normals) if rock_pcd.has_normals() else np.zeros_like(rock_points)
                bottom_normal_array = np.asarray(bottom_pcd.normals) if bottom_pcd.has_normals() else np.zeros_like(bottom_points)
                combined_normals = np.vstack((rock_normal_array, bottom_normal_array))
            self.prepared_mesh_data = {
                "rock_pcd": rock_pcd,
                "bottom_pcd": bottom_pcd,
                "combined_points": combined_points,
                "combined_colors": combined_colors,
                "combined_normals": combined_normals,
                "preparation_result": None,
            }

    def _write_marker_las(self, path: Path) -> None:
        pcd = self._require_pcd()
        points = np.asarray(pcd.points, dtype=float)
        header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header)
        las.x = points[:, 0] + self.file_handler.x_mean
        las.y = points[:, 1] + self.file_handler.y_mean
        las.z = points[:, 2] + self.file_handler.z_mean

        red = np.full(len(points), 32768, dtype=np.uint16)
        green = np.full(len(points), 32768, dtype=np.uint16)
        blue = np.full(len(points), 32768, dtype=np.uint16)
        intensity = np.zeros(len(points), dtype=np.uint16)
        classification = np.zeros(len(points), dtype=np.uint8)

        def mark(indices: list[int] | np.ndarray | None, rgb: tuple[int, int, int], class_value: int, intensity_value: int) -> None:
            if indices is None:
                return
            valid = np.asarray(indices, dtype=int)
            valid = valid[(valid >= 0) & (valid < len(points))]
            if len(valid) == 0:
                return
            red[valid] = rgb[0]
            green[valid] = rgb[1]
            blue[valid] = rgb[2]
            classification[valid] = class_value
            intensity[valid] = intensity_value

        mark(self.rock_seeds, (65535, 0, 0), 1, 1)
        mark(self.pedestal_seeds, (0, 0, 65535), 2, 2)
        mark(self.basal_points, (0, 65535, 0), 9, 3)
        if self.manual_basal_points is not None:
            mark(self.manual_basal_points, (0, 65535, 0), 9, 4)
        if self.auto_basal_points is not None:
            mark(self.auto_basal_points, (0, 49152, 32768), 9, 5)

        las.red = red
        las.green = green
        las.blue = blue
        las.intensity = intensity
        las.classification = classification
        path.parent.mkdir(parents=True, exist_ok=True)
        las.write(path)

    def _project_manifest(self, ui_state: dict[str, Any] | None, artifacts: dict[str, Any], app_build: str | None) -> dict[str, Any]:
        return _to_jsonable({
            "format": PROJECT_FORMAT,
            "schema_version": PROJECT_SCHEMA_VERSION,
            "app_build": app_build,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_filename": _safe_filename(
                (ui_state or {}).get("project_filename"),
                self.current_pbr_file or "rock_detection_project",
                PROJECT_ARCHIVE_SUFFIX,
            ),
            "ui_state": ui_state or {},
            "workflow_state": {
                "current_file": self.current_pbr_file,
                "epsg_code": self.epsg_code,
                "point_count": len(self.pcd.points) if self.pcd is not None else 0,
                "status": self.status.__dict__,
                "seeds": {
                    "rock": self.rock_seeds,
                    "pedestal": self.pedestal_seeds,
                },
                "interface_source": self.interface_source,
                "display_interface_source": self.display_interface_source,
                "manual_basal_points": self.manual_basal_points,
                "auto_basal_points": self.auto_basal_points,
                "basal_points": self.basal_points,
                "manual_basal_parts_metadata": self.manual_basal_parts_metadata,
                "auto_basal_parts_metadata": self.auto_basal_parts_metadata,
                "basal_parts_metadata": self.basal_parts_metadata,
                "interface_preview_metadata": self.interface_preview_metadata,
                "interface_edit_draft": self._serialize_interface_draft() if self.interface_edit_draft else None,
                "normals_display_ready": self.normals_display_ready,
                "summary": self.summary(),
            },
            "artifacts": artifacts,
            "provenance": {
                "last_segmentation_mode": self.status.last_segmentation_mode,
                "analysis_completed": self.status.analysis_completed,
                "mesh_completed": self.status.mesh_completed,
                "mesh_prepared": self.status.mesh_prepared,
            },
        })

    def export_project(
        self,
        ui_state: dict[str, Any] | None = None,
        filename: str | None = None,
        app_build: str | None = None,
    ) -> Path:
        if self.pcd is None:
            raise ValueError("Load a point cloud before saving a project.")

        project_filename = _safe_filename(filename or (ui_state or {}).get("project_filename"), self.current_pbr_file or "rock_detection_project", PROJECT_ARCHIVE_SUFFIX)
        project_dir = self.output_dir / "projects"
        project_dir.mkdir(parents=True, exist_ok=True)
        archive_path = project_dir / project_filename
        artifacts: dict[str, Any] = {}

        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            working_state = temp_dir / "working_point_cloud.npz"
            self._write_working_state_npz(working_state)
            artifacts["working_state"] = {"path": "state/working_point_cloud.npz"}

            marker_las = temp_dir / "seeds_interface.las"
            self._write_marker_las(marker_las)
            artifacts["seeds_interface"] = {"path": "assets/intermediate/seeds_interface.las"}

            segmented_state: Path | None = None
            if (
                self.segmented_pcd is not None
                or self.segmented_labels is not None
                or self.voxel_segmented_points is not None
                or self.voxel_segmented_labels is not None
            ):
                segmented_state = temp_dir / "segmented_state.npz"
                self._write_segmented_state_npz(segmented_state)
                artifacts["segmented_state"] = {"path": "state/segmented_state.npz"}

            prepared_state: Path | None = None
            if self.prepared_mesh_data:
                prepared_state = temp_dir / "prepared_mesh.npz"
                self._write_prepared_mesh_npz(prepared_state)
                artifacts["prepared_mesh_state"] = {"path": "state/prepared_mesh.npz"}

            raw_member = None
            if self.input_path is not None and Path(self.input_path).exists():
                raw_name = _safe_archive_name(Path(self.input_path).name, "raw_point_cloud.las")
                raw_member = f"assets/raw/{raw_name}"
                artifacts["raw_point_cloud"] = {"path": raw_member, "filename": raw_name}

            if self.segmented_pcd_file_path and Path(self.segmented_pcd_file_path).exists():
                segmented_name = _safe_archive_name(Path(self.segmented_pcd_file_path).name, "segmented.las")
                artifacts["segmented_point_cloud"] = {"path": f"assets/segmented/{segmented_name}", "filename": segmented_name}
            if self.mesh_path and Path(self.mesh_path).exists():
                mesh_name = _safe_archive_name(Path(self.mesh_path).name, "mesh.ply")
                artifacts["mesh"] = {"path": f"assets/mesh/{mesh_name}", "filename": mesh_name}
            if self.analysis_csv_path and Path(self.analysis_csv_path).exists():
                analysis_name = _safe_archive_name(Path(self.analysis_csv_path).name, "analysis.csv")
                artifacts["analysis"] = {"path": f"assets/analysis/{analysis_name}", "filename": analysis_name}

            manifest = self._project_manifest({**(ui_state or {}), "project_filename": project_filename}, artifacts, app_build)

            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("project.json", json.dumps(manifest, indent=2))
                archive.write(working_state, artifacts["working_state"]["path"])
                archive.write(marker_las, artifacts["seeds_interface"]["path"])
                if segmented_state is not None:
                    archive.write(segmented_state, artifacts["segmented_state"]["path"])
                if prepared_state is not None:
                    archive.write(prepared_state, artifacts["prepared_mesh_state"]["path"])
                if raw_member and self.input_path is not None:
                    archive.write(Path(self.input_path), raw_member)
                if "segmented_point_cloud" in artifacts:
                    archive.write(Path(self.segmented_pcd_file_path), artifacts["segmented_point_cloud"]["path"])
                if "mesh" in artifacts:
                    archive.write(Path(self.mesh_path), artifacts["mesh"]["path"])
                if "analysis" in artifacts:
                    archive.write(Path(self.analysis_csv_path), artifacts["analysis"]["path"])

        return archive_path

    def _copy_project_member(
        self,
        archive: zipfile.ZipFile,
        member: str | None,
        target_dir: Path,
        default_name: str,
    ) -> Path | None:
        if not member or member not in archive.namelist():
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / _safe_archive_name(Path(member).name, default_name)
        with archive.open(member) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return target

    def _restore_interface_state(self, workflow_state: dict[str, Any]) -> None:
        self.rock_seeds = [int(idx) for idx in workflow_state.get("seeds", {}).get("rock", [])]
        self.pedestal_seeds = [int(idx) for idx in workflow_state.get("seeds", {}).get("pedestal", [])]
        self.interface_source = workflow_state.get("interface_source")
        self.display_interface_source = workflow_state.get("display_interface_source")

        def index_array(key: str) -> np.ndarray | None:
            value = workflow_state.get(key)
            if value is None:
                return None
            return np.asarray(value, dtype=int)

        self.manual_basal_points = index_array("manual_basal_points")
        self.auto_basal_points = index_array("auto_basal_points")
        self.basal_points = index_array("basal_points")
        self.manual_basal_parts_metadata = deepcopy(workflow_state.get("manual_basal_parts_metadata") or {
            "parts": [],
            "close_loop": False,
            "num_parts": 0,
            "has_lateral_parts": False,
            "palette": [],
        })
        self.auto_basal_parts_metadata = deepcopy(workflow_state.get("auto_basal_parts_metadata") or {
            "parts": [],
            "close_loop": False,
            "num_parts": 0,
            "has_lateral_parts": False,
            "palette": [],
        })
        self.basal_parts_metadata = deepcopy(workflow_state.get("basal_parts_metadata") or {
            "parts": [],
            "close_loop": False,
            "num_parts": 0,
            "has_lateral_parts": False,
            "palette": [],
        })
        self.interface_preview_metadata = deepcopy(workflow_state.get("interface_preview_metadata"))
        self.manual_dense_basal_parts = _metadata_dense_parts(self.manual_basal_parts_metadata)
        self.manual_dense_basal_parts_is_lateral = _metadata_lateral_flags(self.manual_basal_parts_metadata)
        self.dense_basal_parts = _metadata_dense_parts(self.basal_parts_metadata)
        self.dense_basal_parts_is_lateral = _metadata_lateral_flags(self.basal_parts_metadata)
        if self._has_manual_interface():
            self.display_interface_source = "manual"
        elif self.auto_basal_points is not None and len(self.auto_basal_points) > 0:
            self.display_interface_source = "auto"

        draft = workflow_state.get("interface_edit_draft")
        if draft:
            self.interface_edit_draft = {
                "source": draft.get("source"),
                "parts": deepcopy(draft.get("parts", []) or []),
                "close_loop": bool(draft.get("close_loop", True)),
                "include_indices": list(draft.get("include_indices", []) or []),
                "exclude_indices": list(draft.get("exclude_indices", []) or []),
                "effective_indices": list(draft.get("effective_indices", []) or []),
                "preview_metadata": deepcopy(draft.get("metadata", {}) or {}),
                "history": [],
            }
        self.normals_display_ready = bool(workflow_state.get("normals_display_ready", False))

    def import_project(self, archive_path: Path) -> dict[str, Any]:
        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                names = archive.namelist()
                _validate_archive_members(names)
                if "project.json" not in names:
                    raise ValueError("Project archive does not contain project.json.")
                manifest = json.loads(archive.read("project.json").decode("utf-8"))
                if manifest.get("format") != PROJECT_FORMAT:
                    raise ValueError("This file is not a Rock Detection 3D project.")
                if int(manifest.get("schema_version", 0)) != PROJECT_SCHEMA_VERSION:
                    raise ValueError(f"Unsupported project schema version: {manifest.get('schema_version')}.")

                artifacts = manifest.get("artifacts", {}) or {}
                workflow_state = manifest.get("workflow_state", {}) or {}

                self.reset_runtime()
                raw_path = self._copy_project_member(
                    archive,
                    (artifacts.get("raw_point_cloud") or {}).get("path"),
                    self.upload_dir,
                    "raw_point_cloud.las",
                )
                if raw_path is not None:
                    self.input_path = raw_path
                    self.current_pbr_file = workflow_state.get("current_file") or raw_path.stem
                    self.pcd, _, loaded_epsg = self.file_handler.load_las_as_open3d_point_cloud(raw_path)
                    self.epsg_code = workflow_state.get("epsg_code", loaded_epsg)
                    self._snapshot_raw_view()

                with tempfile.TemporaryDirectory() as temp_name:
                    temp_dir = Path(temp_name)
                    working_state = self._copy_project_member(
                        archive,
                        (artifacts.get("working_state") or {}).get("path"),
                        temp_dir,
                        "working_point_cloud.npz",
                    )
                    if working_state is not None:
                        self._restore_working_state_npz(working_state)

                    segmented_state = self._copy_project_member(
                        archive,
                        (artifacts.get("segmented_state") or {}).get("path"),
                        temp_dir,
                        "segmented_state.npz",
                    )
                    if segmented_state is not None:
                        self._restore_segmented_state_npz(segmented_state)

                    prepared_state = self._copy_project_member(
                        archive,
                        (artifacts.get("prepared_mesh_state") or {}).get("path"),
                        temp_dir,
                        "prepared_mesh.npz",
                    )
                    if prepared_state is not None:
                        self._restore_prepared_mesh_npz(prepared_state)

                if self.pcd is None:
                    raise ValueError("Project archive does not include restorable point-cloud data.")

                self.current_pbr_file = workflow_state.get("current_file") or self.current_pbr_file or "point_cloud"
                self.epsg_code = workflow_state.get("epsg_code", self.epsg_code)
                self._restore_interface_state(workflow_state)
                self._project_status_from_dict(workflow_state.get("status"))

                segmented_path = self._copy_project_member(
                    archive,
                    (artifacts.get("segmented_point_cloud") or {}).get("path"),
                    self.output_dir,
                    f"{self.current_pbr_file}_segmented.las",
                )
                self.segmented_pcd_file_path = str(segmented_path) if segmented_path is not None else None

                mesh_path = self._copy_project_member(
                    archive,
                    (artifacts.get("mesh") or {}).get("path"),
                    self.output_dir,
                    f"{self.current_pbr_file}_mesh.ply",
                )
                if mesh_path is not None:
                    self.mesh_path = str(mesh_path)
                    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
                    if mesh is not None and len(mesh.triangles) > 0:
                        self.mesh_processor.reconstructed_mesh = mesh

                analysis_path = self._copy_project_member(
                    archive,
                    (artifacts.get("analysis") or {}).get("path"),
                    self.output_dir,
                    f"{self.current_pbr_file}_analysis.csv",
                )
                self.analysis_csv_path = str(analysis_path) if analysis_path is not None else None

                if self.segmented_pcd is None or self.segmented_labels is None:
                    self.status.segmentation_ready = False
                if self.voxel_segmented_points is None or self.voxel_segmented_labels is None:
                    self.status.voxel_segmentation_ready = False
                else:
                    self.status.voxel_segmentation_ready = True
                if not self.prepared_mesh_data:
                    self.status.mesh_prepared = False
                if not self.mesh_path:
                    self.status.mesh_completed = False
                if not self.analysis_csv_path:
                    self.status.analysis_completed = False
                self.status.point_cloud_loaded = True
                self.status.seeds_ready = bool(self.rock_seeds and self.pedestal_seeds)
                self.status.manual_interface_ready = self._has_manual_interface()
                self.status.auto_interface_ready = bool(self.status.auto_interface_ready or self._has_auto_interface())
                self.status.interface_ready = bool(self.status.manual_interface_ready or self.status.auto_interface_ready or (self.basal_points is not None and len(self.basal_points)))

                return {
                    "summary": self.summary(),
                    "ui_state": manifest.get("ui_state", {}) or {},
                    "project_filename": manifest.get("project_filename") or _safe_filename(self.current_pbr_file, "rock_detection_project", PROJECT_ARCHIVE_SUFFIX),
                }
        except zipfile.BadZipFile as exc:
            raise ValueError("Project file is not a valid ZIP archive.") from exc

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
        self.seed_view_colors = np.full((len(self.seed_view_points), 3), 0.5, dtype=float)
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

    def _interface_draft_summary(self) -> dict[str, Any] | None:
        draft = self.interface_edit_draft
        if not draft:
            return None
        parts = draft.get("parts", []) or []
        anchor_count = sum(len(part.get("selected_indices", []) or []) for part in parts)
        return {
            "part_count": len(parts),
            "anchor_count": int(anchor_count),
            "include_count": int(len(draft.get("include_indices", []) or [])),
            "exclude_count": int(len(draft.get("exclude_indices", []) or [])),
            "effective_count": int(len(draft.get("effective_indices", []) or [])),
            "can_undo": bool(draft.get("history")),
            "close_loop": bool(draft.get("close_loop", True)),
        }

    def _ordered_auto_interface_indices(self, indices: np.ndarray) -> list[int]:
        pcd = self._require_pcd()
        unique_indices = np.unique(np.asarray(indices, dtype=int))
        if len(unique_indices) <= 2:
            return unique_indices.astype(int).tolist()

        points = np.asarray(pcd.points, dtype=float)[unique_indices]
        finite_mask = np.all(np.isfinite(points), axis=1)
        if not np.all(finite_mask):
            unique_indices = unique_indices[finite_mask]
            points = points[finite_mask]
        if len(unique_indices) <= 2:
            return unique_indices.astype(int).tolist()

        centroid = np.mean(points, axis=0)
        centered = points - centroid
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis_a = vh[0]
            axis_b = vh[1] if vh.shape[0] > 1 else np.array([0.0, 1.0, 0.0])
            proj_a = centered @ axis_a
            proj_b = centered @ axis_b
            angles = np.arctan2(proj_b, proj_a)
            order = np.argsort(angles)
        except np.linalg.LinAlgError:
            order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
        return unique_indices[order].astype(int).tolist()

    @staticmethod
    def _resample_anchor_indices(ordered_indices: list[int]) -> list[int]:
        if len(ordered_indices) <= DRAFT_MIN_ANCHORS:
            return list(dict.fromkeys(int(idx) for idx in ordered_indices))
        target_count = min(len(ordered_indices), max(DRAFT_MIN_ANCHORS, min(DRAFT_MAX_ANCHORS, DRAFT_TARGET_ANCHORS)))
        positions = np.floor(np.arange(target_count) * len(ordered_indices) / target_count).astype(int)
        anchors = [int(ordered_indices[int(pos)]) for pos in positions]
        return list(dict.fromkeys(anchors))

    @staticmethod
    def _unique_ordered_indices(indices: list[int] | np.ndarray) -> list[int]:
        return list(dict.fromkeys(int(idx) for idx in indices))

    @staticmethod
    def _squared_distance(a: np.ndarray, b: np.ndarray) -> float:
        delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        return float(np.dot(delta, delta))

    @staticmethod
    def _point_to_segment_distance_sq(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        segment = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        denom = float(np.dot(segment, segment))
        if denom <= 1e-12:
            return WebWorkflowSession._squared_distance(point, start)
        t = float(np.dot(np.asarray(point, dtype=float) - start, segment) / denom)
        t = max(0.0, min(1.0, t))
        closest = np.asarray(start, dtype=float) + segment * t
        return WebWorkflowSession._squared_distance(point, closest)

    @staticmethod
    def _point_to_segment_t(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        segment = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        denom = float(np.dot(segment, segment))
        if denom <= 1e-12:
            return 0.0
        t = float(np.dot(np.asarray(point, dtype=float) - start, segment) / denom)
        return float(max(0.0, min(1.0, t)))

    def _valid_ordered_indices(self, indices: list[int] | np.ndarray, point_count: int, label: str) -> list[int]:
        ordered: list[int] = []
        previous: int | None = None
        for idx in ([] if indices is None else list(indices)):
            valid = self._validate_index(idx, point_count, label)
            if previous is None or valid != previous:
                ordered.append(valid)
            previous = valid
        return ordered

    def _clean_brush_stroke_indices(self, ordered_indices: list[int]) -> list[int]:
        pcd = self._require_pcd()
        points = np.asarray(pcd.points, dtype=float)
        unique_indices: list[int] = []
        seen: set[int] = set()
        for idx in ordered_indices:
            if idx in seen:
                continue
            point = points[idx]
            if np.all(np.isfinite(point)):
                unique_indices.append(int(idx))
                seen.add(int(idx))
        if len(unique_indices) <= 2:
            return unique_indices

        coords = points[np.asarray(unique_indices, dtype=int)]
        segment_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        positive_lengths = segment_lengths[segment_lengths > 1e-12]
        if len(positive_lengths) == 0:
            return unique_indices[:1]

        median_step = float(np.median(positive_lengths))
        q75_step = float(np.percentile(positive_lengths, 75))
        jump_threshold = max(median_step * BRUSH_JUMP_MEDIAN_MULTIPLIER, q75_step * 6.0)
        if not np.isfinite(jump_threshold) or jump_threshold <= 0:
            return unique_indices

        runs: list[list[int]] = []
        current = [unique_indices[0]]
        for offset, length in enumerate(segment_lengths, start=1):
            if length > jump_threshold:
                if len(current) >= 2:
                    runs.append(current)
                current = [unique_indices[offset]]
            else:
                current.append(unique_indices[offset])
        if len(current) >= 2:
            runs.append(current)
        if not runs:
            return unique_indices
        return max(
            runs,
            key=lambda run: float(np.sum(np.linalg.norm(np.diff(points[np.asarray(run, dtype=int)], axis=0), axis=1))) if len(run) > 1 else 0.0,
        )

    def _sample_brush_anchor_indices(self, ordered_indices: list[int]) -> list[int]:
        cleaned = self._clean_brush_stroke_indices(ordered_indices)
        if len(cleaned) < 2:
            raise ValueError("Brush Add needs at least two visible points along the stroke.")
        if len(cleaned) <= BRUSH_ADD_MAX_ANCHORS:
            return cleaned

        points = np.asarray(self._require_pcd().points, dtype=float)
        coords = points[np.asarray(cleaned, dtype=int)]
        segment_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        total_length = float(np.sum(segment_lengths))
        if not np.isfinite(total_length) or total_length <= 1e-12:
            return [cleaned[0], cleaned[-1]]

        positive_lengths = segment_lengths[segment_lengths > 1e-12]
        median_step = float(np.median(positive_lengths)) if len(positive_lengths) else total_length
        desired_spacing = max(median_step * 8.0, total_length / max(1, BRUSH_ADD_MAX_ANCHORS - 1))
        target_count = int(np.ceil(total_length / desired_spacing)) + 1
        target_count = min(BRUSH_ADD_MAX_ANCHORS, max(min(BRUSH_ADD_MIN_ANCHORS, len(cleaned)), target_count))

        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        target_distances = np.linspace(0.0, total_length, target_count)
        sampled: list[int] = []
        for distance in target_distances:
            position = int(np.searchsorted(cumulative, distance, side="left"))
            position = min(max(position, 0), len(cleaned) - 1)
            sampled.append(cleaned[position])
        sampled[0] = cleaned[0]
        sampled[-1] = cleaned[-1]
        sampled = self._unique_ordered_indices(sampled)
        if len(sampled) < 2:
            sampled = [cleaned[0], cleaned[-1]]
        return sampled

    def _sample_sparse_path_controls(self, ordered_indices: list[int], max_count: int) -> list[int]:
        unique_indices = self._unique_ordered_indices(ordered_indices)
        if len(unique_indices) <= max_count:
            return unique_indices

        points = np.asarray(self._require_pcd().points, dtype=float)
        coords = points[np.asarray(unique_indices, dtype=int)]
        segment_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        total_length = float(np.sum(segment_lengths))
        if not np.isfinite(total_length) or total_length <= 1e-12:
            positions = np.floor(np.linspace(0, len(unique_indices) - 1, max_count)).astype(int)
            sampled = [unique_indices[int(pos)] for pos in positions]
        else:
            cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
            sampled = []
            for distance in np.linspace(0.0, total_length, max_count):
                position = int(np.searchsorted(cumulative, distance, side="left"))
                position = min(max(position, 0), len(unique_indices) - 1)
                sampled.append(unique_indices[position])

        sampled[0] = unique_indices[0]
        sampled[-1] = unique_indices[-1]
        sampled = self._unique_ordered_indices(sampled)
        if len(sampled) < 2:
            sampled = [unique_indices[0], unique_indices[-1]]
        return sampled

    def _splice_brush_anchors_into_draft(
        self,
        brush_anchors: list[int],
        brush_selected_indices: set[int] | None = None,
        ordered_stroke_indices: list[int] | None = None,
        target_part_index: int | None = None,
        target_edge_index: int | None = None,
        target_anchor_index: int | None = None,
        target_source_index: int | None = None,
        start_target_part_index: int | None = None,
        start_target_edge_index: int | None = None,
        start_target_anchor_index: int | None = None,
        start_target_edge_t: float | None = None,
        start_target_source_index: int | None = None,
        end_target_part_index: int | None = None,
        end_target_edge_index: int | None = None,
        end_target_anchor_index: int | None = None,
        end_target_edge_t: float | None = None,
        end_target_source_index: int | None = None,
        replace_direction: Literal["forward", "opposite"] | None = None,
    ) -> dict[str, Any]:
        if not self.interface_edit_draft:
            raise ValueError("Create an interface draft before brushing points.")
        if len(brush_anchors) < 2:
            raise ValueError("Brush Add needs at least two sampled anchors.")

        pcd = self._require_pcd()
        points = np.asarray(pcd.points, dtype=float)
        parts = [
            {
                "selected_indices": self._unique_ordered_indices(part.get("selected_indices", []) or []),
                "is_lateral": bool(part.get("is_lateral", False)),
            }
            for part in self.interface_edit_draft.get("parts", []) or []
        ]
        parts = [part for part in parts if len(part["selected_indices"]) >= 2]
        if not parts:
            self.interface_edit_draft["parts"] = [{"selected_indices": brush_anchors, "is_lateral": False}]
            self.interface_edit_draft["close_loop"] = False
            return {
                "brush_add_mode": "new_path",
                "path_edited": True,
                "path_part_count": 1,
                "sampled_anchor_count": len(brush_anchors),
                "inserted_anchor_count": len(brush_anchors),
                "removed_anchor_count": 0,
                "guided_target_used": False,
            }

        sample_coords = points[np.asarray(brush_anchors, dtype=int)]
        start_target = self._resolve_brush_endpoint_target(
            parts,
            sample_coords[0],
            start_target_part_index if start_target_part_index is not None else target_part_index,
            start_target_edge_index if start_target_edge_index is not None else target_edge_index,
            start_target_anchor_index if start_target_anchor_index is not None else target_anchor_index,
            start_target_edge_t,
            start_target_source_index if start_target_source_index is not None else target_source_index,
        )
        end_target = self._resolve_brush_endpoint_target(
            parts,
            sample_coords[-1],
            end_target_part_index,
            end_target_edge_index,
            end_target_anchor_index,
            end_target_edge_t,
            end_target_source_index,
        )
        if start_target and end_target:
            return self._splice_brush_path_into_draft(
                parts,
                brush_anchors,
                start_target,
                end_target,
            )

        probe_coords = [sample_coords[0], sample_coords[-1], np.mean(sample_coords, axis=0)]
        best: tuple[float, int, int, bool] | None = None
        guided_target_used = False

        if start_target:
            part_idx = int(start_target["part_idx"])
            edge_idx = int(start_target["edge_idx"])
            indices = parts[part_idx]["selected_indices"]
            start_idx = indices[edge_idx]
            end_idx = indices[(edge_idx + 1) % len(indices)]
            start = points[start_idx]
            end = points[end_idx]
            forward_score = self._squared_distance(sample_coords[0], start) + self._squared_distance(sample_coords[-1], end)
            reverse_score = self._squared_distance(sample_coords[-1], start) + self._squared_distance(sample_coords[0], end)
            reverse = reverse_score < forward_score
            best = (0.0, part_idx, edge_idx, reverse)
            guided_target_used = bool(start_target.get("guided"))

        for part_idx, part in enumerate(parts):
            if guided_target_used:
                break
            indices = part["selected_indices"]
            closes_part = bool(self.interface_edit_draft.get("close_loop", True)) and len(parts) == 1
            edge_count = len(indices) if closes_part else len(indices) - 1
            for edge_idx in range(edge_count):
                start_idx = indices[edge_idx]
                end_idx = indices[(edge_idx + 1) % len(indices)]
                start = points[start_idx]
                end = points[end_idx]
                segment_score = float(np.mean([self._point_to_segment_distance_sq(probe, start, end) for probe in probe_coords]))
                forward_score = self._squared_distance(sample_coords[0], start) + self._squared_distance(sample_coords[-1], end)
                reverse_score = self._squared_distance(sample_coords[-1], start) + self._squared_distance(sample_coords[0], end)
                reverse = reverse_score < forward_score
                score = segment_score + 0.15 * min(forward_score, reverse_score)
                if best is None or score < best[0]:
                    best = (score, part_idx, edge_idx, reverse)

        if best is None:
            raise ValueError("Could not find an existing interface path segment for Brush Add.")

        _, part_idx, edge_idx, reverse = best
        target_part = parts[part_idx]
        existing = target_part["selected_indices"]
        oriented = list(reversed(brush_anchors)) if reverse else list(brush_anchors)
        existing_set = set(existing)
        insert_anchors = [idx for idx in oriented if idx not in existing_set]
        if len(insert_anchors) < 1:
            raise ValueError("Brush Add did not add any new interface anchors.")

        insert_at = edge_idx + 1
        target_part["selected_indices"] = self._unique_ordered_indices(existing[:insert_at] + insert_anchors + existing[insert_at:])
        self.interface_edit_draft["parts"] = parts
        result = {
            "brush_add_mode": "insert_fallback",
            "replacement_selection": "different_endpoint_parts" if start_target and end_target else "nearest_edge_insert",
            "fallback_reason": "different_endpoint_parts" if start_target and end_target else "nearest_edge_insert",
            "path_edited": True,
            "path_part_count": len(parts),
            "sampled_anchor_count": len(brush_anchors),
            "inserted_anchor_count": len(insert_anchors),
            "removed_anchor_count": 0,
            "target_part_index": int(part_idx),
            "target_edge_index": int(edge_idx),
            "guided_target_used": guided_target_used,
        }
        return result

    def _edge_count_for_part(self, part: dict[str, Any], part_count: int) -> int:
        indices = part.get("selected_indices", []) or []
        if len(indices) < 2:
            return 0
        closes_part = bool(self.interface_edit_draft.get("close_loop", True)) and part_count == 1
        return len(indices) if closes_part else len(indices) - 1

    def _resolve_brush_endpoint_target(
        self,
        parts: list[dict[str, Any]],
        endpoint: np.ndarray,
        target_part_index: int | None,
        target_edge_index: int | None,
        target_anchor_index: int | None = None,
        target_edge_t: float | None = None,
        target_source_index: int | None = None,
    ) -> dict[str, Any] | None:
        pcd = self._require_pcd()
        points = np.asarray(pcd.points, dtype=float)
        point_count = len(points)
        if target_source_index is not None:
            source_target = self._resolve_brush_source_target(
                parts,
                int(target_source_index),
                guided=target_part_index is not None or target_edge_index is not None,
            )
            if source_target is not None:
                return source_target

        if target_part_index is not None and target_edge_index is not None:
            part_idx = int(target_part_index)
            edge_idx = int(target_edge_index)
            if 0 <= part_idx < len(parts) and 0 <= edge_idx < self._edge_count_for_part(parts[part_idx], len(parts)):
                target: dict[str, Any] = {"part_idx": part_idx, "edge_idx": edge_idx, "guided": True}
                indices = parts[part_idx]["selected_indices"]
                if target_anchor_index is not None:
                    anchor_idx = int(target_anchor_index)
                    if 0 <= anchor_idx < len(indices):
                        target["anchor_idx"] = anchor_idx
                if target_edge_t is not None and np.isfinite(float(target_edge_t)):
                    target["edge_t"] = float(max(0.0, min(1.0, float(target_edge_t))))
                if target_source_index is not None:
                    source_idx = int(target_source_index)
                    if 0 <= source_idx < point_count:
                        target["source_idx"] = source_idx
                return target

        best: tuple[float, int, int] | None = None
        for part_idx, part in enumerate(parts):
            indices = part["selected_indices"]
            for edge_idx in range(self._edge_count_for_part(part, len(parts))):
                start = points[indices[edge_idx]]
                end = points[indices[(edge_idx + 1) % len(indices)]]
                score = self._point_to_segment_distance_sq(endpoint, start, end)
                if best is None or score < best[0]:
                    best = (score, part_idx, edge_idx)
        if best is None:
            return None
        _, part_idx, edge_idx = best
        indices = parts[part_idx]["selected_indices"]
        start = points[indices[edge_idx]]
        end = points[indices[(edge_idx + 1) % len(indices)]]
        start_distance = self._squared_distance(endpoint, start)
        end_distance = self._squared_distance(endpoint, end)
        anchor_idx = edge_idx if start_distance <= end_distance else (edge_idx + 1) % len(indices)
        return {"part_idx": part_idx, "edge_idx": edge_idx, "anchor_idx": int(anchor_idx), "guided": False}

    def _resolve_brush_source_target(
        self,
        parts: list[dict[str, Any]],
        source_idx: int,
        guided: bool = False,
    ) -> dict[str, Any] | None:
        pcd = self._require_pcd()
        points = np.asarray(pcd.points, dtype=float)
        if source_idx < 0 or source_idx >= len(points):
            return None
        tree = cKDTree(points)
        source_point = points[int(source_idx)]
        best: tuple[int, float, int, int, float] | None = None
        for part_idx, part in enumerate(parts):
            indices = part.get("selected_indices", []) or []
            edge_count = self._edge_count_for_part(part, len(parts))
            for edge_idx in range(edge_count):
                start_idx = int(indices[edge_idx])
                end_idx = int(indices[(edge_idx + 1) % len(indices)])
                start = points[start_idx]
                end = points[end_idx]
                dense_indices = set(self._edge_dense_source_indices(points, tree, start_idx, end_idx))
                on_dense_edge = int(source_idx) in dense_indices
                distance = self._point_to_segment_distance_sq(source_point, start, end)
                edge_t = self._point_to_segment_t(source_point, start, end)
                candidate = (0 if on_dense_edge else 1, distance, part_idx, edge_idx, edge_t)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            return None

        _, _, part_idx, edge_idx, edge_t = best
        indices = parts[part_idx]["selected_indices"]
        start_idx = int(indices[edge_idx])
        end_anchor_idx = (edge_idx + 1) % len(indices)
        end_idx = int(indices[end_anchor_idx])
        start_distance = self._squared_distance(source_point, points[start_idx])
        end_distance = self._squared_distance(source_point, points[end_idx])
        anchor_idx = int(edge_idx if start_distance <= end_distance else end_anchor_idx)
        return {
            "part_idx": int(part_idx),
            "edge_idx": int(edge_idx),
            "anchor_idx": anchor_idx,
            "edge_t": float(edge_t),
            "source_idx": int(source_idx),
            "guided": bool(guided),
            "source_guided": True,
        }

    def _materialize_endpoint_source_anchor(
        self,
        part: dict[str, Any],
        part_count: int,
        target: dict[str, Any],
        sibling: dict[str, Any] | None,
    ) -> None:
        if "source_idx" not in target:
            return
        indices = part["selected_indices"]
        source_idx = int(target["source_idx"])
        if source_idx in indices:
            target["anchor_idx"] = int(indices.index(source_idx))
            return

        edge_count = self._edge_count_for_part(part, part_count)
        if edge_count <= 0:
            return
        edge_idx = self._edge_index_for_source_point(part, part_count, source_idx)
        if edge_idx is None:
            edge_idx = max(0, min(edge_count - 1, int(target["edge_idx"])))
        else:
            target["edge_idx"] = int(edge_idx)
        insert_at = edge_idx + 1
        indices.insert(insert_at, source_idx)
        target["anchor_idx"] = int(insert_at)
        target["edge_idx"] = int(insert_at)

        if sibling is not None:
            if "anchor_idx" in sibling and int(sibling["anchor_idx"]) >= insert_at:
                sibling["anchor_idx"] = int(sibling["anchor_idx"]) + 1
            if int(sibling.get("edge_idx", -1)) > edge_idx:
                sibling["edge_idx"] = int(sibling["edge_idx"]) + 1
            elif int(sibling.get("edge_idx", -1)) == edge_idx:
                sibling_t = float(sibling.get("edge_t", 0.0))
                target_t = float(target.get("edge_t", 0.5))
                if sibling_t > target_t:
                    sibling["edge_idx"] = int(sibling["edge_idx"]) + 1

    def _materialize_endpoint_source_anchors(
        self,
        part: dict[str, Any],
        part_count: int,
        start_target: dict[str, Any],
        end_target: dict[str, Any],
    ) -> None:
        start_t = float(start_target.get("edge_t", 0.0))
        end_t = float(end_target.get("edge_t", 1.0))
        targets = [
            (start_target, end_target, start_t),
            (end_target, start_target, end_t),
        ]
        targets.sort(key=lambda item: (int(item[0].get("edge_idx", 0)), item[2]), reverse=True)
        for target, sibling, _ in targets:
            self._materialize_endpoint_source_anchor(part, part_count, target, sibling)

    def _edge_index_for_source_point(
        self,
        part: dict[str, Any],
        part_count: int,
        source_idx: int,
    ) -> int | None:
        indices = part["selected_indices"]
        edge_count = self._edge_count_for_part(part, part_count)
        if edge_count <= 0:
            return None

        pcd = self._require_pcd()
        points = np.asarray(pcd.points, dtype=float)
        tree = cKDTree(points)
        source_point = points[int(source_idx)]
        best: tuple[float, int] | None = None
        for edge_idx in range(edge_count):
            start_idx = int(indices[edge_idx])
            end_idx = int(indices[(edge_idx + 1) % len(indices)])
            dense_indices = self._edge_dense_source_indices(points, tree, start_idx, end_idx)
            if int(source_idx) in set(dense_indices):
                return int(edge_idx)
            start = points[start_idx]
            end = points[end_idx]
            score = self._point_to_segment_distance_sq(source_point, start, end)
            if best is None or score < best[0]:
                best = (score, edge_idx)
        return int(best[1]) if best is not None else None

    @staticmethod
    def _path_edge_sequence(start_edge: int, end_edge: int, edge_count: int) -> list[int]:
        if edge_count <= 0:
            return []
        start = int(start_edge) % edge_count
        end = int(end_edge) % edge_count
        edges = [start]
        while edges[-1] != end:
            edges.append((edges[-1] + 1) % edge_count)
            if len(edges) > edge_count:
                break
        return edges

    def _edge_dense_source_indices(
        self,
        points: np.ndarray,
        tree: cKDTree,
        start_idx: int,
        end_idx: int,
    ) -> list[int]:
        try:
            dense = _dense_basal_points(points, [int(start_idx), int(end_idx)], False)
            if dense.size == 0:
                return self._unique_ordered_indices([start_idx, end_idx])
            _, indices = tree.query(dense)
            return self._unique_ordered_indices(np.asarray(indices, dtype=int).tolist())
        except Exception:
            return self._unique_ordered_indices([start_idx, end_idx])

    @staticmethod
    def _point_to_polyline_distance_sq(point: np.ndarray, polyline: np.ndarray) -> float:
        if len(polyline) == 0:
            return float("inf")
        if len(polyline) == 1:
            return WebWorkflowSession._squared_distance(point, polyline[0])
        return float(min(
            WebWorkflowSession._point_to_segment_distance_sq(point, polyline[idx - 1], polyline[idx])
            for idx in range(1, len(polyline))
        ))

    @staticmethod
    def _positive_median(values: np.ndarray) -> float | None:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite) & (finite > 1e-9)]
        if finite.size == 0:
            return None
        return float(np.median(finite))

    def _brush_overlap_threshold(
        self,
        points: np.ndarray,
        stroke_indices: list[int],
        existing: list[int],
        closed: bool,
    ) -> float:
        candidates: list[float] = []
        if len(stroke_indices) >= 2:
            stroke_coords = points[np.asarray(stroke_indices, dtype=int)]
            stroke_step = self._positive_median(np.linalg.norm(np.diff(stroke_coords, axis=0), axis=1))
            if stroke_step is not None:
                candidates.append(stroke_step * 3.0)

        edge_indices = list(existing)
        if closed and len(existing) > 2:
            edge_indices = [*edge_indices, existing[0]]
        if len(edge_indices) >= 2:
            control_coords = points[np.asarray(edge_indices, dtype=int)]
            control_step = self._positive_median(np.linalg.norm(np.diff(control_coords, axis=0), axis=1))
            if control_step is not None:
                candidates.append(control_step * 0.08)

        if not candidates:
            return 0.05
        return float(max(0.01, min(0.3, max(candidates))))

    def _score_brush_overlap_by_edge(
        self,
        existing: list[int],
        closed: bool,
        brush_selected_indices: set[int],
        ordered_stroke_indices: list[int],
        brush_anchors: list[int],
    ) -> tuple[list[dict[str, Any]], float]:
        points = np.asarray(self._require_pcd().points, dtype=float)
        tree = cKDTree(points)
        edge_count = len(existing) if closed else len(existing) - 1
        stroke_source = ordered_stroke_indices if len(ordered_stroke_indices) >= 2 else brush_anchors
        stroke_coords = points[np.asarray(stroke_source, dtype=int)]
        threshold = self._brush_overlap_threshold(points, stroke_source, existing, closed)
        selected = set(int(idx) for idx in brush_selected_indices)

        scores: list[dict[str, Any]] = []
        for edge_idx in range(edge_count):
            start_idx = int(existing[edge_idx])
            end_idx = int(existing[(edge_idx + 1) % len(existing)])
            dense_indices = self._edge_dense_source_indices(points, tree, start_idx, end_idx)
            dense_set = set(dense_indices)
            hit_count = len(dense_set.intersection(selected))

            if dense_indices:
                sample_count = min(24, len(dense_indices))
                sample_positions = np.linspace(0, len(dense_indices) - 1, sample_count).astype(int)
                sample_indices = [dense_indices[int(pos)] for pos in sample_positions]
                sample_coords = points[np.asarray(sample_indices, dtype=int)]
            else:
                sample_coords = np.asarray([(points[start_idx] + points[end_idx]) * 0.5], dtype=float)

            distances = np.asarray([
                np.sqrt(self._point_to_polyline_distance_sq(sample, stroke_coords))
                for sample in sample_coords
            ], dtype=float)
            finite_distances = distances[np.isfinite(distances)]
            mean_distance = float(np.mean(finite_distances)) if finite_distances.size else float("inf")
            near_count = int(np.sum(finite_distances <= threshold)) if finite_distances.size else 0
            overlap_weight = int(hit_count * 4 + near_count)
            scores.append({
                "edge": int(edge_idx),
                "dense_indices": dense_indices,
                "hit_count": int(hit_count),
                "near_count": int(near_count),
                "overlap_weight": int(overlap_weight),
                "mean_distance": mean_distance,
                "positive": bool(hit_count > 0 or near_count > 0),
            })
        return scores, threshold

    def _summarize_brush_overlap_arc(
        self,
        edges: list[int],
        edge_scores: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not edges:
            return {
                "edges": [],
                "edge_count": 0,
                "overlap_edge_count": 0,
                "selected_hit_count": 0,
                "near_sample_count": 0,
                "overlap_weight": 0,
                "overlap_fraction": 0.0,
                "mean_distance": float("inf"),
            }
        selected_hit_count = int(sum(edge_scores[edge]["hit_count"] for edge in edges))
        near_sample_count = int(sum(edge_scores[edge]["near_count"] for edge in edges))
        overlap_weight = int(sum(edge_scores[edge]["overlap_weight"] for edge in edges))
        overlap_edge_count = int(sum(1 for edge in edges if edge_scores[edge]["positive"]))
        distances = [
            float(edge_scores[edge]["mean_distance"])
            for edge in edges
            if np.isfinite(edge_scores[edge]["mean_distance"])
        ]
        mean_distance = float(np.mean(distances)) if distances else float("inf")
        return {
            "edges": [int(edge) for edge in edges],
            "edge_count": int(len(edges)),
            "overlap_edge_count": overlap_edge_count,
            "selected_hit_count": selected_hit_count,
            "near_sample_count": near_sample_count,
            "overlap_weight": overlap_weight,
            "overlap_fraction": float(overlap_edge_count / max(len(edges), 1)),
            "mean_distance": mean_distance,
        }

    def _select_brush_replacement_arc(
        self,
        existing: list[int],
        closed: bool,
        start_edge: int,
        end_edge: int,
        brush_selected_indices: set[int],
        ordered_stroke_indices: list[int],
        brush_anchors: list[int],
        replace_direction: Literal["forward", "opposite"] | None = None,
        start_anchor_idx: int | None = None,
        end_anchor_idx: int | None = None,
    ) -> dict[str, Any] | None:
        edge_count = len(existing) if closed else len(existing) - 1
        if edge_count <= 0:
            return None
        edge_scores, threshold = self._score_brush_overlap_by_edge(
            existing,
            closed,
            brush_selected_indices,
            ordered_stroke_indices,
            brush_anchors,
        )

        anchor_candidates = False
        if start_anchor_idx is not None and end_anchor_idx is not None and start_anchor_idx != end_anchor_idx:
            start_anchor = int(start_anchor_idx) % len(existing)
            end_anchor = int(end_anchor_idx) % len(existing)
            anchor_candidates = True
            if closed:
                candidate_specs = [
                    (
                        "forward",
                        start_anchor,
                        (end_anchor - 1) % edge_count,
                        self._path_edge_sequence(start_anchor, (end_anchor - 1) % edge_count, edge_count),
                    ),
                    (
                        "opposite",
                        end_anchor,
                        (start_anchor - 1) % edge_count,
                        self._path_edge_sequence(end_anchor, (start_anchor - 1) % edge_count, edge_count),
                    ),
                ]
            else:
                arc_start = int(min(start_anchor, end_anchor))
                arc_end = int(max(start_anchor, end_anchor) - 1)
                candidate_specs = [
                    ("forward", arc_start, arc_end, list(range(arc_start, arc_end + 1)))
                ] if arc_start <= arc_end else []
        elif closed:
            candidate_specs = [
                ("forward", int(start_edge), int(end_edge), self._path_edge_sequence(start_edge, end_edge, edge_count)),
                ("opposite", int(end_edge), int(start_edge), self._path_edge_sequence(end_edge, start_edge, edge_count)),
            ]
        else:
            arc_start = int(min(start_edge, end_edge))
            arc_end = int(max(start_edge, end_edge))
            candidate_specs = [("forward", arc_start, arc_end, list(range(arc_start, arc_end + 1)))]

        max_local_edges = max(3, int(np.ceil(edge_count * 0.35)))
        candidates: list[dict[str, Any]] = []
        for direction, arc_start, arc_end, edges in candidate_specs:
            summary = self._summarize_brush_overlap_arc(edges, edge_scores)
            overlap_fraction = float(summary["overlap_fraction"])
            broad_without_overlap = summary["edge_count"] > max_local_edges and overlap_fraction < 0.5
            weak_endpoint_only = (
                summary["edge_count"] > 3
                and summary["overlap_edge_count"] <= 1
                and summary["selected_hit_count"] == 0
            )
            acceptable = bool(summary["overlap_weight"] > 0 and not broad_without_overlap and not weak_endpoint_only)
            candidates.append({
                **summary,
                "direction": direction,
                "start_edge": int(arc_start),
                "end_edge": int(arc_end),
                "acceptable": acceptable,
                "anchor_candidate": bool(anchor_candidates),
                "max_local_edge_count": int(max_local_edges),
                "overlap_threshold": float(threshold),
                "fallback_reason": "no_overlapping_arc" if summary["overlap_weight"] <= 0 else (
                    "replacement_not_local" if broad_without_overlap else (
                        "endpoint_only_overlap" if weak_endpoint_only else None
                    )
                ),
            })

        acceptable_candidates = [candidate for candidate in candidates if candidate["acceptable"]]
        if not acceptable_candidates and anchor_candidates:
            shortest_endpoint_arc = [
                candidate
                for candidate in candidates
                if candidate["edge_count"] > 0
            ]
            if shortest_endpoint_arc:
                selected = min(
                    shortest_endpoint_arc,
                    key=lambda candidate: (
                        int(candidate["edge_count"]),
                        -float(candidate["overlap_weight"]),
                        float(candidate["mean_distance"]) if np.isfinite(candidate["mean_distance"]) else float("inf"),
                    ),
                )
                selected = {
                    **selected,
                    "acceptable": True,
                    "fallback_reason": "endpoint_shortest_local_arc",
                }
                return {
                    "selected": selected,
                    "candidates": candidates,
                    "fallback_reason": None,
                }

        if not acceptable_candidates:
            return {
                "selected": None,
                "candidates": candidates,
                "fallback_reason": next(
                    (candidate["fallback_reason"] for candidate in candidates if candidate["fallback_reason"]),
                    "no_overlapping_arc",
                ),
            }

        if replace_direction in {"forward", "opposite"}:
            directed = [
                candidate
                for candidate in acceptable_candidates
                if candidate["direction"] == replace_direction
            ]
            if directed:
                acceptable_candidates = directed

        selected = max(
            acceptable_candidates,
            key=lambda candidate: (
                int(candidate["selected_hit_count"] > 0),
                float(candidate["overlap_fraction"]),
                float(candidate["overlap_weight"]) / max(int(candidate["edge_count"]), 1),
                -int(candidate["edge_count"]),
            ),
        )
        return {
            "selected": selected,
            "candidates": candidates,
            "fallback_reason": None,
        }

    @staticmethod
    def _replacement_edge_from_target_anchor(
        target: dict[str, Any],
        role: Literal["start", "end"],
        anchor_count: int,
        closed: bool,
    ) -> int:
        edge_idx = int(target["edge_idx"])
        if "anchor_idx" not in target or anchor_count < 2:
            return edge_idx

        anchor_idx = max(0, min(anchor_count - 1, int(target["anchor_idx"])))
        if closed:
            return anchor_idx % anchor_count if role == "start" else (anchor_idx - 1) % anchor_count

        max_edge = max(0, anchor_count - 2)
        if role == "start":
            return max(0, min(max_edge, anchor_idx))
        return max(0, min(max_edge, anchor_idx - 1))

    def _orient_brush_anchors_for_connection(
        self,
        brush_anchors: list[int],
        start_anchor: int,
        end_anchor: int,
    ) -> list[int]:
        points = np.asarray(self._require_pcd().points, dtype=float)
        sample_coords = points[np.asarray(brush_anchors, dtype=int)]
        start = points[int(start_anchor)]
        end = points[int(end_anchor)]
        forward_score = self._squared_distance(sample_coords[0], start) + self._squared_distance(sample_coords[-1], end)
        reverse_score = self._squared_distance(sample_coords[-1], start) + self._squared_distance(sample_coords[0], end)
        return list(reversed(brush_anchors)) if reverse_score < forward_score else list(brush_anchors)

    def _polyline_length_for_indices(self, indices: list[int]) -> float:
        if len(indices) < 2:
            return 0.0
        points = np.asarray(self._require_pcd().points, dtype=float)
        coords = points[np.asarray(indices, dtype=int)]
        return self._polyline_length_for_points(coords)

    @staticmethod
    def _polyline_length_for_points(points: np.ndarray) -> float:
        coords = np.asarray(points, dtype=float)
        if len(coords) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))

    def _interpolated_length_for_indices(self, indices: list[int]) -> float:
        if len(indices) < 2:
            return 0.0
        points = np.asarray(self._require_pcd().points, dtype=float)
        try:
            dense = _dense_basal_points(points, [int(idx) for idx in indices], False)
        except Exception:
            return self._polyline_length_for_indices(indices)
        if dense.size == 0:
            return self._polyline_length_for_indices(indices)
        return self._polyline_length_for_points(dense)

    def _brush_path_between(self, brush_anchors: list[int], start_idx: int, end_idx: int) -> list[int]:
        oriented = self._orient_brush_anchors_for_connection(brush_anchors, start_idx, end_idx)
        interior = [int(idx) for idx in oriented if int(idx) not in {int(start_idx), int(end_idx)}]
        return self._unique_ordered_indices([int(start_idx), *interior, int(end_idx)])

    @staticmethod
    def _closed_nodes_between(existing: list[int], start_anchor_idx: int, end_anchor_idx: int) -> list[int]:
        if not existing:
            return []
        n = len(existing)
        idx = int(start_anchor_idx) % n
        end = int(end_anchor_idx) % n
        nodes = [int(existing[idx])]
        while idx != end:
            idx = (idx + 1) % n
            nodes.append(int(existing[idx]))
            if len(nodes) > n:
                break
        return nodes

    def _open_path_to_anchor(self, indices: list[int], anchor_idx: int) -> list[int]:
        anchor_idx = max(0, min(len(indices) - 1, int(anchor_idx)))
        left = [int(idx) for idx in indices[:anchor_idx + 1]]
        right = [int(idx) for idx in reversed(indices[anchor_idx:])]
        return left if self._interpolated_length_for_indices(left) >= self._interpolated_length_for_indices(right) else right

    def _open_path_from_anchor(self, indices: list[int], anchor_idx: int) -> list[int]:
        anchor_idx = max(0, min(len(indices) - 1, int(anchor_idx)))
        left = [int(idx) for idx in reversed(indices[:anchor_idx + 1])]
        right = [int(idx) for idx in indices[anchor_idx:]]
        return right if self._interpolated_length_for_indices(right) >= self._interpolated_length_for_indices(left) else left

    def _splice_brush_path_into_same_part(
        self,
        parts: list[dict[str, Any]],
        brush_anchors: list[int],
        start_target: dict[str, Any],
        end_target: dict[str, Any],
    ) -> dict[str, Any]:
        part_idx = int(start_target["part_idx"])
        target_part = parts[part_idx]
        closed = bool(self.interface_edit_draft.get("close_loop", True)) and len(parts) == 1
        self._materialize_endpoint_source_anchors(target_part, len(parts), start_target, end_target)

        existing = target_part["selected_indices"]
        if "anchor_idx" not in start_target or "anchor_idx" not in end_target:
            raise ValueError("Brush Add could not resolve both endpoint anchors.")

        start_anchor = int(start_target["anchor_idx"])
        end_anchor = int(end_target["anchor_idx"])
        if start_anchor == end_anchor:
            raise ValueError("Brush Add endpoints snapped to the same interface point.")

        start_idx = int(existing[start_anchor])
        end_idx = int(existing[end_anchor])
        brush_path = self._brush_path_between(brush_anchors, start_idx, end_idx)
        if len(brush_path) < 2:
            raise ValueError("Brush Add did not add enough points for a splice.")

        candidate_lengths: dict[str, float] = {}
        removed_anchor_count_override: int | None = None
        removed_segment_length_override: float | None = None
        preserved_segment_length_override: float | None = None
        if closed:
            start_to_end = self._closed_nodes_between(existing, start_anchor, end_anchor)
            end_to_start = self._closed_nodes_between(existing, end_anchor, start_anchor)
            start_to_end_length = self._interpolated_length_for_indices(start_to_end)
            end_to_start_length = self._interpolated_length_for_indices(end_to_start)
            candidate_lengths = {
                "start_to_end": float(start_to_end_length),
                "end_to_start": float(end_to_start_length),
            }
            keep_start_to_end = start_to_end_length >= end_to_start_length
            kept_nodes = start_to_end if keep_start_to_end else end_to_start
            removed_nodes = end_to_start if keep_start_to_end else start_to_end
            bridge_interior = list(reversed(brush_path))[1:-1] if keep_start_to_end else brush_path[1:-1]
            new_indices = self._unique_ordered_indices([*kept_nodes, *bridge_interior])
            close_loop = True
            splice_case = "closed_keep_start_to_end" if keep_start_to_end else "closed_keep_end_to_start"
        else:
            low = min(start_anchor, end_anchor)
            high = max(start_anchor, end_anchor)
            low_idx = int(existing[low])
            high_idx = int(existing[high])
            bridge_low_to_high = brush_path if brush_path[0] == low_idx else list(reversed(brush_path))
            start_is_end = start_anchor in {0, len(existing) - 1}
            end_is_end = end_anchor in {0, len(existing) - 1}
            if start_is_end or end_is_end:
                if {start_anchor, end_anchor} == {0, len(existing) - 1}:
                    if existing[-1] == brush_path[0] and existing[0] == brush_path[-1]:
                        bridge_after_existing = brush_path[1:-1]
                    elif existing[-1] == brush_path[-1] and existing[0] == brush_path[0]:
                        bridge_after_existing = list(reversed(brush_path))[1:-1]
                    else:
                        bridge_after_existing = bridge_low_to_high[1:-1]
                    new_indices = self._unique_ordered_indices([*existing, *bridge_after_existing])
                    close_loop = True
                    splice_case = "open_both_ends_preserve_original_close_with_brush"
                    removed_nodes = []
                    kept_nodes = list(existing)
                    candidate_lengths = {
                        "original_path": float(self._interpolated_length_for_indices(existing)),
                        "brush_path": float(self._interpolated_length_for_indices(brush_path)),
                    }
                else:
                    endpoint_anchor = start_anchor if start_is_end else end_anchor
                    interior_anchor = end_anchor if start_is_end else start_anchor
                    endpoint_idx = int(existing[endpoint_anchor])
                    before = [int(idx) for idx in existing[:interior_anchor + 1]]
                    after = [int(idx) for idx in existing[interior_anchor:]]
                    before_length = self._interpolated_length_for_indices(before)
                    after_length = self._interpolated_length_for_indices(after)
                    keep_before = before_length >= after_length
                    kept_nodes = before if keep_before else after
                    removed_nodes = after if keep_before else before
                    kept_contains_endpoint = (
                        (keep_before and endpoint_anchor == 0)
                        or ((not keep_before) and endpoint_anchor == len(existing) - 1)
                    )
                    if kept_contains_endpoint:
                        connector = self._brush_path_between(brush_anchors, int(kept_nodes[-1]), int(kept_nodes[0]))
                        new_indices = self._unique_ordered_indices([*kept_nodes, *connector[1:-1]])
                        close_loop = True
                        splice_case = "open_one_end_keep_longer_endpoint_piece_close_with_brush"
                    elif keep_before:
                        connector = self._brush_path_between(brush_anchors, int(kept_nodes[-1]), endpoint_idx)
                        new_indices = self._unique_ordered_indices([*kept_nodes, *connector[1:]])
                        close_loop = False
                        splice_case = "open_one_end_keep_longer_opposite_piece_append_brush"
                    else:
                        connector = self._brush_path_between(brush_anchors, endpoint_idx, int(kept_nodes[0]))
                        new_indices = self._unique_ordered_indices([*connector[:-1], *kept_nodes])
                        close_loop = False
                        splice_case = "open_one_end_keep_longer_opposite_piece_prepend_brush"
                    removed_anchor_count_override = max(0, len(removed_nodes) - 1)
                    candidate_lengths = {
                        "before_including_interior_snap": float(before_length),
                        "after_including_interior_snap": float(after_length),
                        "kept_piece": float(self._interpolated_length_for_indices(kept_nodes)),
                        "removed_piece": float(self._interpolated_length_for_indices(removed_nodes)),
                        "brush_path": float(self._interpolated_length_for_indices(brush_path)),
                    }
            else:
                bounded_nodes = [int(idx) for idx in existing[low:high + 1]]
                prefix_nodes = [int(idx) for idx in existing[:low + 1]]
                suffix_nodes = [int(idx) for idx in existing[high:]]
                outside_nodes = [*prefix_nodes, *suffix_nodes]
                bounded_length = self._interpolated_length_for_indices(bounded_nodes)
                outside_length = (
                    self._interpolated_length_for_indices(prefix_nodes)
                    + self._interpolated_length_for_indices(suffix_nodes)
                )
                if bounded_length <= outside_length:
                    new_indices = self._unique_ordered_indices([
                        *prefix_nodes,
                        *bridge_low_to_high[1:-1],
                        *suffix_nodes,
                    ])
                    removed_nodes = bounded_nodes
                    kept_nodes = outside_nodes
                    close_loop = False
                    splice_case = "open_interior_remove_shorter_bounded_interval"
                    removed_anchor_count_override = max(0, len(bounded_nodes) - 2)
                    removed_segment_length_override = float(bounded_length)
                    preserved_segment_length_override = float(outside_length)
                else:
                    bridge_high_to_low = list(reversed(bridge_low_to_high))
                    new_indices = self._unique_ordered_indices([
                        *bounded_nodes,
                        *bridge_high_to_low[1:-1],
                    ])
                    removed_nodes = outside_nodes
                    kept_nodes = bounded_nodes
                    close_loop = True
                    splice_case = "open_interior_remove_shorter_outside_route"
                    removed_anchor_count_override = max(0, len(prefix_nodes) + len(suffix_nodes) - 2)
                    removed_segment_length_override = float(outside_length)
                    preserved_segment_length_override = float(bounded_length)
                candidate_lengths = {
                    "bounded_interval": float(bounded_length),
                    "outside_terminal_route": float(outside_length),
                    "brush_path": float(self._interpolated_length_for_indices(brush_path)),
                }

        if len(new_indices) < 2:
            raise ValueError("Brush Add splice would leave fewer than two interface anchors.")

        target_part["selected_indices"] = new_indices
        self.interface_edit_draft["parts"] = parts
        self.interface_edit_draft["close_loop"] = close_loop
        removed_anchor_count = (
            int(removed_anchor_count_override)
            if removed_anchor_count_override is not None
            else max(0, len(removed_nodes) - 2)
        )
        result = {
            "brush_add_mode": "splice",
            "replacement_selection": "longest_segment_splice",
            "splice_case": splice_case,
            "path_edited": True,
            "path_part_count": len(parts),
            "sampled_anchor_count": len(brush_anchors),
            "inserted_anchor_count": max(0, len(brush_path) - 2),
            "removed_anchor_count": int(removed_anchor_count),
            "preserved_anchor_count": int(len(kept_nodes)),
            "removed_segment_length": float(
                removed_segment_length_override
                if removed_segment_length_override is not None
                else self._interpolated_length_for_indices(removed_nodes)
            ),
            "preserved_segment_length": float(
                preserved_segment_length_override
                if preserved_segment_length_override is not None
                else self._interpolated_length_for_indices(kept_nodes)
            ),
            "candidate_lengths": candidate_lengths,
            "start_target_part_index": int(part_idx),
            "end_target_part_index": int(part_idx),
            "start_target_source_index": int(start_idx),
            "end_target_source_index": int(end_idx),
            "start_target_anchor_index": int(start_anchor),
            "end_target_anchor_index": int(end_anchor),
            "start_target_at_open_end": bool(not closed and start_anchor in {0, len(existing) - 1}),
            "end_target_at_open_end": bool(not closed and end_anchor in {0, len(existing) - 1}),
            "guided_target_used": bool(start_target.get("guided") or end_target.get("guided")),
        }
        return {
            **result,
        }

    def _splice_brush_path_between_parts(
        self,
        parts: list[dict[str, Any]],
        brush_anchors: list[int],
        start_target: dict[str, Any],
        end_target: dict[str, Any],
    ) -> dict[str, Any]:
        start_part_idx = int(start_target["part_idx"])
        end_part_idx = int(end_target["part_idx"])
        self._materialize_endpoint_source_anchor(parts[start_part_idx], len(parts), start_target, None)
        self._materialize_endpoint_source_anchor(parts[end_part_idx], len(parts), end_target, None)
        if "anchor_idx" not in start_target or "anchor_idx" not in end_target:
            raise ValueError("Brush Add could not resolve both endpoint anchors.")

        start_indices = parts[start_part_idx]["selected_indices"]
        end_indices = parts[end_part_idx]["selected_indices"]
        start_anchor = int(start_target["anchor_idx"])
        end_anchor = int(end_target["anchor_idx"])
        start_path = self._open_path_to_anchor(start_indices, start_anchor)
        end_path = self._open_path_from_anchor(end_indices, end_anchor)
        brush_path = self._brush_path_between(brush_anchors, start_path[-1], end_path[0])
        insert_anchors = [idx for idx in brush_path[1:-1] if idx not in set(start_path).union(end_path)]
        if len(insert_anchors) < 1:
            raise ValueError("Brush Add did not add any new interface anchors.")

        merged = self._unique_ordered_indices([*start_path, *insert_anchors, *end_path])
        merged_part = {
            "selected_indices": merged,
            "is_lateral": bool(parts[start_part_idx].get("is_lateral", False) or parts[end_part_idx].get("is_lateral", False)),
        }
        next_parts = [
            part
            for idx, part in enumerate(parts)
            if idx not in {start_part_idx, end_part_idx}
        ]
        insert_at = min(start_part_idx, end_part_idx, len(next_parts))
        next_parts.insert(insert_at, merged_part)
        removed_anchor_count = max(0, len(start_indices) + len(end_indices) - len(start_path) - len(end_path))
        self.interface_edit_draft["parts"] = next_parts
        self.interface_edit_draft["close_loop"] = False
        result = {
            "brush_add_mode": "splice",
            "replacement_selection": "connect_parts_splice",
            "path_edited": True,
            "path_part_count": len(next_parts),
            "sampled_anchor_count": len(brush_anchors),
            "inserted_anchor_count": len(insert_anchors),
            "removed_anchor_count": int(removed_anchor_count),
            "preserved_anchor_count": int(len(start_path) + len(end_path)),
            "removed_segment_length": 0.0,
            "preserved_segment_length": float(self._interpolated_length_for_indices(start_path) + self._interpolated_length_for_indices(end_path)),
            "start_target_part_index": int(start_part_idx),
            "end_target_part_index": int(end_part_idx),
            "start_target_source_index": int(start_path[-1]),
            "end_target_source_index": int(end_path[0]),
            "start_target_at_open_end": bool(start_anchor in {0, len(start_indices) - 1}),
            "end_target_at_open_end": bool(end_anchor in {0, len(end_indices) - 1}),
            "guided_target_used": bool(start_target.get("guided") or end_target.get("guided")),
        }
        return result

    def _splice_brush_path_into_draft(
        self,
        parts: list[dict[str, Any]],
        brush_anchors: list[int],
        start_target: dict[str, Any],
        end_target: dict[str, Any],
    ) -> dict[str, Any]:
        if int(start_target["part_idx"]) == int(end_target["part_idx"]):
            return self._splice_brush_path_into_same_part(parts, brush_anchors, start_target, end_target)
        return self._splice_brush_path_between_parts(parts, brush_anchors, start_target, end_target)

    def _connect_brush_between_draft_parts(
        self,
        parts: list[dict[str, Any]],
        brush_anchors: list[int],
        start_target: dict[str, Any],
        end_target: dict[str, Any],
    ) -> dict[str, Any]:
        start_part_idx = int(start_target["part_idx"])
        end_part_idx = int(end_target["part_idx"])
        if start_part_idx == end_part_idx:
            raise ValueError("Endpoint part merge requires two different parts.")

        self._materialize_endpoint_source_anchor(parts[start_part_idx], len(parts), start_target, None)
        self._materialize_endpoint_source_anchor(parts[end_part_idx], len(parts), end_target, None)
        start_indices = parts[start_part_idx]["selected_indices"]
        end_indices = parts[end_part_idx]["selected_indices"]
        if "anchor_idx" not in start_target or "anchor_idx" not in end_target:
            raise ValueError("Brush Add could not resolve both endpoint anchors.")

        start_path = self._open_path_to_anchor(start_indices, int(start_target["anchor_idx"]))
        end_path = self._open_path_from_anchor(end_indices, int(end_target["anchor_idx"]))
        oriented = self._orient_brush_anchors_for_connection(
            brush_anchors,
            start_path[-1],
            end_path[0],
        )
        protected = set(start_path).union(end_path)
        insert_anchors = [idx for idx in oriented if idx not in protected]
        if len(insert_anchors) < 1:
            raise ValueError("Brush Add did not add any new interface anchors.")

        merged = self._unique_ordered_indices(start_path + insert_anchors + end_path)
        merged_part = {
            "selected_indices": merged,
            "is_lateral": bool(parts[start_part_idx].get("is_lateral", False) or parts[end_part_idx].get("is_lateral", False)),
        }
        next_parts = [
            part
            for idx, part in enumerate(parts)
            if idx not in {start_part_idx, end_part_idx}
        ]
        insert_at = min(start_part_idx, end_part_idx, len(next_parts))
        next_parts.insert(insert_at, merged_part)
        self.interface_edit_draft["parts"] = next_parts
        self.interface_edit_draft["close_loop"] = False
        return {
            "brush_add_mode": "replace",
            "replacement_selection": "connect_parts",
            "path_edited": True,
            "path_part_count": len(next_parts),
            "sampled_anchor_count": len(brush_anchors),
            "inserted_anchor_count": len(insert_anchors),
            "removed_anchor_count": 0,
            "start_target_part_index": int(start_part_idx),
            "end_target_part_index": int(end_part_idx),
            "start_target_anchor_index": int(start_target["anchor_idx"]),
            "end_target_anchor_index": int(end_target["anchor_idx"]),
            "guided_target_used": bool(start_target.get("guided") or end_target.get("guided")),
        }

    def _add_brush_bridge_part_to_draft(
        self,
        parts: list[dict[str, Any]],
        brush_anchors: list[int],
        start_target: dict[str, Any],
        end_target: dict[str, Any],
    ) -> dict[str, Any]:
        points = np.asarray(self._require_pcd().points, dtype=float)

        def endpoint_source(target: dict[str, Any]) -> int:
            if "source_idx" in target:
                return int(target["source_idx"])
            part_idx = int(target["part_idx"])
            anchor_idx = int(target.get("anchor_idx", target.get("edge_idx", 0)))
            indices = parts[part_idx]["selected_indices"]
            anchor_idx = max(0, min(len(indices) - 1, anchor_idx))
            return int(indices[anchor_idx])

        start_idx = endpoint_source(start_target)
        end_idx = endpoint_source(end_target)
        oriented = self._orient_brush_anchors_for_connection(brush_anchors, start_idx, end_idx)
        bridge_indices = self._unique_ordered_indices([start_idx, *oriented, end_idx])
        if len(bridge_indices) < 2:
            raise ValueError("Brush Add did not add enough points for a bridge.")

        original_anchor_count = int(sum(len(part.get("selected_indices", []) or []) for part in parts))
        bridge_part = {
            "selected_indices": bridge_indices,
            "is_lateral": bool(
                parts[int(start_target["part_idx"])].get("is_lateral", False)
                if 0 <= int(start_target["part_idx"]) < len(parts)
                else False
            ),
            "source": "brush_add",
        }
        self.interface_edit_draft["parts"] = [*parts, bridge_part]
        self.interface_edit_draft["close_loop"] = False
        return {
            "brush_add_mode": "add_bridge",
            "replacement_selection": "endpoint_bridge",
            "path_edited": True,
            "path_part_count": len(parts) + 1,
            "sampled_anchor_count": len(brush_anchors),
            "inserted_anchor_count": max(0, len(bridge_indices) - 2),
            "removed_anchor_count": 0,
            "preserved_anchor_count": original_anchor_count,
            "start_target_part_index": int(start_target["part_idx"]),
            "end_target_part_index": int(end_target["part_idx"]),
            "start_target_source_index": int(start_idx),
            "end_target_source_index": int(end_idx),
            "guided_target_used": bool(start_target.get("guided") or end_target.get("guided")),
        }

    def _replace_brush_anchors_in_draft_part(
        self,
        parts: list[dict[str, Any]],
        brush_anchors: list[int],
        brush_selected_indices: set[int],
        ordered_stroke_indices: list[int],
        start_target: dict[str, Any],
        end_target: dict[str, Any],
        replace_direction: Literal["forward", "opposite"] | None = None,
    ) -> dict[str, Any]:
        part_idx = int(start_target["part_idx"])
        target_part = parts[part_idx]
        existing = target_part["selected_indices"]
        closed = bool(self.interface_edit_draft.get("close_loop", True)) and len(parts) == 1
        self._materialize_endpoint_source_anchors(target_part, len(parts), start_target, end_target)
        existing = target_part["selected_indices"]
        n = len(existing)
        start_edge = self._replacement_edge_from_target_anchor(start_target, "start", n, closed)
        end_edge = self._replacement_edge_from_target_anchor(end_target, "end", n, closed)
        fallback_edge = start_edge
        selection_result = self._select_brush_replacement_arc(
            existing,
            closed,
            start_edge,
            end_edge,
            brush_selected_indices,
            ordered_stroke_indices,
            brush_anchors,
            replace_direction,
            int(start_target["anchor_idx"]) if "anchor_idx" in start_target else None,
            int(end_target["anchor_idx"]) if "anchor_idx" in end_target else None,
        )
        selected_arc = selection_result.get("selected") if selection_result else None

        if not selected_arc:
            fallback_end = (fallback_edge + 1) % n if closed else fallback_edge + 1
            fallback_oriented = self._orient_brush_anchors_for_connection(
                brush_anchors,
                existing[fallback_edge],
                existing[fallback_end],
            )
            fallback_insert = [idx for idx in fallback_oriented if idx not in set(existing)]
            if len(fallback_insert) < 1:
                raise ValueError("Brush Add did not add any new interface anchors.")
            insert_at = fallback_edge + 1
            target_part["selected_indices"] = self._unique_ordered_indices(
                existing[:insert_at] + fallback_insert + existing[insert_at:]
            )
            self.interface_edit_draft["parts"] = parts
            diagnostics = selection_result or {"candidates": [], "fallback_reason": "no_overlapping_arc"}
            return {
                "brush_add_mode": "insert_fallback",
                "replacement_selection": diagnostics.get("fallback_reason", "no_overlapping_arc"),
                "fallback_reason": diagnostics.get("fallback_reason", "no_overlapping_arc"),
                "path_edited": True,
                "path_part_count": len(parts),
                "sampled_anchor_count": len(brush_anchors),
                "inserted_anchor_count": len(fallback_insert),
                "removed_anchor_count": 0,
                "target_part_index": int(part_idx),
                "target_edge_index": int(fallback_edge),
                "start_target_anchor_index": int(start_target["anchor_idx"]) if "anchor_idx" in start_target else None,
                "end_target_anchor_index": int(end_target["anchor_idx"]) if "anchor_idx" in end_target else None,
                "replacement_candidates": diagnostics.get("candidates", []),
                "guided_target_used": bool(start_target.get("guided") or end_target.get("guided")),
            }

        start_edge = int(selected_arc["start_edge"])
        end_edge = int(selected_arc["end_edge"])
        replaced_edges = [int(edge) for edge in selected_arc["edges"]]

        end_after = (end_edge + 1) % n if closed else min(end_edge + 1, n - 1)
        oriented = self._orient_brush_anchors_for_connection(brush_anchors, existing[start_edge], existing[end_after])
        existing_set = set(existing)
        insert_anchors = [idx for idx in oriented if idx not in existing_set]
        if len(insert_anchors) < 1:
            raise ValueError("Brush Add did not add any new interface anchors.")

        if not closed:
            new_indices = existing[:start_edge + 1] + insert_anchors + existing[end_edge + 1:]
            removed_anchor_count = max(0, end_edge - start_edge)
        else:
            kept_nodes = self._closed_nodes_between(existing, end_after, start_edge)
            new_indices = kept_nodes + insert_anchors
            removed_anchor_count = max(0, n - len(kept_nodes))

        new_indices = self._unique_ordered_indices(new_indices)
        if len(new_indices) < 2:
            raise ValueError("Brush Add replacement would leave fewer than two interface anchors.")
        target_part["selected_indices"] = new_indices
        self.interface_edit_draft["parts"] = parts
        return {
            "brush_add_mode": "replace",
            "replacement_selection": "overlap",
            "path_edited": True,
            "path_part_count": len(parts),
            "sampled_anchor_count": len(brush_anchors),
            "inserted_anchor_count": len(insert_anchors),
            "removed_anchor_count": int(removed_anchor_count),
            "target_part_index": int(part_idx),
            "target_edge_index": int(start_edge),
            "start_target_part_index": int(part_idx),
            "start_target_edge_index": int(start_edge),
            "start_target_anchor_index": int(start_target["anchor_idx"]) if "anchor_idx" in start_target else None,
            "end_target_part_index": int(part_idx),
            "end_target_edge_index": int(end_edge),
            "end_target_anchor_index": int(end_target["anchor_idx"]) if "anchor_idx" in end_target else None,
            "replaced_edge_indices": [int(edge) for edge in replaced_edges],
            "replacement_edge_count": int(len(replaced_edges)),
            "overlap_edge_count": int(selected_arc["overlap_edge_count"]),
            "overlap_fraction": float(selected_arc["overlap_fraction"]),
            "selected_overlap_count": int(selected_arc["selected_hit_count"]),
            "near_overlap_count": int(selected_arc["near_sample_count"]),
            "replacement_candidates": selection_result.get("candidates", []) if selection_result else [],
            "guided_target_used": bool(start_target.get("guided") or end_target.get("guided")),
        }

    @staticmethod
    def _split_ordered_path_by_removed(point_indices: np.ndarray, removed: set[int], closed: bool = False) -> tuple[list[list[int]], int]:
        runs: list[list[int]] = []
        current: list[int] = []
        ordered = point_indices.astype(int).tolist()
        removed_positions = [pos for pos, point_idx in enumerate(ordered) if int(point_idx) in removed]
        if not removed_positions:
            return [ordered] if len(dict.fromkeys(ordered)) >= 2 else [], 0

        bridge_gap = max(3, min(25, len(ordered) // 100))
        pad = max(1, min(5, len(ordered) // 500))
        remove_mask = np.zeros(len(ordered), dtype=bool)
        start = removed_positions[0]
        previous = removed_positions[0]
        for pos in removed_positions[1:]:
            if pos - previous > bridge_gap:
                remove_mask[max(0, start - pad):min(len(ordered), previous + pad + 1)] = True
                start = pos
            previous = pos
        remove_mask[max(0, start - pad):min(len(ordered), previous + pad + 1)] = True

        for pos, point_idx in enumerate(ordered):
            if remove_mask[pos]:
                if len(dict.fromkeys(current)) >= 2:
                    runs.append(current)
                current = []
            else:
                current.append(int(point_idx))
        if len(dict.fromkeys(current)) >= 2:
            runs.append(current)
        if closed and len(runs) > 1 and ordered and not remove_mask[0] and not remove_mask[-1]:
            runs = [runs[-1] + runs[0], *runs[1:-1]]
        return runs, int(np.sum(remove_mask))

    def _sparse_part_from_remaining_path(
        self,
        remaining_indices: list[int] | np.ndarray,
        selected_indices: list[int],
        is_lateral: bool = False,
    ) -> dict[str, Any] | None:
        remaining = self._unique_ordered_indices(remaining_indices)
        if len(remaining) < 2:
            return None

        remaining_positions = {int(idx): pos for pos, idx in enumerate(remaining)}
        preserved_controls = [
            int(idx)
            for idx in selected_indices
            if int(idx) in remaining_positions
        ]
        preserved_controls.sort(key=lambda idx: remaining_positions[idx])

        candidate_anchors = [remaining[0], *preserved_controls, remaining[-1]]
        candidate_anchors = self._unique_ordered_indices(candidate_anchors)
        if len(candidate_anchors) < 2:
            candidate_anchors = [remaining[0], remaining[-1]]

        anchors = self._sample_sparse_path_controls(candidate_anchors, BRUSH_REMOVE_MAX_ANCHORS)
        if len(anchors) < 2:
            return None
        return {"selected_indices": anchors, "is_lateral": bool(is_lateral)}

    def _rewrite_draft_paths_after_brush_remove(self, removed: set[int]) -> dict[str, Any]:
        if not self.interface_edit_draft:
            raise ValueError("Create an interface draft before brushing points.")

        metadata, _, _ = self._draft_effective_interface(self.interface_edit_draft)
        new_parts: list[dict[str, Any]] = []
        path_removed_count = 0
        path_edited = False
        editable_parts = [part for part in metadata.get("parts", []) or [] if part.get("source") != "brush_add"]
        closed_single_part = bool(self.interface_edit_draft.get("close_loop", True)) and len(editable_parts) == 1

        for part in metadata.get("parts", []) or []:
            if part.get("source") == "brush_add":
                continue

            is_lateral = bool(part.get("is_lateral", False))
            point_indices = np.asarray(part.get("point_indices", []), dtype=int)
            selected_indices = self._unique_ordered_indices(part.get("selected_indices", []) or [])
            touched = bool(len(point_indices)) and any(int(idx) in removed for idx in point_indices)

            if not touched:
                if len(selected_indices) >= 2:
                    new_parts.append({"selected_indices": selected_indices, "is_lateral": is_lateral})
                continue

            path_edited = True
            remaining_runs, removed_from_path = self._split_ordered_path_by_removed(point_indices, removed, closed_single_part)
            path_removed_count += removed_from_path
            for run in remaining_runs:
                new_part = self._sparse_part_from_remaining_path(run, selected_indices, is_lateral)
                if new_part is not None:
                    new_parts.append(new_part)

        if path_edited:
            self.interface_edit_draft["parts"] = new_parts
            self.interface_edit_draft["close_loop"] = False

        return {
            "path_edited": path_edited,
            "path_removed_count": path_removed_count,
            "path_part_count": len(new_parts),
            "control_anchor_count": int(sum(len(part.get("selected_indices", []) or []) for part in new_parts)),
        }

    def _push_draft_history(self) -> None:
        if not self.interface_edit_draft:
            return
        snapshot = deepcopy({
            "parts": self.interface_edit_draft.get("parts", []),
            "close_loop": self.interface_edit_draft.get("close_loop", True),
            "include_indices": self.interface_edit_draft.get("include_indices", []),
            "exclude_indices": self.interface_edit_draft.get("exclude_indices", []),
        })
        history = self.interface_edit_draft.setdefault("history", [])
        history.append(snapshot)
        del history[:-DRAFT_HISTORY_LIMIT]

    def _draft_effective_interface(
        self,
        draft: dict[str, Any],
    ) -> tuple[dict[str, Any], list[np.ndarray], np.ndarray]:
        pcd = self._require_pcd()
        point_count = len(pcd.points)
        parts = draft.get("parts", []) or []
        if not parts:
            raise ValueError("The interface draft has no control anchors.")

        part_indices_list, lateral_flags = self._validate_interface_parts(parts)
        metadata, dense_parts, _ = self._compute_basal_metadata(
            part_indices_list,
            lateral_flags,
            bool(draft.get("close_loop", True)),
        )
        include = {
            self._validate_index(idx, point_count, "draft include point")
            for idx in draft.get("include_indices", []) or []
        }
        exclude = {
            self._validate_index(idx, point_count, "draft exclude point")
            for idx in draft.get("exclude_indices", []) or []
        }
        include.difference_update(exclude)

        filtered_parts: list[dict[str, Any]] = []
        filtered_dense_parts: list[np.ndarray] = []
        effective_indices: list[int] = []
        for part in metadata.get("parts", []) or []:
            point_indices = np.asarray(part.get("point_indices", []), dtype=int)
            dense_points = np.asarray(part.get("dense_points", []), dtype=float)
            if len(point_indices) and dense_points.shape[0] == len(point_indices):
                keep_mask = np.array([int(idx) not in exclude for idx in point_indices], dtype=bool)
                point_indices = point_indices[keep_mask]
                dense_points = dense_points[keep_mask]
            else:
                point_indices = point_indices[[int(idx) not in exclude for idx in point_indices]]
                dense_points = np.asarray(self._require_pcd().points)[point_indices] if len(point_indices) else np.empty((0, 3))
            if len(point_indices) == 0:
                continue
            filtered = deepcopy(part)
            filtered["id"] = len(filtered_parts) + 1
            filtered["point_indices"] = point_indices.astype(int).tolist()
            filtered["dense_points"] = dense_points.astype(float).tolist()
            filtered["num_points"] = int(len(point_indices))
            filtered_parts.append(filtered)
            filtered_dense_parts.append(dense_points)
            effective_indices.extend(point_indices.astype(int).tolist())

        if include:
            points = np.asarray(pcd.points, dtype=float)
            include_indices = np.asarray(sorted(include), dtype=int)
            brush_color = list(INTERFACE_GREEN)
            filtered_parts.append({
                "id": len(filtered_parts) + 1,
                "is_lateral": False,
                "selected_indices": include_indices.astype(int).tolist(),
                "original_points": points[include_indices].astype(float).tolist(),
                "dense_points": points[include_indices].astype(float).tolist(),
                "point_indices": include_indices.astype(int).tolist(),
                "num_points": int(len(include_indices)),
                "color": brush_color,
                "source": "brush_add",
            })
            filtered_dense_parts.append(points[include_indices])
            effective_indices.extend(include_indices.astype(int).tolist())

        if not effective_indices:
            raise ValueError("The interface draft removed every interface point.")

        effective_metadata = {
            "parts": filtered_parts,
            "close_loop": bool(draft.get("close_loop", True)),
            "num_parts": len(filtered_parts),
            "has_lateral_parts": any(bool(part.get("is_lateral")) for part in filtered_parts),
            "palette": [list(color) for color in INTERFACE_PART_COLOR_CYCLE],
            "draft": True,
            "include_indices": sorted(include),
            "exclude_indices": sorted(exclude),
        }
        return effective_metadata, filtered_dense_parts, np.unique(np.asarray(effective_indices, dtype=int))

    def _refresh_interface_draft_preview(self) -> dict[str, Any]:
        if not self.interface_edit_draft:
            raise ValueError("Create an interface draft before editing it.")
        pcd = self._require_pcd()
        metadata, dense_parts, basal_indices = self._draft_effective_interface(self.interface_edit_draft)
        colors = self._build_basal_color_array(metadata)
        self.interface_preview_view_points = np.asarray(pcd.points).copy()
        self.interface_preview_view_colors = colors
        self.interface_preview_view_normals = self._ensure_view_normals(pcd).copy()
        self.interface_preview_metadata = metadata
        self.interface_preview_dense_parts = dense_parts
        self.interface_preview_basal_points = basal_indices
        self.interface_edit_draft["effective_indices"] = basal_indices.astype(int).tolist()
        self.interface_edit_draft["preview_metadata"] = deepcopy(metadata)
        self.status.interface_draft_ready = True
        self.normals_display_ready = False
        return {
            "basal_point_count": int(len(basal_indices)),
            "metadata": metadata,
        }

    def get_interface_draft(self) -> dict[str, Any]:
        if not self.interface_edit_draft:
            return {"draft": None, "summary": self.summary()}
        self._refresh_interface_draft_preview()
        return {
            "draft": self._serialize_interface_draft(),
            "summary": self.summary(),
        }

    def _serialize_interface_draft(self) -> dict[str, Any]:
        draft = self.interface_edit_draft or {}
        return {
            "source": draft.get("source"),
            "parts": deepcopy(draft.get("parts", []) or []),
            "close_loop": bool(draft.get("close_loop", True)),
            "include_indices": list(draft.get("include_indices", []) or []),
            "exclude_indices": list(draft.get("exclude_indices", []) or []),
            "effective_indices": list(draft.get("effective_indices", []) or []),
            "metadata": deepcopy(draft.get("preview_metadata", {}) or {}),
            "summary": self._interface_draft_summary(),
        }

    def _resampled_part_from_indices(self, indices: np.ndarray, is_lateral: bool = False) -> dict[str, Any]:
        ordered = self._ordered_auto_interface_indices(np.asarray(indices, dtype=int))
        anchors = self._resample_anchor_indices(ordered)
        if len(anchors) < 2:
            raise ValueError("The interface source is too small to create an editable draft.")
        return {"selected_indices": anchors, "is_lateral": bool(is_lateral)}

    def _draft_parts_from_metadata(self, metadata: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[int], list[int]]:
        if not metadata:
            return [], [], []
        parts: list[dict[str, Any]] = []
        include_indices = set(int(idx) for idx in metadata.get("include_indices", []) or [])
        exclude_indices = set(int(idx) for idx in metadata.get("exclude_indices", []) or [])
        for part in metadata.get("parts", []) or []:
            selected = [int(idx) for idx in part.get("selected_indices", []) or []]
            point_indices = [int(idx) for idx in part.get("point_indices", []) or []]
            if part.get("source") == "brush_add":
                include_indices.update(point_indices or selected)
                continue
            if len(dict.fromkeys(selected)) >= 2:
                parts.append({
                    "selected_indices": list(dict.fromkeys(selected)),
                    "is_lateral": bool(part.get("is_lateral", False)),
                })
            elif len(point_indices) >= 2:
                parts.append(self._resampled_part_from_indices(
                    np.asarray(point_indices, dtype=int),
                    bool(part.get("is_lateral", False)),
                ))
        return parts, sorted(include_indices), sorted(exclude_indices)

    def create_interface_draft_from_source(self, source: Literal["auto", "manual"]) -> dict[str, Any]:
        pcd = self._require_pcd()
        source_name = str(source).lower()
        source_indices: np.ndarray | None
        draft_parts: list[dict[str, Any]]
        include_indices: list[int] = []
        exclude_indices: list[int] = []
        close_loop = True

        if source_name == "auto":
            source_indices = self.auto_basal_points if self.auto_basal_points is not None else (
                self.basal_points if self.interface_source == "auto" else None
            )
            draft_parts = []
        elif source_name == "manual":
            if not self._has_manual_interface() or self.manual_basal_points is None:
                raise ValueError("Save a manual interface before editing the manual interface.")
            source_indices = self.manual_basal_points
            draft_parts, include_indices, exclude_indices = self._draft_parts_from_metadata(self.manual_basal_parts_metadata)
            close_loop = bool((self.manual_basal_parts_metadata or {}).get("close_loop", True))
        else:
            raise ValueError("Interface draft source must be 'auto' or 'manual'.")

        if source_indices is None or len(source_indices) < 2:
            if source_name == "auto":
                raise ValueError("Run regular region growing first so an automatic interface is available.")
            raise ValueError("Save a manual interface with at least two interface points before editing it.")
        if not draft_parts:
            draft_parts = [self._resampled_part_from_indices(np.asarray(source_indices, dtype=int))]
        anchor_count = sum(len(part.get("selected_indices", []) or []) for part in draft_parts)
        self.interface_edit_draft = {
            "source": source_name,
            "parts": draft_parts,
            "close_loop": close_loop,
            "include_indices": include_indices,
            "exclude_indices": exclude_indices,
            "history": [],
            "source_point_count": int(len(source_indices)),
        }
        colors = np.full((len(pcd.points), 3), 0.5, dtype=float)
        colors[np.asarray(source_indices, dtype=int)] = INTERFACE_GREEN
        self.interface_view_points = np.asarray(pcd.points).copy()
        self.interface_view_colors = colors
        self.interface_view_normals = self._ensure_view_normals(pcd).copy()
        preview = self._refresh_interface_draft_preview()
        return {
            "draft": self._serialize_interface_draft(),
            "source": source_name,
            "source_point_count": int(len(source_indices)),
            "auto_point_count": int(len(source_indices)) if source_name == "auto" else 0,
            "anchor_count": int(anchor_count),
            **preview,
            "summary": self.summary(),
        }

    def create_interface_draft_from_auto(self) -> dict[str, Any]:
        return self.create_interface_draft_from_source("auto")

    def update_interface_draft_anchors(self, parts: list[dict[str, Any]], close_loop: bool) -> dict[str, Any]:
        if not self.interface_edit_draft:
            raise ValueError("Create an interface draft before editing anchors.")
        part_indices_list, _ = self._validate_interface_parts(parts)
        normalized_parts: list[dict[str, Any]] = []
        for idx, indices in enumerate(part_indices_list):
            unique_indices = list(dict.fromkeys(int(point_idx) for point_idx in indices))
            if len(unique_indices) < 2:
                raise ValueError(f"Draft part {idx + 1} needs at least two unique anchors.")
            normalized_parts.append({
                "selected_indices": unique_indices,
                "is_lateral": bool(parts[idx].get("is_lateral", False)),
            })
        self._push_draft_history()
        self.interface_edit_draft["parts"] = normalized_parts
        self.interface_edit_draft["close_loop"] = bool(close_loop)
        preview = self._refresh_interface_draft_preview()
        return {"draft": self._serialize_interface_draft(), **preview, "summary": self.summary()}

    def brush_interface_draft(
        self,
        mode: Literal["add", "remove"],
        selected_indices: list[int],
        stroke_indices: list[int] | None = None,
        target_part_index: int | None = None,
        target_edge_index: int | None = None,
        target_anchor_index: int | None = None,
        target_source_index: int | None = None,
        start_target_part_index: int | None = None,
        start_target_edge_index: int | None = None,
        start_target_anchor_index: int | None = None,
        start_target_edge_t: float | None = None,
        start_target_source_index: int | None = None,
        end_target_part_index: int | None = None,
        end_target_edge_index: int | None = None,
        end_target_anchor_index: int | None = None,
        end_target_edge_t: float | None = None,
        end_target_source_index: int | None = None,
        replace_direction: Literal["forward", "opposite"] | None = None,
    ) -> dict[str, Any]:
        if not self.interface_edit_draft:
            raise ValueError("Create an interface draft before brushing points.")
        pcd = self._require_pcd()
        point_count = len(pcd.points)
        valid_indices = {
            self._validate_index(idx, point_count, "draft brush point")
            for idx in (selected_indices or [])
        }
        ordered_stroke_indices = self._valid_ordered_indices(stroke_indices or selected_indices or [], point_count, "draft brush stroke point")
        if not valid_indices:
            valid_indices.update(ordered_stroke_indices)
        if not valid_indices:
            raise ValueError("Brush selection did not include any valid visible points.")
        previous_draft = deepcopy(self.interface_edit_draft)
        self._push_draft_history()
        include = set(int(idx) for idx in self.interface_edit_draft.get("include_indices", []) or [])
        exclude = set(int(idx) for idx in self.interface_edit_draft.get("exclude_indices", []) or [])
        path_edit_result = {
            "path_edited": False,
            "path_removed_count": 0,
            "path_part_count": len(self.interface_edit_draft.get("parts", []) or []),
            "sampled_anchor_count": 0,
            "inserted_anchor_count": 0,
            "removed_anchor_count": 0,
            "control_anchor_count": int(sum(
                len(part.get("selected_indices", []) or [])
                for part in self.interface_edit_draft.get("parts", []) or []
            )),
            "guided_target_used": False,
            "brush_add_mode": "none",
        }
        try:
            if mode == "add":
                brush_anchors = self._sample_brush_anchor_indices(ordered_stroke_indices)
                path_edit_result = self._splice_brush_anchors_into_draft(
                    brush_anchors,
                    valid_indices,
                    ordered_stroke_indices,
                    target_part_index,
                    target_edge_index,
                    target_anchor_index,
                    target_source_index,
                    start_target_part_index,
                    start_target_edge_index,
                    start_target_anchor_index,
                    start_target_edge_t,
                    start_target_source_index,
                    end_target_part_index,
                    end_target_edge_index,
                    end_target_anchor_index,
                    end_target_edge_t,
                    end_target_source_index,
                    replace_direction,
                )
                exclude.difference_update(valid_indices)
                exclude.difference_update(brush_anchors)
            elif mode == "remove":
                path_edit_result = self._rewrite_draft_paths_after_brush_remove(valid_indices)
                exclude.update(valid_indices)
                include.difference_update(valid_indices)
            else:
                raise ValueError("Brush mode must be 'add' or 'remove'.")
            self.interface_edit_draft["include_indices"] = sorted(include)
            self.interface_edit_draft["exclude_indices"] = sorted(exclude)
            preview = self._refresh_interface_draft_preview()
            draft_summary = self._interface_draft_summary() or {}
        except Exception:
            self.interface_edit_draft = previous_draft
            raise
        return {
            "mode": mode,
            "changed_count": int(len(valid_indices)),
            **path_edit_result,
            "control_anchor_count": int(draft_summary.get("anchor_count", 0)),
            "effective_interface_count": int(draft_summary.get("effective_count", 0)),
            "draft": self._serialize_interface_draft(),
            **preview,
            "summary": self.summary(),
        }

    def undo_interface_draft(self) -> dict[str, Any]:
        if not self.interface_edit_draft:
            raise ValueError("No interface draft is available to undo.")
        history = self.interface_edit_draft.get("history", [])
        if not history:
            raise ValueError("There are no draft edits to undo.")
        previous = history.pop()
        self.interface_edit_draft.update(previous)
        self.interface_edit_draft["history"] = history
        preview = self._refresh_interface_draft_preview()
        return {"draft": self._serialize_interface_draft(), **preview, "summary": self.summary()}

    def clear_interface_draft(self) -> dict[str, Any]:
        self.interface_edit_draft = None
        self.status.interface_draft_ready = False
        self._clear_interface_preview_state()
        self.normals_display_ready = False
        return self.summary()

    def commit_interface_draft(self) -> dict[str, Any]:
        if not self.interface_edit_draft:
            raise ValueError("Create an interface draft before saving it as manual interface.")
        pcd = self._require_pcd()
        metadata, dense_parts, basal_indices = self._draft_effective_interface(self.interface_edit_draft)
        colors = self._build_basal_color_array(metadata)
        self.normals_display_ready = False
        pcd.colors = o3d.utility.Vector3dVector(colors)
        self._snapshot_interface_view()
        self.manual_basal_points = basal_indices.copy()
        self.manual_dense_basal_parts = [part.copy() for part in dense_parts]
        self.manual_dense_basal_parts_is_lateral = [bool(part.get("is_lateral")) for part in metadata.get("parts", [])]
        self.manual_basal_parts_metadata = deepcopy(metadata)
        self.basal_points = basal_indices.copy()
        self.dense_basal_parts = [part.copy() for part in dense_parts]
        self.dense_basal_parts_is_lateral = [bool(part.get("is_lateral")) for part in metadata.get("parts", [])]
        self.basal_parts_metadata = deepcopy(metadata)
        self.interface_source = "manual"
        self.display_interface_source = "manual"
        self.interface_edit_draft = None
        self.status.interface_draft_ready = False
        self.status.manual_interface_ready = True
        self.status.interface_ready = True
        self._clear_interface_preview_state()
        return {
            "basal_point_count": int(len(basal_indices)),
            "metadata": metadata,
            "summary": self.summary(),
        }

    def set_interface(self, parts: list[dict[str, Any]], close_loop: bool) -> dict[str, Any]:
        pcd = self._require_pcd()
        part_indices_list, lateral_flags = self._validate_interface_parts(parts)

        metadata, dense_parts, basal_indices = self._compute_basal_metadata(part_indices_list, lateral_flags, close_loop)
        colors = self._build_basal_color_array(metadata)
        self.normals_display_ready = False
        pcd.colors = o3d.utility.Vector3dVector(colors)
        self._snapshot_interface_view()
        self.manual_basal_points = basal_indices.copy()
        self.manual_dense_basal_parts = [part.copy() for part in dense_parts]
        self.manual_dense_basal_parts_is_lateral = [bool(part["is_lateral"]) for part in metadata["parts"]]
        self.manual_basal_parts_metadata = deepcopy(metadata)
        self.basal_points = basal_indices
        self.dense_basal_parts = dense_parts
        self.dense_basal_parts_is_lateral = [bool(part["is_lateral"]) for part in metadata["parts"]]
        self.basal_parts_metadata = metadata
        self.interface_source = "manual"
        self.display_interface_source = "manual"
        self._clear_interface_preview_state()
        self.status.manual_interface_ready = True
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

    def _interface_path_indices_from_metadata(
        self,
        metadata: dict[str, Any] | None,
        point_count: int | None = None,
    ) -> np.ndarray:
        if not metadata:
            return np.asarray([], dtype=int)
        indices: list[int] = []
        for part in metadata.get("parts", []) or []:
            part_indices = part.get("point_indices", []) or part.get("selected_indices", []) or []
            indices.extend(int(idx) for idx in part_indices)
        if not indices:
            return np.asarray([], dtype=int)
        path_indices = np.asarray(indices, dtype=int)
        if point_count is not None:
            path_indices = path_indices[(path_indices >= 0) & (path_indices < point_count)]
        return np.unique(path_indices.astype(int))

    @staticmethod
    def _sample_interface_indices(indices: np.ndarray, max_count: int = 500) -> np.ndarray:
        unique = np.unique(np.asarray(indices, dtype=int).reshape(-1))
        if len(unique) <= max_count:
            return unique
        sample_positions = np.linspace(0, len(unique) - 1, max_count).round().astype(int)
        return unique[sample_positions]

    def _display_interface_state(
        self,
        *,
        metadata_override: dict[str, Any] | None = None,
        source_override: Literal["manual", "auto"] | None = None,
    ) -> dict[str, Any] | None:
        if metadata_override is not None:
            indices = self._interface_path_indices_from_metadata(
                metadata_override,
                point_count=len(self.pcd.points) if self.pcd is not None else None,
            )
            return {
                "source": source_override or self.display_interface_source or self.interface_source or "manual",
                "metadata": metadata_override,
                "indices": indices,
            }

        preferred_sources: list[Literal["manual", "auto"]] = []
        if self.display_interface_source in {"manual", "auto"}:
            preferred_sources.append(self.display_interface_source)
        preferred_sources.extend(["manual", "auto"])

        for source in dict.fromkeys(preferred_sources):
            if source == "manual" and self._has_manual_interface():
                return {
                    "source": "manual",
                    "metadata": self.manual_basal_parts_metadata,
                    "indices": self._interface_path_indices_from_metadata(
                        self.manual_basal_parts_metadata,
                        point_count=len(self.pcd.points) if self.pcd is not None else None,
                    ),
                }
            if source == "auto" and self._has_auto_interface():
                indices = np.asarray(self.auto_basal_points, dtype=int).reshape(-1)
                if self.pcd is not None:
                    indices = indices[(indices >= 0) & (indices < len(self.pcd.points))]
                return {
                    "source": "auto",
                    "metadata": self.auto_basal_parts_metadata,
                    "indices": np.unique(indices.astype(int)),
                }
        return None

    def _interface_overlay_markers(
        self,
        state: dict[str, Any] | None = None,
        *,
        max_markers: int = 500,
    ) -> list[dict[str, Any]]:
        if self.pcd is None:
            return []
        overlay = state if state is not None else self._display_interface_state()
        if not overlay:
            return []

        points = np.asarray(self.pcd.points)
        metadata = overlay.get("metadata") or {}
        source = str(overlay.get("source") or "interface")
        markers: list[dict[str, Any]] = []

        for part in metadata.get("parts", []) or []:
            color = part.get("color") or list(INTERFACE_GREEN)
            point_indices = np.asarray(part.get("point_indices", []) or part.get("selected_indices", []) or [], dtype=int)
            if len(point_indices):
                valid = point_indices[(point_indices >= 0) & (point_indices < len(points))]
                for idx in self._sample_interface_indices(valid, max_count=max_markers):
                    markers.append({
                        "index": int(idx),
                        "point": points[int(idx)].astype(float).tolist(),
                        "color": color,
                        "label": f"Interface {part.get('id', source)}",
                    })
            elif part.get("dense_points"):
                dense_points = np.asarray(part.get("dense_points"), dtype=float)
                if dense_points.ndim == 2 and dense_points.shape[1] == 3:
                    stride = max(1, int(np.ceil(len(dense_points) / max_markers)))
                    for dense_idx, point in enumerate(dense_points[::stride]):
                        markers.append({
                            "index": -1,
                            "point": point.astype(float).tolist(),
                            "color": color,
                            "label": f"Interface {part.get('id', source)}",
                        })
            if len(markers) >= max_markers:
                return markers[:max_markers]
        if markers:
            return markers

        indices = np.asarray(overlay.get("indices", []), dtype=int)
        valid_indices = indices[(indices >= 0) & (indices < len(points))]
        color = list(INTERFACE_GREEN)
        label = "Interface auto" if source == "auto" else "Interface manual"
        for idx in self._sample_interface_indices(valid_indices, max_count=max_markers - len(markers)):
            markers.append({
                "index": int(idx),
                "point": points[int(idx)].astype(float).tolist(),
                "color": color,
                "label": label,
            })
        return markers

    def _overlay_interface_colors(
        self,
        colors: np.ndarray,
        source_indices: np.ndarray | None = None,
        state: dict[str, Any] | None = None,
    ) -> np.ndarray:
        color_array = np.asarray(colors, dtype=float).copy()
        overlay = state if state is not None else self._display_interface_state()
        if not overlay:
            return color_array
        interface_indices = np.asarray(overlay.get("indices", []), dtype=int).reshape(-1)
        if len(interface_indices) == 0:
            return color_array

        if source_indices is None:
            valid = interface_indices[(interface_indices >= 0) & (interface_indices < len(color_array))]
            if len(valid):
                color_array[valid] = np.asarray(INTERFACE_GREEN, dtype=float)
            return color_array

        source_array = np.asarray(source_indices, dtype=int).reshape(-1)
        if len(source_array) != len(color_array):
            return color_array
        interface_set = set(int(idx) for idx in interface_indices if int(idx) >= 0)
        if not interface_set:
            return color_array
        mask = np.fromiter((int(idx) in interface_set for idx in source_array), dtype=bool, count=len(source_array))
        if np.any(mask):
            color_array[mask] = np.asarray(INTERFACE_GREEN, dtype=float)
        return color_array

    def _markers_with_interface(
        self,
        markers: list[dict[str, Any]] | None = None,
        state: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return [*(markers or []), *self._interface_overlay_markers(state)]

    @staticmethod
    def _colors_from_segmentation_labels(labels: np.ndarray | None, point_count: int) -> np.ndarray:
        colors = np.full((point_count, 3), 0.5, dtype=float)
        if labels is None:
            return colors
        label_array = np.asarray(labels, dtype=int)
        usable = min(point_count, len(label_array))
        if usable == 0:
            return colors
        color_slice = colors[:usable]
        label_slice = label_array[:usable]
        color_slice[label_slice == 1] = [1.0, 0.0, 0.0]
        color_slice[label_slice == 0] = [0.0, 0.0, 1.0]
        return colors

    @staticmethod
    def _colors_from_branch_ids(
        labels: np.ndarray | None,
        branch_ids: np.ndarray | None,
        branches: list[dict[str, Any]] | None,
        point_count: int,
    ) -> np.ndarray:
        colors = WebWorkflowSession._colors_from_segmentation_labels(labels, point_count)
        if branch_ids is None:
            return colors
        branch_array = np.asarray(branch_ids, dtype=int)
        if branch_array.ndim != 1 or len(branch_array) != point_count:
            return colors
        metadata_by_id: dict[int, dict[str, Any]] = {}
        for branch in branches or []:
            try:
                branch_id = int(branch.get("branch_id"))
            except (TypeError, ValueError):
                continue
            metadata_by_id[branch_id] = branch
        for branch_id in np.unique(branch_array[branch_array >= 0]):
            branch = metadata_by_id.get(int(branch_id), {})
            color = np.asarray(
                branch.get("color") or BRANCH_COLOR_PALETTE[int(branch_id) % len(BRANCH_COLOR_PALETTE)],
                dtype=float,
            )
            if color.shape != (3,):
                color = np.asarray(BRANCH_COLOR_PALETTE[int(branch_id) % len(BRANCH_COLOR_PALETTE)], dtype=float)
            colors[branch_array == int(branch_id)] = np.clip(color, 0.0, 1.0)
        return colors

    def _snapshot_voxel_segmentation(self) -> None:
        if self.segmenter is None:
            return
        points = getattr(self.segmenter, "voxel_region_points", None)
        labels = getattr(self.segmenter, "voxel_region_labels", None)
        if points is None or labels is None:
            self.status.voxel_segmentation_ready = False
            return
        point_array = np.asarray(points, dtype=float)
        label_array = np.asarray(labels, dtype=int)
        if point_array.ndim != 2 or point_array.shape[1] != 3 or len(point_array) == 0:
            self.status.voxel_segmentation_ready = False
            return
        normals = getattr(self.segmenter, "voxel_region_normals", None)
        normal_array = np.asarray(normals, dtype=float) if normals is not None else None
        if normal_array is not None and normal_array.shape != point_array.shape:
            normal_array = None
        branch_ids = getattr(self.segmenter, "voxel_region_branch_ids", None)
        branch_array = np.asarray(branch_ids, dtype=int) if branch_ids is not None else None
        if branch_array is not None and (branch_array.ndim != 1 or len(branch_array) != len(point_array)):
            branch_array = None
        branches = deepcopy(getattr(self.segmenter, "voxel_region_branches", []) or [])
        self.voxel_segmented_points = point_array.copy()
        self.voxel_segmented_labels = label_array.copy()
        self.voxel_segmented_branch_ids = branch_array.copy() if branch_array is not None else None
        self.voxel_segmented_branches = branches
        self.voxel_segmented_colors = self._colors_from_branch_ids(
            label_array,
            self.voxel_segmented_branch_ids,
            self.voxel_segmented_branches,
            len(point_array),
        )
        self.voxel_segmented_normals = normal_array.copy() if normal_array is not None else None
        self.status.voxel_segmentation_ready = True

    @staticmethod
    def _branch_summaries_for(
        branch_ids: np.ndarray | None,
        branches: list[dict[str, Any]] | None,
        point_count: int,
    ) -> list[dict[str, Any]]:
        if branch_ids is None:
            return []
        branch_array = np.asarray(branch_ids, dtype=int)
        if branch_array.ndim != 1 or len(branch_array) != point_count:
            return []

        summaries: list[dict[str, Any]] = []
        metadata_by_id: dict[int, dict[str, Any]] = {}
        for branch in branches or []:
            try:
                branch_id = int(branch.get("branch_id"))
            except (TypeError, ValueError):
                continue
            metadata_by_id[branch_id] = branch

        for branch_id in sorted(int(value) for value in np.unique(branch_array) if int(value) >= 0):
            metadata = dict(metadata_by_id.get(branch_id, {}))
            metadata.setdefault("branch_id", branch_id)
            metadata.setdefault("label", f"Seed {branch_id + 1}")
            metadata.setdefault("seed_index", branch_id)
            metadata.setdefault("class_label", "seed")
            metadata.setdefault("region_index", None)
            metadata.setdefault("color", BRANCH_COLOR_PALETTE[branch_id % len(BRANCH_COLOR_PALETTE)])
            metadata["node_count"] = int(np.sum(branch_array == branch_id))
            summaries.append(metadata)
        return summaries

    def _voxel_branch_summaries(self) -> list[dict[str, Any]]:
        point_count = 0 if self.voxel_segmented_points is None else len(self.voxel_segmented_points)
        return self._branch_summaries_for(
            self.voxel_segmented_branch_ids,
            self.voxel_segmented_branches,
            point_count,
        )

    def _segmented_branch_summaries(self) -> list[dict[str, Any]]:
        point_count = 0
        if self.segmented_pcd is not None:
            point_count = len(self.segmented_pcd.points)
        elif self.pcd is not None:
            point_count = len(self.pcd.points)
        return self._branch_summaries_for(
            self.segmented_branch_ids,
            self.segmented_branches,
            point_count,
        )

    def _voxel_seed_markers(self) -> list[dict[str, Any]]:
        if self.voxel_segmented_points is None or len(self.voxel_segmented_points) == 0:
            return []

        voxel_points = np.asarray(self.voxel_segmented_points, dtype=float)
        markers: list[dict[str, Any]] = []
        branch_summaries = self._voxel_branch_summaries()
        if branch_summaries:
            for branch in branch_summaries:
                try:
                    idx = int(branch.get("seed_index"))
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(voxel_points):
                    color = branch.get("color") or BRANCH_COLOR_PALETTE[int(branch["branch_id"]) % len(BRANCH_COLOR_PALETTE)]
                    markers.append({
                        "index": idx,
                        "point": voxel_points[idx].astype(float).tolist(),
                        "color": color,
                        "label": f"{branch.get('label', 'Seed')} ({int(branch.get('node_count', 0))} nodes)",
                    })
            return markers

        def seed_indices_from_segmenter(name: str) -> np.ndarray | None:
            if self.segmenter is None:
                return None
            seeds = getattr(self.segmenter, name, None)
            if seeds is None:
                return None
            indices = np.asarray(seeds, dtype=int).reshape(-1)
            valid = indices[(indices >= 0) & (indices < len(voxel_points))]
            return np.unique(valid.astype(int))

        def nearest_voxel_indices(source_indices: list[int]) -> np.ndarray:
            if self.pcd is None or not source_indices:
                return np.asarray([], dtype=int)
            dense_points = np.asarray(self.pcd.points)
            source = np.asarray(source_indices, dtype=int)
            source = source[(source >= 0) & (source < len(dense_points))]
            if len(source) == 0:
                return np.asarray([], dtype=int)
            tree = cKDTree(voxel_points)
            _, nearest = tree.query(dense_points[source], k=1)
            return np.unique(np.asarray(nearest, dtype=int))

        def add_markers(indices: np.ndarray | None, color: list[float], label: str) -> None:
            if indices is None or len(indices) == 0:
                return
            for idx in indices:
                point = voxel_points[int(idx)]
                markers.append({
                    "index": int(idx),
                    "point": point.astype(float).tolist(),
                    "color": color,
                    "label": label,
                })

        rock_indices = seed_indices_from_segmenter("rock_seeds")
        pedestal_indices = seed_indices_from_segmenter("pedestal_seeds")
        if rock_indices is None:
            rock_indices = nearest_voxel_indices(self.rock_seeds)
        if pedestal_indices is None:
            pedestal_indices = nearest_voxel_indices(self.pedestal_seeds)

        add_markers(rock_indices, [1, 0.05, 0.02], "Rock seed")
        add_markers(pedestal_indices, [0.0, 0.24, 1.0], "Pedestal seed")
        return markers

    def _has_manual_interface(self) -> bool:
        return (
            self.manual_basal_points is not None
            and len(self.manual_basal_points) > 0
            and bool((self.manual_basal_parts_metadata or {}).get("parts"))
        )

    def _has_auto_interface(self) -> bool:
        return self.auto_basal_points is not None and len(self.auto_basal_points) > 0

    def _manual_basal_coords_for_segmentation(self) -> np.ndarray | None:
        if not self._has_manual_interface() or self.manual_basal_points is None or len(self.manual_basal_points) == 0:
            return None
        return np.asarray(self._require_pcd().points)[self.manual_basal_points]

    def _activate_manual_interface_for_segmentation(self) -> None:
        if not self._has_manual_interface() or self.manual_basal_points is None:
            raise ValueError("Save a manual interface before running ICRG.")
        self.basal_points = self.manual_basal_points.copy()
        self.dense_basal_parts = [part.copy() for part in self.manual_dense_basal_parts]
        self.dense_basal_parts_is_lateral = list(self.manual_dense_basal_parts_is_lateral)
        self.basal_parts_metadata = deepcopy(self.manual_basal_parts_metadata)
        self.interface_source = "manual"
        self.display_interface_source = "manual"
        self.status.interface_ready = True

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

    def _set_automatic_interface_from_segmentation(self, labels: np.ndarray | None = None) -> np.ndarray:
        label_source = labels if labels is not None else self.segmented_labels
        if label_source is None:
            raise ValueError("Run segmentation before detecting an automatic interface.")
        pcd = self._require_pcd()
        points = np.asarray(pcd.points)
        label_array = np.asarray(label_source, dtype=int).reshape(-1)
        if len(label_array) != len(points):
            raise ValueError("Automatic interface labels do not match the point cloud.")
        colors = np.asarray(pcd.colors)
        if colors.shape != points.shape:
            colors = np.full((len(points), 3), 0.5, dtype=float)
        else:
            colors = colors.copy()

        basal_mask = np.asarray(self.detect_basal_points_optimized(points, label_array), dtype=bool)
        basal_indices = np.flatnonzero(basal_mask)
        if len(basal_indices):
            colors[basal_indices] = np.asarray(INTERFACE_GREEN, dtype=float)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            self.interface_view_points = points.copy()
            self.interface_view_colors = colors.copy()
            self.interface_view_normals = self._ensure_view_normals(pcd).copy()
            if self.segmented_pcd is not None:
                self.segmented_pcd.colors = o3d.utility.Vector3dVector(colors)

        self.basal_points = basal_indices.astype(int)
        self.auto_basal_points = self.basal_points.copy()
        self.dense_basal_parts = []
        self.dense_basal_parts_is_lateral = []
        self.basal_parts_metadata = {
            "parts": [],
            "close_loop": False,
            "num_parts": 0,
            "has_lateral_parts": False,
            "palette": [],
        }
        self.auto_basal_parts_metadata = deepcopy(self.basal_parts_metadata)
        self.interface_source = "auto"
        self.display_interface_source = "auto"
        self.status.interface_ready = bool(len(basal_indices))
        self.status.auto_interface_ready = bool(len(basal_indices))
        self.interface_edit_draft = None
        self.status.interface_draft_ready = False
        return self.basal_points

    def segment(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._run_segmentation(params, use_manual_interface=False)

    def segment_icrg(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._run_segmentation(params, use_manual_interface=True)

    def segment_region_growing(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._run_segmentation(
            params,
            use_manual_interface=False,
            complete_label_propagation=False,
        )

    def segment_icrg_region_growing(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._run_segmentation(
            params,
            use_manual_interface=True,
            complete_label_propagation=False,
        )

    def label_propagation(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._complete_label_propagation(params)

    def _run_segmentation(
        self,
        params: dict[str, Any] | None = None,
        *,
        use_manual_interface: bool,
        complete_label_propagation: bool = True,
    ) -> dict[str, Any]:
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

        if use_manual_interface:
            basal_coords = self._manual_basal_coords_for_segmentation()
            if basal_coords is None:
                raise ValueError("Save a manual interface before running ICRG.")
            self._activate_manual_interface_for_segmentation()
        else:
            basal_coords = None
        used_manual_interface_constraint = basal_coords is not None
        segmentation_mode: Literal["rg", "icrg"] = "icrg" if used_manual_interface_constraint else "rg"
        self.voxel_segmented_points = None
        self.voxel_segmented_colors = None
        self.voxel_segmented_normals = None
        self.voxel_segmented_labels = None
        self.voxel_segmented_branch_ids = None
        self.voxel_segmented_branches = []
        self.segmented_branch_ids = None
        self.segmented_branches = []
        self.segmented_pcd = None
        self.segmented_labels = None
        self.region_growing_dense_labels = None
        self.region_growing_dense_branch_ids = None
        self.segmented_pcd_file_path = None
        self.status.voxel_segmentation_ready = False
        self.status.segmentation_ready = False
        self.status.last_segmentation_mode = segmentation_mode
        self.status.mesh_prepared = False
        self.status.mesh_completed = False
        self.status.analysis_completed = False
        self.prepared_mesh_data = None
        self.normals_display_ready = False
        self.noise_removal_history = []
        self.mesh_processor.reconstructed_mesh = None
        self.mesh_processor.temp_mesh_path = None
        self.mesh_path = None
        self.analysis_csv_path = None

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
        self._snapshot_voxel_segmentation()
        self.region_growing_dense_labels = np.asarray(self.segmenter.labels, dtype=int).copy()
        branch_ids = getattr(self.segmenter, "branch_ids", None)
        self.region_growing_dense_branch_ids = (
            np.asarray(branch_ids, dtype=int).copy() if branch_ids is not None else None
        )

        if not complete_label_propagation:
            auto_interface_indices = np.asarray([], dtype=int)
            if not used_manual_interface_constraint:
                auto_interface_indices = self._set_automatic_interface_from_segmentation(
                    self.region_growing_dense_labels
                )
            voxel_labels = np.asarray(self.voxel_segmented_labels, dtype=int)
            return {
                "label_counts": {
                    "unlabeled": int(np.sum(voxel_labels == -1)),
                    "pedestal": int(np.sum(voxel_labels == 0)),
                    "rock": int(np.sum(voxel_labels == 1)),
                },
                "used_manual_interface_constraint": used_manual_interface_constraint,
                "segmentation_mode": segmentation_mode,
                "auto_interface_generated": bool(len(auto_interface_indices)),
                "auto_interface_point_count": int(len(auto_interface_indices)),
                "view": "voxel_segmented",
                "summary": self.summary(),
            }

        return self._complete_label_propagation(params)

    def _complete_label_propagation(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.segmenter is None or self.region_growing_dense_labels is None:
            raise ValueError("Run region growing before label propagation.")
        params = params or {}
        segmentation_mode: Literal["rg", "icrg"] = self.status.last_segmentation_mode or "rg"
        used_manual_interface_constraint = segmentation_mode == "icrg"
        label_propagation_distance = float(params.get(
            "label_propagation_distance",
            _deep_get(self.config, "region_growing.label_propagation_distance", 0.05),
        ))

        self.segmenter.labels = self.region_growing_dense_labels.copy()
        if self.region_growing_dense_branch_ids is not None:
            self.segmenter.branch_ids = self.region_growing_dense_branch_ids.copy()
        self.segmenter.conditional_label_propagation(
            distance_threshold=label_propagation_distance
        )
        colored_pcd = self.segmenter.color_point_cloud()
        labels = np.asarray(self.segmenter.labels)
        labels[labels == -1] = 0
        branch_ids = getattr(self.segmenter, "branch_ids", None)
        branch_array = np.asarray(branch_ids, dtype=int) if branch_ids is not None else None
        if branch_array is not None and (branch_array.ndim != 1 or len(branch_array) != len(labels)):
            branch_array = None
        self.segmented_branch_ids = branch_array.copy() if branch_array is not None else None
        self.segmented_branches = deepcopy(getattr(self.segmenter, "_branch_summary")() if hasattr(self.segmenter, "_branch_summary") else [])

        self.pcd = colored_pcd
        self._ensure_view_normals(self.pcd)
        self.segmented_pcd = colored_pcd
        self.segmented_labels = labels
        self.status.segmentation_ready = True
        self.status.last_segmentation_mode = segmentation_mode
        self.status.mesh_prepared = False
        self.status.mesh_completed = False
        self.status.analysis_completed = False
        self.prepared_mesh_data = None
        self.normals_display_ready = False
        self.noise_removal_history = []
        self.mesh_processor.reconstructed_mesh = None
        self.mesh_processor.temp_mesh_path = None
        self.mesh_path = None
        self.analysis_csv_path = None
        auto_interface_indices = np.asarray([], dtype=int)
        if not used_manual_interface_constraint:
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
            "segmentation_mode": segmentation_mode,
            "auto_interface_generated": bool(len(auto_interface_indices)),
            "auto_interface_point_count": int(len(auto_interface_indices)),
            "download": self.segmented_pcd_file_path,
            "summary": self.summary(),
        }

    def prepare_mesh(self) -> dict[str, Any]:
        if self.segmented_pcd is None or self.segmented_labels is None:
            raise ValueError("Run segmentation before preparing the mesh.")
        pcd = self._require_pcd()
        if self.status.last_segmentation_mode == "icrg":
            self._activate_manual_interface_for_segmentation()
        elif self.interface_source == "manual" or self.basal_points is None or len(self.basal_points) == 0:
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

    def viewer_payload(
        self,
        view_name: str,
        mesh_url: str | None = None,
        color_mode: str | None = None,
    ) -> dict[str, Any]:
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
            indices = np.arange(len(self.raw_view_points), dtype=int)
            interface_state = self._display_interface_state()
            return self._point_payload(
                self.raw_view_points,
                self._overlay_interface_colors(self.raw_view_colors, indices, interface_state),
                indices,
                markers=self._markers_with_interface([], interface_state),
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
            indices = np.arange(len(points), dtype=int)
            interface_state = self._display_interface_state()
            return self._point_payload(
                points,
                self._overlay_interface_colors(colors, indices, interface_state),
                indices,
                markers=self._markers_with_interface(
                    self._current_markers(include_interface=False),
                    interface_state,
                ),
                normals=self._cached_normals_for_points(points, normals),
            )
        elif view_name == "interface":
            if self.interface_preview_view_points is not None and self.interface_preview_view_colors is not None:
                points = self.interface_preview_view_points
                colors = self.interface_preview_view_colors
                normals = self.interface_preview_view_normals
                interface_metadata = self.interface_preview_metadata
                interface_state = self._display_interface_state(
                    metadata_override=interface_metadata,
                    source_override=self.interface_source,
                )
            else:
                points = self.interface_view_points
                colors = self.interface_view_colors
                normals = self.interface_view_normals
                interface_metadata = self.basal_parts_metadata
                interface_state = self._display_interface_state()
            if points is None or colors is None:
                points = self.seed_view_points if self.seed_view_points is not None else self.raw_view_points
                colors = self.seed_view_colors if self.seed_view_colors is not None else self.raw_view_colors
                normals = self.seed_view_normals if self.seed_view_normals is not None else self.raw_view_normals
            if points is None or colors is None:
                pcd = self._require_pcd()
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                normals = self._ensure_view_normals(pcd)
            indices = np.arange(len(points), dtype=int)
            payload = self._point_payload(
                points,
                self._overlay_interface_colors(colors, indices, interface_state),
                indices,
                markers=self._markers_with_interface(
                    self._current_markers(include_interface=False),
                    interface_state,
                ),
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
        elif view_name == "voxel_segmented":
            if self.voxel_segmented_points is None or self.voxel_segmented_colors is None:
                raise ValueError("No voxelized segmentation result is available.")
            interface_state = self._display_interface_state()
            payload = self._point_payload(
                self.voxel_segmented_points,
                self.voxel_segmented_colors,
                np.arange(len(self.voxel_segmented_points), dtype=int),
                markers=self._markers_with_interface(self._voxel_seed_markers(), interface_state),
                normals=self.voxel_segmented_normals,
            )
            if self.voxel_segmented_labels is not None:
                labels = np.asarray(self.voxel_segmented_labels, dtype=int)
                payload["label_counts"] = {
                    "unlabeled": int(np.sum(labels == -1)),
                    "pedestal": int(np.sum(labels == 0)),
                    "rock": int(np.sum(labels == 1)),
                }
            seed_branches = self._voxel_branch_summaries()
            if seed_branches:
                payload["seed_branches"] = seed_branches
            return payload
        elif view_name == "segmented":
            pcd = self.segmented_pcd
            if pcd is not None and color_mode == "multi_seed" and self.segmented_branch_ids is not None:
                points = np.asarray(pcd.points)
                colors = self._colors_from_branch_ids(
                    self.segmented_labels,
                    self.segmented_branch_ids,
                    self.segmented_branches,
                    len(points),
                )
                indices = np.arange(len(points), dtype=int)
                interface_state = self._display_interface_state()
                colors = self._overlay_interface_colors(colors, indices, interface_state)
                payload = self._point_payload(
                    points,
                    colors,
                    indices,
                    markers=self._markers_with_interface([], interface_state),
                    normals=self._ensure_view_normals(pcd),
                )
                seed_branches = self._segmented_branch_summaries()
                if seed_branches:
                    payload["seed_branches"] = seed_branches
                payload["segmented_color_mode"] = "multi_seed"
                return payload
            if pcd is not None:
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                if colors.shape != points.shape:
                    colors = np.full((len(points), 3), 0.5, dtype=float)
                else:
                    colors = colors.copy()
                indices = np.arange(len(points), dtype=int)
                interface_state = self._display_interface_state()
                colors = self._overlay_interface_colors(colors, indices, interface_state)
                payload = self._point_payload(
                    points,
                    colors,
                    indices,
                    markers=self._markers_with_interface([], interface_state),
                    normals=self._ensure_view_normals(pcd),
                )
                seed_branches = self._segmented_branch_summaries()
                if seed_branches:
                    payload["seed_branches"] = seed_branches
                payload["segmented_color_mode"] = "two_color"
                return payload
            markers = []
        elif view_name == "mesh_prepared":
            if not self.prepared_mesh_data:
                raise ValueError("No prepared mesh point cloud is available.")
            payload = self._point_payload(
                self.prepared_mesh_data["combined_points"],
                self.prepared_mesh_data["combined_colors"],
                np.arange(len(self.prepared_mesh_data["combined_points"]), dtype=int),
                markers=self._markers_with_interface([]),
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
            markers = self._markers_with_interface(self._current_markers(include_interface=False))

        if pcd is None:
            raise ValueError(f"View '{view_name}' is not available yet.")

        payload = self._point_payload(
            np.asarray(pcd.points),
            np.asarray(pcd.colors),
            np.arange(len(pcd.points), dtype=int),
            markers=markers,
            normals=self._ensure_view_normals(pcd),
        )
        if view_name == "segmented":
            seed_branches = self._segmented_branch_summaries()
            if seed_branches:
                payload["seed_branches"] = seed_branches
            payload["segmented_color_mode"] = "two_color"
        return payload

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
