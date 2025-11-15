import logging
import multiprocessing
import os
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import open3d as o3d
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from basal_points_algo import BasalPointAlgorithm
from database_manager import DatabaseManager
from geometric_analyzer import GeometricAnalyzer
from mesh_processor import MeshProcessor
from point_cloud_io import PointCloudFileHandler
from visualization import PointCloudVisualization
from utils import filter_point_cloud
from scipy.spatial import cKDTree

# ---- New: YAML-backed configuration ---------------------------------
try:
    import yaml  # PyYAML
except Exception:  # pragma: no cover
    yaml = None

DEFAULT_CONFIG = {
    "users": ["Deep Rodge", "Zhiang Chen", "Ramon Arrowsmith"],
    "thresholds": {
        "smoothness": 0.9,
        "curvature": 0.1,
        "basal_proximity": 0.05,
    },
    "filters": {
        "sor": True,
        "vertical": True,
        "k_neighbors": 10,
        "std_ratio": 2.0,
        "vertical_std": 1.0,
    },
    "normals": {
        "method": "PyMeshLab",  # PyMeshLab | Open3D
        "k": 200,
    },
    "paths": {
        # Supports placeholders: {input_dir}, {pbr}, {ts}
        "pcd_dir": "{input_dir}",
        "mesh_dir": "{input_dir}",
        "csv_dir": "{input_dir}",
    },
    "visualization": {
        "alpha_view_rods": True,
    },
    "region_growing": {
        "voxel_size": 0.02,
        "neighbor_count": 50,
        "distance_threshold": 0.05,
    },
}

INTERFACE_PART_COLOR_CYCLE = [
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 1.0, 0.0),  # Yellow
    (1.0, 0.0, 1.0),  # Magenta
]


def _compute_part_color(part_index: int, is_lateral: bool) -> np.ndarray:
    base = np.array(INTERFACE_PART_COLOR_CYCLE[part_index % len(INTERFACE_PART_COLOR_CYCLE)], dtype=float)
    if is_lateral:
        base = np.clip(base + 0.3, 0.0, 1.0)
    return base

def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base

def _deep_get(dct: dict, dotted: str, default=None):
    cur = dct
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def _find_config_path() -> Optional[Path]:
    # 1) Explicit env var
    env = os.environ.get("ROCK3D_CONFIG")
    if env and Path(env).exists():
        return Path(env)
    # 2) Walk upward from current working directory and script location
    def _search_up(start: Path) -> Optional[Path]:
        resolved = start.resolve()
        for parent in [resolved, *resolved.parents]:
            candidate = parent / "config.yaml"
            if candidate.exists():
                return candidate
        return None

    for root in (Path.cwd(), Path(__file__).resolve().parent):
        found = _search_up(root)
        if found:
            return found

    # 3) ~/.config/rock3d/config.yaml
    p = Path.home() / ".config" / "rock3d" / "config.yaml"
    if p.exists():
        return p
    return None

class AppConfig:
    def __init__(self, path: Optional[Path] = None):
        self.path: Optional[Path] = path or _find_config_path()
        self.cfg = dict(DEFAULT_CONFIG)  # shallow copy
        if yaml and self.path and self.path.exists():
            try:
                with open(self.path, "r") as f:
                    user_cfg = yaml.safe_load(f) or {}
                _deep_update(self.cfg, user_cfg)
            except Exception as exc:  # pragma: no cover
                logging.warning(f"Failed to load config {self.path}: {exc}")

    def get(self, key: str, default=None):
        return _deep_get(self.cfg, key, default)

# ---- End YAML-backed configuration ----------------------------------

@dataclass(frozen=True)
class UIStyle:
    """Centralized UI dimensions and spacing for easy customization."""

    window_width: int = 400
    window_height: int = 900
    sidebar_min_width: int = 200
    grid_spacing: int = 12
    panel_margin: int = 4
    button_min_width: int = 100
    slider_min_width: int = 100
    segmentation_slider_width: int = 140
    threshold_spin_width: int = 72
    threshold_slider_height: int = 18
    threshold_slider_handle: int = 12
    instructions_height: int = 160
    log_min_height: int = 220
    footer_spacing: int = 8
    dialog_manual_width: int = 420
    dialog_manual_height: int = 260
    dialog_multipart_width: int = 480
    dialog_multipart_height: int = 300
    segmentation_max_width: int = 240
    status_panel_height: int = 120


STYLE = UIStyle()


def region_growing(controller, pcd, rock_seeds, pedestal_seeds):
    """Proxy to existing region growing implementation."""
    from RegionGrowing import RegionGrowingSegmentation

    rock_seed_indices = list(map(int, rock_seeds))
    pedestal_seed_indices = list(map(int, pedestal_seeds))

    controller.segmenter = RegionGrowingSegmentation(
        pcd,
        downsample=True,
        voxel_size=getattr(controller, "region_voxel_size", 0.02),
        num_neighbors=getattr(controller, "region_neighbor_count", 50),
        smoothness_threshold=controller.smoothness_threshold,
        distance_threshold=getattr(controller, "region_distance_threshold", 0.05),
        curvature_threshold=controller.curvature_threshold,
        rock_seeds=rock_seed_indices,
        pedestal_seeds=pedestal_seed_indices,
        basal_points=np.asarray(pcd.points)[controller.basal_points] if np.any(controller.basal_points) else None,
        basal_proximity_threshold=controller.basal_proximity_threshold,
        stepwise_visualize=False,
    )

    segmented_pcd, _ = controller.segmenter.segment()
    controller.segmenter.conditional_label_propagation()

    colored_pcd = controller.segmenter.color_point_cloud()
    labels = np.asarray(controller.segmenter.labels)
    labels[labels == -1] = 1
    return colored_pcd, labels


@dataclass
class WorkflowState:
    seeds_ready: bool = False
    basal_ready: bool = False
    segmentation_ready: bool = False
    mesh_prepared: bool = False
    mesh_completed: bool = False
    analysis_completed: bool = False


class ManualSeedDialog(QDialog):
    """Dialog guiding the user through manual seed selection."""

    def __init__(self, parent, style: UIStyle):
        super().__init__(parent)
        self.setWindowTitle("Manual Seed Selection")
        self.resize(style.dialog_manual_width, style.dialog_manual_height)

        self.layout = QVBoxLayout(self)

        self.instructions = QLabel(
            "Pick seeds using [shift + left click].\n"
            "Use [shift + right click] to undo."
        )
        self.instructions.setWordWrap(True)
        self.layout.addWidget(self.instructions)

        self.step_label = QLabel("Step 1/2: Select rock seeds")
        self.layout.addWidget(self.step_label)

        button_layout = QHBoxLayout()
        self.next_button = QPushButton("Next")
        self.next_button.setDefault(True)
        self.done_button = QPushButton("Done")
        self.done_button.setEnabled(False)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.done_button)
        self.layout.addLayout(button_layout)

        self.next_button.clicked.connect(self.advance_to_pedestal)

    def advance_to_pedestal(self):
        self.step_label.setText("Step 2/2: Select pedestal seeds")
        self.next_button.setEnabled(False)
        self.done_button.setEnabled(True)


class MultiPartInterfaceDialog(QDialog):
    """Dialog to guide multi-part interface selection without handling point picking."""

    def __init__(self, parent, style: UIStyle):
        super().__init__(parent)
        self.setWindowTitle("Multi-Part Interface Input")
        self.resize(style.dialog_multipart_width, style.dialog_multipart_height)

        self.current_part = 0
        self.num_parts = 0
        self.points_per_part: List[List[int]] = []
        self.lateral_flags: List[bool] = []
        self.close_loop = False

        self.main_layout = QVBoxLayout(self)

        intro = QLabel("Specify the number of interface constraint parts, then collect points for each part.")
        intro.setWordWrap(True)
        self.main_layout.addWidget(intro)

        self.part_count_container = QWidget()
        form = QFormLayout(self.part_count_container)
        self.part_count_combo = QSpinBox()
        self.part_count_combo.setRange(2, 6)
        self.part_count_combo.setValue(2)
        form.addRow("Number of parts", self.part_count_combo)
        self.main_layout.addWidget(self.part_count_container)

        self.instructions = QLabel("")
        self.instructions.setWordWrap(True)
        self.main_layout.addWidget(self.instructions)

        self.lateral_checkbox = QCheckBox("Part is lateral")
        self.main_layout.addWidget(self.lateral_checkbox)

        self.close_loop_checkbox = QCheckBox("Close loop on final part")
        self.main_layout.addWidget(self.close_loop_checkbox)

        button_layout = QHBoxLayout()
        self.next_button = QPushButton("Start Selection")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.cancel_button)
        self.main_layout.addLayout(button_layout)

        self.cancel_button.clicked.connect(self.reject)
        self.show_count_step()

    def show_count_step(self):
        """Reset dialog to initial state where user picks number of parts."""
        self.current_part = 0
        self.num_parts = 0
        self.points_per_part = []
        self.lateral_flags = []
        self.close_loop = False
        self.part_count_container.setVisible(True)
        self.instructions.setVisible(False)
        self.lateral_checkbox.setVisible(False)
        self.close_loop_checkbox.setVisible(False)
        self.lateral_checkbox.setChecked(False)
        self.close_loop_checkbox.setChecked(True)
        self.next_button.setText("Start Selection")

    def show_part_step(self, part_index: int, total_parts: int):
        """Configure the dialog for point collection of the given part."""
        self.current_part = part_index
        self.num_parts = total_parts
        self.part_count_container.setVisible(False)
        self.instructions.setVisible(True)
        instruction_text = (
            f"Collect points for part {part_index} of {total_parts}.\n"
            "Use the 3D viewer to pick points with [Shift + Left Click] and undo with [Shift + Right Click].\n"
            "Check 'Part is lateral' when the segment is lateral rather than the interface contact."
        )
        self.lateral_checkbox.setVisible(True)
        self.lateral_checkbox.setChecked(False)
        self.lateral_checkbox.setText(f"Part {part_index} is lateral")
        is_final = part_index == total_parts
        self.close_loop_checkbox.setVisible(is_final)
        self.close_loop_checkbox.setChecked(True)
        if is_final:
            instruction_text += "\nEnable 'Close loop on final part' to connect the last and first points." 
        self.instructions.setText(instruction_text)
        self.next_button.setText("Finish" if is_final else "Save Part")



class InterfacePointConfirmationDialog(QDialog):
    """Dialog reminding users how to finish interface selection."""

    def __init__(self, parent, message: str):
        super().__init__(parent)
        self.setWindowTitle("Finalize Interface Constraint Points")
        layout = QVBoxLayout(self)

        instructions = QLabel(message)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        button_row = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.done_button = QPushButton("Done")
        self.cancel_button.clicked.connect(self.reject)
        self.done_button.clicked.connect(self.accept)
        button_row.addStretch(1)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.done_button)
        layout.addLayout(button_row)


class RefactoredMainWindow(QMainWindow):
    """Single-window refactored GUI keeping multiprocessing visualization intact."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Rock Segmentation Tool V2")
        self.style = STYLE

        # New: Load configuration
        self.config = AppConfig()

        self.state = WorkflowState()
        self.current_user: Optional[str] = None
        # Use users from config
        self.users = list(self.config.get("users", DEFAULT_CONFIG["users"]))

        # Runtime state matching legacy implementation
        self.pcd = None
        self.segmenter = None
        self.process: Optional[multiprocessing.Process] = None
        self._active_processes = set()
        self.rock_seeds: Optional[Sequence[int]] = None
        self.pedestal_seeds: Optional[Sequence[int]] = None
        self.poc_points: np.ndarray = np.empty((0, 3))
        self.basal_points: Optional[np.ndarray] = None
        self.basal_parts: List[List[int]] = []
        self.basal_parts_is_lateral: List[bool] = []
        self.point_pick_queue = None
        self.close_picking_event = None
        self.epsg_code = None
        self.noise_removal_history: List[dict] = []
        self.mesh_reconstruction_stage = 0
        self.prepared_mesh_data = None
        self.current_normal_method = 'pymeshlab'
        self.dense_basal_parts = None
        self.dense_basal_parts_is_lateral = None
        self.segmented_pcd_file_path = "--"
        self.mesh_path = "--"
        self.analysis_csv_path = "--"
        self.current_pbr_file = None
        self.input_path: Optional[Path] = None
        self.output_folder: Optional[Path] = None
        self.segmented_pcd = None
        self.segmented_labels = None
        self._interface_dialog: Optional[QDialog] = None
        self._mesh_dialog: Optional[QDialog] = None
        self._busy_depth = 0

        self.basal_parts_metadata = {
            'parts': [],
            'close_loop': False,
            'num_parts': 0,
            'has_lateral_parts': False,
            'palette': [],
        }

        # Thresholds (from config)
        self.smoothness_threshold = float(self.config.get("thresholds.smoothness", DEFAULT_CONFIG["thresholds"]["smoothness"]))
        self.curvature_threshold = float(self.config.get("thresholds.curvature", DEFAULT_CONFIG["thresholds"]["curvature"]))
        self.basal_proximity_threshold = float(self.config.get("thresholds.basal_proximity", DEFAULT_CONFIG["thresholds"]["basal_proximity"]))

        rg_defaults = DEFAULT_CONFIG["region_growing"]
        self.region_voxel_size = float(self.config.get("region_growing.voxel_size", rg_defaults["voxel_size"]))
        self.region_neighbor_count = int(self.config.get("region_growing.neighbor_count", rg_defaults["neighbor_count"]))
        self.region_distance_threshold = float(self.config.get("region_growing.distance_threshold", rg_defaults["distance_threshold"]))

        # Filter settings (always applied)
        self.filter_k_neighbors = int(self.config.get("filters.k_neighbors", DEFAULT_CONFIG["filters"]["k_neighbors"]))
        self.filter_std_ratio = float(self.config.get("filters.std_ratio", DEFAULT_CONFIG["filters"]["std_ratio"]))
        self.filter_vertical_std = float(self.config.get("filters.vertical_std", DEFAULT_CONFIG["filters"]["vertical_std"]))

        # Services
        self.visualizer = PointCloudVisualization()
        self.file_handler = PointCloudFileHandler()
        self.mesh_processor = MeshProcessor()
        self.geometric_analyzer = GeometricAnalyzer()
        self.db_manager = DatabaseManager()

        self._build_ui()
        self._connect_signals()
        self._update_user(self.user_combo.currentIndex())
        self._set_initial_states()
        self._finalize_window_size()
        # Log config source once UI is ready
        logging.info("Config: %s", self.config.path if self.config.path else "defaults")

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(self.style.footer_spacing)
        root_layout.setContentsMargins(self.style.panel_margin, self.style.panel_margin, self.style.panel_margin, self.style.panel_margin)

        self._build_top_bar(root_layout)

        main_body = QHBoxLayout()
        main_body.setSpacing(self.style.grid_spacing)
        root_layout.addLayout(main_body, 1)

        self._build_main_columns(main_body)
        self._build_footer(root_layout)

    def _build_top_bar(self, parent_layout: QVBoxLayout):
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(self.style.grid_spacing // 2)

        row1 = QHBoxLayout()
        row1.setSpacing(self.style.grid_spacing)
        row1.addWidget(QLabel("User"))
        self.user_combo = QComboBox()
        self.user_combo.addItems(self.users)
        row1.addWidget(self.user_combo)
        row1.addStretch(1)

        self.instructions_toggle = QPushButton("Show Instructions")
        self.instructions_toggle.setCheckable(True)
        self.instructions_toggle.setChecked(False)
        self.instructions_toggle.toggled.connect(self._toggle_instructions)
        self._style_buttons(self.instructions_toggle)
        row1.addWidget(self.instructions_toggle)

        row2 = QHBoxLayout()
        row2.setSpacing(self.style.grid_spacing)
        self.current_file_label = QLabel("Current File: None")
        self.epsg_label = QLabel("EPSG: --")
        row2.addWidget(self.current_file_label)
        row2.addWidget(self.epsg_label)
        row2.addStretch(1)

        top_layout.addLayout(row1)
        top_layout.addLayout(row2)
        parent_layout.addWidget(top_widget)

    def _ensure_interface_dialog(self) -> QDialog:
        if self._interface_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("Interface Constraint Tools")
            dialog.setModal(False)
            layout = QVBoxLayout(dialog)
            intro = QLabel("Choose an interface constraint workflow action.")
            intro.setWordWrap(True)
            layout.addWidget(intro)
            layout.addWidget(self.single_interface_button)
            layout.addWidget(self.multi_interface_button)
            layout.addStretch(1)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            self._style_buttons(close_btn)
            layout.addWidget(close_btn)
            self._interface_dialog = dialog
        return self._interface_dialog

    def show_interface_tools_dialog(self):
        dialog = self._ensure_interface_dialog()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _ensure_mesh_dialog(self) -> QDialog:
        if self._mesh_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("Mesh Workflow")
            dialog.setModal(False)
            layout = QVBoxLayout(dialog)
            layout.addWidget(self.prepare_bottom_button)
            normal_row = QHBoxLayout()
            normal_row.addWidget(self.normal_k_label)
            normal_row.addWidget(self.normal_k_spin)
            normal_row.addWidget(self.normal_method_combo)
            normal_row.addWidget(self.compute_normals_button)
            layout.addLayout(normal_row)
            noise_row = QHBoxLayout()
            self.mesh_remove_noise_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self.mesh_undo_noise_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            noise_row.addWidget(self.mesh_remove_noise_button, 1)
            noise_row.addWidget(self.mesh_undo_noise_button, 1)
            layout.addLayout(noise_row)
            layout.addWidget(self.complete_reconstruction_button)
            layout.addWidget(self.save_mesh_button)
            layout.addStretch(1)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            self._style_buttons(close_btn)
            layout.addWidget(close_btn)
            self._mesh_dialog = dialog
        return self._mesh_dialog

    def show_mesh_workflow_dialog(self):
        dialog = self._ensure_mesh_dialog()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _build_main_columns(self, parent_layout: QHBoxLayout):
        self.main_panel = QWidget()
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(self.style.grid_spacing)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        self.main_panel.setLayout(columns_layout)

        self.data_panel = self._create_data_panel()
        self.interface_panel = self._create_interface_panel()
        self.segmentation_panel = self._create_segmentation_panel()
        self.mesh_panel = self._create_mesh_panel()

        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setSpacing(self.style.grid_spacing)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.data_panel)
        left_layout.addWidget(self.segmentation_panel)
        self.instructions_group = self._create_instructions_panel()
        left_layout.addWidget(self.instructions_group)
        left_layout.addStretch(1)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setSpacing(self.style.grid_spacing)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.interface_panel)
        right_layout.addWidget(self.mesh_panel)
        self.status_panel = self._create_status_panel()
        right_layout.addWidget(self.status_panel)
        right_layout.addStretch(1)

        columns_layout.addWidget(left_column)
        columns_layout.addWidget(right_column)

        parent_layout.addWidget(self.main_panel, 3)
        self._toggle_instructions(False)

    def _create_status_panel(self) -> QGroupBox:
        panel = QGroupBox("Workflow Progress")
        panel.setMaximumWidth(self.style.segmentation_max_width)
        panel.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(self.style.panel_margin, self.style.panel_margin, self.style.panel_margin, self.style.panel_margin)
        layout.setSpacing(self.style.grid_spacing)

        self.status_list = QListWidget()
        self.status_items: List[tuple[str, QListWidgetItem]] = []
        for field_name in WorkflowState.__dataclass_fields__:
            label_text = self._format_status_label(field_name)
            item = QListWidgetItem(f"• {label_text}")
            item.setFlags(Qt.ItemIsEnabled)
            item.setData(Qt.UserRole, label_text)
            self.status_list.addItem(item)
            self.status_items.append((field_name, item))
        self.status_list.setFixedHeight(self.style.status_panel_height)
        layout.addWidget(self.status_list)

        return panel

    def _create_instructions_panel(self) -> QGroupBox:
        panel = QGroupBox("Guided Steps")
        panel.setVisible(False)
        panel.setMaximumWidth(self.style.segmentation_max_width)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(self.style.panel_margin, self.style.panel_margin, self.style.panel_margin, self.style.panel_margin)
        layout.setSpacing(self.style.grid_spacing)

        self.instructions_label = QLabel("Welcome! Load a point cloud to begin.")
        self.instructions_label.setWordWrap(True)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(self.instructions_label)
        container_layout.addStretch(1)
        scroll.setWidget(container)
        scroll.setMinimumHeight(self.style.instructions_height)

        layout.addWidget(scroll)
        return panel

    def _toggle_instructions(self, checked: bool):
        group = getattr(self, "instructions_group", None)
        if group is not None:
            group.setVisible(checked)
        if hasattr(self, "instructions_toggle"):
            self.instructions_toggle.setText("Hide Instructions" if checked else "Show Instructions")

    def _finalize_window_size(self):
        self.adjustSize()
        tight_size = self.sizeHint()
        self.resize(tight_size)
        self.setMinimumSize(tight_size)
        self._position_window()

    def _position_window(self):
        app = QApplication.instance()
        if app is None:
            return
        desktop = app.desktop()
        screen_rect = desktop.availableGeometry(self)
        x = screen_rect.x() + screen_rect.width() - self.width()
        self.move(x, screen_rect.y())

    def _build_footer(self, root_layout: QVBoxLayout):
        footer = QHBoxLayout()
        footer.setSpacing(self.style.footer_spacing)
        self.last_message_label = QLabel("Ready")
        self.last_message_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        footer.addWidget(self.last_message_label, 1)

        self.export_summary_label = QLabel("PCD: -- | Mesh: -- | CSV: --")
        self.export_summary_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.export_summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        footer.addWidget(self.export_summary_label, 1)
        root_layout.addLayout(footer)
        self._update_export_summary()

    # Panel Builders ----------------------------------------------------
    def _create_data_panel(self) -> QGroupBox:
        panel = QGroupBox("Dataset and Preprocessing")
        panel.setMaximumWidth(self.style.segmentation_max_width)
        panel.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(self.style.panel_margin, self.style.panel_margin, self.style.panel_margin, self.style.panel_margin)
        layout.setSpacing(self.style.grid_spacing)

        self.load_file_button = QPushButton("Load Point Cloud")
        self.load_database_button = QPushButton("Load Database")
        self.mark_false_positive_button = QPushButton("Log False Positive")
        self.remove_noise_button = QPushButton("Remove Noise")
        self.undo_noise_button = QPushButton("Undo Noise")
        self._style_buttons(
            self.load_file_button,
            self.load_database_button,
            self.mark_false_positive_button,
            self.remove_noise_button,
            self.undo_noise_button,
        )

        layout.addWidget(self.load_file_button)
        layout.addWidget(self.load_database_button)
        layout.addWidget(self.mark_false_positive_button)
        layout.addWidget(self.remove_noise_button)
        layout.addWidget(self.undo_noise_button)

        return panel

    def _create_interface_panel(self) -> QGroupBox:
        panel = QGroupBox("Interface Constraint Workflow")
        panel.setMaximumWidth(self.style.segmentation_max_width)
        panel.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(self.style.panel_margin, self.style.panel_margin, self.style.panel_margin, self.style.panel_margin)
        layout.setSpacing(self.style.grid_spacing)
        self.auto_seed_button = QPushButton("Preview Auto Seeds")
        self.manual_seed_button = QPushButton("Start Manual Seed Selection")
        self.open_interface_tools_button = QPushButton("Interface Constraint Tools...")
        self.single_interface_button = QPushButton("Single Complete Interface Input")
        self.multi_interface_button = QPushButton("Multi-Part Interface Input")
        self._style_buttons(
            self.auto_seed_button,
            self.manual_seed_button,
            self.open_interface_tools_button,
            self.single_interface_button,
            self.multi_interface_button,
        )

        layout.addWidget(self.auto_seed_button)
        layout.addWidget(self.manual_seed_button)
        layout.addWidget(self.open_interface_tools_button)

        return panel

    def _create_segmentation_panel(self) -> QGroupBox:
        panel = QGroupBox("Segmentation Controls")
        panel.setMaximumWidth(self.style.segmentation_max_width)
        panel.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(self.style.panel_margin, self.style.panel_margin, self.style.panel_margin, self.style.panel_margin)
        layout.setSpacing(self.style.grid_spacing)

        # Use config-backed thresholds for initial slider positions
        self.smoothness_slider, self.smoothness_spin = self._add_slider(layout, "Smoothness", self.smoothness_threshold)
        self.curvature_slider, self.curvature_spin = self._add_slider(layout, "Curvature", self.curvature_threshold)
        self.proximity_slider, self.proximity_spin = self._add_slider(layout, "Interface Proximity", self.basal_proximity_threshold)

        self.run_region_growing_button = QPushButton("Run Region Growing")
        self.save_pcd_button = QPushButton("Save Segmented Point Cloud")
        self._style_buttons(self.run_region_growing_button, self.save_pcd_button)

        layout.addWidget(self.run_region_growing_button)
        layout.addWidget(self.save_pcd_button)
        return panel

    def _create_mesh_panel(self) -> QGroupBox:
        panel = QGroupBox("Mesh and Analysis")
        panel.setMaximumWidth(self.style.segmentation_max_width)
        panel.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(self.style.panel_margin, self.style.panel_margin, self.style.panel_margin, self.style.panel_margin)
        layout.setSpacing(self.style.grid_spacing)

        self.mesh_workflow_button = QPushButton("Mesh Workflow...")
        self.prepare_bottom_button = QPushButton("Prepare Bottom Face")
        self.normal_k_label = QLabel("k")
        self.normal_k_spin = QSpinBox()
        self.normal_k_spin.setRange(3, 200)
        self.normal_k_spin.setValue(int(self.config.get("normals.k", DEFAULT_CONFIG["normals"]["k"])))
        self.normal_method_combo = QComboBox()
        self.normal_method_combo.addItems(["PyMeshLab", "Open3D"])
        # Set initial normal method from config
        _method = str(self.config.get("normals.method", DEFAULT_CONFIG["normals"]["method"]))
        _idx = self.normal_method_combo.findText(_method, Qt.MatchFixedString)
        if _idx >= 0:
            self.normal_method_combo.setCurrentIndex(_idx)
        self.compute_normals_button = QPushButton("Compute Normals")
        self.mesh_remove_noise_button = QPushButton("Remove Noise")
        self.mesh_undo_noise_button = QPushButton("Undo Noise")
        self.complete_reconstruction_button = QPushButton("Complete Reconstruction")
        self.save_mesh_button = QPushButton("Save Mesh")
        self.compute_geometric_button = QPushButton("Compute Geometric Analysis")
        self.analysis_next_pbr_button = QPushButton("Load Next PBR")
        self.restart_button = QPushButton("Restart Workflow")
        self._style_buttons(
            self.mesh_workflow_button,
            self.complete_reconstruction_button,
            self.save_mesh_button,
            self.compute_geometric_button,
            self.analysis_next_pbr_button,
            self.prepare_bottom_button,
            self.compute_normals_button,
            self.mesh_remove_noise_button,
            self.mesh_undo_noise_button,
            self.restart_button,
        )

        layout.addWidget(self.mesh_workflow_button)
        layout.addWidget(self.compute_geometric_button)
        layout.addWidget(self.analysis_next_pbr_button)
        layout.addWidget(self.restart_button)
        return panel

    def _add_slider(self, layout: QVBoxLayout, name: str, initial: float) -> tuple[QSlider, QDoubleSpinBox]:
        label = QLabel(f"{name}: {initial:.2f}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(initial * 100))
        slider.setFixedWidth(self.style.segmentation_slider_width)
        slider.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._style_threshold_slider(slider)

        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.01)
        spin.setValue(initial)
        spin.setFixedWidth(self.style.threshold_spin_width)
        spin.setAlignment(Qt.AlignRight)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(self.style.grid_spacing // 2)
        row.addWidget(slider)
        row.addWidget(spin)
        row.addStretch(1)

        def update_from_slider(value: int):
            float_value = value / 100.0
            spin.blockSignals(True)
            spin.setValue(float_value)
            spin.blockSignals(False)
            label.setText(f"{name}: {float_value:.2f}")

        def update_from_spin(value: float):
            slider.setValue(int(round(value * 100)))

        slider.valueChanged.connect(update_from_slider)
        spin.valueChanged.connect(update_from_spin)

        layout.addWidget(label)
        layout.addLayout(row)
        return slider, spin

    def _style_threshold_slider(self, slider: QSlider):
        groove_height = max(2, self.style.threshold_slider_height // 4)
        handle_diameter = max(8, self.style.threshold_slider_handle)
        slider.setFixedHeight(self.style.threshold_slider_height)
        vertical_margin = max(0, (handle_diameter - groove_height) // 2)
        slider.setStyleSheet(
            f"""
            QSlider::groove:horizontal {{
                background-color: #4a4a4a;
                height: {groove_height}px;
                border-radius: {groove_height // 2}px;
                margin: 0px 6px;
            }}
            QSlider::sub-page:horizontal {{
                background-color: #2d89ef;
                border-radius: {groove_height // 2}px;
            }}
            QSlider::add-page:horizontal {{
                background-color: #5a5a5a;
                border-radius: {groove_height // 2}px;
            }}
            QSlider::handle:horizontal {{
                background-color: #d6d6d6;
                border: 1px solid #1f1f1f;
                width: {handle_diameter}px;
                height: {handle_diameter}px;
                margin: -{vertical_margin}px 0px;
                border-radius: {handle_diameter // 2}px;
            }}
            """
        )

    def _style_buttons(self, *buttons: QPushButton):
        for btn in buttons:
            if isinstance(btn, QPushButton):
                btn.setMinimumWidth(self.style.button_min_width)
                btn.setAutoDefault(False)
                btn.setDefault(False)

    # ------------------------------------------------------------------
    # Signal Connections
    # ------------------------------------------------------------------
    def _connect_signals(self):
        self.user_combo.currentIndexChanged.connect(self._update_user)

        # Data panel
        self.load_file_button.clicked.connect(lambda: self.load_las_file(None))
        self.load_database_button.clicked.connect(self.load_database)
        self.mark_false_positive_button.clicked.connect(self.mark_as_false_positive)

        # Interface panel
        self.auto_seed_button.clicked.connect(self.continue_to_select_seeds)
        self.manual_seed_button.clicked.connect(self.start_manual_selection)
        self.open_interface_tools_button.clicked.connect(self.show_interface_tools_dialog)
        self.single_interface_button.clicked.connect(self.input_interface_contacts)
        self.multi_interface_button.clicked.connect(self.start_multi_part_interface_input)

        # Segmentation panel
        self.run_region_growing_button.clicked.connect(self.run_region_growing)
        self.save_pcd_button.clicked.connect(self.save_point_cloud)
        self.smoothness_slider.valueChanged.connect(lambda v: setattr(self, 'smoothness_threshold', v / 100.0))
        self.smoothness_spin.valueChanged.connect(lambda v: setattr(self, 'smoothness_threshold', float(v)))
        self.curvature_slider.valueChanged.connect(lambda v: setattr(self, 'curvature_threshold', v / 100.0))
        self.curvature_spin.valueChanged.connect(lambda v: setattr(self, 'curvature_threshold', float(v)))
        self.proximity_slider.valueChanged.connect(lambda v: setattr(self, 'basal_proximity_threshold', v / 100.0))
        self.proximity_spin.valueChanged.connect(lambda v: setattr(self, 'basal_proximity_threshold', float(v)))

        # Mesh panel
        self.mesh_workflow_button.clicked.connect(self.show_mesh_workflow_dialog)
        self.prepare_bottom_button.clicked.connect(self.reconstruct_mesh)
        self.compute_normals_button.clicked.connect(self._compute_normals_only)
        self.complete_reconstruction_button.clicked.connect(self._complete_reconstruction_only)
        self.save_mesh_button.clicked.connect(self.save_mesh)

        # Mesh panel (analysis controls)
        self.compute_geometric_button.clicked.connect(self.perform_geometric_analysis)
        self.analysis_next_pbr_button.clicked.connect(self.load_next_pbr)

        # Utility actions
        self.restart_button.clicked.connect(self.restart_application)
        self.remove_noise_button.clicked.connect(self.remove_noise_iterative)
        self.undo_noise_button.clicked.connect(self.undo_remove_noise)
        self.mesh_remove_noise_button.clicked.connect(self.remove_noise_iterative)
        self.mesh_undo_noise_button.clicked.connect(self.undo_remove_noise)

    def _set_initial_states(self):
        self.remove_noise_button.setEnabled(False)
        self.undo_noise_button.setEnabled(False)
        self.mesh_remove_noise_button.setEnabled(False)
        self.mesh_undo_noise_button.setEnabled(False)
        self.auto_seed_button.setEnabled(False)
        self.manual_seed_button.setEnabled(False)
        self.open_interface_tools_button.setEnabled(False)
        self.run_region_growing_button.setEnabled(False)
        self.save_pcd_button.setEnabled(False)
        self.single_interface_button.setEnabled(False)
        self.multi_interface_button.setEnabled(False)
        self.prepare_bottom_button.setEnabled(False)
        self.compute_normals_button.setEnabled(False)
        self.complete_reconstruction_button.setEnabled(False)
        self.save_mesh_button.setEnabled(False)
        self.mesh_workflow_button.setEnabled(False)
        self.compute_geometric_button.setEnabled(False)
        self.mark_false_positive_button.setEnabled(False)
        self.analysis_next_pbr_button.setEnabled(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def log(self, message: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")
        sys.stdout.flush()
        self.last_message_label.setText(message)

    def set_instruction(self, text: str):
        self.instructions_label.setText(text)
        self.log(text)

    def _update_user(self, index):
        self.current_user = self.user_combo.currentText()
        self.log(f"Current user: {self.current_user}")

    @staticmethod
    def _format_status_label(field_name: str) -> str:
        if field_name == "basal_ready":
            return "Interface constraints ready"
        label = field_name.replace('_', ' ')
        return label.capitalize() if label else label

    def _update_status_indicators(self):
        if not hasattr(self, "status_items"):
            return
        for field_name, item in self.status_items:
            label_text = item.data(Qt.UserRole) or self._format_status_label(field_name)
            prefix = "✓ " if getattr(self.state, field_name, False) else "• "
            item.setText(prefix + label_text)

    def _resolve_output_dir(self, config_key: str) -> Path:
        """Resolve an output directory using config placeholders."""
        template = self.config.get(config_key, _deep_get(DEFAULT_CONFIG, config_key, "{input_dir}"))
        if not template:
            template = "{input_dir}"

        base_dir = self.output_folder or (self.input_path.parent if self.input_path else Path.cwd())
        context = {
            "input_dir": str(base_dir),
            "pbr": self.current_pbr_file or "output",
            "ts": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        try:
            resolved = template.format(**context)
        except KeyError as exc:  # pragma: no cover
            logging.warning("Missing placeholder %s in config path %s", exc, config_key)
            resolved = template

        path = Path(resolved)
        if not path.is_absolute():
            path = Path(base_dir) / path

        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover
            logging.warning("Could not create directory %s: %s", path, exc)

        return path

    def _prepare_analysis_csv(self):
        """Ensure the analysis CSV exists for the current input file."""
        if not self.input_path:
            self.analysis_csv_path = "--"
            return

        try:
            csv_dir = self._resolve_output_dir('paths.csv_dir')
            parent_name = self.input_path.parent.name or self.input_path.stem
            csv_path = csv_dir / f"{parent_name}_geometric_analysis_results.csv"
            self.analysis_csv_path = self.geometric_analyzer.initialize_placeholder_entry(
                pbr_name=self.current_pbr_file or self.input_path.stem,
                input_path=self.input_path,
                segmented_path=self.segmented_pcd_file_path,
                mesh_path=self.mesh_path,
                user=self.current_user,
                output_csv=csv_path,
            )
            self.log(f"Initialized analysis CSV: {self.analysis_csv_path}")
        except Exception as exc:
            logging.error("Failed to prepare analysis CSV", exc_info=True)
            self.analysis_csv_path = "--"

    def _set_busy(self, busy: bool):
        app = QApplication.instance()
        if app is None:
            return
        if busy:
            if self._busy_depth == 0:
                QApplication.setOverrideCursor(Qt.WaitCursor)
            self._busy_depth += 1
        else:
            if self._busy_depth > 0:
                self._busy_depth -= 1
            if self._busy_depth == 0:
                QApplication.restoreOverrideCursor()
        self._update_export_summary()

    def _update_export_summary(self):
        if not hasattr(self, "export_summary_label"):
            return
        entries = [
            ("PCD", self.segmented_pcd_file_path or "--"),
            ("Mesh", self.mesh_path or "--"),
            ("CSV", self.analysis_csv_path or "--"),
        ]
        metrics = QFontMetrics(self.export_summary_label.font())
        available_width = self.export_summary_label.width()
        if available_width <= 0:
            available_width = self.export_summary_label.sizeHint().width()
        if available_width <= 0:
            available_width = 240
        per_section = max(60, (available_width - 8 * (len(entries) - 1)) // len(entries))
        display_parts = []
        tooltip_lines = []
        for name, value in entries:
            text_value = value if value and value.strip() else "--"
            tooltip_lines.append(f"{name}: {text_value}")
            elided = metrics.elidedText(text_value, Qt.ElideMiddle, per_section)
            display_parts.append(f"{name}: {elided}")
        self.export_summary_label.setText(" | ".join(display_parts))
        self.export_summary_label.setToolTip("\n".join(tooltip_lines))

    def _stop_active_process(self):
        """Terminate and clean up any active multiprocessing visualization."""
        if self.process:
            try:
                self.process.terminate()
                self.process.join(timeout=1.0)
            except Exception as exc:  # pragma: no cover
                logging.warning("Failed to stop process: %s", exc)
            finally:
                if self.process in self._active_processes:
                    self._active_processes.remove(self.process)
                self.process = None

    # ------------------------------------------------------------------
    # Visualization Handling (unchanged behavior)
    # ------------------------------------------------------------------
    def start_visualization_process(self, target, args):
        self._stop_active_process()
        self.process = multiprocessing.Process(target=target, args=args)
        self._active_processes.add(self.process)
        self.process.start()

    def _refresh_visualization(self):
        if self.pcd is None:
            QMessageBox.warning(self, "No Point Cloud", "Load a point cloud first.")
            return
        self.start_visualization_process(
            target=self.visualizer.show_point_cloud,
            args=(
                np.asarray(self.pcd.points),
                np.asarray(self.pcd.colors),
                self.current_pbr_file or "Point Cloud",
            ),
        )

    # ------------------------------------------------------------------
    # Data Loading & Preprocessing
    # ------------------------------------------------------------------
    def load_las_file(self, file_name: Optional[str]):
        if file_name is None:
            dialog = QFileDialog(self)
            dialog.setWindowTitle("Open LAS/LAZ File")
            dialog.setNameFilter("Point Cloud Files (*.las *.laz)")
            dialog.setFileMode(QFileDialog.ExistingFile)
            if dialog.exec_() == QDialog.Accepted:
                file_name = dialog.selectedFiles()[0]
            else:
                return

        try:
            self.input_path = Path(file_name)
            self.current_pbr_file = self.input_path.stem
            self.output_folder = self.input_path.parent
            self.current_file_label.setText(f"Current File: {self.current_pbr_file}")

            self.segmented_pcd = None
            self.segmented_labels = None
            self.prepared_mesh_data = None
            self.mesh_reconstruction_stage = 0
            self.noise_removal_history.clear()
            self.segmented_pcd_file_path = "--"
            self.mesh_path = "--"
            self.analysis_csv_path = "--"
            self.state = WorkflowState()
            self._update_status_indicators()
            self._set_initial_states()
            self._update_export_summary()

            self.pcd, _, self.epsg_code = self.file_handler.load_las_as_open3d_point_cloud(file_name)
            self.epsg_label.setText(f"EPSG: {self.epsg_code or '--'}")

            self._prepare_analysis_csv()
            self._update_export_summary()

            self.mark_false_positive_button.setEnabled(True)
            self.auto_seed_button.setEnabled(True)
            self.manual_seed_button.setEnabled(True)
            self.open_interface_tools_button.setEnabled(False)
            self.mesh_workflow_button.setEnabled(False)
            self.set_instruction("Point cloud loaded. Configure seeds to continue.")
            self.log(f"Loaded point cloud: {file_name}")
            self._refresh_visualization()
        except Exception as exc:
            logging.error("Error loading LAS/LAZ", exc_info=True)
            QMessageBox.critical(self, "Load Error", str(exc))

    def load_database(self):
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Open Database File")
        dialog.setNameFilter("CSV Files (*.csv)")
        if dialog.exec_() == QDialog.Accepted:
            file_name = dialog.selectedFiles()[0]
        else:
            return

        try:
            self.db_manager.load_database(file_name)
            self.log(f"Database loaded: {file_name}")
            self.analysis_next_pbr_button.setEnabled(True)
        except Exception as exc:
            logging.error("Error loading database", exc_info=True)
            QMessageBox.critical(self, "Database Error", str(exc))

    def load_next_pbr(self):
        try:
            next_entry = self.db_manager.get_next_unprocessed()
            if next_entry is None:
                QMessageBox.information(self, "Database", "No unprocessed PBRs remain.")
                return

            file_path = next_entry['las_path']
            self.load_las_file(file_path)
            self.log(f"Loaded next PBR: {file_path}")
        except Exception as exc:
            logging.error("Error getting next PBR", exc_info=True)
            QMessageBox.critical(self, "Next PBR Error", str(exc))

    def mark_as_false_positive(self):
        if not self.current_pbr_file or not self.input_path:
            QMessageBox.warning(self, "No Current File", "Load a point cloud first.")
            return

        false_dir = self.output_folder / "false_positives"
        false_dir.mkdir(exist_ok=True)
        target_file = false_dir / f"{self.current_pbr_file}{self.input_path.suffix}"

        try:
            shutil.move(str(self.input_path), str(target_file))
            csv_path = self.output_folder / "false_positives.csv"
            self.db_manager.mark_false_positive(self.current_pbr_file)
            with open(csv_path, 'a', newline='') as file:
                file.write(f"{self.current_pbr_file},{datetime.now().isoformat()}\n")
            self.log(f"Marked {self.current_pbr_file} as false positive")
            self.load_next_pbr()
        except Exception as exc:
            logging.error("Error marking false positive", exc_info=True)
            QMessageBox.critical(self, "False Positive", str(exc))

    # ------------------------------------------------------------------
    # Seed Selection
    # ------------------------------------------------------------------
    def continue_to_select_seeds(self):
        if self.pcd is None:
            QMessageBox.warning(self, "No Point Cloud", "Load a point cloud first.")
            return

        self.set_instruction("Preprocessing point cloud and computing auto seeds...")
        QApplication.processEvents()

        filtered_pcd, _ = filter_point_cloud(
            self.pcd,
            filter_type='sor',
            use_vertical_filter=True,
            k_neighbors=self.filter_k_neighbors,
            std_ratio=self.filter_std_ratio,
            vertical_std=self.filter_vertical_std,
        )
        self.pcd = filtered_pcd
        self.log("Applied SOR + vertical filtering using configured defaults")

        points = np.asarray(self.pcd.points)
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        centroid = (min_bound + max_bound) / 2.0
        distances = np.linalg.norm(points[:, :2] - centroid[:2], axis=1)
        highest_point_index = np.argmax(points[:, 2] - distances)
        bottommost_point_index = np.argmin(points[:, 2])

        self.rock_seeds = [highest_point_index]
        self.pedestal_seeds = [bottommost_point_index]

        colors = np.full(points.shape, [0.5, 0.5, 0.5])
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        seed_points = [
            (points[highest_point_index], [1, 0, 0]),
            (points[bottommost_point_index], [0, 0, 1]),
        ]
        self.start_visualization_process(
            target=self.visualizer.show_point_cloud,
            args=(points, colors, self.current_pbr_file or "Point Cloud", False, seed_points),
        )

        self.state.seeds_ready = True
        self._update_status_indicators()
        self.single_interface_button.setEnabled(True)
        self.multi_interface_button.setEnabled(True)
        self.run_region_growing_button.setEnabled(True)
        self.open_interface_tools_button.setEnabled(True)
        self.set_instruction("Seeds prepared. Configure interface constraints or run region growing.")

    def start_manual_selection(self):
        if self.pcd is None:
            QMessageBox.warning(self, "No Point Cloud", "Load a point cloud first.")
            return

        dialog = ManualSeedDialog(self, self.style)

        def select_rock():
            self.pick_points(self.pcd)
            self.log("Manual seed selection: rock seeds")

        def select_pedestal():
            self.rock_seeds = self.get_selected_points_close_window()
            if not self.rock_seeds:
                QMessageBox.warning(self, "Selection", "No rock seeds selected.")
                return
            self.pick_points(self.pcd)
            self.log("Manual seed selection: pedestal seeds")

        def finalize():
            self.pedestal_seeds = self.get_selected_points_close_window()
            if not self.pedestal_seeds:
                QMessageBox.warning(self, "Selection", "No pedestal seeds selected.")
                return
            dialog.accept()

        dialog.next_button.clicked.connect(lambda: select_pedestal())
        dialog.done_button.clicked.connect(lambda: finalize())

        select_rock()
        if dialog.exec_() == QDialog.Accepted:
            self.state.seeds_ready = True
            self._update_status_indicators()
            self.single_interface_button.setEnabled(True)
            self.multi_interface_button.setEnabled(True)
            self.run_region_growing_button.setEnabled(True)
            self.open_interface_tools_button.setEnabled(True)
            self.set_instruction("Manual seeds captured. Proceed to interface constraint selection or segmentation.")

    # ------------------------------------------------------------------
    # Point Picking Helpers
    # ------------------------------------------------------------------
    def pick_points(self, pcd):
        self._stop_active_process()

        self.point_pick_queue = multiprocessing.Queue()
        self.close_picking_event = multiprocessing.Event()

        self.start_visualization_process(
            target=self.visualizer.show_point_cloud_picking,
            args=(
                np.asarray(pcd.points),
                np.asarray(pcd.colors),
                self.point_pick_queue,
                self.close_picking_event,
                self.current_pbr_file or "Point Cloud",
            ),
        )

    def get_selected_points_close_window(self, timeout: float = 5.0) -> List[int]:
        selected_points: List[int] = []
        try:
            if self.close_picking_event:
                self.close_picking_event.set()
            if self.point_pick_queue is not None:
                if timeout is None:
                    selected_points = self.point_pick_queue.get()
                else:
                    selected_points = self.point_pick_queue.get(timeout=timeout)
        except Exception as exc:
            logging.warning("Point picking did not return any points: %s", exc)
        finally:
            self._stop_active_process()
            self.point_pick_queue = None
            self.close_picking_event = None
        return selected_points

    def _cancel_point_selection(self):
        """Abort any in-progress point picking session and close its visualization."""
        if self.point_pick_queue is None and self.close_picking_event is None:
            return
        try:
            if self.close_picking_event:
                self.close_picking_event.set()
        except Exception as exc:  # pragma: no cover
            logging.debug("Failed to signal picking event: %s", exc)
        finally:
            self._stop_active_process()
            self.point_pick_queue = None
            self.close_picking_event = None

    # ------------------------------------------------------------------
    # Interface Constraint Input
    # ------------------------------------------------------------------
    def _reset_basal_state(self):
        """Clear stored interface constraint data so new selections overwrite previous results."""
        self.poc_points = np.empty((0, 3))
        self.basal_parts = []
        self.basal_parts_is_lateral = []
        self.basal_points = None
        self.dense_basal_parts = None
        self.dense_basal_parts_is_lateral = None
        self.basal_parts_metadata = {
            'parts': [],
            'close_loop': False,
            'num_parts': 0,
            'has_lateral_parts': False,
            'palette': [],
        }
        self.state.basal_ready = False
        self._update_status_indicators()
        self.prepare_bottom_button.setEnabled(False)
        self.mesh_workflow_button.setEnabled(False)

    def _compute_basal_metadata(self, part_indices_list: List[List[int]], lateral_flags: List[bool], close_loop: bool):
        if self.pcd is None:
            raise ValueError("Load a point cloud before selecting interface constraint points.")
        if not part_indices_list or not any(part for part in part_indices_list):
            raise ValueError("No interface constraint support points were provided.")

        algorithm = BasalPointAlgorithm(self.pcd)
        points = np.asarray(self.pcd.points)
        tree = cKDTree(points)

        global_close_loop = bool(close_loop)
        multi_part_close_loop = global_close_loop and len(part_indices_list) > 1
        single_part_close_loop = global_close_loop and len(part_indices_list) == 1
        first_part_indices = np.asarray(part_indices_list[0], dtype=int) if part_indices_list else np.empty(0, dtype=int)
        if multi_part_close_loop and first_part_indices.size == 0:
            raise ValueError("Close loop requested but the first part has no points.")

        metadata = {
            'parts': [],
            'close_loop': global_close_loop,
            'num_parts': 0,
            'has_lateral_parts': False,
        }

        dense_parts: List[np.ndarray] = []
        all_indices: List[int] = []

        for idx, indices in enumerate(part_indices_list):
            if not indices:
                raise ValueError(f"Part {idx + 1} has no selected points. Please select at least two points.")
            if len(indices) < 2:
                raise ValueError(f"Select at least two points for part {idx + 1}.")

            is_lateral = bool(lateral_flags[idx] if idx < len(lateral_flags) else False)
            close_current = single_part_close_loop

            selected_indices = np.asarray(indices, dtype=int)
            selected_points = points[selected_indices]

            # When closing a multi-part loop, connect the last part back to the first part's first point
            run_indices = selected_indices
            if multi_part_close_loop and idx == len(part_indices_list) - 1:
                run_indices = np.concatenate([selected_indices, first_part_indices[:1]])
                close_current = False  # manual connection handled via appended point

            dense_part = algorithm.run(run_indices, show_progress=False, close_loop=close_current)
            if dense_part is None or len(dense_part) == 0:
                raise ValueError(f"Failed to generate interface curve for part {idx + 1}.")

            dense_part = np.asarray(dense_part)
            _, point_indices = tree.query(dense_part)
            point_indices = np.asarray(point_indices, dtype=int)

            part_color = _compute_part_color(idx, is_lateral)

            metadata['parts'].append({
                'id': idx + 1,
                'is_lateral': is_lateral,
                'selected_indices': selected_indices.tolist(),
                'original_points': selected_points.tolist(),
                'dense_points': dense_part.tolist(),
                'point_indices': point_indices.tolist(),
                'num_points': int(len(point_indices)),
                'color': part_color.tolist(),
            })

            dense_parts.append(dense_part)
            all_indices.extend(point_indices.tolist())

        metadata['num_parts'] = len(metadata['parts'])
        metadata['has_lateral_parts'] = any(part['is_lateral'] for part in metadata['parts'])
        metadata['palette'] = [list(color) for color in INTERFACE_PART_COLOR_CYCLE]

        if not all_indices:
            raise ValueError("No interface constraint points were generated from the selected inputs.")

        basal_indices = np.unique(np.asarray(all_indices, dtype=int))
        return metadata, dense_parts, basal_indices

    def _build_basal_color_array(self, metadata) -> np.ndarray:
        points = np.asarray(self.pcd.points)
        colors = np.full((len(points), 3), 0.5, dtype=float)
        parts = metadata.get('parts', []) or []
        for idx, part in enumerate(parts):
            idxs = np.asarray(part.get('point_indices', []), dtype=int)
            if idxs.size == 0:
                continue
            stored_color = part.get('color')
            if stored_color is None:
                computed = _compute_part_color(idx, part.get('is_lateral', False))
                stored_color = computed.tolist()
                part['color'] = stored_color
            color_array = np.asarray(stored_color, dtype=float)
            colors[idxs] = color_array
        return colors

    def _finalize_interface_selection(self, part_indices_list: List[List[int]], lateral_flags: List[bool], close_loop: bool):
        metadata, dense_parts, basal_indices = self._compute_basal_metadata(part_indices_list, lateral_flags, close_loop)
        colors = self._build_basal_color_array(metadata)
        points_array = np.asarray(self.pcd.points)
        seed_markers: List[tuple[np.ndarray, Sequence[float]]] = []
        for part in metadata.get('parts', []) or []:
            stored_color = part.get('color')
            if stored_color is None:
                stored_color = _compute_part_color(part.get('id', 0) - 1, part.get('is_lateral', False)).tolist()
                part['color'] = stored_color
            for idx in part.get('selected_indices', []) or []:
                if 0 <= idx < len(points_array):
                    seed_markers.append((points_array[idx].copy(), list(stored_color)))

        self._reset_basal_state()
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        self.start_visualization_process(
            target=self.visualizer.show_point_cloud,
            args=(
                np.asarray(self.pcd.points),
                colors,
                self.current_pbr_file or "Point Cloud",
                False,
                seed_markers or None,
            ),
        )

        self.basal_points = basal_indices
        self.dense_basal_parts = dense_parts
        self.dense_basal_parts_is_lateral = [part['is_lateral'] for part in metadata['parts']]
        self.basal_parts_metadata = metadata
        self.basal_parts = [list(map(int, part)) for part in part_indices_list]
        self.basal_parts_is_lateral = [
            bool(lateral_flags[idx] if idx < len(lateral_flags) else False)
            for idx in range(len(self.basal_parts))
        ]
        self.poc_points = np.vstack(dense_parts) if dense_parts else np.empty((0, 3))

        self.state.basal_ready = True
        self._update_status_indicators()
        self.run_region_growing_button.setEnabled(True)
        self.prepare_bottom_button.setEnabled(True)
        self.mesh_workflow_button.setEnabled(True)
        self.set_instruction("Interface constraints ready. Adjust thresholds or run region growing as needed.")

    def input_interface_contacts(self):
        if self.pcd is None:
            QMessageBox.warning(self, "No Point Cloud", "Load a point cloud first.")
            return
        self.pick_points(self.pcd)
        self.set_instruction(
            "Selecting interface constraint points...\n"
            "Use Shift + Left Click to add points and Shift + Right Click to undo."
        )

        message = (
            "Select interface points in the 3D viewer using Shift + Left Click.\n"
            "Use Shift + Right Click to undo the last point.\n\n"
            "Click Done to generate the interface curve or Cancel to exit."
        )
        dialog = InterfacePointConfirmationDialog(self, message)

        if dialog.exec_() == QDialog.Accepted:
            selected_points = self.get_selected_points_close_window(timeout=30.0)
            if not selected_points:
                QMessageBox.warning(self, "Interface Points", "No points were selected. Please try again.")
                self.set_instruction("No interface points were selected. Open interface tools to try again.")
                return
            try:
                self._finalize_interface_selection([list(map(int, selected_points))], [False], True)
            except Exception as exc:
                logging.error("Interface point estimation failed", exc_info=True)
                QMessageBox.critical(self, "Interface Selection", str(exc))
                self.set_instruction("Interface estimation failed. Start the selection again when ready.")
        else:
            self._cancel_point_selection()
            self.set_instruction("Interface point selection canceled. Open interface tools to try again when ready.")

    def start_multi_part_interface_input(self):
        if self.pcd is None:
            QMessageBox.warning(self, "No Point Cloud", "Load a point cloud first.")
            return

        dialog = MultiPartInterfaceDialog(self, self.style)
        state = {
            'num_parts': None,
            'current_part': 0,
            'parts': [],
            'lateral_flags': [],
        }

        def handle_cancel():
            if state['current_part']:
                self.set_instruction("Multi-part interface selection canceled. Open interface tools to try again when ready.")
            self._cancel_point_selection()

        def begin_part(part_index: int):
            dialog.show_part_step(part_index, state['num_parts'])
            self.set_instruction(
                f"Selecting interface points for part {part_index} of {state['num_parts']}\n"
                "Use Shift + Left Click to add points and Shift + Right Click to undo."
            )
            self.pick_points(self.pcd)

        def on_next_clicked():
            # Step 1: choose number of parts
            if state['num_parts'] is None:
                state['num_parts'] = int(dialog.part_count_combo.value())
                state['current_part'] = 1
                begin_part(state['current_part'])
                return

            # Step 2+: capture selected points for current part
            selected_points = self.get_selected_points_close_window(timeout=30.0)
            if not selected_points:
                QMessageBox.warning(self, "Interface Points", "No points were selected. Please pick points in the viewer before continuing.")
                begin_part(state['current_part'])
                return

            state['parts'].append(list(map(int, selected_points)))
            state['lateral_flags'].append(dialog.lateral_checkbox.isChecked())

            if state['current_part'] == state['num_parts']:
                close_loop = dialog.close_loop_checkbox.isChecked()
                try:
                    self._finalize_interface_selection(state['parts'], state['lateral_flags'], close_loop)
                    dialog.accept()
                except Exception as exc:
                    logging.error("Multi-part interface estimation failed", exc_info=True)
                    QMessageBox.critical(self, "Interface Selection", str(exc))
                    # allow reselection of the current (final) part
                    if state['parts']:
                        state['parts'].pop()
                    if state['lateral_flags']:
                        state['lateral_flags'].pop()
                    begin_part(state['current_part'])
            else:
                state['current_part'] += 1
                begin_part(state['current_part'])

        dialog.rejected.connect(handle_cancel)
        dialog.next_button.clicked.connect(on_next_clicked)
        dialog.show_count_step()
        dialog.exec_()

    # ------------------------------------------------------------------
    # Segmentation & Saving
    # ------------------------------------------------------------------
    def run_region_growing(self):
        if not self.rock_seeds or not self.pedestal_seeds:
            QMessageBox.warning(self, "Seeds", "Seed selection required before segmentation.")
            return

        try:
            self._set_busy(True)
            self.set_instruction("Running region growing...")
            colored_pcd, labels = region_growing(self, self.pcd, self.rock_seeds, self.pedestal_seeds)
            self.segmented_pcd = colored_pcd
            self.segmented_labels = labels
            self.start_visualization_process(
                target=self.visualizer.show_point_cloud,
                args=(
                    np.asarray(colored_pcd.points),
                    np.asarray(colored_pcd.colors),
                    self.current_pbr_file or "Segmented",
                ),
            )

            self.state.segmentation_ready = True
            self._update_status_indicators()
            self.save_pcd_button.setEnabled(True)
            self.prepare_bottom_button.setEnabled(True)
            self.compute_normals_button.setEnabled(True)
            self.complete_reconstruction_button.setEnabled(True)
            self.mesh_workflow_button.setEnabled(True)
            self.set_instruction("Segmentation complete. Proceed to mesh reconstruction or save results.")
        except Exception as exc:
            logging.error("Region growing failed", exc_info=True)
            QMessageBox.critical(self, "Segmentation Error", str(exc))
        finally:
            self._set_busy(False)

    def save_point_cloud(self):
        if self.segmented_pcd is None:
            QMessageBox.warning(self, "No Segmentation", "Run segmentation before saving.")
            return

        default_name = f"{self.current_pbr_file}_segmented.las" if self.current_pbr_file else "segmented.las"
        default_dir = self._resolve_output_dir('paths.pcd_dir')
        default_path = str(default_dir / default_name)
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Point Cloud", default_path, "LAS Files (*.las)")
        if not file_name:
            return

        basal_metadata = getattr(self, 'basal_parts_metadata', None) or self.basal_points
        self.segmented_pcd_file_path = self.file_handler.save_point_cloud(
            self.pcd,
            file_name,
            self.segmented_labels,
            basal_metadata,
            plain=False,
        )
        self._update_export_summary()
        self.set_instruction(f"Segmented point cloud saved to {file_name}")

    # ------------------------------------------------------------------
    # Mesh Reconstruction
    # ------------------------------------------------------------------
    def reconstruct_mesh(self):
        if self.segmented_pcd is None:
            QMessageBox.warning(self, "Segmentation Required", "Segment point cloud first.")
            return

        try:
            self._set_busy(True)
            self.set_instruction("Preparing bottom face...")

            if self.basal_points is None:
                self.basal_points = self.detect_basal_points_optimized(
                    np.asarray(self.pcd.points),
                    self.segmented_labels,
                )

            result = self.mesh_processor.prepare_bottom_face(
                self.pcd,
                labels=self.segmented_labels,
                basal_points=self.basal_points,
                dense_basal_parts=getattr(self, 'dense_basal_parts', None),
                dense_basal_parts_is_lateral=getattr(self, 'dense_basal_parts_is_lateral', None),
                use_dbscan_cleaning=False,
                basal_parts_metadata=self.basal_parts_metadata,
            )

            rock_pcd, bottom_pcd, combined_points, combined_colors, combined_normals = self.mesh_processor.compute_normals_for_visualization(
                result.rock_points,
                result.bottom_points,
            )

            self.prepared_mesh_data = {
                'rock_pcd': rock_pcd,
                'bottom_pcd': bottom_pcd,
                'combined_points': combined_points,
                'combined_colors': combined_colors,
                'combined_normals': combined_normals,
                'preparation_result': result,
            }

            self.start_visualization_process(
                target=self.visualizer.show_point_cloud,
                args=(combined_points, combined_colors, "Prepared Mesh", False, None, True, combined_normals),
            )

            self.mesh_reconstruction_stage = 1
            self.state.mesh_prepared = True
            self._update_status_indicators()
            self.remove_noise_button.setEnabled(True)
            self.mesh_remove_noise_button.setEnabled(True)
            self.mesh_undo_noise_button.setEnabled(False)
            self.compute_normals_button.setEnabled(True)
            self.complete_reconstruction_button.setEnabled(True)
            self.save_mesh_button.setEnabled(True)
            self.set_instruction("Bottom face prepared. Adjust normals or remove noise as needed.")
        except Exception as exc:
            logging.error("Mesh preparation failed", exc_info=True)
            QMessageBox.critical(self, "Mesh Preparation", str(exc))
        finally:
            self._set_busy(False)

    def _compute_normals_only(self):
        if not self.prepared_mesh_data:
            QMessageBox.warning(self, "Preparation Required", "Prepare bottom face first.")
            return

        try:
            self._set_busy(True)
            k = self.normal_k_spin.value()
            method = self.normal_method_combo.currentText().lower()
            self.set_instruction(f"Computing normals using {method} (k={k})...")

            prep = self.prepared_mesh_data['preparation_result']
            if method == 'pymeshlab':
                compute_fn = self.mesh_processor.compute_normals_for_visualization
            else:
                compute_fn = self.mesh_processor.compute_normals_for_visualization_separate

            rock_pcd, bottom_pcd, combined_points, combined_colors, combined_normals = compute_fn(
                prep.rock_points,
                prep.bottom_points,
                k=k,
            )

            self.prepared_mesh_data.update({
                'rock_pcd': rock_pcd,
                'bottom_pcd': bottom_pcd,
                'combined_points': combined_points,
                'combined_colors': combined_colors,
                'combined_normals': combined_normals,
            })

            self.current_normal_method = method
            self.start_visualization_process(
                target=self.visualizer.show_point_cloud,
                args=(combined_points, combined_colors, f"Normals ({method})", False, None, True, combined_normals),
            )
        except Exception as exc:
            logging.error("Normal computation failed", exc_info=True)
            QMessageBox.critical(self, "Normals", str(exc))
        finally:
            self._set_busy(False)

    def _complete_reconstruction_only(self):
        if not self.prepared_mesh_data:
            QMessageBox.warning(self, "Preparation Required", "Prepare bottom face first.")
            return

        try:
            self._set_busy(True)
            self._stop_active_process()
            self.set_instruction("Completing mesh reconstruction...")
            rock_pcd = self.prepared_mesh_data['rock_pcd']
            bottom_pcd = self.prepared_mesh_data['bottom_pcd']
            if not rock_pcd.has_normals() or not bottom_pcd.has_normals():
                QMessageBox.warning(self, "Normals Required", "Compute normals before running reconstruction.")
                return

            payload = {
                'rock_points': np.asarray(rock_pcd.points, dtype=np.float64).copy(),
                'rock_normals': np.asarray(rock_pcd.normals, dtype=np.float64).copy(),
                'bottom_points': np.asarray(bottom_pcd.points, dtype=np.float64).copy(),
                'bottom_normals': np.asarray(bottom_pcd.normals, dtype=np.float64).copy(),
                'depth': 8,
            }

            ctx = multiprocessing.get_context("spawn")
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            process = ctx.Process(
                target=MeshProcessor.poisson_worker_entrypoint,
                args=(child_conn, payload),
                name="poisson_reconstruction_worker",
            )
            process.start()
            child_conn.close()

            worker_result = None
            try:
                while process.is_alive():
                    process.join(timeout=0.1)
                    QApplication.processEvents()

                if parent_conn.poll():
                    try:
                        worker_result = parent_conn.recv()
                    except EOFError:
                        worker_result = {'success': False, 'message': 'Worker terminated before sending results.'}
                else:
                    worker_result = {'success': False, 'message': 'Worker exited without returning a result.'}
            finally:
                parent_conn.close()
                if process.is_alive():
                    process.terminate()
                    process.join()

            if not worker_result or not worker_result.get('success'):
                message = worker_result.get('message', 'Mesh reconstruction failed.') if worker_result else 'Mesh reconstruction failed.'
                traceback_txt = worker_result.get('traceback') if worker_result else None
                if traceback_txt:
                    logging.error("Poisson worker traceback:\n%s", traceback_txt)
                self.mesh_processor.last_error_message = message
                QMessageBox.warning(self, "Mesh Reconstruction", message)
                return

            mesh_path = worker_result.get('mesh_path')
            if not mesh_path or not os.path.exists(mesh_path):
                message = "Worker completed but produced no mesh output."
                self.mesh_processor.last_error_message = message
                QMessageBox.warning(self, "Mesh Reconstruction", message)
                return

            mesh = o3d.io.read_triangle_mesh(mesh_path)
            if mesh is None or len(mesh.triangles) == 0:
                message = "Worker returned an empty mesh."
                self.mesh_processor.last_error_message = message
                QMessageBox.warning(self, "Mesh Reconstruction", message)
                return

            self.reconstructed_mesh = mesh
            self.mesh_processor.reconstructed_mesh = mesh
            self.mesh_processor.temp_mesh_path = mesh_path
            self.mesh_reconstruction_stage = 2
            self.state.mesh_completed = True
            self._update_status_indicators()
            self.compute_geometric_button.setEnabled(True)
            self.save_mesh_button.setEnabled(True)
            self.mesh_remove_noise_button.setEnabled(False)
            self.mesh_undo_noise_button.setEnabled(False)
            self.remove_noise_button.setEnabled(False)
            self.undo_noise_button.setEnabled(False)
            self.set_instruction("Mesh reconstruction complete.")

            if getattr(self.mesh_processor, 'temp_mesh_path', None):
                self.start_visualization_process(
                    target=self.visualizer.show_point_cloud,
                    args=(
                        self.mesh_processor.temp_mesh_path,
                        None,
                        "Reconstructed Mesh",
                        True,
                        None,
                        False,
                        None,
                    ),
                )
        except Exception as exc:
            logging.error("Mesh reconstruction failed", exc_info=True)
            QMessageBox.critical(self, "Mesh Reconstruction", str(exc))
        finally:
            self._set_busy(False)

    def save_mesh(self):
        if not hasattr(self.mesh_processor, 'reconstructed_mesh') or self.mesh_processor.reconstructed_mesh is None:
            QMessageBox.warning(self, "No Mesh", "Complete reconstruction before saving.")
            return

        default_name = f"{self.current_pbr_file}_mesh.ply" if self.current_pbr_file else "mesh.ply"
        default_dir = self._resolve_output_dir('paths.mesh_dir')
        default_path = str(default_dir / default_name)
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Mesh", default_path, "PLY Files (*.ply)")
        if not file_name:
            return

        self.mesh_path = self.mesh_processor.save_mesh(file_name)
        self._update_export_summary()
        self.set_instruction(f"Mesh saved to {file_name}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def remove_noise_iterative(self):
        if self.mesh_reconstruction_stage != 1 or not self.prepared_mesh_data:
            QMessageBox.warning(self, "Not Ready", "Prepare mesh before removing noise.")
            return

        try:
            self.set_instruction("Removing noise...")
            self.noise_removal_history.append(self.prepared_mesh_data.copy())

            rock_pcd = self.prepared_mesh_data['rock_pcd']
            bottom_pcd = self.prepared_mesh_data['bottom_pcd']

            filtered_rock_pcd, bottom_pcd, combined_points, combined_colors, combined_normals = self.mesh_processor.apply_noise_removal(
                rock_pcd,
                bottom_pcd,
            )

            self.prepared_mesh_data.update({
                'rock_pcd': filtered_rock_pcd,
                'bottom_pcd': bottom_pcd,
                'combined_points': combined_points,
                'combined_colors': combined_colors,
                'combined_normals': combined_normals,
            })

            self.start_visualization_process(
                target=self.visualizer.show_point_cloud,
                args=(combined_points, combined_colors, "Noise Removed", False, None, True, combined_normals),
            )

            self.undo_noise_button.setEnabled(True)
            self.mesh_undo_noise_button.setEnabled(True)
            self.set_instruction("Noise removed. You can undo or proceed.")
        except Exception as exc:
            logging.error("Noise removal failed", exc_info=True)
            QMessageBox.critical(self, "Noise Removal", str(exc))

    def undo_remove_noise(self):
        if not self.noise_removal_history:
            QMessageBox.information(self, "Undo", "No noise removal steps to undo.")
            return

        self.prepared_mesh_data = self.noise_removal_history.pop()
        self.start_visualization_process(
            target=self.visualizer.show_point_cloud,
            args=(
                self.prepared_mesh_data['combined_points'],
                self.prepared_mesh_data['combined_colors'],
                "Undo Noise Removal",
                False,
                None,
                True,
                self.prepared_mesh_data['combined_normals'],
            ),
        )
        if not self.noise_removal_history:
            self.undo_noise_button.setEnabled(False)
            self.mesh_undo_noise_button.setEnabled(False)
        self.set_instruction("Noise removal undone.")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def _show_geometric_analysis_popup(self, results, csv_path: str):
        dialog = QDialog(self)
        dialog.setWindowTitle("Geometric Analysis Summary")
        outer_layout = QVBoxLayout(dialog)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        header = QLabel("Key Metrics")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        metrics_form = QFormLayout()

        def make_value_label(value_text: str) -> QLabel:
            value_label = QLabel(value_text)
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            return value_label

        def add_row(label_text: str, value_text: str):
            metrics_form.addRow(QLabel(label_text), make_value_label(value_text))

        def fmt_float(val, precision: int = 3):
            return f"{val:.{precision}f}" if val is not None else "--"

        def fmt_vector(vec):
            if vec is None:
                return "--"
            arr = np.asarray(vec).reshape(-1)
            return ", ".join(f"{v:.3f}" for v in arr)

        add_row("User", results.get('user') or (self.current_user or "--"))
        add_row("PBR", self.current_pbr_file or "--")
        add_row("Height", fmt_float(results.get('height')))
        add_row("Width", fmt_float(results.get('width')))
        add_row("Length", fmt_float(results.get('length')))
        add_row("Height/Width", fmt_float(results.get('height_width_ratio')))
        add_row("Length/Width", fmt_float(results.get('length_width_ratio')))
        add_row("Alpha Angle (°)", fmt_float(results.get('alpha_angle')))
        add_row("Alpha Rectangular (°)", fmt_float(results.get('alpha_rectangular')))
        add_row("Beta Angle (°)", fmt_float(results.get('beta_angle')))
        add_row("Center of Mass", fmt_vector(results.get('center_of_mass')))
        add_row("Alpha Plane Normal", fmt_vector(results.get('alpha_plane_normal')))

        thresholds = QFormLayout()
        thresholds_header = QLabel("Thresholds")
        thresholds_header.setStyleSheet("font-weight: bold;")

        layout.addLayout(metrics_form)
        layout.addWidget(thresholds_header)

        thresholds.addRow(QLabel("Smoothness"), make_value_label(fmt_float(self.smoothness_threshold)))
        thresholds.addRow(QLabel("Curvature"), make_value_label(fmt_float(self.curvature_threshold)))
        thresholds.addRow(QLabel("Interface Proximity"), make_value_label(fmt_float(self.basal_proximity_threshold)))
        layout.addLayout(thresholds)

        paths_header = QLabel("Saved Outputs")
        paths_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(paths_header)

        paths_form = QFormLayout()

        def add_path(label_text: str, path_value):
            value = str(path_value) if path_value else "--"
            paths_form.addRow(QLabel(label_text), make_value_label(value))

        add_path("Input Point Cloud", self.input_path)
        add_path("Segmented Point Cloud", self.segmented_pcd_file_path)
        add_path("Mesh", self.mesh_path)
        add_path("Analysis CSV", csv_path)
        layout.addLayout(paths_form)
        layout.addStretch(1)

        scroll.setWidget(content_widget)
        outer_layout.addWidget(scroll)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        outer_layout.addWidget(button_box)

        dialog.resize(self.style.dialog_manual_width, self.style.dialog_manual_height + 160)
        dialog.exec_()

    def _launch_alpha_view(self, results, pedestal_points: Optional[np.ndarray]):
        mesh_file_path: Optional[str] = None

        candidate = getattr(self.mesh_processor, 'temp_mesh_path', None)
        if candidate and Path(candidate).exists():
            mesh_file_path = candidate
        elif self.mesh_path and self.mesh_path not in {"--", "Reconstructed Mesh was not saved"} and Path(self.mesh_path).exists():
            mesh_file_path = self.mesh_path
        elif getattr(self.mesh_processor, 'reconstructed_mesh', None) is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_mesh:
                    o3d.io.write_triangle_mesh(temp_mesh.name, self.mesh_processor.reconstructed_mesh)
                    mesh_file_path = temp_mesh.name
                self.mesh_processor.temp_mesh_path = mesh_file_path
            except Exception as exc:  # pragma: no cover
                logging.warning("Failed to create temporary mesh for alpha view: %s", exc)
                mesh_file_path = None

        if not mesh_file_path:
            logging.warning("Alpha view skipped: no mesh file available.")
            return

        alpha_point = results.get('min_alpha_basal_point')
        pedestal = None
        if pedestal_points is not None and len(pedestal_points) > 0:
            pedestal = np.asarray(pedestal_points)

        title = f"Alpha View - {self.current_pbr_file or ''} (α={results.get('alpha_angle', 0.0):.1f}°)"

        try:
            self.start_visualization_process(
                target=self.visualizer.show_mesh_with_alpha_view,
                args=(
                    mesh_file_path,
                    np.asarray(results['center_of_mass']),
                    np.asarray(results['alpha_plane_normal']),
                    title,
                    pedestal,
                    np.asarray(alpha_point) if alpha_point is not None else None,
                    True,
                    bool(self.config.get("visualization.alpha_view_rods", True)),
                ),
            )
        except Exception as exc:  # pragma: no cover
            logging.warning("Failed to start alpha view visualization: %s", exc)

    def perform_geometric_analysis(self):
        if not hasattr(self.mesh_processor, 'reconstructed_mesh') or self.mesh_processor.reconstructed_mesh is None:
            QMessageBox.warning(self, "No Mesh", "Complete reconstruction before analysis.")
            return

        try:
            self._set_busy(True)
            self._stop_active_process()
            self.set_instruction("Computing geometric properties...")

            pedestal_points = None
            if hasattr(self, 'segmenter') and self.segmenter is not None and hasattr(self.segmenter, 'labels'):
                labels = np.asarray(self.segmenter.labels)
                if labels.size and np.any(labels == 0):
                    pedestal_points = np.asarray(self.pcd.points)[labels == 0]

            basal_coords = np.asarray(self.pcd.points)[self.basal_points] if self.basal_points is not None else None
            if basal_coords is None or len(basal_coords) == 0:
                QMessageBox.warning(self, "Interface Constraint Required", "Define interface constraint points before running analysis.")
                return

            results = self.geometric_analyzer.compute_geometric_properties(
                self.mesh_processor.reconstructed_mesh,
                basal_coords,
                pedestal_points,
                lateral_flags=None,
            )
            results['user'] = self.current_user or self.user_combo.currentText()

            csv_path = self.geometric_analyzer.save_results(
                results,
                self.current_pbr_file,
                self.input_path,
                self.segmented_pcd_file_path,
                self.mesh_path,
                self.smoothness_threshold,
                self.curvature_threshold,
                self.basal_proximity_threshold,
                self.current_user,
                self.epsg_code,
                output_csv=self.analysis_csv_path if self.analysis_csv_path and self.analysis_csv_path != "--" else None,
            )

            self.analysis_csv_path = csv_path
            self._update_export_summary()
            self.state.analysis_completed = True
            self._update_status_indicators()
            self.set_instruction(f"Geometric analysis complete. Results saved to {csv_path}.")

            self._launch_alpha_view(results, pedestal_points)
            QApplication.processEvents()
            self._show_geometric_analysis_popup(results, csv_path)
        except Exception as exc:
            logging.error("Geometric analysis failed", exc_info=True)
            QMessageBox.critical(self, "Geometric Analysis", str(exc))
        finally:
            self._set_busy(False)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def restart_application(self):
        if QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        self._busy_depth = 0
        if self.process:
            self.process.terminate()
            self.process.join(timeout=1.0)
            self.process = None

        self.__init__()

    @staticmethod
    def detect_basal_points_optimized(points, labels, k=30, threshold=0.35):
        tree = cKDTree(points)
        _, indices = tree.query(points, k=k)
        neighborhood_labels = labels[indices]
        rock_ratios = np.sum(neighborhood_labels == 1, axis=1) / k
        return (threshold <= rock_ratios) & (rock_ratios <= (1 - threshold))

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        self._update_export_summary()


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="open3d")
    os.environ['OPEN3D_VERBOSE'] = '0'
    if sys.platform == 'darwin':
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    multiprocessing.set_start_method("spawn", force=True)
    app = QApplication(sys.argv)
    window = RefactoredMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
