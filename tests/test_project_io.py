import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import open3d as o3d

from rock_detection_3d.web_workflow import (
    PROJECT_FORMAT,
    PROJECT_SCHEMA_VERSION,
    WebWorkflowSession,
)


class ProjectIOTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_root.name)

    def tearDown(self):
        self.temp_root.cleanup()

    def make_session(self, name="sample") -> WebWorkflowSession:
        session = WebWorkflowSession(session_id=f"{name}_session", run_dir=self.root / f"{name}_run")
        points = np.asarray([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ])
        colors = np.full((len(points), 3), 0.5)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        session.pcd = pcd
        session.current_pbr_file = name
        session.status.point_cloud_loaded = True
        session.input_path = session.upload_dir / f"{name}.las"
        session.file_handler.save_point_cloud(pcd, session.input_path, plain=True)
        session._snapshot_raw_view()
        return session

    def add_seeds_and_interface(self, session: WebWorkflowSession) -> None:
        session.manual_seeds([1], [0])
        metadata = {
            "parts": [{
                "id": 1,
                "is_lateral": False,
                "selected_indices": [2, 3],
                "original_points": [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
                "dense_points": [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
                "point_indices": [2, 3],
                "num_points": 2,
                "color": [0.0, 1.0, 0.0],
            }],
            "close_loop": False,
            "num_parts": 1,
            "has_lateral_parts": False,
            "palette": [[0.0, 1.0, 0.0]],
        }
        session.manual_basal_points = np.asarray([2, 3], dtype=int)
        session.basal_points = np.asarray([2, 3], dtype=int)
        session.manual_basal_parts_metadata = metadata
        session.basal_parts_metadata = metadata
        session.manual_dense_basal_parts = [np.asarray(metadata["parts"][0]["dense_points"])]
        session.dense_basal_parts = [np.asarray(metadata["parts"][0]["dense_points"])]
        session.manual_dense_basal_parts_is_lateral = [False]
        session.dense_basal_parts_is_lateral = [False]
        session.interface_source = "manual"
        session.status.manual_interface_ready = True
        session.status.interface_ready = True

    def add_outputs(self, session: WebWorkflowSession) -> None:
        labels = np.asarray([0, 1, 1, 1, 0, 1], dtype=int)
        session.segmented_pcd = session.pcd
        session.segmented_labels = labels
        session.status.segmentation_ready = True
        session.status.last_segmentation_mode = "rg"
        session.segmented_pcd_file_path = session.file_handler.save_point_cloud(
            session.pcd,
            session.output_dir / f"{session.current_pbr_file}_segmented.las",
            labels=labels,
            basal_data=session.basal_parts_metadata,
        )

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray([[0, 1, 2]], dtype=int))
        session.mesh_processor.reconstructed_mesh = mesh
        session.mesh_path = session.mesh_processor.save_mesh(session.output_dir / f"{session.current_pbr_file}_mesh.ply")
        session.status.mesh_completed = True

        analysis_path = session.output_dir / f"{session.current_pbr_file}_analysis.csv"
        analysis_path.write_text("metric,value\nheight,1.0\n", encoding="utf-8")
        session.analysis_csv_path = str(analysis_path)
        session.status.analysis_completed = True

    def add_prepared_mesh_view(self, session: WebWorkflowSession) -> None:
        rock_pcd = o3d.geometry.PointCloud()
        rock_pcd.points = o3d.utility.Vector3dVector(np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=float))
        rock_pcd.colors = o3d.utility.Vector3dVector(np.full((2, 3), [1.0, 0.0, 0.0], dtype=float))
        bottom_pcd = o3d.geometry.PointCloud()
        bottom_pcd.points = o3d.utility.Vector3dVector(np.asarray([[0.0, 0.0, 0.0]], dtype=float))
        bottom_pcd.colors = o3d.utility.Vector3dVector(np.full((1, 3), [0.0, 1.0, 0.0], dtype=float))
        combined_points = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=float)
        session.prepared_mesh_data = {
            "rock_pcd": rock_pcd,
            "bottom_pcd": bottom_pcd,
            "combined_points": combined_points,
            "combined_colors": np.full((3, 3), 0.5, dtype=float),
            "combined_normals": np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=float), (3, 1)),
            "preparation_result": None,
        }
        session.status.mesh_prepared = True

    def interface_marker_indices(self, payload):
        return [
            int(marker["index"])
            for marker in payload["markers"]
            if str(marker["label"]).startswith("Interface")
        ]

    def test_project_round_trip_restores_state_and_ui(self):
        source = self.make_session("roundtrip")
        self.add_seeds_and_interface(source)
        ui_state = {
            "project_filename": "roundtrip.rd3dproj",
            "active_view": "interface",
            "segment_params": {"smoothness_threshold": 0.8},
            "point_size": 0.04,
        }

        archive_path = source.export_project(ui_state=ui_state, filename="roundtrip.rd3dproj", app_build="test-build")
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            self.assertIn("project.json", names)
            self.assertIn("state/working_point_cloud.npz", names)
            self.assertIn("assets/intermediate/seeds_interface.las", names)
            manifest = json.loads(archive.read("project.json").decode("utf-8"))
            self.assertEqual(PROJECT_FORMAT, manifest["format"])
            self.assertEqual(PROJECT_SCHEMA_VERSION, manifest["schema_version"])

        dest = WebWorkflowSession(session_id="dest", run_dir=self.root / "dest_run")
        imported = dest.import_project(archive_path)

        self.assertEqual(imported["ui_state"]["active_view"], "interface")
        self.assertEqual(dest.current_pbr_file, "roundtrip")
        self.assertEqual(dest.rock_seeds, [1])
        self.assertEqual(dest.pedestal_seeds, [0])
        self.assertTrue(dest.status.point_cloud_loaded)
        self.assertTrue(dest.status.manual_interface_ready)
        self.assertEqual(dest.basal_points.tolist(), [2, 3])
        self.assertEqual(self.interface_marker_indices(dest.viewer_payload("raw")), [2, 3])
        np.testing.assert_allclose(np.asarray(dest.pcd.points), np.asarray(source.pcd.points))

    def test_raw_only_project_imports_cleanly(self):
        source = self.make_session("rawonly")
        archive_path = source.export_project(ui_state={}, filename="rawonly.rd3dproj", app_build="test-build")

        dest = WebWorkflowSession(session_id="raw_dest", run_dir=self.root / "raw_dest_run")
        imported = dest.import_project(archive_path)

        self.assertEqual(imported["summary"]["current_file"], "rawonly")
        self.assertTrue(dest.status.point_cloud_loaded)
        self.assertFalse(dest.status.segmentation_ready)
        self.assertIsNone(dest.segmented_pcd_file_path)

    def test_project_output_artifacts_restore_download_paths(self):
        source = self.make_session("outputs")
        self.add_seeds_and_interface(source)
        self.add_outputs(source)
        archive_path = source.export_project(ui_state={}, filename="outputs.rd3dproj", app_build="test-build")

        dest = WebWorkflowSession(session_id="outputs_dest", run_dir=self.root / "outputs_dest_run")
        dest.import_project(archive_path)

        self.assertTrue(Path(dest.segmented_pcd_file_path).exists())
        self.assertTrue(Path(dest.mesh_path).exists())
        self.assertTrue(Path(dest.analysis_csv_path).exists())

    def test_manual_seed_edit_preserves_segmented_view_colors(self):
        session = self.make_session("seededit")
        segmented_colors = np.asarray([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ], dtype=float)
        session.pcd.colors = o3d.utility.Vector3dVector(segmented_colors)
        session.segmented_pcd = session.pcd
        session.segmented_labels = np.asarray([0, 1, 1, 1, 0, 1], dtype=int)
        session.status.segmentation_ready = True
        session.status.last_segmentation_mode = "rg"

        session.manual_seeds([1, 3], [0, 4])

        np.testing.assert_allclose(np.asarray(session.segmented_pcd.colors), segmented_colors)
        payload = session.viewer_payload("segmented")
        np.testing.assert_allclose(np.asarray(payload["colors"], dtype=float), segmented_colors)
        self.assertEqual(session.rock_seeds, [1, 3])
        self.assertEqual(session.pedestal_seeds, [0, 4])

    def test_segmented_view_supports_multi_seed_color_mode(self):
        session = self.make_session("segmentedbranches")
        two_color = np.asarray([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ], dtype=float)
        session.pcd.colors = o3d.utility.Vector3dVector(two_color)
        session.segmented_pcd = session.pcd
        session.segmented_labels = np.asarray([0, 1, 1, 1, 0, 1], dtype=int)
        session.segmented_branch_ids = np.asarray([1, 0, 0, 2, 1, 2], dtype=int)
        session.segmented_branches = [
            {"branch_id": 0, "seed_index": 1, "class_label": "rock", "label": "Rock seed 1", "color": [0.93, 0.18, 0.14]},
            {"branch_id": 1, "seed_index": 0, "class_label": "pedestal", "label": "Pedestal seed 1", "color": [0.10, 0.36, 0.95]},
            {"branch_id": 2, "seed_index": 3, "class_label": "rock", "label": "Rock seed 2", "color": [0.96, 0.62, 0.05]},
        ]
        session.status.segmentation_ready = True
        session.status.last_segmentation_mode = "rg"

        default_payload = session.viewer_payload("segmented", color_mode="two_color")
        np.testing.assert_allclose(np.asarray(default_payload["colors"], dtype=float), two_color)
        self.assertEqual(
            [(branch["label"], branch["node_count"]) for branch in default_payload["seed_branches"]],
            [("Rock seed 1", 2), ("Pedestal seed 1", 2), ("Rock seed 2", 2)],
        )

        multi_payload = session.viewer_payload("segmented", color_mode="multi_seed")
        np.testing.assert_allclose(np.asarray(multi_payload["colors"][0], dtype=float), [0.10, 0.36, 0.95])
        np.testing.assert_allclose(np.asarray(multi_payload["colors"][1], dtype=float), [0.93, 0.18, 0.14])
        np.testing.assert_allclose(np.asarray(multi_payload["colors"][3], dtype=float), [0.96, 0.62, 0.05])
        self.assertEqual(
            [(branch["label"], branch["node_count"]) for branch in multi_payload["seed_branches"]],
            [("Rock seed 1", 2), ("Pedestal seed 1", 2), ("Rock seed 2", 2)],
        )

        archive_path = session.export_project(ui_state={}, filename="segmentedbranches.rd3dproj", app_build="test-build")
        dest = WebWorkflowSession(session_id="segmentedbranches_dest", run_dir=self.root / "segmentedbranches_dest_run")
        dest.import_project(archive_path)

        imported_payload = dest.viewer_payload("segmented", color_mode="multi_seed")
        self.assertEqual(imported_payload["seed_branches"], multi_payload["seed_branches"])
        np.testing.assert_allclose(np.asarray(imported_payload["colors"], dtype=float), np.asarray(multi_payload["colors"], dtype=float))

    def test_voxel_segmented_view_round_trips_with_project(self):
        source = self.make_session("voxelview")
        source.rock_seeds = [3]
        source.pedestal_seeds = [0]
        source.voxel_segmented_points = np.asarray([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=float)
        source.voxel_segmented_labels = np.asarray([-1, 0, 1], dtype=int)
        source.voxel_segmented_branch_ids = np.asarray([-1, 1, 0], dtype=int)
        source.voxel_segmented_branches = [
            {
                "branch_id": 0,
                "seed_index": 2,
                "region_index": 1,
                "class_label": "rock",
                "label": "Rock seed 1",
                "color": [0.93, 0.18, 0.14],
                "node_count": 1,
            },
            {
                "branch_id": 1,
                "seed_index": 1,
                "region_index": 0,
                "class_label": "pedestal",
                "label": "Pedestal seed 1",
                "color": [0.1, 0.36, 0.95],
                "node_count": 1,
            },
        ]
        source.voxel_segmented_colors = source._colors_from_branch_ids(
            source.voxel_segmented_labels,
            source.voxel_segmented_branch_ids,
            source.voxel_segmented_branches,
            len(source.voxel_segmented_points),
        )
        source.voxel_segmented_normals = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=float), (3, 1))
        source.status.segmentation_ready = True
        source.status.voxel_segmentation_ready = True

        payload = source.viewer_payload("voxel_segmented")
        self.assertEqual(payload["total_points"], 3)
        self.assertEqual(payload["label_counts"], {"unlabeled": 1, "pedestal": 1, "rock": 1})
        self.assertEqual(payload["colors"][0], [0.5, 0.5, 0.5])
        self.assertEqual(payload["colors"][1], [0.1, 0.36, 0.95])
        self.assertEqual(payload["colors"][2], [0.93, 0.18, 0.14])
        self.assertEqual(
            [(branch["label"], branch["node_count"]) for branch in payload["seed_branches"]],
            [("Rock seed 1", 1), ("Pedestal seed 1", 1)],
        )
        self.assertEqual(
            [(marker["label"], marker["point"]) for marker in payload["markers"]],
            [
                ("Rock seed 1 (1 nodes)", [1.0, 0.0, 0.0]),
                ("Pedestal seed 1 (1 nodes)", [0.5, 0.0, 0.0]),
            ],
        )

        archive_path = source.export_project(ui_state={}, filename="voxelview.rd3dproj", app_build="test-build")
        dest = WebWorkflowSession(session_id="voxel_dest", run_dir=self.root / "voxel_dest_run")
        dest.import_project(archive_path)

        self.assertTrue(dest.status.voxel_segmentation_ready)
        imported_payload = dest.viewer_payload("voxel_segmented")
        self.assertEqual(imported_payload["label_counts"], payload["label_counts"])
        self.assertEqual(imported_payload["seed_branches"], payload["seed_branches"])
        self.assertEqual(
            [(marker["label"], marker["point"]) for marker in imported_payload["markers"]],
            [
                ("Rock seed 1 (1 nodes)", [1.0, 0.0, 0.0]),
                ("Pedestal seed 1 (1 nodes)", [0.5, 0.0, 0.0]),
            ],
        )

    def test_icrg_segmented_view_overlays_manual_interface_path(self):
        session = self.make_session("icrgview")
        self.add_seeds_and_interface(session)
        segmented = o3d.geometry.PointCloud()
        segmented.points = o3d.utility.Vector3dVector(np.asarray(session.pcd.points))
        segmented.colors = o3d.utility.Vector3dVector(np.asarray([
            [0.0, 0.2, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.2, 1.0],
            [1.0, 0.0, 0.0],
        ], dtype=float))
        session.segmented_pcd = segmented
        session.segmented_labels = np.asarray([0, 1, 1, 1, 0, 1], dtype=int)
        session.status.segmentation_ready = True
        session.status.last_segmentation_mode = "icrg"

        payload = session.viewer_payload("segmented")

        self.assertEqual(payload["colors"][2], [0.0, 1.0, 0.0])
        self.assertEqual(payload["colors"][3], [0.0, 1.0, 0.0])
        interface_markers = [marker for marker in payload["markers"] if marker["label"].startswith("Interface")]
        self.assertEqual([marker["index"] for marker in interface_markers], [2, 3])

    def test_manual_interface_overlays_all_point_cloud_views(self):
        session = self.make_session("manualallviews")
        self.add_seeds_and_interface(session)
        session.display_interface_source = "manual"
        segmented = o3d.geometry.PointCloud()
        segmented.points = o3d.utility.Vector3dVector(np.asarray(session.pcd.points))
        segmented.colors = o3d.utility.Vector3dVector(np.full((6, 3), 0.5, dtype=float))
        session.segmented_pcd = segmented
        session.segmented_labels = np.asarray([0, 1, 1, 1, 0, 1], dtype=int)
        session.status.segmentation_ready = True
        session.status.last_segmentation_mode = "icrg"
        session.voxel_segmented_points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        session.voxel_segmented_colors = np.full((2, 3), 0.5, dtype=float)
        session.voxel_segmented_labels = np.asarray([0, 1], dtype=int)
        session.status.voxel_segmentation_ready = True
        self.add_prepared_mesh_view(session)

        for view_name in ["raw", "seeds", "interface", "voxel_segmented", "segmented", "mesh_prepared"]:
            with self.subTest(view=view_name):
                payload = session.viewer_payload(view_name)
                self.assertEqual(self.interface_marker_indices(payload), [2, 3])

        raw_payload = session.viewer_payload("raw")
        self.assertEqual(raw_payload["colors"][2], [0.0, 1.0, 0.0])
        self.assertEqual(raw_payload["colors"][3], [0.0, 1.0, 0.0])

    def test_regular_region_growing_sets_auto_interface_overlay(self):
        session = self.make_session("autorg")
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
        auto_indices = session._set_automatic_interface_from_segmentation(labels)

        self.assertGreater(len(auto_indices), 0)
        self.assertEqual(session.display_interface_source, "auto")
        payload = session.viewer_payload("raw")
        interface_markers = [marker for marker in payload["markers"] if marker["label"] == "Interface auto"]
        self.assertEqual(len(interface_markers), len(auto_indices))

    def test_missing_optional_artifact_reference_does_not_crash(self):
        source = self.make_session("missing")
        self.add_seeds_and_interface(source)
        self.add_outputs(source)
        archive_path = source.export_project(ui_state={}, filename="missing.rd3dproj", app_build="test-build")
        rewritten = self.root / "missing_without_segmented.rd3dproj"

        with zipfile.ZipFile(archive_path) as original, zipfile.ZipFile(rewritten, "w") as target:
            for name in original.namelist():
                if name.startswith("assets/segmented/"):
                    continue
                target.writestr(name, original.read(name))

        dest = WebWorkflowSession(session_id="missing_dest", run_dir=self.root / "missing_dest_run")
        dest.import_project(rewritten)

        self.assertIsNone(dest.segmented_pcd_file_path)
        self.assertTrue(dest.status.segmentation_ready)

    def test_rejects_invalid_project_archives(self):
        dest = WebWorkflowSession(session_id="bad_dest", run_dir=self.root / "bad_dest_run")

        no_manifest = self.root / "no_manifest.rd3dproj"
        with zipfile.ZipFile(no_manifest, "w") as archive:
            archive.writestr("readme.txt", "missing")
        with self.assertRaisesRegex(ValueError, "project.json"):
            dest.import_project(no_manifest)

        unsupported = self.root / "unsupported.rd3dproj"
        with zipfile.ZipFile(unsupported, "w") as archive:
            archive.writestr("project.json", json.dumps({"format": PROJECT_FORMAT, "schema_version": 999}))
        with self.assertRaisesRegex(ValueError, "Unsupported project schema"):
            dest.import_project(unsupported)

        unsafe = self.root / "unsafe.rd3dproj"
        with zipfile.ZipFile(unsafe, "w") as archive:
            archive.writestr("project.json", json.dumps({"format": PROJECT_FORMAT, "schema_version": PROJECT_SCHEMA_VERSION}))
            archive.writestr("../bad.txt", "bad")
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            dest.import_project(unsafe)


if __name__ == "__main__":
    unittest.main()
