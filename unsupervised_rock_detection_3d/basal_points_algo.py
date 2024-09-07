import sys
import os
import multiprocessing
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
import open3d as o3d
import numpy as np
import laspy

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BasalPointAlgorithm:
    def __init__(self, pcd):
        self.pcd = pcd

    def run(self, basal_points):
        logging.debug("Starting vector-directed algorithm")
        points = np.asarray(self.pcd.points)
        dense_points = []
        visited_points = set()
        num_points = len(basal_points)

        for i in range(num_points):
            p1_idx = basal_points[i]
            p2_idx = basal_points[(i + 1) % num_points]
            p1 = points[p1_idx]
            p2 = points[p2_idx]

            logging.debug(f"Processing basal point pair: {p1_idx} -> {p2_idx}")

            current_point = p1
            dense_points.append(current_point)
            visited_points.add(tuple(current_point))

            while not np.allclose(current_point, p2, atol=1e-3):
                direction = p2 - current_point
                distance_to_p2 = np.linalg.norm(direction)
                direction /= distance_to_p2

                vectors = points - current_point
                projections = np.dot(vectors, direction)
                mask = (projections > 0) & (projections < distance_to_p2)
                candidates = points[mask]

                if len(candidates) == 0:
                    logging.debug(
                        "No candidates found in the direction. Breaking loop."
                    )
                    break

                distances = np.linalg.norm(candidates - current_point, axis=1)
                sorted_indices = np.argsort(distances)

                for idx in sorted_indices:
                    next_point = candidates[idx]
                    if tuple(next_point) not in visited_points:
                        break
                else:
                    logging.debug("No unvisited points found. Breaking loop.")
                    break

                if np.linalg.norm(next_point - current_point) < 1e-6:
                    logging.debug(
                        "Next point too close to current point. Breaking loop."
                    )
                    break

                if np.allclose(next_point, p2, atol=1e-3):
                    dense_points.append(p2)
                    logging.debug(f"Reached target point: {p2}. Moving to next pair.")
                    break

                dense_points.append(next_point)
                visited_points.add(tuple(next_point))
                current_point = next_point

                if len(dense_points) >= 2000:
                    logging.debug(f"Reached 2000 dense points. Stopping algorithm.")
                    return np.array(dense_points)

        logging.debug(
            f"Vector-directed algorithm completed with {len(dense_points)} dense points"
        )
        # Save Basal information for testing purposes
        np.save("basal_points_28.npy", np.array(dense_points))
        return np.array(dense_points)


class BasalPointSelection(QMainWindow):
    def __init__(self, las_file_path):
        super().__init__()
        self.setWindowTitle("Basal Points Selection with PyQt and Open3D")
        self.pcd = None
        self.process = None
        self.basal_points = []

        self.load_las_file(las_file_path)
        self.init_ui()

    def init_ui(self):
        logging.debug("Initializing UI components")
        layout = QVBoxLayout()

        self.continue_button = QPushButton("Select Basal Points")
        self.continue_button.clicked.connect(self.start_manual_selection)
        layout.addWidget(self.continue_button)

        self.run_button = QPushButton("Run Basal Point Pairing Algorithm")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_basal_point_pairing_algorithm)
        layout.addWidget(self.run_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_las_file(self, las_file_path):
        logging.debug(f"Loading LAS file from {las_file_path}")
        pc = laspy.read(las_file_path)
        x, y, z = pc.x, pc.y, pc.z

        self.x_mean = np.mean(x)
        self.y_mean = np.mean(y)
        self.z_mean = np.mean(z)

        xyz = np.vstack((x - self.x_mean, y - self.y_mean, z - self.z_mean)).transpose()
        if all(
            dim in pc.point_format.dimension_names for dim in ["red", "green", "blue"]
        ):
            r = np.uint8(pc.red / 65535.0 * 255)
            g = np.uint8(pc.green / 65535.0 * 255)
            b = np.uint8(pc.blue / 65535.0 * 255)
            rgb = np.vstack((r, g, b)).transpose() / 255.0
        else:
            rgb = np.zeros((len(x), 3))

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(xyz)
        self.pcd.colors = o3d.utility.Vector3dVector(rgb)

        logging.debug("LAS file loaded successfully")

    def start_manual_selection(self):
        logging.debug("Starting manual selection of basal points")
        self.continue_button.setEnabled(False)
        self.run_button.setEnabled(True)
        self.seed_selection_vis = o3d.visualization.VisualizerWithEditing()
        self.seed_selection_vis.create_window()
        self.pcd.paint_uniform_color([0.5, 0.5, 0.5])
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
        )
        self.seed_selection_vis.add_geometry(self.pcd)
        self.seed_selection_vis.run()
        self.basal_points = self.seed_selection_vis.get_picked_points()
        self.seed_selection_vis.destroy_window()
        logging.debug(
            f"Manual selection of basal points completed: {self.basal_points}"
        )

    def run_basal_point_pairing_algorithm(self):
        logging.debug("Running basal point pairing algorithm")
        algorithm = BasalPointAlgorithm(self.pcd)
        predicted_points = algorithm.run(self.basal_points)
        self.highlight_points(predicted_points)
        logging.debug("Basal point pairing algorithm completed")

    def highlight_points(self, points):
        logging.debug("Highlighting points")
        colors = np.full((len(self.pcd.points), 3), [0.5, 0.5, 0.5])
        for point in points:
            idx = np.argmin(np.linalg.norm(np.asarray(self.pcd.points) - point, axis=1))
            colors[idx] = [1, 0, 0]

        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors)

        if self.process:
            self.process.terminate()

        self.process = multiprocessing.Process(
            target=self.show_point_cloud, args=(points, colors)
        )
        self.process.start()
        logging.debug("Points highlighted and visualization updated")

    @staticmethod
    def show_point_cloud(points, colors=None):
        logging.debug("Visualizing point cloud")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
        )
        o3d.visualization.draw_geometries([pcd])


# Main entry point for the application
def main():
    logging.debug("Starting application")
    multiprocessing.set_start_method("spawn")
    app = QApplication(sys.argv)
    window = BasalPointSelection("box_pbr_annotation/pbr28.las")
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
