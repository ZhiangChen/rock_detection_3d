import sys
import os
import multiprocessing
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QLabel,
)
import open3d as o3d
import numpy as np
import laspy

pcd_file_path = "pbr28.las"


# Visualize point cloud
def show_point_cloud(points, colors=None):
    """
    Visualize the point cloud using Open3D.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
    )
    o3d.visualization.draw_geometries([pcd])


# Point picking function
def pick_points(self, pcd):
    """
    Pick points from the point cloud for seed selection.
    """
    self.seed_selection_vis = o3d.visualization.VisualizerWithEditing()
    self.seed_selection_vis.create_window()
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
    )
    self.seed_selection_vis.add_geometry(pcd)
    self.seed_selection_vis.run()  # user picks points
    picked_points = self.seed_selection_vis.get_picked_points()
    return picked_points


# Load LAS file and convert to Open3D point cloud
def load_las_as_open3d_point_cloud(self, las_file_path, evaluate=False):
    """
    Load a LAS file and convert it to an Open3D point cloud.
    """
    pc = laspy.read(las_file_path)
    x, y, z = pc.x, pc.y, pc.z
    ground_truth_labels = None
    if evaluate and "Original cloud index" in pc.point_format.dimension_names:
        ground_truth_labels = np.int_(pc["Original cloud index"])

    # Store the mean values for recentering later
    self.x_mean = np.mean(x)
    self.y_mean = np.mean(y)
    self.z_mean = np.mean(z)

    # Recenter the point cloud
    xyz = np.vstack((x - self.x_mean, y - self.y_mean, z - self.z_mean)).transpose()
    if all(dim in pc.point_format.dimension_names for dim in ["red", "green", "blue"]):
        r = np.uint8(pc.red / 65535.0 * 255)
        g = np.uint8(pc.green / 65535.0 * 255)
        b = np.uint8(pc.blue / 65535.0 * 255)
        rgb = np.vstack((r, g, b)).transpose() / 255.0
    else:
        rgb = np.zeros((len(x), 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    self.pcd_colors = pcd.colors

    return pcd, ground_truth_labels


# Basal Point Pairing Algorithm
def basal_point_algorithm(basal_points, point_cloud):
    """
    Pair up the input basal points and find adjacent basal points iteratively.
    """
    points = np.asarray(point_cloud.points)
    dense_points = []

    for i in range(len(basal_points) - 1):
        p1 = points[basal_points[i]]
        p2 = points[basal_points[i + 1]]

        current_point = p1
        dense_points.append(current_point)

        while np.linalg.norm(p2 - current_point) > 1e-3:
            direction = p2 - current_point
            distance = np.linalg.norm(direction)
            direction /= distance

            vectors = points - current_point
            projections = np.dot(vectors, direction)
            mask = projections > 0  # Filter points in the direction of the vector
            candidates = points[mask]
            candidate_vectors = vectors[mask]
            dists = np.linalg.norm(candidate_vectors, axis=1)
            min_idx = np.argmin(dists)
            next_point = candidates[min_idx]

            if np.linalg.norm(next_point - current_point) < 1e-6:
                # Break if the next point is essentially the same as the current point
                break

            dense_points.append(next_point)
            current_point = next_point

    return np.array(dense_points)


# Main window class for the GUI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basal Points Selection with PyQt and Open3D")
        self.pcd = None
        self.process = None
        self.load_las_file()
        self.basal_points = []

        self.init_ui()

    def init_ui(self):
        """
        Initialize the UI components.
        """
        layout = QVBoxLayout()

        # Button to continue to seed selection
        self.continue_button = QPushButton("Select Basal Points")
        self.continue_button.clicked.connect(self.start_manual_selection)
        layout.addWidget(self.continue_button)

        # Button to run the basal point pairing algorithm
        self.run_button = QPushButton("Run Basal Point Pairing Algorithm")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_basal_point_pairing_algorithm)
        layout.addWidget(self.run_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_las_file(self):
        """
        Load a LAS file and display the point cloud.
        """
        self.pcd, _ = load_las_as_open3d_point_cloud(self, pcd_file_path)
        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors)

        if self.process:
            self.process.terminate()

        self.process = multiprocessing.Process(
            target=show_point_cloud, args=(points, colors)
        )
        self.process.start()

    def start_manual_selection(self):
        """
        Start manual selection of basal points.
        """
        self.continue_button.setEnabled(False)
        self.run_button.setEnabled(True)
        if self.process:
            self.process.terminate()
        pick_points(self, self.pcd)

    def run_basal_point_pairing_algorithm(self):
        """
        Run the Basal Point Pairing algorithm with the selected basal points.
        """
        self.basal_points = self.seed_selection_vis.get_picked_points()
        self.seed_selection_vis.destroy_window()
        predicted_points = basal_point_algorithm(self.basal_points, self.pcd)
        self.highlight_points(predicted_points)

    def highlight_points(self, points):
        """
        Highlight the selected points in red.
        """
        colors = np.full((len(self.pcd.points), 3), [0.5, 0.5, 0.5])
        for point in points:
            idx = np.argmin(np.linalg.norm(np.asarray(self.pcd.points) - point, axis=1))
            colors[idx] = [1, 0, 0]  # Red

        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors)

        if self.process:
            self.process.terminate()

        self.process = multiprocessing.Process(
            target=show_point_cloud, args=(points, colors)
        )
        self.process.start()


# Main entry point for the application
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
