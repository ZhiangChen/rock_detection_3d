import sys
import os
import multiprocessing
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
import open3d as o3d
import numpy as np
import laspy
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BasalPointAlgorithm:
    def __init__(self, pcd):
        self.pcd = pcd
        self.vis = None
        self.current_dense_points = []

    def visualize_progress(self, dense_points):
        """Visualize the current progress of basal point estimation"""
        logging.debug(f"Visualizing progress with {len(dense_points)} points")
        
        # Create colors array (grey for all points, red for dense points)
        colors = np.full((len(self.pcd.points), 3), [0.5, 0.5, 0.5])
        
        # Find indices of dense points in the point cloud
        points = np.asarray(self.pcd.points)
        tree = cKDTree(points)
        _, indices = tree.query(dense_points)
        
        # Color dense points red
        colors[indices] = [1, 0, 0]
        
        # Update point cloud visualization
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(points)
        pcd_vis.colors = o3d.utility.Vector3dVector(colors)
        pcd_vis.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
        )
        
        # Show the visualization
        o3d.visualization.draw_geometries([pcd_vis])

    def smooth_path(self, points, smoothing_factor=0.1):
        """
        Smooth the path using spline interpolation.
        """
        # Separate x, y, z coordinates
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        # Fit the spline
        tck, u = splprep([x, y, z], s=smoothing_factor)
        
        # Generate new points
        new_points = np.array(splev(u, tck)).T
        
        return new_points

    def run(self, basal_points, show_progress=False, close_loop=True):
        """
        Run the basal point estimation algorithm
        
        Args:
            basal_points: Can be either indices into the point cloud or actual points
            show_progress: Boolean to control visualization
            close_loop: Boolean to control whether to close the loop between last and first point
        """
        logging.debug("Starting basal point estimation")
        points = np.asarray(self.pcd.points)
        dense_points = []
        visited_points = set()
        
        # Check if basal_points are actual points or indices
        if isinstance(basal_points[0], (int, np.integer)):
            # If they're indices, use them directly
            basal_indices = basal_points
            basal_points_coords = points[basal_indices]
        else:
            # If they're points, use them directly and create corresponding indices
            basal_points_coords = np.array(basal_points)
            # Find closest points in the point cloud for these coordinates
            tree = cKDTree(points)
            _, basal_indices = tree.query(basal_points_coords)

        num_points = len(basal_points_coords)
        end_range = num_points if close_loop else num_points - 1

        MAX_STEP_SIZE = 0.1
        MAX_ITERATIONS = 1000
        MIN_PROGRESS = 0.01

        for i in range(end_range):
            p1 = basal_points_coords[i]
            p2 = basal_points_coords[0] if (i == num_points - 1 and close_loop) else basal_points_coords[i + 1]
            
            pair_dense_points = []
            logging.debug(f"Processing pair {i+1}/{num_points}")

            current_point = p1
            if not dense_points or not np.allclose(current_point, dense_points[-1], atol=1e-3):
                pair_dense_points.append(current_point)
            visited_points.add(tuple(current_point))

            iteration_count = 0
            initial_distance = np.linalg.norm(p2 - current_point)
            previous_distance = initial_distance

            while not np.allclose(current_point, p2, atol=1e-3):
                iteration_count += 1
                if iteration_count > MAX_ITERATIONS:
                    break

                direction = p2 - current_point
                distance_to_p2 = np.linalg.norm(direction)

                if distance_to_p2 < 1e-6:
                    break
                
                direction /= distance_to_p2
                vectors = points - current_point
                projections = np.dot(vectors, direction)
                
                mask = (projections > 0) & (projections < min(distance_to_p2, MAX_STEP_SIZE))
                candidates = points[mask]

                if len(candidates) == 0:
                    break

                distances_to_target = np.linalg.norm(candidates - p2, axis=1)
                distances_to_current = np.linalg.norm(candidates - current_point, axis=1)
                
                progress_mask = distances_to_target < (distance_to_p2 - MIN_PROGRESS)
                if not any(progress_mask):
                    break
                    
                candidates = candidates[progress_mask]
                distances_to_current = distances_to_current[progress_mask]
                
                sorted_indices = np.argsort(distances_to_current)

                for idx in sorted_indices:
                    next_point = candidates[idx]
                    if tuple(next_point) not in visited_points:
                        break
                else:
                    break

                new_distance = np.linalg.norm(next_point - p2)
                if new_distance >= previous_distance:
                    break
                previous_distance = new_distance

                pair_dense_points.append(next_point)
                visited_points.add(tuple(next_point))
                current_point = next_point

                # # Show progress if requested
                # if show_progress and len(pair_dense_points) % 10 == 0:  # Update every 10 points
                #     self.visualize_progress(np.array(dense_points + pair_dense_points))

            if not np.allclose(current_point, p2, atol=1e-3):
                pair_dense_points.append(p2)
                
            dense_points.extend(pair_dense_points)

            # Show progress after completing each pair if requested
            if show_progress:
                self.visualize_progress(np.array(dense_points))

        logging.debug(f"Found {len(dense_points)} dense points")
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
