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
from basal_points_algo import (
    BasalPointSelection,
    BasalPointAlgorithm,
)


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
    if self.process:
        self.process.terminate()
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


# Region growing segmentation
def region_growing(self, pcd, rock_seeds, pedestal_seeds):
    """
    Run the region growing segmentation on the point cloud using the selected seeds.
    """
    from RegionGrowing import RegionGrowingSegmentation

    rock_seed_indices = list(map(int, rock_seeds))
    pedestal_seed_indices = list(map(int, pedestal_seeds))

    self.segmenter = RegionGrowingSegmentation(
        pcd,
        downsample=False,
        rock_seeds=rock_seed_indices,
        pedestal_seeds=pedestal_seed_indices,
    )

    segmented_pcd, labels = self.segmenter.segment()
    self.segmenter.conditional_label_propagation()
    labels[labels == -1] = 1
    colored_pcd = self.segmenter.color_point_cloud()

    return colored_pcd


# Main window class for the GUI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Region Growing with PyQt and Open3D")
        self.pcd = None
        self.process = None
        self.rock_seeds = []
        self.pedestal_seeds = []
        self.poc_points = []
        self.basal_point_selection = (
            None  # Placeholder for BasalPointSelection instance
        )
        self.basal_points = []  # Store basal points for multiple estimations

        self.init_ui()

    def init_ui(self):
        """
        Initialize the UI components.
        """
        layout = QVBoxLayout()

        # Button to load a LAS file
        self.load_button = QPushButton("Load LAS File")
        self.load_button.clicked.connect(self.load_las_file)
        layout.addWidget(self.load_button)

        # Button to continue to seed selection
        self.continue_button = QPushButton("Continue to Select Seeds")
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self.continue_to_select_seeds)
        layout.addWidget(self.continue_button)

        # Button to manually select seeds
        self.manual_selection_button = QPushButton("Select Seeds Manually")
        self.manual_selection_button.setVisible(False)
        self.manual_selection_button.clicked.connect(self.start_manual_selection)
        layout.addWidget(self.manual_selection_button)

        # Button to continue with the selected seeds
        self.continue_with_seeds_button = QPushButton("Continue with Selected Seeds")
        self.continue_with_seeds_button.setVisible(False)
        self.continue_with_seeds_button.clicked.connect(
            self.run_region_growing_with_selected_seeds
        )
        layout.addWidget(self.continue_with_seeds_button)

        # Label to show instructions
        self.instructions_label = QLabel("")
        layout.addWidget(self.instructions_label)

        # Button to proceed to the next step
        self.next_button = QPushButton("Next")
        self.next_button.setVisible(False)
        self.next_button.clicked.connect(self.select_pedestal_seeds)
        layout.addWidget(self.next_button)

        # Button to run the region growing algorithm
        self.run_button = QPushButton("Run Region Growing")
        self.run_button.setVisible(False)
        self.run_button.clicked.connect(self.run_region_growing)
        layout.addWidget(self.run_button)

        # Button to input points of contact
        self.input_poc_button = QPushButton("Input Points of Contact")
        self.input_poc_button.setVisible(False)
        self.input_poc_button.clicked.connect(self.input_point_of_contacts)
        layout.addWidget(self.input_poc_button)

        # Button to estimate basal points
        self.estimate_basal_points_button = QPushButton("Estimate Basal Points")
        self.estimate_basal_points_button.setVisible(False)
        self.estimate_basal_points_button.clicked.connect(self.estimate_basal_points)
        layout.addWidget(self.estimate_basal_points_button)

        # Button to add more basal points
        self.add_more_basal_points_button = QPushButton("Add More Basal Points")
        self.add_more_basal_points_button.setVisible(False)
        self.add_more_basal_points_button.clicked.connect(self.add_more_basal_points)
        layout.addWidget(self.add_more_basal_points_button)

        # Button to save the segmented point cloud
        self.save_pcd_button = QPushButton("Save Point Cloud")
        self.save_pcd_button.setVisible(False)
        self.save_pcd_button.clicked.connect(self.save_point_cloud)
        layout.addWidget(self.save_pcd_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_las_file(self):
        """
        Load a LAS file and display the point cloud.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open LAS File",
            "",
            "LAS Files (*.las);;All Files (*)",
            options=options,
        )
        if file_name:
            self.pcd, _ = load_las_as_open3d_point_cloud(self, file_name)
            points = np.asarray(self.pcd.points)
            colors = np.asarray(self.pcd.colors)

            if self.process:
                self.process.terminate()

            self.process = multiprocessing.Process(
                target=show_point_cloud, args=(points, colors)
            )
            self.process.start()

            self.continue_button.setEnabled(True)

    def continue_to_select_seeds(self):
        """
        Continue to seed selection after loading the point cloud.
        """
        self.continue_button.setEnabled(False)
        voxel_size = 0.01
        self.pcd = self.pcd.voxel_down_sample(voxel_size)

        # Seed selection logic
        points = np.asarray(self.pcd.points)
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        centroid_x = (min_bound[0] + max_bound[0]) / 2.0
        centroid_y = (min_bound[1] + max_bound[1]) / 2.0

        distances = np.linalg.norm(
            points[:, :2] - np.array([centroid_x, centroid_y]), axis=1
        )
        highest_point_index = np.argmax(points[:, 2] - distances)  # seed for rock
        bottommost_point_index = np.argmin(points[:, 2])  # seed for pedestal

        self.rock_seeds = [highest_point_index]
        self.pedestal_seeds = [bottommost_point_index]

        colors = np.full(points.shape, [0.5, 0.5, 0.5])

        # Highlight seeds in different colors
        colors[highest_point_index] = [1, 0, 0]  # Red for rock
        colors[bottommost_point_index] = [0, 0, 1]  # Blue for pedestal

        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors)

        if self.process:
            self.process.terminate()

        self.process = multiprocessing.Process(
            target=show_point_cloud, args=(points, colors)
        )
        self.process.start()

        # Make buttons for manual seed selection and continuing with selected seeds visible
        self.manual_selection_button.setVisible(True)
        self.continue_with_seeds_button.setVisible(True)
        self.continue_button.setVisible(False)

    def get_selected_points_close_window(self):
        """
        Retrieve selected points and close the visualization window.
        """
        selected_points = self.seed_selection_vis.get_picked_points()
        if self.seed_selection_vis:
            self.seed_selection_vis.destroy_window()
            self.seed_selection_vis = None

        return selected_points

    def start_manual_selection(self):
        """
        Start manual selection of seeds.
        """
        self.hide_buttons_except_next()
        self.instructions_label.setText(
            "Currently selecting seeds for Rock.\n \n"
            "1) Please pick points using [shift + left click].\n"
            "2) Press [shift + right click] to undo point picking.\n"
            "3) After picking points, press 'Next' to select Pedestal seeds."
        )
        self.next_button.setVisible(True)
        self.seed_selection_vis = None
        if self.process:
            self.process.terminate()

        pick_points(self, self.pcd)

    def select_pedestal_seeds(self):
        """
        Select pedestal seeds after rock seeds have been selected.
        """
        self.next_button.setVisible(False)
        self.rock_seeds = self.get_selected_points_close_window()
        self.instructions_label.setText(
            "Currently selecting seeds for Pedestal.\n \n"
            "1) Please pick points using [shift + left click].\n"
            "2) Press [shift + right click] to undo point picking.\n"
            "3) After picking points, press the below button to Run Region Growing."
        )
        self.run_button.setVisible(True)
        if self.process:
            self.process.terminate()

        pick_points(self, self.pcd)

    def run_region_growing(self):
        """
        Run the region growing algorithm with the selected seeds.
        """
        self.pedestal_seeds = self.get_selected_points_close_window()
        self.instructions_label.setText("Running Region Growing. Please wait...")
        QApplication.processEvents()

        colored_pcd = region_growing(
            self, self.pcd, self.rock_seeds, self.pedestal_seeds
        )
        self.instructions_label.setText("")
        self.input_poc_button.setVisible(True)
        self.save_pcd_button.setVisible(True)
        self.process = multiprocessing.Process(
            target=show_point_cloud,
            args=(np.asarray(colored_pcd.points), np.asarray(colored_pcd.colors)),
        )
        self.process.start()

    def run_region_growing_with_selected_seeds(self):
        """
        Run the region growing algorithm with automatically selected seeds.
        """
        if self.process:
            self.process.terminate()
        self.hide_all_buttons()
        self.instructions_label.setText("Running Region Growing. Please wait...")
        QApplication.processEvents()

        colored_pcd = region_growing(
            self, self.pcd, self.rock_seeds, self.pedestal_seeds
        )
        self.instructions_label.setText("")
        self.input_poc_button.setVisible(True)
        self.save_pcd_button.setVisible(True)
        self.process = multiprocessing.Process(
            target=show_point_cloud,
            args=(np.asarray(colored_pcd.points), np.asarray(colored_pcd.colors)),
        )
        self.process.start()

    def input_point_of_contacts(self):
        """
        Input points of contact for the region growing algorithm.
        """
        if self.process:
            self.process.terminate()
        self.hide_buttons_for_poc()
        self.instructions_label.setText(
            "Selecting Points of Contact.\n \n"
            "1) Please pick points using [shift + left click].\n"
            "2) Press [shift + right click] to undo point picking.\n"
            "3) After picking points, press 'Next' to estimate basal points."
        )
        self.estimate_basal_points_button.setVisible(True)
        if self.process:
            self.process.terminate()
        pick_points(self, self.pcd)

    def estimate_basal_points(self):
        """
        Estimate basal points based on the selected points of contact.
        """
        self.poc_points = self.get_selected_points_close_window()
        self.instructions_label.setText("Estimating basal points. Please wait...")
        QApplication.processEvents()

        # Use BasalPointAlgorithm to estimate the basal points
        algorithm = BasalPointAlgorithm(self.pcd)
        basal_points = algorithm.run(self.poc_points)

        # Add the newly estimated basal points to the list
        self.basal_points = basal_points

        # Highlight the basal points in the point cloud
        self.highlight_points(self.basal_points)

        self.instructions_label.setText(
            "Basal points estimation completed. Would you like to add more basal points or run region growing again?"
        )
        self.estimate_basal_points_button.setVisible(False)
        self.add_more_basal_points_button.setVisible(True)
        self.run_button.setVisible(True)
        # self.save_pcd_button.setVisible(True)
        show_point_cloud(np.asarray(self.pcd.points), np.asarray(self.pcd.colors))

    def add_more_basal_points(self):
        self.add_more_basal_points_button.setVisible(False)
        self.run_button.setVisible(False)
        self.input_point_of_contacts()

    def highlight_points(self, points):
        """
        Highlight the points in the point cloud.
        """
        colors = np.full((len(self.pcd.points), 3), [0.5, 0.5, 0.5])
        for point in points:
            idx = np.argmin(np.linalg.norm(np.asarray(self.pcd.points) - point, axis=1))
            colors[idx] = [1, 0, 0]  # Red for basal points

        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors)

        if self.process:
            self.process.terminate()

        self.process = multiprocessing.Process(
            target=show_point_cloud, args=(points, colors)
        )
        self.process.start()

    def save_point_cloud(self):
        """
        Save the segmented point cloud to a LAS file.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Point Cloud",
            "",
            "LAS Files (*.las);;All Files (*)",
            options=options,
        )
        if file_name:
            if not file_name.lower().endswith(".las"):
                file_name += ".las"
            header = laspy.LasHeader(point_format=3, version="1.2")
            las = laspy.LasData(header)

            points = np.asarray(self.pcd.points)

            # Undo the recentering
            points[:, 0] += self.x_mean
            points[:, 1] += self.y_mean
            points[:, 2] += self.z_mean

            las.x = points[:, 0]
            las.y = points[:, 1]
            las.z = points[:, 2]

            colors = (np.asarray(self.pcd.colors) * 65535).astype(np.uint16)
            las.red = colors[:, 0]
            las.green = colors[:, 1]
            las.blue = colors[:, 2]

            labels = np.asarray(self.segmenter.labels, dtype=np.uint16)
            las.intensity = labels

            las.write(file_name)

    def hide_buttons_except_next(self):
        """
        Hide all buttons except the 'Next' button.
        """
        self.load_button.setVisible(False)
        self.continue_button.setVisible(False)
        self.manual_selection_button.setVisible(False)
        self.continue_with_seeds_button.setVisible(False)
        self.next_button.setVisible(True)
        self.run_button.setVisible(False)

    def hide_buttons_for_poc(self):
        """
        Hide all buttons except those needed for point of contact selection.
        """
        self.load_button.setVisible(False)
        self.continue_button.setVisible(False)
        self.manual_selection_button.setVisible(False)
        self.continue_with_seeds_button.setVisible(False)
        self.next_button.setVisible(False)
        self.run_button.setVisible(False)
        self.input_poc_button.setVisible(False)
        self.save_pcd_button.setVisible(False)

    def hide_all_buttons(self):
        """
        Hide all buttons.
        """
        self.load_button.setVisible(False)
        self.continue_button.setVisible(False)
        self.manual_selection_button.setVisible(False)
        self.continue_with_seeds_button.setVisible(False)
        self.next_button.setVisible(False)
        self.run_button.setVisible(False)
        self.input_poc_button.setVisible(False)
        self.save_pcd_button.setVisible(False)

    def closeEvent(self, event):
        """
        Handle the close event to terminate any running processes.
        """
        if self.process:
            self.process.terminate()
        super().closeEvent(event)


# Main entry point for the application
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
