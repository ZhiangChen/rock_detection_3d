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
    QSlider,
    QHBoxLayout,
    QFormLayout,
)
from PyQt5.QtCore import Qt
import open3d as o3d
import numpy as np
import laspy
from basal_points_algo import (
    BasalPointSelection,
    BasalPointAlgorithm,
)
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import os
import time
from geomdl import BSpline
from geomdl import utilities

def show_point_cloud(points_or_mesh_path, colors=None, is_mesh=False, seed_points=None, point_show_normal=False):
    """
    Visualize the point cloud or mesh using Open3D.
    """
    geometries = []
    
    if not is_mesh:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_or_mesh_path)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Estimate normals for the point cloud to enhance visualization
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
        )
        geometries.append(pcd)
        
        # Add spheres for seed points if provided
        if seed_points is not None:
            for point, color in seed_points:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                sphere.translate(point)
                sphere.paint_uniform_color(color)
                geometries.append(sphere)
    else:
        # Load the mesh from the file
        geometry = o3d.io.read_triangle_mesh(points_or_mesh_path)
        geometries.append(geometry)
        
    o3d.visualization.draw_geometries(geometries, point_show_normal = point_show_normal)

# Point picking function
def show_point_cloud_picking(points, colors, queue, close_event):
    """
    Show point cloud for picking points and send picked points through queue.
    """
    try:
        # Create visualization window
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()

        # Create point cloud and add to visualizer
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals for better visualization
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
        )
        
        vis.add_geometry(pcd)
        
        # Create a list to store picked points in order
        picked_points = []
        
        def pick_points_callback(vis):
            if close_event.is_set():
                # If close event is set, get current picked points and close window
                picked = vis.get_picked_points()
                picked_points.extend([p for p in picked if p not in picked_points])
                queue.put(picked_points)
                vis.close()
                return True
            
            picked = vis.get_picked_points()
            if picked and picked[-1] not in picked_points:
                picked_points.append(picked[-1])
            return False
        
        # Register the callback
        vis.register_animation_callback(pick_points_callback)
        
        # Run the visualizer
        vis.run()
        
        # Clean up
        vis.destroy_window()
        
    except Exception as e:
        print(f"Error in point picking visualization: {e}")
        queue.put([])  # Send empty list in case of error


# Load LAS file and convert to Open3D point cloud
def load_las_as_open3d_point_cloud(self, las_file_path, evaluate=False):
    """
    Load a LAS file and convert it to an Open3D point cloud.
    """

    # Read LAS file using laspy
    pc = laspy.read(las_file_path)
    x, y, z = pc.x, pc.y, pc.z
    ground_truth_labels = None

    # Check if ground truth labels are available for evaluation
    if evaluate and "Original cloud index" in pc.point_format.dimension_names:
        ground_truth_labels = np.int_(pc["Original cloud index"])

    # Store the mean values for recentering later
    self.x_mean = np.mean(x)
    self.y_mean = np.mean(y)
    self.z_mean = np.mean(z)

    # Recenter the point cloud
    xyz = np.vstack((x - self.x_mean, y - self.y_mean, z - self.z_mean)).transpose()

    # Check if RGB color information is available in the LAS file
    if all(dim in pc.point_format.dimension_names for dim in ["red", "green", "blue"]):
        r = np.uint8(pc.red / 65535.0 * 255)
        g = np.uint8(pc.green / 65535.0 * 255)
        b = np.uint8(pc.blue / 65535.0 * 255)
        rgb = np.vstack((r, g, b)).transpose() / 255.0
    else:
        rgb = np.zeros((len(x), 3))

    # Create Open3D PointCloud object and set points and colors
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

    # Convert seed indices to integers
    rock_seed_indices = list(map(int, rock_seeds))
    pedestal_seed_indices = list(map(int, pedestal_seeds))

    # Initialize the region growing segmentation with the selected seeds and basal points
    print('---------', self.basal_points)
    self.segmenter = RegionGrowingSegmentation(
        pcd,
        downsample=False,
        smoothness_threshold=self.smoothness_threshold,
        distance_threshold=0.05,
        curvature_threshold=self.curvature_threshold,
        rock_seeds=rock_seed_indices,
        pedestal_seeds=pedestal_seed_indices,
        basal_points=np.asarray(pcd.points)[self.basal_points] if np.any(self.basal_points) else None,
        basal_proximity_threshold=self.basal_proximity_threshold 
    )

    # Segment the point cloud and perform conditional label propagation
    segmented_pcd, labels = self.segmenter.segment()
    #if not np.any(self.basal_points):
    self.segmenter.conditional_label_propagation()

    # Assign a default label for unlabeled points
    labels[labels == -1] = 1

    # Color the segmented point cloud
    colored_pcd = self.segmenter.color_point_cloud()

    # Highlight proximity points if basal points are available
    # if np.any(self.basal_points):
    #     colored_pcd = self.segmenter.highlight_proximity_points(colored_pcd)

    return colored_pcd


# Main window class for the GUI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Region Growing with PyQt and Open3D")
        self.pcd = None
        self.process = None
        self.rock_seeds = None
        self.pedestal_seeds = None
        self.poc_points = []
        self.basal_point_selection = None
        self.seed_selection_vis = None
        self.basal_points = None
        self.point_pick_queue = None 
        self.close_picking_event = None

        self.smoothness_threshold = 0.99  # Initial default value for the smoothness threshold (0-1)
        self.curvature_threshold = 0.15   # Initial default value for the curvature threshold (0-1)
        self.basal_proximity_threshold = 0.05  # Default value for basal proximity threshold (0-1)

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
        self.continue_with_seeds_button.clicked.connect(lambda: self.show_sliders(smoothness_threshold=0.99, curvature_threshold=0.15))
        layout.addWidget(self.continue_with_seeds_button)

        # Label to show instructions
        self.instructions_label = QLabel("")
        layout.addWidget(self.instructions_label)

        # Button to proceed to the next step
        self.next_button = QPushButton("Next")
        self.next_button.setVisible(False)
        self.next_button.clicked.connect(self.select_pedestal_seeds)
        layout.addWidget(self.next_button)

        self.slider_layout = QFormLayout()

        # Descriptions for sliders
        smoothness_description = QLabel("Controls surface smoothness variation; higher values include only smoother points.\n")
        curvature_description = QLabel("Sets the curvature limit; higher values allow more curved points.\n")
        #proximity_description = QLabel("Controls how close points should be to basal points to stop region growing.\n")


        # Smoothness Threshold Slider
        smoothness_slider_layout = QHBoxLayout()
        self.smoothness_slider = QSlider(Qt.Horizontal)
        self.smoothness_slider.setRange(0, 100)  # Slider range 0-100, but mapped to 0-1
        self.smoothness_slider.setValue(int(self.smoothness_threshold * 100))
        self.smoothness_slider.setMinimumWidth(300)
        self.smoothness_slider.valueChanged.connect(self.update_smoothness_threshold)
        smoothness_value_label = QLabel(f"{self.smoothness_threshold:.2f}")
        self.smoothness_slider.valueChanged.connect(lambda: smoothness_value_label.setText(f"{self.smoothness_threshold:.2f}"))
        smoothness_slider_layout.addWidget(self.smoothness_slider)
        smoothness_slider_layout.addWidget(smoothness_value_label)
        self.slider_layout.addRow("Smoothness Threshold", smoothness_slider_layout)
        self.slider_layout.addRow(smoothness_description)

        # Curvature Threshold Slider
        curvature_slider_layout = QHBoxLayout()
        self.curvature_slider = QSlider(Qt.Horizontal)
        self.curvature_slider.setRange(0, 100)  # Slider range 0-100, but mapped to 0-1
        self.curvature_slider.setValue(int(self.curvature_threshold * 100))
        self.curvature_slider.setMinimumWidth(300)
        self.curvature_slider.valueChanged.connect(self.update_curvature_threshold)
        curvature_value_label = QLabel(f"{self.curvature_threshold:.2f}")
        self.curvature_slider.valueChanged.connect(lambda: curvature_value_label.setText(f"{self.curvature_threshold:.2f}"))
        curvature_slider_layout.addWidget(self.curvature_slider)
        curvature_slider_layout.addWidget(curvature_value_label)
        self.slider_layout.addRow("Curvature Threshold", curvature_slider_layout)
        self.slider_layout.addRow(curvature_description)

       
        # Store the proximity slider label, slider, and proximity value label references
        self.proximity_slider_label = QLabel("Basal Proximity Threshold")

        proximity_slider_layout = QHBoxLayout()
        self.proximity_slider = QSlider(Qt.Horizontal)
        self.proximity_slider.setRange(0, 100)  # Slider range 0-100, but mapped to 0-1
        self.proximity_slider.setValue(int(self.basal_proximity_threshold * 100))
        self.proximity_slider.setMinimumWidth(300)
        self.proximity_slider.valueChanged.connect(self.update_basal_proximity_threshold)

        # Store reference to the value label that shows the slider's current value
        self.proximity_value_label = QLabel(f"{self.basal_proximity_threshold:.2f}")
        self.proximity_slider.valueChanged.connect(lambda: self.proximity_value_label.setText(f"{self.basal_proximity_threshold:.2f}"))

        # Add the slider and value label to the layout
        proximity_slider_layout.addWidget(self.proximity_slider)
        proximity_slider_layout.addWidget(self.proximity_value_label)

        # Add the proximity slider label and layout to the form layout
        self.slider_layout.addRow(self.proximity_slider_label, proximity_slider_layout)




        slider_widget = QWidget()
        slider_widget.setLayout(self.slider_layout)
        slider_widget.setVisible(False)
        self.slider_widget = slider_widget

        layout.addWidget(slider_widget)

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

        # Button to reselect basal points
        self.add_more_basal_points_button = QPushButton("Reselect Basal Points")
        self.add_more_basal_points_button.setVisible(False)
        self.add_more_basal_points_button.clicked.connect(self.add_more_basal_points)
        layout.addWidget(self.add_more_basal_points_button)


        # Add new button for rerunning region growing
        self.rerun_region_growing_button = QPushButton("Rerun Region Growing with Different Parameters")
        self.rerun_region_growing_button.setVisible(False)
        self.rerun_region_growing_button.clicked.connect(self.show_parameter_adjustment_window)
        layout.addWidget(self.rerun_region_growing_button)

        # Button to save the segmented point cloud
        self.save_pcd_button = QPushButton("Save Point Cloud")
        self.save_pcd_button.setVisible(False)
        self.save_pcd_button.clicked.connect(self.save_point_cloud)
        layout.addWidget(self.save_pcd_button)

        # Button to reconstruct mesh
        self.reconstruct_mesh_button = QPushButton("Reconstruct Mesh")
        self.reconstruct_mesh_button.setVisible(False)
        self.reconstruct_mesh_button.clicked.connect(self.reconstruct_mesh)
        layout.addWidget(self.reconstruct_mesh_button)

        # Button to save mesh
        self.save_mesh_button = QPushButton("Save Mesh")
        self.save_mesh_button.setVisible(False)
        self.save_mesh_button.clicked.connect(self.save_mesh)
        layout.addWidget(self.save_mesh_button)

        # Add new button for jumping to basal line selection
        self.jump_to_basal_button = QPushButton("Jump to Basal Line Selection")
        self.jump_to_basal_button.setVisible(False)
        self.jump_to_basal_button.clicked.connect(self.input_point_of_contacts)
        layout.addWidget(self.jump_to_basal_button)

        # Set the main layout and central widget for the window
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def update_smoothness_threshold(self, value):
        print("update_smoothness_threshold", value)
        self.smoothness_threshold = value / 100.0  # Scale slider value to 0-1

    def update_curvature_threshold(self, value):
        self.curvature_threshold = value / 100.0

    def update_basal_proximity_threshold(self, value):
        self.basal_proximity_threshold = value / 100.0  

    def show_sliders(self, smoothness_threshold=0.99, curvature_threshold=0.15):
        """
        Show sliders for parameter adjustment.
        """
        self.hide_all_buttons()
        self.slider_widget.setVisible(True)

        # Update the smoothness and curvature sliders
        self.smoothness_slider.setValue(int(smoothness_threshold * 100))
        self.curvature_slider.setValue(int(curvature_threshold * 100))

        # Show both the run button and jump to basal button
        self.run_button.setVisible(True)
        #self.jump_to_basal_button.setVisible(True)

        # Only show the proximity slider layout if basal points are available
        if self.basal_points is not None and len(self.basal_points) > 0:
            self.proximity_slider.setValue(int(self.basal_proximity_threshold * 100))
            self.proximity_slider_label.setVisible(True)  # Show label for proximity slider
            self.proximity_slider.setVisible(True)  # Show the proximity slider itself
            self.proximity_value_label.setVisible(True)  # Show the proximity value label
        else:
            self.proximity_slider_label.setVisible(False)
            self.proximity_slider.setVisible(False)
            self.proximity_value_label.setVisible(False)

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
            # Load the LAS file as an Open3D point cloud
            self.pcd, _ = load_las_as_open3d_point_cloud(self, file_name)
            points = np.asarray(self.pcd.points)
            colors = np.asarray(self.pcd.colors)

            # Close any existing visualization windows
            if self.process:
                self.process.terminate()

            # Start a new process to display the point cloud
            self.process = multiprocessing.Process(
                target=show_point_cloud, args=(points, colors)
            )
            self.process.start()
            # Enable the continue button to proceed to seed selection
            self.continue_button.setEnabled(True)

    def continue_to_select_seeds(self):
        """
        Continue to seed selection after loading the point cloud.
        """
        self.continue_button.setEnabled(False)
        voxel_size = 0.01
        self.pcd = self.pcd.voxel_down_sample(voxel_size)

        # Computer selected seeds for rock and pedestal
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
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create seed points list with their positions and colors
        seed_points = [
            (points[highest_point_index], [1, 0, 0]),     # Red sphere for rock
            (points[bottommost_point_index], [0, 0, 1])   # Blue sphere for pedestal
        ]

        if self.process:
            self.process.terminate()

        self.process = multiprocessing.Process(
            target=show_point_cloud, 
            args=(points, colors, False, seed_points)
        )
        self.process.start()

        # Make buttons for manual seed selection and continuing with selected seeds visible
        self.manual_selection_button.setVisible(True)
        self.continue_with_seeds_button.setVisible(True)
        self.continue_button.setVisible(False)
    
    def pick_points(self, pcd):
        """
        Pick points from the point cloud using a separate process.
        """
        # Terminate any existing visualizations before starting point picking
        if self.process:
            self.process.terminate()

        # Create a queue to get the picked points back from the process
        self.point_pick_queue = multiprocessing.Queue()
        
        # Create an event to signal visualization closure
        self.close_picking_event = multiprocessing.Event()

        # Start point picking in a new process
        self.process = multiprocessing.Process(
            target=show_point_cloud_picking,
            args=(
                np.asarray(pcd.points), 
                np.asarray(pcd.colors), 
                self.point_pick_queue,
                self.close_picking_event
            )
        )
        self.process.start()

    def get_selected_points_close_window(self):
        """
        Retrieve selected points from the point picking process.
        """
        selected_points = []
        try:
            # Set the close event to trigger visualization closure
            if self.close_picking_event:
                self.close_picking_event.set()
                
            # Wait for points with timeout
            selected_points = self.point_pick_queue.get(timeout=5.0)
            print("Selected points:", selected_points)
            
        except Exception as e:
            print(f"Warning: Error while getting picked points: {e}")
        finally:
            # Clean up
            if self.process:
                self.process.terminate()
                self.process.join(timeout=1.0)
                self.process = None
            self.close_picking_event = None
        
        return selected_points
    
    def start_manual_selection(self):
        """
        Start manual selection of seeds.
        """
        # Hide all buttons except the 'Next' button to guide the user through the process
        self.hide_buttons_except_next()
        
        # Update the instructions label with steps for selecting rock seeds
        self.instructions_label.setText(
            "Currently selecting seeds for Rock.\n \n"
            "1) Please pick points using [shift + left click].\n"
            "2) Press [shift + right click] to undo point picking.\n"
            "3) After picking points, press 'Next' to select Pedestal seeds."
        )
        self.next_button.setVisible(True)

        # Call the function to allow the user to pick points
        self.pick_points(self.pcd)

    def select_pedestal_seeds(self):
        """
        Select pedestal seeds after rock seeds have been selected.
        """
        # Get the selected rock seeds and close the visualization
        self.rock_seeds = self.get_selected_points_close_window()
        
        # Show sliders and update instructions only if we got valid rock seeds
        if self.rock_seeds:
            self.show_sliders()
            self.instructions_label.setText(
                "Currently selecting seeds for Pedestal.\n \n"
                "1) Please pick points using [shift + left click].\n"
                "2) Press [shift + right click] to undo point picking.\n"
                "3) After picking points, press the below button to Run Region Growing."
            )
            self.run_button.setVisible(True)
            
            # Start new point picking for pedestal seeds
            self.pick_points(self.pcd)
        else:
            # If no rock seeds were selected, show an error message
            self.instructions_label.setText(
                "No rock seeds were selected. Please try selecting rock seeds again."
            )
            # Restart rock seed selection
            self.start_manual_selection()

    def run_region_growing(self):
        """
        Run the region growing algorithm with the selected seeds or with basal information.
        """
        # Hide all buttons except the 'Next' button to guide the user through the process
        self.hide_all_buttons()

        # Check if basal points are available or not
        if not np.any(self.basal_points):
            # If basal points are not available, it means region growing is being run for the first time after manual seed selection,
            # so get and store the selected seeds
            if self.pedestal_seeds is None:
                self.pedestal_seeds = self.get_selected_points_close_window()
        
        if self.seed_selection_vis is not None:
            self.seed_selection_vis.destroy_window()
            self.seed_selection_vis = None

        # Terminate the existing process if it exists
        if self.process:
            self.process.terminate()
            self.process.join(timeout=1.0)
            self.process = None

        self.instructions_label.setText("Running Region Growing. Please wait...")
        QApplication.processEvents()

        # Run the region growing algorithm with the selected seeds
        colored_pcd = region_growing(
            self, self.pcd, self.rock_seeds, self.pedestal_seeds
        )
        self.instructions_label.setText("")

        # If no basal points are present, it means we are running region growing for the first time after manual seed selection,
        # so show the input point of contact button
        if not np.any(self.basal_points):
            self.input_poc_button.setVisible(True)
        
        # Make buttons visible for next steps
        self.save_pcd_button.setVisible(True)
        self.reconstruct_mesh_button.setVisible(True)
        self.rerun_region_growing_button.setVisible(True)  # Show rerun button


        # Start a new process to show the segmented point cloud
        self.process = multiprocessing.Process(
            target=show_point_cloud,
            args=(np.asarray(colored_pcd.points), np.asarray(colored_pcd.colors)),
        )
        self.process.start()

    def input_point_of_contacts(self):
        """
        Input points of contact for the region growing algorithm.
        """
        # Close all the visualization windows before starting point picking
        if self.process:
            self.process.terminate()

        self.hide_all_buttons()
        self.instructions_label.setText(
            "Selecting Points of Contact.\n \n"
            "1) Please pick points using [shift + left click].\n"
            "2) Press [shift + right click] to undo point picking.\n"
            "3) After picking points, press 'Next' to estimate basal points."
        )
        self.estimate_basal_points_button.setVisible(True)

        if self.process:
            self.process.terminate()

        # Call the function to allow the user to pick points
        self.basal_points = None
        self.pick_points(self.pcd)


    def add_more_basal_points(self):
        """
        Allow the user to reselect or add more basal points.
        """
        # Hide the buttons related to basal points and region growing
        self.add_more_basal_points_button.setVisible(False)
        self.run_button.setVisible(False)

        # Allow the user to reselect points of contact
        self.input_point_of_contacts()


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
        points, colors, self.basal_points = self.highlight_points(self.basal_points)

        # Show parameter adjustment window instead of direct slider setup
        self.show_parameter_adjustment_window()

        show_point_cloud(np.asarray(self.pcd.points), np.asarray(self.pcd.colors))

    def highlight_points(self, points):
        """
        Highlight estimated basal points in the point cloud.
        """
        # Convert point cloud and basal points to numpy arrays for efficiency
        pcd_array = np.asarray(self.pcd.points)
        points_array = np.asarray(points)

        # Default color for all points
        colors = np.full((len(pcd_array), 3), [0.5, 0.5, 0.5])
        basal_mask = np.zeros(len(pcd_array), dtype=bool)

        # Create a KDTree for faster spatial queries
        tree = cKDTree(pcd_array)

        # Find the indices of basal points in the point cloud efficiently
        distances, indices = tree.query(points_array, k=1)
        unique_indices = np.unique(indices)

        # Set colors and basal mask for identified points
        colors[unique_indices] = [1, 0, 0]  # Red for basal points
        basal_mask[unique_indices] = True  # Mark these points as basal

        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd_array, colors, basal_mask

    def save_point_cloud(self, plain = False):
        """
        Saves the point cloud to a LAS file with color-coded labels.
        Colors: Blue (pedestal), Red (rock), Green (basal points)
        
        Args:
            plain (bool): If True, saves all points in red without classification colors
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

            points = np.asarray(self.pcd.points)
            labels = np.asarray(self.segmenter.labels)

            # Assign intensity based on labels
            intensity = labels.copy()
            if not plain:
                # Prepare colors for rock, pedestal, and basal points
                colors = np.zeros((points.shape[0], 3), dtype=np.float64)
                colors[labels == 0] = [0, 0, 1]  # Blue for pedestal
                colors[labels == 1] = [1, 0, 0]  # Red for rock
            
                if self.basal_points is None:
                    self.basal_points = self.detect_basal_points_optimized(points, labels)
                    print(f"Detected {np.sum(self.basal_points)} basal points.")

                if self.basal_points is not None:
                    basal_mask = np.asarray(self.basal_points, dtype=bool)
                    # Label basal points with intensity 2
                    intensity[basal_mask] = 2
                    colors[basal_mask] = [0, 1, 0]  # Green for basal points
                    basal_count = np.sum(basal_mask)
                else:
                    basal_count = 0

                print(f"Labeled {basal_count} basal points in the point cloud.")
            else:
                colors = np.zeros((points.shape[0], 3), dtype=np.float64)
                colors[:] = [1, 0, 0]

            # Undo the recentering
            points[:, 0] += self.x_mean
            points[:, 1] += self.y_mean
            points[:, 2] += self.z_mean

            # Create a new LAS file
            header = laspy.LasHeader(point_format=3, version="1.2")
            las = laspy.LasData(header)

            # Assign coordinates and intensity
            las.x = points[:, 0]
            las.y = points[:, 1]
            las.z = points[:, 2]
            las.intensity = intensity

            # Assign colors
            colors = (colors * 65535).astype(np.uint16)
            las.red = colors[:, 0]
            las.green = colors[:, 1]
            las.blue = colors[:, 2]

            # Write the LAS file
            las.write(file_name)


    @staticmethod
    def detect_basal_points_optimized(points, labels, k=30, threshold=0.35):
        """
        Detects basal points using neighborhood analysis.
        """
        tree = cKDTree(points)
        distances, indices = tree.query(points, k=k)
        
        neighborhood_labels = labels[indices]
        rock_ratios = np.sum(neighborhood_labels == 1, axis=1) / k
        
        basal_points = (threshold <= rock_ratios) & (rock_ratios <= (1 - threshold))
        return basal_points

    def reconstruct_mesh(self):
        """
        Reconstructs a 3D mesh from the segmented point cloud.
        Process:
        1. Detects basal points if not already detected
        2. Filters points to keep rock and basal points
        3. Generates bottom face using NURBS interpolation
        4. Performs Poisson reconstruction
        5. Visualizes the resulting mesh
        """
        self.hide_all_buttons()
        self.instructions_label.setText("Reconstructing mesh. Please wait...")
        QApplication.processEvents()

        # Perform basal detection if not available
        if self.basal_points is None:
            points = np.asarray(self.pcd.points)
            labels = np.asarray(self.segmenter.labels)
            self.basal_points = self.detect_basal_points_optimized(points, labels)
            print(f"Detected {np.sum(self.basal_points)} basal points.")

        # Filter the point cloud to keep only rock and basal points
        rock_points = np.asarray(self.segmenter.labels) == 1
        filtered_indices = np.logical_or(rock_points, self.basal_points)
        filtered_points = np.asarray(self.pcd.points)[filtered_indices]
        filtered_colors = np.asarray(self.pcd.colors)[filtered_indices]

        # Create a new point cloud with only rock and basal points
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

        # Perform open face interpolation
        basal_indices = np.where(self.basal_points[filtered_indices])[0]
        rock_indices = np.where(rock_points[filtered_indices])[0]
        bottom_points = self.generate_bottom_face_points(
            filtered_pcd,
            basal_indices,
            degree_u=3,
            degree_v=3,
            control_points_u=12,
            control_points_v=12
        )
        
        if bottom_points is None:
            print("Failed to generate bottom face points")
            return

        # Create separate point clouds for rock and bottom
        rock_points = np.asarray(filtered_pcd.points)[rock_indices]
        
        # Process rock points
        rock_pcd = o3d.geometry.PointCloud()
        rock_pcd.points = o3d.utility.Vector3dVector(rock_points)
        rock_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        # Process bottom points
        bottom_pcd = o3d.geometry.PointCloud()
        bottom_pcd.points = o3d.utility.Vector3dVector(bottom_points)
        bottom_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        # Combine points and normals
        combined_points = np.vstack((rock_points, bottom_points))
        combined_normals = np.vstack((
            np.asarray(rock_pcd.normals),
            np.asarray(bottom_pcd.normals)
        ))
        
        # Create final point cloud
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(combined_points)
        new_pcd.normals = o3d.utility.Vector3dVector(combined_normals)
        
        # Get the center point for consistent orientation
        center = new_pcd.get_center()
        new_pcd.orient_normals_towards_camera_location(center)
        new_pcd.normals = o3d.utility.Vector3dVector(-np.asarray(new_pcd.normals))
        


        # Change the color of all the points in interpolated_pcd to red
        new_pcd.paint_uniform_color([1, 0, 0])

        # Reconstruct and save mesh
        self.reconstructed_mesh = self.poisson_reconstruction(new_pcd)
        self.instructions_label.setText("Mesh reconstruction completed.")
        self.save_mesh_button.setVisible(True)
        QApplication.processEvents()

        # Visualize mesh
        if self.process:
            self.process.terminate()
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
            self.temp_mesh_path = temp_file.name
        o3d.io.write_triangle_mesh(self.temp_mesh_path, self.reconstructed_mesh)

        self.process = multiprocessing.Process(
            target=show_point_cloud, 
            args=(self.temp_mesh_path, None, True)
        )
        self.process.start()

    @staticmethod
    def generate_bottom_face_points(pcd, basal_indices, degree_u=3, degree_v=3, 
                                  control_points_u=8, control_points_v=8):
        """
        Generates points for the bottom face using NURBS surface fitting.
        """
        points = np.asarray(pcd.points)
        basal_points = points[basal_indices]
        
        # Calculate transformation matrix for 2D projection
        center = np.mean(basal_points, axis=0)
        centered_points = basal_points - center
        U, S, Vh = np.linalg.svd(centered_points)
        normal = Vh[2]
        
        # Create transformation matrix
        u = np.cross(normal, [0, 0, 1])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(normal, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        transform_matrix = np.vstack((u, v)).T

        def create_boundary_grid(basal_points, num_u, num_v):
            """Create a grid that maintains connection with basal points"""
            # Project points to 2D
            points_2d = np.dot(basal_points - center, transform_matrix)
            
            # Calculate bounds
            x_min, x_max = np.min(points_2d[:, 0]), np.max(points_2d[:, 0])
            y_min, y_max = np.min(points_2d[:, 1]), np.max(points_2d[:, 1])
            
            # Create regular grid
            x_grid = np.linspace(x_min, x_max, num_u)
            y_grid = np.linspace(y_min, y_max, num_v)
            xx, yy = np.meshgrid(x_grid, y_grid)
            
            # Initialize grid points
            grid_points = np.zeros((num_u * num_v, 3))
            
            # Build KD-tree for nearest neighbor search
            tree = cKDTree(points_2d)
            
            # For each grid point, interpolate height from nearest basal points
            for i in range(num_u * num_v):
                x, y = xx.flatten()[i], yy.flatten()[i]
                query_point = np.array([x, y])
                
                # Find nearest neighbors
                distances, indices = tree.query(query_point, k=8)
                
                # Calculate weights based on distance
                weights = 1.0 / (distances + 1e-10)**2
                weights = weights / np.sum(weights)
                
                # Get actual 3D points
                nearest_points = basal_points[indices]
                
                # Interpolate position using weighted average
                interpolated_point = np.sum(nearest_points * weights[:, np.newaxis], axis=0)
                
                # Store the interpolated point
                grid_points[i] = interpolated_point - center
            
            return grid_points.reshape(num_u, num_v, 3).tolist()

        def ensure_boundary_connection(surface_points, basal_points):
            """Ensure interpolated points connect with basal points"""
            tree = cKDTree(basal_points)
            
            # Find nearest basal point for each surface point
            distances, indices = tree.query(surface_points, k=1)
            
            # Calculate mean distance to determine connection threshold
            mean_dist = np.mean(distances)
            connection_threshold = mean_dist * 0.5
            
            # Adjust points that are close to basal points
            adjusted_points = surface_points.copy()
            for i in range(len(surface_points)):
                if distances[i] < connection_threshold:
                    # Smoothly blend between surface and basal point
                    blend_factor = distances[i] / connection_threshold
                    adjusted_points[i] = (blend_factor * surface_points[i] + 
                                        (1 - blend_factor) * basal_points[indices[i]])
            
            return adjusted_points

        # Create initial grid with boundary connection
        print("Creating boundary-aware grid...")
        grid_points = create_boundary_grid(basal_points, control_points_u, control_points_v)
        
        # Fit NURBS surface
        print("Fitting NURBS surface...")
        try:
            surf = BSpline.Surface()
            surf.degree_u = min(degree_u, control_points_u - 1)
            surf.degree_v = min(degree_v, control_points_v - 1)
            surf.ctrlpts_size_u = control_points_u
            surf.ctrlpts_size_v = control_points_v
            
            # Flatten grid points for NURBS
            ctrlpts_list = [point for row in grid_points for point in row]
            surf.ctrlpts = ctrlpts_list
            
            # Generate knot vectors
            surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, surf.ctrlpts_size_u)
            surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, surf.ctrlpts_size_v)
            
        except Exception as e:
            print(f"Error in NURBS fitting: {str(e)}")
            return None

        # Generate surface points
        print("Generating surface points...")
        try:
            surf.delta = 0.02  # Finer sampling
            surf.evaluate()
            surface_points = np.array(surf.evalpts)
            
            # Transform points back to original coordinate system
            surface_points = surface_points + center
            
            # Ensure connection with basal points
            surface_points = ensure_boundary_connection(surface_points, basal_points)
            
            # Add additional points near the boundary
            tree = cKDTree(basal_points)
            boundary_points = []
            
            for basal_point in basal_points:
                # Add points slightly inward from each basal point
                inward_vector = center - basal_point
                inward_vector = inward_vector / np.linalg.norm(inward_vector)
                
                # Add multiple points at different distances
                for d in [0.02, 0.1]:  # Adjust these distances as needed
                    new_point = basal_point + inward_vector * d
                    boundary_points.append(new_point)
            
            # Combine all points
            all_points = np.vstack((surface_points, boundary_points))
            
            return all_points
            
        except Exception as e:
            print(f"Error in surface generation: {str(e)}")
            return None

    @staticmethod
    def poisson_reconstruction(pcd):
        """
        Performs Poisson surface reconstruction on the point cloud.
        """
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, linear_fit=False
        )
        return mesh
    
    def save_mesh(self):
        """
        Saves the reconstructed mesh to a PLY file and cleans up temporary files.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mesh",
            "",
            "PLY Files (*.ply);;All Files (*)",
            options=options,
        )
        if file_name:
            if not file_name.lower().endswith(".ply"):
                file_name += ".ply"
            o3d.io.write_triangle_mesh(file_name, self.reconstructed_mesh)
            self.instructions_label.setText(f"Mesh saved to {file_name}")
            os.unlink(self.temp_mesh_path)

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
        self.reconstruct_mesh_button.setVisible(False)
        self.save_mesh_button.setVisible(False)
        self.rerun_region_growing_button.setVisible(False)  # Hide rerun button
        self.slider_widget.setVisible(False)
        self.estimate_basal_points_button.setVisible(False)
        self.jump_to_basal_button.setVisible(False)  # Add this line

    def closeEvent(self, event):
        """
        Handle the close event to terminate any running processes.
        """
        if self.process:
            self.process.terminate()
        super().closeEvent(event)

    def show_parameter_adjustment_window(self):
        """
        Show window with parameter sliders and relevant buttons for rerunning region growing.
        """
        self.hide_all_buttons()
        
        # Use existing show_sliders function with current parameter values
        self.show_sliders(
            smoothness_threshold=0.9,
            curvature_threshold=0.1
        )
        
        # Show additional buttons
        if np.any(self.basal_points):
            self.add_more_basal_points_button.setVisible(True)
        
        self.instructions_label.setText(
            "Basal points estimation completed. Would you like to add more basal points or run region growing again?"
        )


# Main entry point for the application
if __name__ == "__main__":
    # Set the start method for multiprocessing to 'spawn' (required for some platforms)
    multiprocessing.set_start_method("spawn")

    # Create the application instance and main window
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Start the application's event loop
    sys.exit(app.exec_())
